"""Sanitized, bounded OTLP tracing for investigation/follow-up runs.

Explicit LangChain callback spans only — no global auto-instrumentation.
All exported attributes pass ``trace._redact`` before leaving the process.
Exporter failures never alter application outcomes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import (
    Span,
    Status,
    StatusCode,
    Tracer,
    set_span_in_context,
    use_span,
)

from casepilot.backend.cost_guard import strip_reasoning_message
from casepilot.backend.trace import _redact
from casepilot.shared.config import Settings

logger = logging.getLogger(__name__)

_TRACER_NAME = "casepilot.backend"
_ATTR_RUN_ID = "casepilot.run_id"
_ATTR_CASE_ID = "casepilot.case_id"
_ATTR_ATTEMPT_ID = "casepilot.attempt_id"
_ATTR_MODE = "casepilot.mode"
_ATTR_OUTCOME = "casepilot.outcome"
_ATTR_ERROR = "casepilot.error"
_ATTR_USAGE = "casepilot.usage"
_ATTR_COST_USD = "casepilot.cost_usd"
_ATTR_MODEL_INPUT = "casepilot.model.input"
_ATTR_MODEL_OUTPUT = "casepilot.model.output"
_ATTR_TOOL_NAME = "casepilot.tool.name"
_ATTR_TOOL_INPUT = "casepilot.tool.input"
_ATTR_TOOL_OUTPUT = "casepilot.tool.output"

# Bounded async export (ms / queue depth). Failures must not stall the agent path.
_MAX_QUEUE_SIZE = 512
_SCHEDULE_DELAY_MS = 1000.0
_MAX_EXPORT_BATCH = 64
_EXPORT_TIMEOUT_MS = 2000.0
_OTLP_TIMEOUT_S = 2.0
_SHUTDOWN_FLUSH_MS = 3000

_provider: TracerProvider | None = None
_tracer: Tracer | None = None
_enabled: bool = False
_configured: bool = False


def _is_loopback_endpoint(endpoint: str) -> bool:
    host = urlparse(endpoint).hostname
    return host in {"127.0.0.1", "localhost", "::1"}


def _safe_json(value: Any) -> str:
    """Redact nested payloads then JSON-serialize for span attributes."""
    return json.dumps(_redact(value), default=str)


def _safe_str(value: Any) -> str:
    redacted = _redact(value)
    if isinstance(redacted, str):
        return redacted
    return json.dumps(redacted, default=str)


def _message_public(msg: BaseMessage) -> dict[str, Any]:
    """Serialize a chat message for export (public content only).

    AIMessages pass through ``strip_reasoning_message`` before dump so content-block
    ``type: reasoning`` parts and any ``*reason*`` provider keys never leave the process.
    Nested secrets still go through ``_redact`` via ``_safe_json``.
    """
    public_msg: BaseMessage = strip_reasoning_message(msg) if isinstance(msg, AIMessage) else msg
    if hasattr(public_msg, "model_dump"):
        data = public_msg.model_dump()
    else:
        data = {
            "type": getattr(public_msg, "type", type(public_msg).__name__),
            "content": getattr(public_msg, "content", None),
            "additional_kwargs": getattr(public_msg, "additional_kwargs", {}) or {},
            "response_metadata": getattr(public_msg, "response_metadata", {}) or {},
        }
    # Drop raw provider/id noise; keep public content + tool call structure.
    if isinstance(data, dict):
        data.pop("id", None)
        return data
    return {"content": data}


def _llm_public_outputs(response: LLMResult) -> list[Any]:
    """Public assistant content only — not raw provider response objects."""
    outs: list[Any] = []
    for gens in response.generations or []:
        for gen in gens:
            if isinstance(gen, ChatGeneration) and gen.message is not None:
                outs.append(_message_public(gen.message))
            else:
                text = getattr(gen, "text", None)
                outs.append({"text": text if text is not None else str(gen)})
    return outs


class _SafeExporter(SpanExporter):
    """Swallow exporter errors; log a single diagnostic (no per-span spam)."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner
        self._warned = False

    def export(self, spans: Any) -> SpanExportResult:
        try:
            return self._inner.export(spans)
        except Exception as exc:  # noqa: BLE001 — telemetry must not raise
            if not self._warned:
                logger.warning("tracing export unavailable: %s", exc)
                self._warned = True
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        try:
            self._inner.shutdown()
        except Exception:  # noqa: BLE001
            pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            return bool(self._inner.force_flush(timeout_millis))
        except Exception:  # noqa: BLE001
            return False


def configure_tracing(
    settings: Settings,
    *,
    exporter: SpanExporter | None = None,
) -> None:
    """Install process-local tracer provider. Safe to call more than once.

    ``exporter`` is for tests (e.g. in-memory). Production uses loopback OTLP HTTP.
    Never raises; invalid/non-loopback endpoints disable export.
    """
    global _provider, _tracer, _enabled, _configured

    # Replace prior provider cleanly (tests reconfigure often).
    if _configured:
        shutdown_tracing()

    _configured = True
    if not settings.tracing_enabled:
        _enabled = False
        _provider = None
        _tracer = None
        return

    try:
        if exporter is None:
            endpoint = settings.tracing_otlp_endpoint.strip()
            if not _is_loopback_endpoint(endpoint):
                logger.warning(
                    "tracing disabled: OTLP endpoint must be loopback, got %s",
                    endpoint,
                )
                _enabled = False
                _provider = None
                _tracer = None
                return
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            raw: SpanExporter = OTLPSpanExporter(
                endpoint=endpoint,
                timeout=_OTLP_TIMEOUT_S,
            )
            use_batch = True
        else:
            raw = exporter
            use_batch = False

        safe = _SafeExporter(raw)
        provider = TracerProvider(resource=Resource.create({"service.name": "casepilot-backend"}))
        if use_batch:
            provider.add_span_processor(
                BatchSpanProcessor(
                    safe,
                    max_queue_size=_MAX_QUEUE_SIZE,
                    schedule_delay_millis=_SCHEDULE_DELAY_MS,
                    max_export_batch_size=_MAX_EXPORT_BATCH,
                    export_timeout_millis=_EXPORT_TIMEOUT_MS,
                )
            )
        else:
            # Tests: export finished spans immediately into the in-memory exporter.
            provider.add_span_processor(SimpleSpanProcessor(safe))

        _provider = provider
        _tracer = provider.get_tracer(_TRACER_NAME)
        _enabled = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("tracing configuration failed: %s", exc)
        _enabled = False
        _provider = None
        _tracer = None


def shutdown_tracing(timeout_millis: int = _SHUTDOWN_FLUSH_MS) -> None:
    """Bounded flush + shutdown. Never raises; never changes app exit outcome."""
    global _provider, _tracer, _enabled, _configured
    provider = _provider
    _provider = None
    _tracer = None
    _enabled = False
    _configured = False
    if provider is None:
        return
    try:
        provider.force_flush(timeout_millis)
    except Exception:  # noqa: BLE001
        pass
    try:
        provider.shutdown()
    except Exception:  # noqa: BLE001
        pass


def tracing_enabled() -> bool:
    """True when a live exporter-backed provider is configured."""
    return _enabled and _tracer is not None


class _SafeTracingCallbackHandler(BaseCallbackHandler):
    """LangChain callbacks → sanitized child spans under the investigation root."""

    raise_error = False  # never break the agent path

    def __init__(self, root: Span, tracer: Tracer) -> None:
        super().__init__()
        self._root = root
        self._tracer = tracer
        self._spans: dict[UUID, Span] = {}

    def _parent_context(self, parent_run_id: UUID | None) -> Any:
        parent = self._spans.get(parent_run_id) if parent_run_id else None
        if parent is not None and parent.is_recording():
            return set_span_in_context(parent)
        return set_span_in_context(self._root)

    def _start(self, name: str, run_id: UUID, parent_run_id: UUID | None) -> Span:
        span = self._tracer.start_span(name, context=self._parent_context(parent_run_id))
        self._spans[run_id] = span
        return span

    def _end(self, run_id: UUID, *, error: BaseException | None = None) -> Span | None:
        span = self._spans.pop(run_id, None)
        if span is None:
            return None
        if error is not None:
            span.set_status(Status(StatusCode.ERROR, _safe_str(str(error))))
            span.set_attribute(_ATTR_ERROR, _safe_str(str(error)))
        span.end()
        return span

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            span = self._start("model", run_id, parent_run_id)
            flat = [_message_public(m) for batch in messages for m in batch]
            span.set_attribute(_ATTR_MODEL_INPUT, _safe_json(flat))
        except Exception:  # noqa: BLE001
            pass

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            span = self._start("model", run_id, parent_run_id)
            span.set_attribute(_ATTR_MODEL_INPUT, _safe_json(prompts))
        except Exception:  # noqa: BLE001
            pass

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            span = self._spans.get(run_id)
            if span is not None:
                span.set_attribute(_ATTR_MODEL_OUTPUT, _safe_json(_llm_public_outputs(response)))
            self._end(run_id)
        except Exception:  # noqa: BLE001
            self._end(run_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._end(run_id, error=error)
        except Exception:  # noqa: BLE001
            pass

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            name = ""
            if isinstance(serialized, dict):
                name = str(serialized.get("name") or serialized.get("id") or "")
            name = name or str(kwargs.get("name") or "tool")
            span = self._start(f"tool:{name}", run_id, parent_run_id)
            span.set_attribute(_ATTR_TOOL_NAME, name)
            payload: Any = inputs if inputs is not None else input_str
            span.set_attribute(_ATTR_TOOL_INPUT, _safe_json(payload))
        except Exception:  # noqa: BLE001
            pass

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            span = self._spans.get(run_id)
            if span is not None:
                content = getattr(output, "content", output)
                span.set_attribute(_ATTR_TOOL_OUTPUT, _safe_json(content))
            self._end(run_id)
        except Exception:  # noqa: BLE001
            self._end(run_id)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._end(run_id, error=error)
        except Exception:  # noqa: BLE001
            pass


@dataclass
class InvestigationTrace:
    """Active investigation/follow-up root span + LangChain callbacks."""

    run_id: str
    case_id: str
    attempt_id: str | None = None
    callbacks: list[BaseCallbackHandler] = field(default_factory=list)
    _root: Span | None = field(default=None, repr=False)

    def set_mode(self, mode: str) -> None:
        """Record execution mode (e.g. ``scripted`` / ``live``)."""
        try:
            if self._root is not None and self._root.is_recording():
                self._root.set_attribute(_ATTR_MODE, str(mode))
        except Exception:  # noqa: BLE001 — telemetry must not raise
            pass

    def set_outcome(self, status: str, *, error: str | None = None) -> None:
        """Record final status; optional error text is redacted."""
        try:
            if self._root is None or not self._root.is_recording():
                return
            self._root.set_attribute(_ATTR_OUTCOME, str(status))
            if error:
                self._root.set_attribute(_ATTR_ERROR, _safe_str(error))
                self._root.set_status(Status(StatusCode.ERROR, _safe_str(error)))
            elif str(status).lower() in {"failed", "error", "timed_out", "timeout"}:
                self._root.set_status(Status(StatusCode.ERROR, str(status)))
            else:
                self._root.set_status(Status(StatusCode.OK))
        except Exception:  # noqa: BLE001
            pass

    def set_usage(
        self,
        usage: dict[str, Any] | None,
        *,
        cost_usd: float | None = None,
    ) -> None:
        """Attach normalized usage/cost when supplied; omit rather than invent zeros."""
        try:
            if self._root is None or not self._root.is_recording():
                return
            if usage:
                self._root.set_attribute(_ATTR_USAGE, _safe_json(usage))
            if cost_usd is not None:
                self._root.set_attribute(_ATTR_COST_USD, float(cost_usd))
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def investigation_trace(
    run_id: str,
    case_id: str,
    attempt_id: str | None = None,
) -> Iterator[InvestigationTrace]:
    """Root span for one investigation/follow-up. No-op when tracing is disabled."""
    handle = InvestigationTrace(run_id=run_id, case_id=case_id, attempt_id=attempt_id)
    tracer = _tracer
    if not _enabled or tracer is None:
        yield handle
        return

    attributes: dict[str, Any] = {
        _ATTR_RUN_ID: run_id,
        _ATTR_CASE_ID: case_id,
    }
    if attempt_id is not None:
        attributes[_ATTR_ATTEMPT_ID] = attempt_id

    span = tracer.start_span("investigation", attributes=attributes)
    handle._root = span
    handle.callbacks = [_SafeTracingCallbackHandler(span, tracer)]
    # Make root current so any direct tracer use nests; callbacks set parent explicitly.
    token = use_span(span, end_on_exit=False)
    token.__enter__()
    try:
        yield handle
    except Exception as exc:
        try:
            if span.is_recording():
                span.set_status(Status(StatusCode.ERROR, _safe_str(str(exc))))
                span.set_attribute(_ATTR_ERROR, _safe_str(str(exc)))
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        try:
            token.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
        try:
            span.end()
        except Exception:  # noqa: BLE001
            pass
