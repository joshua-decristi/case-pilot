"""JSONL run traces with secret redaction."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SECRET_KEYS = re.compile(
    r"(authorization|api[_-]?key|token|password|secret|bearer)",
    re.IGNORECASE,
)
# Exact provider CoT field names only — not domain keys like reason/pause_reason.
_REASONING_KEY_NAMES = frozenset(
    {
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "reasoning_text",
    }
)
_BEARER = re.compile(r"Bearer\s+\S+", re.IGNORECASE)


def _is_secret_key(key: str) -> bool:
    """True for credential-like keys; not model usage counters (*_tokens)."""
    k = str(key)
    kl = k.lower()
    # input_tokens / output_tokens contain "token" but are billable usage, not secrets.
    if kl.endswith("_tokens") or kl in {"tokens", "input_tokens", "output_tokens"}:
        return False
    return bool(_SECRET_KEYS.search(k))


def _is_reasoning_key(key: str) -> bool:
    """True for provider chain-of-thought keys (exact, case-insensitive)."""
    return str(key).lower() in _REASONING_KEY_NAMES


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if _is_secret_key(str(k)):
                out[k] = "[REDACTED]"
            elif _is_reasoning_key(str(k)):
                # Never persist private chain-of-thought / reasoning traces.
                out[k] = "[REDACTED_REASONING]"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        return _BEARER.sub("Bearer [REDACTED]", value)
    return value


class TraceWriter:
    def __init__(self, trace_dir: Path, run_id: str) -> None:
        self.trace_dir = trace_dir
        self.run_id = run_id
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.trace_dir / f"{run_id}.jsonl"

    def emit(self, event_type: str, **payload: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "event": event_type,
            **_redact(payload),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
