"""Live-model spend guard — backend sole mutator of shared CostLedger.

Uses explicit reservation API:
  reserve_explicit(key, amount_usd, model, case_id) -> Reservation
  settle_reservation / release_reservation

Eval must NOT reserve/settle. Scripted models never touch the ledger.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

from casepilot.evaluation.cost import (
    ALLOWED_LIVE_MODEL,
    DEFAULT_BUDGET_USD,
    LEDGER_DEFAULT_PATH,
    CostLedger,
    CostPause,
    ModelPrice,
    Reservation,
    RunTokenBudget,
    compute_max_reserve_usd,
    cost_from_usage,
    get_model_price,
    load_verified_prices,
    validate_live_model,
)
from casepilot.shared.config import Settings

LEDGER_PATH_ENV = "CASEPILOT_COST_LEDGER_PATH"

# Reservation assumptions — must match enforced OpenRouter caps (see verified_prices.json).
MAX_OUTPUT_TOKENS = 2000  # combined reasoning+output budget (max_tokens)
# Per-invocation input ceiling (byte≈token). 8k was too small for tools+schema+history.
INPUT_TOKENS_PER_TURN = 48_000
# Generous framing overhead: structured-output schema, tool defs, JSON wrappers.
_SCHEMA_OVERHEAD_TOKENS = 8_192
_ROLE_OVERHEAD_TOKENS = 32

# Enforced provider.max_price USD per 1M tokens (must match verified_prices + live request).
MAX_PRICE_PROMPT_USD_PER_MTOK = 0.23
MAX_PRICE_COMPLETION_USD_PER_MTOK = 2.0


def ledger_path() -> Path:
    raw = os.environ.get(LEDGER_PATH_ENV, "").strip()
    return Path(raw) if raw else LEDGER_DEFAULT_PATH


def get_ledger(
    *,
    budget_usd: float = DEFAULT_BUDGET_USD,
    prices: dict[str, ModelPrice] | None = None,
    prices_path: Path | None = None,
) -> CostLedger:
    """Build ledger. Prices: explicit inject, else verified file (fail-closed on reserve)."""
    table = prices if prices is not None else load_verified_prices(prices_path)
    return CostLedger(
        path=ledger_path(),
        budget_usd=min(float(budget_usd), DEFAULT_BUDGET_USD),
        model=ALLOWED_LIVE_MODEL,
        prices=table,
        prices_path=prices_path,
    )


def is_live_model(model: Any) -> bool:
    return isinstance(model, ChatOpenRouter)


def live_model_price(
    *,
    prices: dict[str, ModelPrice] | None = None,
    prices_path: Path | None = None,
) -> ModelPrice:
    """Verified enforced-cap row for GLM (fail closed if missing)."""
    return get_model_price(ALLOWED_LIVE_MODEL, prices, prices_path=prices_path)


def build_live_chat_model(
    api_key: Any,
    *,
    price: ModelPrice | None = None,
) -> ChatOpenRouter:
    """ChatOpenRouter pinned to GLM with max_price + low reasoning (excluded)."""
    price = price or live_model_price()
    if (
        float(price.max_price_prompt or 0) != MAX_PRICE_PROMPT_USD_PER_MTOK
        or float(price.max_price_completion or 0) != MAX_PRICE_COMPLETION_USD_PER_MTOK
    ):
        raise CostPause(
            "verified max_price must be prompt="
            f"{MAX_PRICE_PROMPT_USD_PER_MTOK} completion="
            f"{MAX_PRICE_COMPLETION_USD_PER_MTOK} "
            f"(got {price.max_price_prompt}/{price.max_price_completion})"
        )
    return ChatOpenRouter(
        model=ALLOWED_LIVE_MODEL,
        api_key=api_key,
        temperature=0,
        max_retries=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        # Combined reasoning+output budget; exclude so traces never see chain-of-thought.
        reasoning={"effort": "low", "exclude": True},
        openrouter_provider=price.openrouter_provider(),
    )


def live_request_payload_preview(model: ChatOpenRouter) -> dict[str, Any]:
    """Offline serialization of provider-bound request params (no network)."""
    params = dict(model._default_params)
    # Never claim secrets; drop nothing sensitive here (key not in params).
    return {
        "model": params.get("model"),
        "max_tokens": params.get("max_tokens"),
        "temperature": params.get("temperature"),
        "reasoning": params.get("reasoning"),
        "provider": params.get("provider"),
    }


def strip_reasoning_message(msg: AIMessage) -> AIMessage:
    """Drop reasoning fields so traces/API never expose private chain-of-thought."""
    meta = {
        k: v for k, v in dict(msg.response_metadata or {}).items() if "reason" not in str(k).lower()
    }
    extra = {
        k: v for k, v in dict(msg.additional_kwargs or {}).items() if "reason" not in str(k).lower()
    }
    content = msg.content
    # Some providers nest reasoning as list parts — keep only non-reasoning text.
    if isinstance(content, list):
        cleaned: list[Any] = []
        for part in content:
            if isinstance(part, dict) and "reason" in str(part.get("type", "")).lower():
                continue
            cleaned.append(part)
        content = cleaned
    return msg.model_copy(
        update={
            "content": content,
            "response_metadata": meta,
            "additional_kwargs": extra,
        }
    )


def estimate_text_tokens(text: str) -> int:
    """Conservative token estimate without a tokenizer: 1 UTF-8 byte ≈ 1 token.

    utf8_bytes/2 under-counts arbitrary Unicode and many tokenizers; byte count
    is a safer upper bound for budget enforcement.
    """
    if not text:
        return 0
    return len(text.encode("utf-8"))


def estimate_request_tokens(
    *,
    messages: list[Any] | None = None,
    system_message: SystemMessage | str | None = None,
    tools: list[Any] | None = None,
    schema_overhead: int = _SCHEMA_OVERHEAD_TOKENS,
) -> int:
    """Bound input size for one model invocation (messages + system + tools + schema)."""
    total = schema_overhead
    if system_message is not None:
        content = (
            system_message
            if isinstance(system_message, str)
            else str(getattr(system_message, "content", system_message) or "")
        )
        total += estimate_text_tokens(content) + _ROLE_OVERHEAD_TOKENS
    for msg in messages or []:
        total += _ROLE_OVERHEAD_TOKENS
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            total += estimate_text_tokens(content)
        elif isinstance(content, list):
            total += estimate_text_tokens(json.dumps(content, default=str))
        else:
            total += estimate_text_tokens(str(content))
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            total += estimate_text_tokens(json.dumps(tool_calls, default=str))
        name = getattr(msg, "name", None)
        if name:
            total += estimate_text_tokens(str(name))
    for tool in tools or []:
        if isinstance(tool, dict):
            total += estimate_text_tokens(json.dumps(tool, default=str))
        else:
            # BaseTool / callable — dump name + description + args schema when present
            payload: dict[str, Any] = {
                "name": getattr(tool, "name", None) or getattr(tool, "__name__", "tool"),
                "description": getattr(tool, "description", "") or "",
            }
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None:
                try:
                    if hasattr(args_schema, "model_json_schema"):
                        payload["schema"] = args_schema.model_json_schema()
                    elif isinstance(args_schema, dict):
                        payload["schema"] = args_schema
                except Exception:  # noqa: BLE001 — best-effort overhead only
                    payload["schema"] = str(args_schema)[:2000]
            total += estimate_text_tokens(json.dumps(payload, default=str))
    return total


def run_token_budget(settings: Settings) -> RunTokenBudget:
    turns = max(1, int(settings.max_model_turns))
    return RunTokenBudget(
        max_input_tokens=turns * INPUT_TOKENS_PER_TURN,
        max_output_tokens=turns * MAX_OUTPUT_TOKENS,
    )


def extract_usage(messages: list[BaseMessage] | None) -> dict[str, Any] | None:
    """Aggregate token/cost metadata from AIMessage stream (OpenRouter-shaped)."""
    if not messages:
        return None
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    saw_in = False
    saw_out = False
    provider_cost = 0.0
    saw_provider_cost = False
    models: list[str] = []
    providers: list[str] = []

    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        meta = dict(msg.response_metadata or {})
        if "cost" in meta and isinstance(meta["cost"], (int, float)):
            provider_cost += float(meta["cost"])
            saw_provider_cost = True
        usage_blob = meta.get("token_usage") or meta.get("usage")
        if isinstance(usage_blob, dict):
            if "cost" in usage_blob and isinstance(usage_blob["cost"], (int, float)):
                provider_cost += float(usage_blob["cost"])
                saw_provider_cost = True
            pt = usage_blob.get("prompt_tokens") or usage_blob.get("input_tokens")
            ct = usage_blob.get("completion_tokens") or usage_blob.get("output_tokens")
            tt = usage_blob.get("total_tokens")
            if pt is not None:
                prompt_tokens += int(pt)
                saw_in = True
            if ct is not None:
                completion_tokens += int(ct)
                saw_out = True
            if tt is not None:
                total_tokens += int(tt)
        um = getattr(msg, "usage_metadata", None)
        if um is not None:
            if isinstance(um, dict):
                inp = um.get("input_tokens")
                out = um.get("output_tokens")
                tot = um.get("total_tokens")
            else:
                inp = getattr(um, "input_tokens", None)
                out = getattr(um, "output_tokens", None)
                tot = getattr(um, "total_tokens", None)
            if inp is not None:
                prompt_tokens += int(inp)
                saw_in = True
            if out is not None:
                completion_tokens += int(out)
                saw_out = True
            if tot is not None:
                total_tokens += int(tot)
        model_name = meta.get("model_name") or meta.get("model")
        if isinstance(model_name, str):
            models.append(model_name)
        provider = meta.get("model_provider") or meta.get("provider")
        if isinstance(provider, str):
            providers.append(provider)

    if not (saw_in or saw_out or saw_provider_cost or models):
        return None
    cost_usd = provider_cost if saw_provider_cost else None
    return {
        "prompt_tokens": prompt_tokens if saw_in else None,
        "completion_tokens": completion_tokens if saw_out else None,
        "total_tokens": total_tokens or None,
        "input_tokens": prompt_tokens if saw_in else None,
        "output_tokens": completion_tokens if saw_out else None,
        "provider_cost_usd": cost_usd,
        "cost_usd": cost_usd,
        "cost": cost_usd,
        "models": models,
        "providers": providers,
    }


@dataclass
class LiveCallTracker:
    """Shared mutable state for middleware + investigation settle path."""

    provider_started: bool = False
    captured_ai_messages: list[AIMessage] = field(default_factory=list)
    last_input_estimate: int = 0
    input_cap: int = INPUT_TOKENS_PER_TURN


def _assert_input_budget(tracker: LiveCallTracker, request: Any) -> None:
    est = estimate_request_tokens(
        messages=getattr(request, "messages", None),
        system_message=getattr(request, "system_message", None),
        tools=getattr(request, "tools", None),
    )
    tracker.last_input_estimate = est
    if est > tracker.input_cap:
        raise CostPause(
            f"model input bound exceeded: estimated {est} tokens "
            f"> per-turn cap {tracker.input_cap} "
            f"(reservation assumes ≤{INPUT_TOKENS_PER_TURN}/turn)"
        )


def _capture_model_response(tracker: LiveCallTracker, response: Any) -> Any:
    """Strip reasoning and retain AIMessages for usage/cost accounting."""
    result_msgs = getattr(response, "result", None)
    if isinstance(result_msgs, list):
        stripped: list[Any] = []
        for m in result_msgs:
            if isinstance(m, AIMessage):
                clean = strip_reasoning_message(m)
                tracker.captured_ai_messages.append(clean)
                stripped.append(clean)
            else:
                stripped.append(m)
        if hasattr(response, "model_copy"):
            return response.model_copy(update={"result": stripped})
        try:
            object.__setattr__(response, "result", stripped)
        except Exception:  # noqa: BLE001
            pass
        return response
    if isinstance(response, AIMessage):
        clean = strip_reasoning_message(response)
        tracker.captured_ai_messages.append(clean)
        return clean
    return response


class LiveCostGuardMiddleware(AgentMiddleware[Any, Any]):
    """Input-cap + usage capture with both sync and async model hooks.

    ``@wrap_model_call`` alone is insufficient for ``agent.ainvoke`` — the async
    path requires ``awrap_model_call``. This class implements both.
    """

    def __init__(self, tracker: LiveCallTracker) -> None:
        super().__init__()
        self.tracker = tracker

    def wrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        _assert_input_budget(self.tracker, request)
        # About to hit the provider — never release reservation after this point.
        self.tracker.provider_started = True
        response = handler(request)
        return _capture_model_response(self.tracker, response)

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        _assert_input_budget(self.tracker, request)
        self.tracker.provider_started = True
        response = await handler(request)
        return _capture_model_response(self.tracker, response)


def make_live_guard_middleware(tracker: LiveCallTracker) -> AgentMiddleware[Any, Any]:
    """Build live cost-guard middleware (sync + async model call wrappers)."""
    return LiveCostGuardMiddleware(tracker)


def guard_live_call(
    *,
    model: Any,
    model_name: str,
    run_id: str,
    case_id: str,
    settings: Settings,
    ledger: CostLedger | None = None,
) -> tuple[CostLedger | None, Reservation | None]:
    """Reserve max cost before a paid call. Scripted → (None, None).

    Uses ``CostLedger.create_reservation`` (alias ``reserve_explicit``).
    """
    if not is_live_model(model):
        return None, None
    validate_live_model(model_name)
    lg = ledger or get_ledger()
    price = get_model_price(model_name, lg.prices, prices_path=lg.prices_path)
    amount = compute_max_reserve_usd(price, run_token_budget(settings))
    # Preferred explicit API (reserve_explicit is an alias of create_reservation).
    reservation = lg.create_reservation(
        key=run_id,
        amount_usd=amount,
        model=model_name,
        case_id=case_id,
    )
    return lg, reservation


def settle_live(
    ledger: CostLedger | None,
    reservation: Reservation | None,
    *,
    messages: list[BaseMessage] | None = None,
    tracker: LiveCallTracker | None = None,
    trace_emit: Any | None = None,
    force_full_reserve: bool = False,
) -> tuple[dict[str, Any] | None, float | None]:
    """Settle after provider involvement. Unknown actual → full reserve + pause."""
    if ledger is None or reservation is None:
        return None, None

    msgs: list[BaseMessage] = list(messages or [])
    if tracker is not None:
        # Prefer captured AI messages (includes partial turns on failure).
        for m in tracker.captured_ai_messages:
            if m not in msgs:
                msgs.append(m)

    usage = extract_usage(msgs)
    price = get_model_price(reservation.model, ledger.prices, prices_path=ledger.prices_path)
    actual: float | None = None
    if not force_full_reserve and usage is not None:
        derived = cost_from_usage(
            price,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
        if derived is not None:
            actual = derived
        elif usage.get("provider_cost_usd") is not None:
            actual = float(usage["provider_cost_usd"])

    provider_model = None
    if usage and usage.get("models"):
        provider_model = usage["models"][-1]

    if trace_emit is not None:
        trace_emit(
            "model_usage",
            model=reservation.model,
            provider="openrouter",
            usage=usage,
            cost_usd=actual,
            reserved_usd=reservation.amount_usd,
            reservation_id=reservation.id,
            force_full_reserve=force_full_reserve,
        )

    if actual is None:
        spend = ledger.settle_reservation(
            reservation.id,
            actual_usd=None,
            usage=usage,
            provider_model=provider_model,
        )
        ledger.mark_pricing_unreliable(
            "no token/cost metadata after live provider involvement; charged full reserve"
        )
        if trace_emit is not None:
            trace_emit("cost_pause", detail="pricing unreliable after missing usage")
        if usage is None:
            usage = {
                "cost_usd": spend,
                "cost": spend,
                "settled_at_full_reserve": True,
            }
        else:
            usage = {**usage, "cost_usd": spend, "cost": spend, "settled_at_full_reserve": True}
        return usage, spend

    spend = ledger.settle_reservation(
        reservation.id,
        actual_usd=actual,
        usage=usage,
        provider_model=provider_model,
    )
    if usage is not None:
        usage = {**usage, "cost_usd": spend, "cost": spend}
    return usage, spend


def release_live(
    ledger: CostLedger | None,
    reservation: Reservation | None,
    *,
    detail: str,
) -> None:
    """Release only when the provider was never contacted."""
    if ledger is None or reservation is None:
        return
    ledger.release_reservation(reservation.id, detail=detail)


def close_live_reservation(
    ledger: CostLedger | None,
    reservation: Reservation | None,
    *,
    tracker: LiveCallTracker | None,
    messages: list[BaseMessage] | None = None,
    trace_emit: Any | None = None,
    detail: str = "",
) -> tuple[dict[str, Any] | None, float | None]:
    """After agent exit: settle if provider may have started, else release."""
    if ledger is None or reservation is None:
        return None, None
    started = bool(tracker and tracker.provider_started)
    if started:
        return settle_live(
            ledger,
            reservation,
            messages=messages,
            tracker=tracker,
            trace_emit=trace_emit,
            force_full_reserve=False,
        )
    release_live(ledger, reservation, detail=detail or "provider never started")
    return None, None
