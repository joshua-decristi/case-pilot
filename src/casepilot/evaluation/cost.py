"""Cross-process cumulative cost ledger — **backend is the sole accounting authority**.

Eval runners must NOT mutate the ledger (no double-charging). They may observe
remaining budget and enforce CLI thresholds.

=============================================================================
BACKEND INTEGRATION — single mutator API (Reservation)
=============================================================================

::

    from casepilot.evaluation.cost import (
        ALLOWED_LIVE_MODEL,       # "z-ai/glm-5.3-flash" only
        DEFAULT_BUDGET_USD,       # 5.0 global envelope
        LEDGER_DEFAULT_PATH,      # evals/reports/cost_ledger.json
        CostLedger,
        CostPause,
        RunTokenBudget,
        compute_max_reserve_usd,
        get_model_price,          # fail-closed without verified prices
        load_verified_prices,
        validate_live_model,
    )

    # Prices: verified config only (env CASEPILOT_VERIFIED_PRICES_PATH or
    # evals/pricing/verified_prices.json). No guessed ceilings in code.
    price = get_model_price(ALLOWED_LIVE_MODEL)
    tokens = RunTokenBudget(max_input_tokens=..., max_output_tokens=...)  # backend limits
    amount = compute_max_reserve_usd(price, tokens)

    ledger = CostLedger(path=LEDGER_DEFAULT_PATH, budget_usd=min(b, DEFAULT_BUDGET_USD))
    res = ledger.create_reservation(
        key=run_id, amount_usd=amount, model=ALLOWED_LIVE_MODEL, case_id=case_id
    )
    try:
        ... provider call ...
        # actual_usd from provider; None ⇒ charge full reserve (possibly billable)
        ledger.settle_reservation(
            res.id, actual_usd=actual_or_None, usage=usage, provider_model=reported
        )
        # If actual > reserved: full actual is recorded AND pricing marked unreliable
        # (future reserves pause) — never clamp true spend.
    except BaseException:
        # Only release when sure no billable provider work occurred.
        ledger.release_reservation(res.id, detail="failed before provider call")
        raise

Crash / stale open reservations (operator only, audited)::

    uv run python -m casepilot.evaluation --recover-stale-reservations \\
        --operator local-ops --reason "process crash" --yes

    # or: ledger.recover_stale_reservations(operator=..., reason=..., confirm=True)
    # Settles each open reservation at FULL reserved amount (conservative).
    # Never refunds/releases uncertain billable spend.

Compat: ``reserve(case_id) -> float`` / ``settle(case_id, reserved, ...)`` /
``release(case_id, reserved, ...)`` remain for existing backend.cost_guard only;
new code should use create_reservation / settle_reservation / release_reservation.
``CONSERVATIVE_RESERVE_USD`` lazy-loads from verified prices (CostPause if missing).
=============================================================================
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALLOWED_LIVE_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_BUDGET_USD = 5.0

# Token ceilings only bound *quantity*; USD comes from verified enforced caps.
# Backend should pass its real max input/output enforcement, not these defaults alone.
DEFAULT_MAX_INPUT_TOKENS = 128_000
DEFAULT_MAX_OUTPUT_TOKENS = 16_000


class CostPause(RuntimeError):
    """Budget exceeded, model disallowed, or pricing unreliable/unverified — do not call."""


@dataclass(frozen=True)
class ModelPrice:
    """Enforced routing-cap USD per 1M tokens for reservation + OpenRouter max_price.

    ``input_usd_per_mtok`` / ``output_usd_per_mtok`` are the **enforced caps** used for
    ledger reservation and for ``provider.max_price`` (prompt/completion). They are
    not claimed exact invoice unit prices. See ``kind`` and list_* fields.
    """

    model: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    source: str = ""
    verified_at: str = ""
    verified: bool = False
    kind: str = "enforced_routing_cap"
    require_parameters: bool = True
    max_price_prompt: float | None = None
    max_price_completion: float | None = None
    list_input_usd_per_mtok: float | None = None
    list_output_usd_per_mtok: float | None = None
    source_page: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.input_usd_per_mtok <= 0 or self.output_usd_per_mtok <= 0:
            raise ValueError("unit prices must be positive")
        # Default max_price mirrors enforced cap rates when omitted.
        if self.max_price_prompt is None:
            object.__setattr__(self, "max_price_prompt", float(self.input_usd_per_mtok))
        if self.max_price_completion is None:
            object.__setattr__(self, "max_price_completion", float(self.output_usd_per_mtok))

    def openrouter_provider(self) -> dict[str, Any]:
        """Backend helper: provider prefs for each live call (no extra pricing system)."""
        return {
            "require_parameters": bool(self.require_parameters),
            "max_price": {
                "prompt": float(self.max_price_prompt or self.input_usd_per_mtok),
                "completion": float(self.max_price_completion or self.output_usd_per_mtok),
            },
        }


@dataclass(frozen=True)
class RunTokenBudget:
    """Hard upper bound on tokens the run may consume (with backend enforcement)."""

    max_input_tokens: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        if self.max_input_tokens <= 0 or self.max_output_tokens <= 0:
            raise ValueError("token budgets must be positive")


# Repo-root defaults so import/load works regardless of process cwd / without .env.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VERIFIED_PRICES_PATH = _REPO_ROOT / "evals" / "pricing" / "verified_prices.json"
# Ledger stays cwd-relative (writable runtime artifact).
DEFAULT_LEDGER_PATH = Path("evals/reports/cost_ledger.json")
LEDGER_DEFAULT_PATH = DEFAULT_LEDGER_PATH
VERIFIED_PRICES_ENV = "CASEPILOT_VERIFIED_PRICES_PATH"


def verified_prices_path(override: Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    env = os.environ.get(VERIFIED_PRICES_ENV, "").strip()
    return Path(env) if env else DEFAULT_VERIFIED_PRICES_PATH


def load_verified_prices(path: Path | None = None) -> dict[str, ModelPrice]:
    """Load verified enforced-cap rows. Missing/unverified → {} (fail closed on use)."""
    p = verified_prices_path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    models = data.get("models")
    if not isinstance(models, dict):
        return {}
    out: dict[str, ModelPrice] = {}
    for mid, raw in models.items():
        if not isinstance(raw, dict):
            continue
        if not raw.get("verified"):
            continue
        inp = raw.get("input_usd_per_mtok")
        outp = raw.get("output_usd_per_mtok")
        if not isinstance(inp, (int, float)) or not isinstance(outp, (int, float)):
            continue
        if float(inp) <= 0 or float(outp) <= 0:
            continue
        max_price = raw.get("max_price") if isinstance(raw.get("max_price"), dict) else {}
        prompt_cap = max_price.get("prompt") if isinstance(max_price, dict) else None
        completion_cap = max_price.get("completion") if isinstance(max_price, dict) else None
        list_in = raw.get("list_input_usd_per_mtok")
        list_out = raw.get("list_output_usd_per_mtok")
        try:
            out[str(mid)] = ModelPrice(
                model=str(mid),
                input_usd_per_mtok=float(inp),
                output_usd_per_mtok=float(outp),
                source=str(raw.get("source") or ""),
                source_page=str(raw.get("source_page") or ""),
                verified_at=str(raw.get("verified_at") or ""),
                verified=True,
                kind=str(raw.get("kind") or "enforced_routing_cap"),
                require_parameters=bool(raw.get("require_parameters", True)),
                max_price_prompt=(
                    float(prompt_cap) if isinstance(prompt_cap, (int, float)) else None
                ),
                max_price_completion=(
                    float(completion_cap) if isinstance(completion_cap, (int, float)) else None
                ),
                list_input_usd_per_mtok=(
                    float(list_in) if isinstance(list_in, (int, float)) else None
                ),
                list_output_usd_per_mtok=(
                    float(list_out) if isinstance(list_out, (int, float)) else None
                ),
                notes=str(raw.get("notes") or ""),
            )
        except ValueError:
            continue
    return out


def validate_live_model(model: str) -> None:
    if model != ALLOWED_LIVE_MODEL:
        raise CostPause(f"model {model!r} not allowed; only {ALLOWED_LIVE_MODEL!r} (no fallbacks)")


def get_model_price(
    model: str,
    prices: dict[str, ModelPrice] | None = None,
    *,
    prices_path: Path | None = None,
) -> ModelPrice:
    """Return verified enforced-cap ModelPrice or CostPause.

    Shape is what backend uses for reservation and ``provider.max_price`` /
    ``require_parameters`` via ``ModelPrice.openrouter_provider()``.
    """
    validate_live_model(model)
    table = prices if prices is not None else load_verified_prices(prices_path)
    price = table.get(model)
    if price is None or not price.verified:
        raise CostPause(
            f"no verified unit price for {model!r}; "
            f"set {VERIFIED_PRICES_ENV} or {DEFAULT_VERIFIED_PRICES_PATH} "
            "with verified=true enforced routing caps (fail closed; "
            "no guessed ceilings)"
        )
    if price.input_usd_per_mtok <= 0 or price.output_usd_per_mtok <= 0:
        raise CostPause(f"verified price row for {model!r} has non-positive rates")
    return price


def compute_max_reserve_usd(price: ModelPrice, tokens: RunTokenBudget) -> float:
    """Max cost = enforced cap rates × token ceilings (backend must enforce ceilings)."""
    if not price.verified:
        raise CostPause("refusing reserve from unverified ModelPrice")
    total = (tokens.max_input_tokens / 1_000_000.0) * price.input_usd_per_mtok + (
        tokens.max_output_tokens / 1_000_000.0
    ) * price.output_usd_per_mtok
    if total <= 0:
        raise CostPause("computed max reserve is non-positive; pricing unreliable")
    return total


def cost_from_usage(
    price: ModelPrice,
    *,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    if input_tokens < 0 or output_tokens < 0:
        return None
    if not price.verified:
        return None
    return (input_tokens / 1_000_000.0) * price.input_usd_per_mtok + (
        output_tokens / 1_000_000.0
    ) * price.output_usd_per_mtok


def default_run_token_budget() -> RunTokenBudget:
    return RunTokenBudget(
        max_input_tokens=DEFAULT_MAX_INPUT_TOKENS,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )


def get_conservative_reserve_usd(
    *,
    prices: dict[str, ModelPrice] | None = None,
    tokens: RunTokenBudget | None = None,
    prices_path: Path | None = None,
) -> float:
    """Default pre-call reserve from verified prices × token ceiling. Fail closed."""
    price = get_model_price(ALLOWED_LIVE_MODEL, prices, prices_path=prices_path)
    return compute_max_reserve_usd(price, tokens or default_run_token_budget())


class _ConservativeReserveProxy:
    """Import-safe stand-in; resolves via verified prices on numeric use."""

    __slots__ = ()

    def _value(self) -> float:
        return get_conservative_reserve_usd()

    def __float__(self) -> float:
        return self._value()

    def __int__(self) -> int:
        return int(self._value())

    def __repr__(self) -> str:
        try:
            return f"<CONSERVATIVE_RESERVE_USD {self._value()!r}>"
        except CostPause:
            return "<CONSERVATIVE_RESERVE_USD unverified/unavailable>"


# Backend may `from ... import CONSERVATIVE_RESERVE_USD` without verified file present.
CONSERVATIVE_RESERVE_USD: Any = _ConservativeReserveProxy()


@dataclass(frozen=True)
class Reservation:
    id: str
    key: str
    amount_usd: float
    model: str
    case_id: str | None = None


@dataclass
class LedgerEntry:
    ts: str
    kind: str  # reserve | settle | release | stale_recover | note
    reservation_id: str | None
    key: str | None
    case_id: str | None
    amount_usd: float
    model: str
    detail: str
    usage: dict[str, Any] | None = None
    provider_model: str | None = None
    operator: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def _interprocess_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")  # noqa: SIM115
    try:
        try:
            import fcntl
        except ImportError:  # pragma: no cover
            deadline = time.time() + 30
            taken = str(lock_path) + ".taken"
            while True:
                try:
                    fd = os.open(taken, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    break
                except FileExistsError:
                    if time.time() > deadline:
                        raise CostPause("could not acquire cost ledger lock") from None
                    time.sleep(0.05)
            try:
                yield
            finally:
                try:
                    os.unlink(taken)
                except OSError:
                    pass
            return

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


@dataclass
class CostLedger:
    """File-locked cumulative spend. Backend-only mutator on live paths.

    Primary API: ``create_reservation`` / ``settle_reservation`` / ``release_reservation`` /
    ``recover_stale_reservations``. Float case_id helpers are compatibility-only.
    """

    path: Path
    budget_usd: float = DEFAULT_BUDGET_USD
    model: str = ALLOWED_LIVE_MODEL
    reserve_usd: float | None = None
    prices: dict[str, ModelPrice] | None = None
    prices_path: Path | None = None

    def __post_init__(self) -> None:
        if self.budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        if self.budget_usd > DEFAULT_BUDGET_USD + 1e-12:
            raise CostPause(
                f"budget_usd {self.budget_usd} exceeds global envelope ${DEFAULT_BUDGET_USD:.2f}"
            )
        validate_live_model(self.model)
        object.__setattr__(self, "path", Path(self.path))
        # Resolve price table once (explicit inject wins; else verified file).
        if self.prices is None:
            object.__setattr__(self, "prices", load_verified_prices(self.prices_path))
        if self.reserve_usd is not None:
            try:
                ru = float(self.reserve_usd)
            except CostPause:
                raise
            if ru <= 0:
                raise CostPause("reserve_usd must be positive (from verified prices × tokens)")
            object.__setattr__(self, "reserve_usd", ru)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock():
            if not self.path.exists():
                self._write_unlocked(self._empty_state())

    def _price_table(self) -> dict[str, ModelPrice]:
        return self.prices if self.prices is not None else {}

    def _default_reserve_amount(self) -> float:
        if self.reserve_usd is not None:
            return float(self.reserve_usd)
        return get_conservative_reserve_usd(
            prices=self._price_table() or None,
            prices_path=self.prices_path,
        )

    def _lock(self) -> Any:
        return _interprocess_lock(self.path.with_suffix(self.path.suffix + ".lock"))

    def _empty_state(self) -> dict[str, Any]:
        return {
            "budget_usd": float(self.budget_usd),
            "spent_usd": 0.0,
            "reserved_usd": 0.0,
            "pricing_reliable": True,
            "allowed_model": ALLOWED_LIVE_MODEL,
            "open_reservations": {},
            "entries": [],
            "notes": [
                "Backend is sole mutator; eval must not double-reserve.",
                "Application policy only — not a provider hard cap.",
                f"Global envelope ${DEFAULT_BUDGET_USD:.2f}; GLM-only {ALLOWED_LIVE_MODEL}.",
                "Unit prices must be verified config — no guessed ceilings in code.",
                "actual>reserve records full actual and marks pricing unreliable.",
                "Stale recovery settles full reserve (never refunds uncertain spend).",
            ],
        }

    def _read_unlocked(self) -> dict[str, Any]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"corrupt cost ledger at {self.path}")
        return data

    def _write_unlocked(self, state: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def snapshot(self) -> dict[str, Any]:
        with self._lock():
            return self._read_unlocked()

    def remaining_usd(self) -> float:
        s = self.snapshot()
        budget = min(float(s.get("budget_usd") or self.budget_usd), self.budget_usd)
        committed = float(s.get("spent_usd") or 0) + float(s.get("reserved_usd") or 0)
        return max(0.0, budget - committed)

    def open_reservations(self) -> dict[str, dict[str, Any]]:
        snap = self.snapshot()
        raw = snap.get("open_reservations") or {}
        return dict(raw) if isinstance(raw, dict) else {}

    def assert_model_allowed(self, model: str) -> None:
        validate_live_model(model)

    def mark_pricing_unreliable(self, reason: str) -> None:
        with self._lock():
            state = self._read_unlocked()
            self._mark_unreliable_unlocked(state, reason)
            self._write_unlocked(state)

    def _mark_unreliable_unlocked(self, state: dict[str, Any], reason: str) -> None:
        state["pricing_reliable"] = False
        state.setdefault("entries", []).append(
            asdict(
                LedgerEntry(
                    ts=_now(),
                    kind="note",
                    reservation_id=None,
                    key=None,
                    case_id=None,
                    amount_usd=0.0,
                    model=self.model,
                    detail=f"pricing_unreliable: {reason}",
                )
            )
        )

    # ----- primary Reservation API -------------------------------------------

    def create_reservation(
        self,
        *,
        key: str,
        amount_usd: float,
        model: str,
        case_id: str | None = None,
    ) -> Reservation:
        """Reserve bounded max cost before a provider call. Raises CostPause to stop."""
        validate_live_model(model)
        get_model_price(model, self._price_table() or None, prices_path=self.prices_path)
        if amount_usd <= 0:
            self.mark_pricing_unreliable("non-positive reservation amount")
            raise CostPause("reservation amount must be positive (from verified prices)")

        with self._lock():
            state = self._read_unlocked()
            if not state.get("pricing_reliable", True):
                raise CostPause("pricing marked unreliable; live calls paused")
            budget = min(float(state.get("budget_usd") or self.budget_usd), float(self.budget_usd))
            state["budget_usd"] = budget
            committed = float(state.get("spent_usd") or 0) + float(state.get("reserved_usd") or 0)
            if committed + amount_usd > budget + 1e-12:
                raise CostPause(
                    f"reservation ${amount_usd:.6f} would exceed budget "
                    f"(committed ${committed:.6f} / ${budget:.6f}); paused"
                )
            rid = str(uuid.uuid4())
            open_res = state.setdefault("open_reservations", {})
            open_res[rid] = {
                "key": key,
                "amount_usd": float(amount_usd),
                "model": model,
                "case_id": case_id,
                "created_at": _now(),
            }
            state["reserved_usd"] = float(state.get("reserved_usd") or 0) + float(amount_usd)
            state.setdefault("entries", []).append(
                asdict(
                    LedgerEntry(
                        ts=_now(),
                        kind="reserve",
                        reservation_id=rid,
                        key=key,
                        case_id=case_id,
                        amount_usd=float(amount_usd),
                        model=model,
                        detail="pre-call reservation from verified unit prices × token ceiling",
                    )
                )
            )
            self._write_unlocked(state)
            return Reservation(
                id=rid,
                key=key,
                amount_usd=float(amount_usd),
                model=model,
                case_id=case_id,
            )

    def settle_reservation(
        self,
        reservation_id: str,
        *,
        actual_usd: float | None = None,
        usage: dict[str, Any] | None = None,
        provider_model: str | None = None,
    ) -> float:
        """Close reservation. ``actual_usd is None`` ⇒ spend full reserved amount.

        If ``actual_usd > reserved``: record **full actual** (never clamp) and mark
        pricing unreliable so future calls pause.
        """
        with self._lock():
            state = self._read_unlocked()
            return self._settle_unlocked(
                state,
                reservation_id,
                actual_usd=actual_usd,
                usage=usage,
                provider_model=provider_model,
            )

    def release_reservation(
        self, reservation_id: str, *, detail: str = "released unused reservation"
    ) -> None:
        """Drop reservation only when no billable provider work occurred."""
        with self._lock():
            state = self._read_unlocked()
            self._release_unlocked(state, reservation_id, detail=detail)

    def recover_stale_reservations(
        self,
        *,
        operator: str,
        reason: str,
        confirm: bool = False,
        reservation_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Operator crash recovery: settle open reservations at **full reserve**.

        Conservative: treats reserved amount as spent (may have been billable).
        Does **not** release/refund. Requires ``confirm=True``. Fully audited.
        """
        if not confirm:
            raise CostPause(
                "recover_stale_reservations requires confirm=True "
                "(explicit operator acknowledgment)"
            )
        op = (operator or "").strip()
        why = (reason or "").strip()
        if not op or not why:
            raise CostPause("operator and reason required for stale recovery")

        with self._lock():
            state = self._read_unlocked()
            open_res = dict(state.get("open_reservations") or {})
            if reservation_ids is not None:
                wanted = set(reservation_ids)
                open_res = {k: v for k, v in open_res.items() if k in wanted}
                missing = wanted - set(open_res)
                if missing:
                    raise CostPause(f"unknown reservation ids for recovery: {sorted(missing)}")

            recovered: list[dict[str, Any]] = []
            for rid, rec in list(open_res.items()):
                reserved = float(rec["amount_usd"])
                # Settle at full reserve — never refund uncertain billable spend.
                spend = self._settle_unlocked(
                    state,
                    rid,
                    actual_usd=reserved,
                    usage={
                        "stale_recovery": True,
                        "operator": op,
                        "reason": why,
                        "original_reserved_usd": reserved,
                    },
                    provider_model=None,
                    entry_kind="stale_recover",
                    detail=(
                        f"stale recovery by {op}: {why}; "
                        f"settled full reserve ${reserved:.6f} (no refund)"
                    ),
                    operator=op,
                    write=False,
                )
                recovered.append(
                    {
                        "reservation_id": rid,
                        "key": rec.get("key"),
                        "case_id": rec.get("case_id"),
                        "spent_usd": spend,
                        "operator": op,
                        "reason": why,
                    }
                )
            # Pause further live calls until operator re-verifies pricing/state.
            self._mark_unreliable_unlocked(
                state,
                f"stale reservation recovery by {op}: {why} "
                f"({len(recovered)} settled at full reserve)",
            )
            self._write_unlocked(state)
            return recovered

    # ----- backend.cost_guard compatibility (float / case_id) ----------------

    def reserve(self, case_id: str, *, amount: float | None = None) -> float:
        """Compat: returns reserved USD. Prefer ``create_reservation``."""
        self.assert_model_allowed(self.model)
        get_model_price(self.model, self._price_table() or None, prices_path=self.prices_path)
        amt = self._default_reserve_amount() if amount is None else float(amount)
        res = self.create_reservation(
            key=case_id, amount_usd=amt, model=self.model, case_id=case_id
        )
        return res.amount_usd

    def settle(
        self,
        case_id: str,
        reserved: float,
        *,
        actual_usd: float | None = None,
        usage: dict[str, Any] | None = None,
        provider_model: str | None = None,
    ) -> float:
        """Compat: settle by case_id + reserved amount. Prefer ``settle_reservation``."""
        with self._lock():
            state = self._read_unlocked()
            rid = self._find_open(state, case_id=case_id, amount=reserved)
            if rid is None:
                raise CostPause(f"no open reservation for case_id={case_id!r} amount={reserved}")
            return self._settle_unlocked(
                state,
                rid,
                actual_usd=actual_usd,
                usage=usage,
                provider_model=provider_model,
            )

    def release(self, case_id: str, reserved: float, *, detail: str = "released") -> None:
        """Compat: release by case_id. Prefer ``release_reservation``."""
        with self._lock():
            state = self._read_unlocked()
            rid = self._find_open(state, case_id=case_id, amount=reserved)
            if rid is None:
                raise CostPause(f"no open reservation for case_id={case_id!r} amount={reserved}")
            self._release_unlocked(state, rid, detail=detail)

    # aliases kept for earlier eval docs / tests
    reserve_explicit = create_reservation

    # ----- internals ---------------------------------------------------------

    @staticmethod
    def _find_open(state: dict[str, Any], *, case_id: str, amount: float) -> str | None:
        open_res = state.get("open_reservations") or {}
        for rid, rec in open_res.items():
            if rec.get("case_id") == case_id and abs(float(rec["amount_usd"]) - amount) < 1e-9:
                return str(rid)
        for rid, rec in open_res.items():
            if rec.get("key") == case_id and abs(float(rec["amount_usd"]) - amount) < 1e-9:
                return str(rid)
        return None

    def _settle_unlocked(
        self,
        state: dict[str, Any],
        reservation_id: str,
        *,
        actual_usd: float | None,
        usage: dict[str, Any] | None,
        provider_model: str | None,
        entry_kind: str = "settle",
        detail: str | None = None,
        operator: str | None = None,
        write: bool = True,
    ) -> float:
        open_res = state.setdefault("open_reservations", {})
        rec = open_res.pop(reservation_id, None)
        if rec is None:
            raise CostPause(f"unknown reservation_id {reservation_id}")
        reserved = float(rec["amount_usd"])
        state["reserved_usd"] = max(0.0, float(state.get("reserved_usd") or 0) - reserved)

        over_reserve = False
        if actual_usd is None:
            spend = reserved
            settle_detail = detail or "settled at full reserve (actual cost unknown)"
        else:
            spend = float(actual_usd)
            if spend < 0:
                raise CostPause("actual_usd must be >= 0")
            if spend > reserved + 1e-12:
                over_reserve = True
                # Truthful actual — never clamp real spend down to reserve.
                settle_detail = detail or (
                    f"actual ${spend:.6f} exceeded reserve ${reserved:.6f}; "
                    "recorded full actual; pricing marked unreliable"
                )
            else:
                settle_detail = detail or "settled at provider/derived actual cost"

        state["spent_usd"] = float(state.get("spent_usd") or 0) + spend
        state.setdefault("entries", []).append(
            asdict(
                LedgerEntry(
                    ts=_now(),
                    kind=entry_kind,
                    reservation_id=reservation_id,
                    key=rec.get("key"),
                    case_id=rec.get("case_id"),
                    amount_usd=spend,
                    model=str(rec.get("model") or self.model),
                    detail=settle_detail,
                    usage=usage,
                    provider_model=provider_model,
                    operator=operator,
                )
            )
        )
        if over_reserve:
            self._mark_unreliable_unlocked(
                state,
                f"provider actual ${spend:.6f} > reserved ${reserved:.6f} "
                f"(reservation {reservation_id})",
            )
        if write:
            self._write_unlocked(state)
        return spend

    def _release_unlocked(self, state: dict[str, Any], reservation_id: str, *, detail: str) -> None:
        open_res = state.setdefault("open_reservations", {})
        rec = open_res.pop(reservation_id, None)
        if rec is None:
            raise CostPause(f"unknown reservation_id {reservation_id}")
        reserved = float(rec["amount_usd"])
        state["reserved_usd"] = max(0.0, float(state.get("reserved_usd") or 0) - reserved)
        state.setdefault("entries", []).append(
            asdict(
                LedgerEntry(
                    ts=_now(),
                    kind="release",
                    reservation_id=reservation_id,
                    key=rec.get("key"),
                    case_id=rec.get("case_id"),
                    amount_usd=reserved,
                    model=str(rec.get("model") or self.model),
                    detail=detail,
                )
            )
        )
        self._write_unlocked(state)
