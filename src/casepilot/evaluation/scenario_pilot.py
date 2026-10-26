"""Scenario-pilot cohort orchestration (P0 answer-quality measurements).

Owns fixed 6×3 slots, read-only ledger preflight, live calibration gate,
HTTP investigation/follow-up execution, safety/coverage assessment hooks,
and honest versioned reports. Does **not** mutate the cost ledger.

Judge spend goes through backend ``run_judge_call`` only. Agent spend is
backend-owned via HTTP. Missing audit/context proof is ``not_assessed``.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from casepilot.backend.cost_guard import (
    INPUT_TOKENS_PER_TURN,
    LEDGER_PATH_ENV,
    MAX_OUTPUT_TOKENS,
)
from casepilot.backend.cost_guard import (
    ledger_path as canonical_ledger_path,
)
from casepilot.backend.evaluation_judge import (
    JudgeOutcome,
    JudgePayload,
    JudgeVerdict,
    judge_prompt_version,
    judge_schema_version,
    run_judge_call,
)
from casepilot.backend.pilot_observability import (
    PhaseObservation,
    PilotObservability,
    WriteAuditSnapshot,
    proposal_facts_from_audit,
    task_facts_from_audit,
)
from casepilot.evaluation.cost import (
    ALLOWED_LIVE_MODEL,
    DEFAULT_BUDGET_USD,
    CostPause,
    RunTokenBudget,
    compute_max_reserve_usd,
    get_model_price,
    load_verified_prices,
    validate_live_model,
)
from casepilot.evaluation.metadata import collect_metadata
from casepilot.evaluation.pilot_assets import (
    CalibrationExample,
    PilotAssets,
    ReferenceDef,
    ScenarioDef,
    load_pilot_assets,
)
from casepilot.evaluation.pilot_coverage import (
    AnswerCitation,
    AttemptSnapshot,
    CoverageAssessment,
    CoverageStatus,
    ExposedFact,
    RepetitionCoverage,
    RetrievalClass,
    TargetFact,
    assess_followup_coverage,
    summarize_cohort_coverage,
)
from casepilot.evaluation.pilot_safety import (
    CitationFact,
    ForbiddenAttemptFact,
    PhaseSafetyEvidence,
    ScenarioSafetyResult,
    SnapshotProvenanceFact,
    TaskStateFact,
    assess_scenario_safety,
    citation_fact,
    collect_proposal_states,
    task_state_fact,
)
from casepilot.shared.config import Settings, get_settings
from casepilot.shared.contracts import (
    BackendRoutes,
    ConversationTurnResponse,
    Evidence,
    InvestigationResult,
    RunStatus,
    SendMessageRequest,
    StartInvestigationRequest,
    StartInvestigationResponse,
    TurnStatus,
)

REPETITIONS = 3
SLOT_COUNT = 18  # 6 scenarios × 3
RUBRIC_VERSION = "p0-v1"
# Frozen manifest: only this scenario's 3 slots satisfy the P0 retrieval gate.
REQUIRED_RETRIEVAL_SCENARIO_ID = "retrieve"
REEVALUATOR_VERSION = "p0-reassess-invalid-quality-v2"

# Conservative call-count framing for preflight only (not a guarantee).
# calibration(8) + baseline judges(18) + agent turns: ~18 investigations +
# ~4 follow-up setups + tool/model headroom + failure paths.
_EST_JUDGE_CALLS = 8 + 18
_EST_AGENT_CALLS = 18 + 8 + 12  # targets + setups + tool-ish headroom
_EST_FAILURE_PAD_CALLS = 6

DISCLAIMERS = (
    "six development scenarios only; not evidence of broad reliability",
    "same-model fallible judge (z-ai/glm-5.3-flash); ratings are not ground truth",
    "authored references and calibration; no held-out reliability claim",
    "safety fail is a measured finding; missing safety evidence is not pass",
    "no automatic billable retries; failed slots are never replaced",
)


class PilotError(RuntimeError):
    """Hard stop before or during pilot (preflight, calibration gate, config)."""


# ---------------------------------------------------------------------------
# Config / adapters
# ---------------------------------------------------------------------------


@dataclass
class PilotConfig:
    """Pilot run configuration. ``live=True`` is the only paid opt-in."""

    live: bool = False
    backend_base_url: str = "http://127.0.0.1:8000"
    report_dir: Path = Path("evals/reports")
    ledger_path: Path | None = None
    assets_path: Path | None = None
    cohort_id: str | None = None
    model: str = ALLOWED_LIVE_MODEL
    timeout_seconds: float = 180.0
    budget_usd: float = DEFAULT_BUDGET_USD
    resume_path: Path | None = None  # existing cohort report to resume unstarted slots
    # Must match the running backend/simulator process paths or audits stay not_assessed.
    backend_db_path: Path | None = None
    simulator_db_path: Path | None = None
    trace_dir: Path | None = None


@dataclass(frozen=True)
class PhaseCapture:
    """One investigation or follow-up phase outcome (HTTP-level)."""

    ok: bool
    run_id: str | None = None
    conversation_id: str | None = None
    turn_id: str | None = None
    attempt_id: str | None = None
    answer_text: str | None = None
    answer_payload: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    latency_s: float | None = None
    cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    provider_model: str | None = None
    # Optional authoritative facts when available via HTTP/store adapters.
    proposals_before: tuple[Any, ...] | None = None
    proposals_after: tuple[Any, ...] | None = None
    tasks_before: tuple[Any, ...] | None = None
    tasks_after: tuple[Any, ...] | None = None
    persisted_snapshots: tuple[Any, ...] | None = None
    citations: tuple[Any, ...] = ()
    forbidden_attempts: tuple[ForbiddenAttemptFact, ...] = ()
    write_audit_captured: bool = False
    # Coverage instrumentation (follow-up target only). None = missing proof.
    pre_turn_model_context: tuple[ExposedFact, ...] | None = None
    attempt_follow_up_snapshots: tuple[AttemptSnapshot, ...] | None = None
    answering_model_exposed: tuple[ExposedFact, ...] | None = None
    answer_citations: tuple[AnswerCitation, ...] | None = None
    gaps: tuple[str, ...] = ()


class BackendAdapter(Protocol):
    """HTTP (or test double) surface for investigation + follow-up only."""

    def start_investigation(self, case_id: str, objective: str) -> PhaseCapture: ...

    def send_followup(
        self,
        run_id: str,
        question: str,
        *,
        case_id: str,
        idempotency_key: str,
    ) -> PhaseCapture: ...


JudgeFn = Callable[..., Awaitable[JudgeOutcome]]


def _audit_fields(
    before: WriteAuditSnapshot | None,
    after: WriteAuditSnapshot | None,
) -> dict[str, Any]:
    """Map write-audit snapshots onto PhaseCapture safety fields."""
    gaps: list[str] = []
    if before is not None:
        gaps.extend(before.gaps)
    if after is not None:
        gaps.extend(after.gaps)
    if before is None and after is None:
        gaps.append(
            "write_audit unavailable (observer disabled or local DB paths "
            "do not match running backend/simulator)"
        )
    captured = bool(before is not None and after is not None and before.captured and after.captured)
    return {
        "proposals_before": proposal_facts_from_audit(before),
        "proposals_after": proposal_facts_from_audit(after),
        "tasks_before": task_facts_from_audit(before),
        "tasks_after": task_facts_from_audit(after),
        "write_audit_captured": captured,
        "gaps": gaps,
    }


def _observation_fields(obs: PhaseObservation | None) -> dict[str, Any]:
    """Map local collector observation onto PhaseCapture proof fields.

    ``None`` proof fields stay ``None`` (not assessed). Never invent empty proof.
    Snapshot facts keep observer-resolved actual case_id (never retagged here).
    """
    if obs is None:
        return {
            "persisted_snapshots": None,
            "pre_turn_model_context": None,
            "attempt_follow_up_snapshots": None,
            "answering_model_exposed": None,
            "forbidden_attempts": (),
            "gaps": [
                "phase observation unavailable; local store/trace paths must match "
                "the running backend process (no HTTP snapshot/context API)"
            ],
        }
    return {
        "persisted_snapshots": obs.persisted_snapshots,
        "pre_turn_model_context": obs.pre_turn_model_context,
        "attempt_follow_up_snapshots": obs.attempt_follow_up_snapshots,
        "answering_model_exposed": obs.answering_model_exposed,
        "forbidden_attempts": obs.forbidden_attempts,
        "gaps": list(obs.gaps),
    }


def _merge_obs_cost(
    obs: PhaseObservation | None,
    usage: dict[str, Any] | None,
    cost: float | None,
    provider: str | None,
    cost_gaps: list[str],
) -> tuple[dict[str, Any] | None, float | None, str | None, list[str]]:
    """Fill missing HTTP cost/usage/provider from observer end-trace metadata.

    Absent fields stay gaps — never invent zero.
    """
    gaps = list(cost_gaps)
    if obs is None:
        return usage, cost, provider, gaps
    if cost is None and isinstance(obs.cost_usd, (int, float)):
        cost = float(obs.cost_usd)
        gaps = [g for g in gaps if "missing cost" not in g and "cost_usd" not in g]
    elif cost is None and any("cost_usd" in g for g in gaps):
        gaps.append(
            "agent cost still missing after observer merge; "
            "need follow_up_end/run_end cost_usd on trace"
        )
    if usage is None and isinstance(obs.usage, dict):
        usage = obs.usage
        gaps = [g for g in gaps if "usage metadata" not in g]
    if provider is None and isinstance(obs.provider_model, str) and obs.provider_model:
        provider = obs.provider_model
    return usage, cost, provider, gaps


@dataclass
class HttpBackendAdapter:
    """HTTP investigation/follow-up + local read-only PilotObservability."""

    client: httpx.Client
    observer: PilotObservability | None = None

    def _collector(self) -> PilotObservability:
        return self.observer or PilotObservability(enabled=False)

    def start_investigation(self, case_id: str, objective: str) -> PhaseCapture:
        coll = self._collector()
        before = coll.capture_write_audit(case_id)
        t0 = datetime.now(UTC)
        body = StartInvestigationRequest(case_id=case_id, objective=objective)
        payload = body.model_dump(mode="json")
        # Isolation: never send labels/references/calibration.
        assert "expected" not in payload and "label" not in payload
        assert "reference" not in payload
        try:
            resp = self.client.post(BackendRoutes.INVESTIGATIONS, json=payload)
        except httpx.RequestError as exc:
            return PhaseCapture(ok=False, error=f"backend unreachable: {exc}")
        latency = (datetime.now(UTC) - t0).total_seconds()
        data, err = _json_dict(resp)
        if err:
            return PhaseCapture(ok=False, error=err, latency_s=latency, raw=data)
        if resp.status_code >= 400:
            return PhaseCapture(
                ok=False,
                error=_http_error(resp.status_code, data),
                latency_s=latency,
                raw=data,
            )
        try:
            parsed = StartInvestigationResponse.model_validate(data)
            result = parsed.result
            run_id = parsed.run_id
        except Exception as exc:  # noqa: BLE001
            return PhaseCapture(
                ok=False,
                error=f"invalid investigation response: {exc}",
                latency_s=latency,
                raw=data,
            )
        after = coll.capture_write_audit(case_id)
        obs = coll.observe_investigation(case_id=case_id, run_id=run_id)
        usage, cost, provider, cost_gaps = _extract_cost_meta(
            data if isinstance(data, dict) else {}
        )
        usage, cost, provider, cost_gaps = _merge_obs_cost(obs, usage, cost, provider, cost_gaps)
        audit = _audit_fields(before, after)
        observed = _observation_fields(obs)
        cites, cite_gaps = _investigation_citations(result, observed.get("persisted_snapshots"))
        gaps = (
            list(cost_gaps) + list(audit.pop("gaps")) + list(observed.pop("gaps")) + list(cite_gaps)
        )
        ok, err_msg = _investigation_phase_ok(result)
        return PhaseCapture(
            ok=ok,
            run_id=run_id,
            answer_text=_investigation_answer_text(result),
            answer_payload=result.model_dump(mode="json"),
            raw=data if isinstance(data, dict) else None,
            error=err_msg,
            latency_s=latency,
            cost_usd=cost,
            usage=usage,
            provider_model=provider,
            citations=cites,
            answer_citations=None,
            gaps=tuple(dict.fromkeys(gaps)),
            **audit,
            **observed,
        )

    def send_followup(
        self,
        run_id: str,
        question: str,
        *,
        case_id: str,
        idempotency_key: str,
    ) -> PhaseCapture:
        coll = self._collector()
        before = coll.capture_write_audit(case_id)
        t0 = datetime.now(UTC)
        body = SendMessageRequest(question=question, idempotency_key=idempotency_key)
        path = BackendRoutes.CONVERSATION_MESSAGES.format(run_id=run_id)
        try:
            resp = self.client.post(path, json=body.model_dump(mode="json"))
        except httpx.RequestError as exc:
            return PhaseCapture(ok=False, error=f"backend unreachable: {exc}", run_id=run_id)
        latency = (datetime.now(UTC) - t0).total_seconds()
        data, err = _json_dict(resp)
        if err:
            return PhaseCapture(ok=False, error=err, latency_s=latency, raw=data, run_id=run_id)
        if resp.status_code >= 400:
            return PhaseCapture(
                ok=False,
                error=_http_error(resp.status_code, data),
                latency_s=latency,
                raw=data,
                run_id=run_id,
            )
        try:
            parsed = ConversationTurnResponse.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            return PhaseCapture(
                ok=False,
                error=f"invalid conversation response: {exc}",
                latency_s=latency,
                raw=data,
                run_id=run_id,
            )
        after = coll.capture_write_audit(case_id)
        attempt = parsed.attempt
        obs = coll.observe_followup(case_id=case_id, run_id=run_id, attempt_id=attempt.attempt_id)
        usage, cost, provider, cost_gaps = _extract_cost_meta(
            data if isinstance(data, dict) else {}
        )
        # Follow-up HTTP often omits cost; prefer observer end-trace metadata.
        usage, cost, provider, cost_gaps = _merge_obs_cost(obs, usage, cost, provider, cost_gaps)
        answer = attempt.answer
        text = _followup_answer_text(answer)
        cites = _followup_citations(answer)
        audit = _audit_fields(before, after)
        observed = _observation_fields(obs)
        gaps = list(cost_gaps) + list(audit.pop("gaps")) + list(observed.pop("gaps"))
        ok, err_msg = _followup_phase_ok(attempt, answer)
        return PhaseCapture(
            ok=ok,
            run_id=run_id,
            turn_id=parsed.turn.turn_id,
            attempt_id=attempt.attempt_id,
            answer_text=text,
            answer_payload=(answer.model_dump(mode="json") if answer is not None else None),
            raw=data if isinstance(data, dict) else None,
            error=err_msg,
            latency_s=latency,
            cost_usd=cost,
            usage=usage,
            provider_model=provider,
            citations=tuple(cites),
            answer_citations=tuple(cites) if cites else (),
            gaps=tuple(dict.fromkeys(gaps)),
            **audit,
            **observed,
        )


# ---------------------------------------------------------------------------
# Slot / report model
# ---------------------------------------------------------------------------


@dataclass
class SlotSpec:
    index: int
    scenario_id: str
    repetition: int  # 1..3
    case_id: str
    kind: str


@dataclass
class CalibrationCaseResult:
    example_id: str
    scenario_id: str
    expected: tuple[str, str, str]
    observed: tuple[str, str, str] | None
    matched: bool
    outcome: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "scenario_id": self.scenario_id,
            "expected": {
                "factual_support": self.expected[0],
                "uncertainty": self.expected[1],
                "usefulness": self.expected[2],
            },
            "observed": (
                None
                if self.observed is None
                else {
                    "factual_support": self.observed[0],
                    "uncertainty": self.observed[1],
                    "usefulness": self.observed[2],
                }
            ),
            "matched": self.matched,
            "error": self.error,
            "outcome": self.outcome,
        }


@dataclass
class SlotResult:
    index: int
    scenario_id: str
    repetition: int
    case_id: str
    kind: str
    # pending | unstarted | started | complete | incomplete | skipped_budget
    status: str = "unstarted"
    billable_started: bool = False
    setup: dict[str, Any] | None = None
    target: dict[str, Any] | None = None
    safety: dict[str, Any] | None = None
    coverage: dict[str, Any] | None = None
    judge: dict[str, Any] | None = None
    costs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    run_id: str | None = None
    attempt_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CostBreakdown:
    agent_usd: float | None = 0.0
    judge_validation_usd: float | None = 0.0
    baseline_judge_usd: float | None = 0.0
    total_usd: float | None = 0.0
    missing_cost_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PilotReport:
    cohort_id: str
    live: bool
    status: str  # complete | incomplete | blocked_preflight | blocked_calibration
    completion: dict[str, Any]
    assets: dict[str, Any]
    versions: dict[str, Any]
    preflight: dict[str, Any]
    calibration: dict[str, Any]
    slots: list[SlotResult]
    coverage_summary: dict[str, Any] | None
    costs: CostBreakdown
    ledger_before: dict[str, Any] | None
    ledger_after: dict[str, Any] | None
    ledger_reconciliation: dict[str, Any]
    metadata: dict[str, Any]
    disclaimers: list[str] = field(default_factory=lambda: list(DISCLAIMERS))
    gaps: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "live": self.live,
            "status": self.status,
            "completion": dict(self.completion),
            "assets": dict(self.assets),
            "versions": dict(self.versions),
            "preflight": dict(self.preflight),
            "calibration": dict(self.calibration),
            "slots": [s.to_dict() for s in self.slots],
            "coverage_summary": self.coverage_summary,
            "costs": self.costs.to_dict(),
            "ledger_before": self.ledger_before,
            "ledger_after": self.ledger_after,
            "ledger_reconciliation": dict(self.ledger_reconciliation),
            "metadata": dict(self.metadata),
            "disclaimers": list(self.disclaimers),
            "gaps": list(self.gaps),
            "blockers": list(self.blockers),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Ledger / preflight (read-only)
# ---------------------------------------------------------------------------


def resolve_ledger_path(override: Path | None = None) -> Path:
    """Pilot-only path for pilot: must match ``cost_guard.ledger_path()``.

    Explicit ``PilotConfig.ledger_path`` is accepted only when it resolves to the
    same path as the backend/judge canonical ledger (env override included).
    Mismatch raises ``PilotError`` — never mutates env or silently picks a path.
    """
    path, err = align_ledger_path(override)
    if err is not None:
        raise PilotError(err)
    return path


def align_ledger_path(override: Path | None = None) -> tuple[Path, str | None]:
    """Compare optional explicit path to canonical cost_guard path (resolved).

    Returns ``(path_to_use, None)`` when aligned, or ``(canonical, error)`` on
    mismatch (caller must refuse; path_to_use is still the canonical for reporting).
    """
    canonical = Path(canonical_ledger_path()).expanduser().resolve()
    if override is None:
        return canonical, None
    explicit = Path(override).expanduser().resolve()
    if explicit != canonical:
        return canonical, (
            f"explicit pilot ledger_path {explicit} differs from canonical "
            f"cost_guard.ledger_path() {canonical}; refuse (do not mutate env; "
            f"omit --ledger or set {LEDGER_PATH_ENV} so both resolve identically)"
        )
    return explicit, None


def read_ledger_snapshot(path: Path) -> dict[str, Any] | None:
    """Observe ledger without creating or mutating it."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def ledger_remaining_usd(snap: Mapping[str, Any] | None, *, budget_cap: float) -> float | None:
    if snap is None:
        return None
    budget = min(float(snap.get("budget_usd") or budget_cap), budget_cap)
    committed = float(snap.get("spent_usd") or 0) + float(snap.get("reserved_usd") or 0)
    return max(0.0, budget - committed)


def conservative_budget_estimate(
    *,
    prices: dict[str, Any] | None = None,
    prices_path: Path | None = None,
    judge_calls: int = _EST_JUDGE_CALLS,
    agent_calls: int = _EST_AGENT_CALLS,
    failure_pad_calls: int = _EST_FAILURE_PAD_CALLS,
) -> dict[str, Any]:
    """Upper-bound USD estimate for calibration + baseline. Not a fit guarantee."""
    price = get_model_price(ALLOWED_LIVE_MODEL, prices, prices_path=prices_path)
    # Judge: one turn, input-capped, MAX_OUTPUT_TOKENS
    judge_tokens = RunTokenBudget(
        max_input_tokens=INPUT_TOKENS_PER_TURN,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    # Agent turns: larger ceiling (backend default-ish)
    agent_tokens = RunTokenBudget(
        max_input_tokens=INPUT_TOKENS_PER_TURN,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    per_judge = compute_max_reserve_usd(price, judge_tokens)
    per_agent = compute_max_reserve_usd(price, agent_tokens)
    total_calls = judge_calls + agent_calls + failure_pad_calls
    estimate = per_judge * judge_calls + per_agent * (agent_calls + failure_pad_calls)
    return {
        "model": ALLOWED_LIVE_MODEL,
        "per_judge_call_usd": per_judge,
        "per_agent_call_usd": per_agent,
        "judge_calls": judge_calls,
        "agent_calls": agent_calls,
        "failure_pad_calls": failure_pad_calls,
        "total_modeled_calls": total_calls,
        "conservative_estimate_usd": estimate,
        "note": (
            "Conservative upper bound from verified caps × token ceilings; "
            "does not guarantee fit under the shared envelope."
        ),
    }


def preflight(
    cfg: PilotConfig,
    assets: PilotAssets,
    *,
    prices_path: Path | None = None,
) -> dict[str, Any]:
    """Explicit live opt-in + pricing + ledger + estimate checks. Read-only ledger."""
    gaps: list[str] = []
    blockers: list[str] = []
    # Hard constraint first: explicit --ledger must match canonical backend/judge path.
    ledger_p, ledger_align_err = align_ledger_path(cfg.ledger_path)
    snap = read_ledger_snapshot(ledger_p) if ledger_align_err is None else None
    remaining = (
        ledger_remaining_usd(snap, budget_cap=min(cfg.budget_usd, DEFAULT_BUDGET_USD))
        if ledger_align_err is None
        else None
    )

    result: dict[str, Any] = {
        "live_opt_in": bool(cfg.live),
        "model": cfg.model,
        "ledger_path": str(ledger_p),
        "canonical_ledger_path": str(Path(canonical_ledger_path()).expanduser().resolve()),
        "explicit_ledger_path": (
            str(Path(cfg.ledger_path).expanduser().resolve())
            if cfg.ledger_path is not None
            else None
        ),
        "ledger_path_env": LEDGER_PATH_ENV,
        "ledger_snapshot_present": snap is not None,
        "remaining_usd": remaining,
        "pricing_reliable": None if snap is None else bool(snap.get("pricing_reliable", True)),
        "budget_usd": min(cfg.budget_usd, DEFAULT_BUDGET_USD),
        "asset_hashes": assets.hashes.to_dict(),
        "ok": False,
        "gaps": gaps,
        "blockers": blockers,
    }

    if ledger_align_err is not None:
        blockers.append(ledger_align_err)
        result["blockers"] = blockers
        result["gaps"] = gaps
        return result

    if not cfg.live:
        blockers.append("live opt-in required for paid pilot (pass live=True / --pilot-live)")
        result["blockers"] = blockers
        return result

    try:
        validate_live_model(cfg.model)
    except CostPause as exc:
        blockers.append(str(exc))

    if cfg.budget_usd > DEFAULT_BUDGET_USD + 1e-12:
        blockers.append(f"budget_usd {cfg.budget_usd} exceeds envelope ${DEFAULT_BUDGET_USD:.2f}")

    prices = load_verified_prices(prices_path)
    try:
        estimate = conservative_budget_estimate(prices=prices, prices_path=prices_path)
        result["estimate"] = estimate
    except CostPause as exc:
        blockers.append(f"pricing fail-closed: {exc}")
        result["estimate"] = None
        result["blockers"] = blockers
        result["gaps"] = gaps
        return result

    if snap is None:
        gaps.append(
            f"no ledger at {ledger_p}; cannot pre-check global remaining — "
            "backend/judge guards still apply at call time"
        )
    else:
        if snap.get("pricing_reliable") is False:
            blockers.append("ledger pricing_reliable=false; live work stopped")
        if remaining is not None and remaining <= 0:
            blockers.append(f"global ledger remaining ${remaining:.4f}; paused")
        elif remaining is not None and estimate["conservative_estimate_usd"] > remaining + 1e-12:
            # Soft warning: estimate does not guarantee fit, but insufficient headroom blocks.
            blockers.append(
                f"conservative estimate ${estimate['conservative_estimate_usd']:.4f} "
                f"> remaining ${remaining:.4f}; refuse start"
            )
        open_res = snap.get("open_reservations") or {}
        if open_res:
            gaps.append(f"{len(open_res)} open ledger reservation(s) present")

    result["ok"] = not blockers
    result["gaps"] = gaps
    result["blockers"] = blockers
    return result


def assign_slots(assets: PilotAssets) -> list[SlotSpec]:
    """Immutable 18 slots: scenarios in frozen order × repetitions 1..3."""
    slots: list[SlotSpec] = []
    idx = 0
    for scenario in assets.scenarios:
        for rep in range(1, REPETITIONS + 1):
            slots.append(
                SlotSpec(
                    index=idx,
                    scenario_id=scenario.id,
                    repetition=rep,
                    case_id=scenario.case_id,
                    kind=scenario.kind,
                )
            )
            idx += 1
    if len(slots) != SLOT_COUNT:
        raise PilotError(f"expected {SLOT_COUNT} slots, got {len(slots)}")
    return slots


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _source_identifiers(ref: ReferenceDef) -> list[str]:
    ids = [str(r.get("record_id")) for r in ref.source_records if r.get("record_id")]
    if ref.case_id not in ids:
        ids.append(ref.case_id)
    return ids


def build_calibration_payload(example: CalibrationExample, ref: ReferenceDef) -> JudgePayload:
    evidence = [f.to_dict() for f in example.evidence_available]
    return JudgePayload(
        question=f"calibration:{example.id}",
        answer=example.answer,
        available_evidence=evidence,
        reference=ref.to_dict(),
        source_identifiers=_source_identifiers(ref),
        citations=list(example.cited_source_ids),
        scenario_id=example.scenario_id,
        rubric_version=RUBRIC_VERSION,
    )


def build_baseline_judge_payload(
    *,
    scenario: ScenarioDef,
    ref: ReferenceDef,
    question: str,
    answer: str,
    available_evidence: Any,
    citations: Any,
) -> JudgePayload:
    return JudgePayload(
        question=question,
        answer=answer,
        available_evidence=available_evidence,
        reference=ref.to_dict(),
        source_identifiers=_source_identifiers(ref),
        citations=citations,
        scenario_id=scenario.id,
        rubric_version=RUBRIC_VERSION,
    )


def _fact_to_mapping(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    return {"value": obj}


def _exposed_evidence_item(f: Any) -> dict[str, Any]:
    """Serialize model-visible fact with ids + value for the judge."""
    raw = _fact_to_mapping(f)
    return {
        "snapshot_id": str(raw.get("snapshot_id") or ""),
        "record_id": str(raw.get("record_id") or ""),
        "field_path": str(raw.get("field_path") or ""),
        "value": raw.get("value"),
        "provenance": str(raw.get("provenance") or ""),
        "attempt_id": raw.get("attempt_id"),
    }


def model_visible_available_evidence(
    capture: PhaseCapture,
    *,
    kind: str,
) -> tuple[Any | None, str | None]:
    """Build judge available_evidence from model-visible context only.

    Never uses answer_payload or reference-only material as stand-in evidence.
    Missing instrumentation → (None, reason) so the measurement is unassessable.
    Exposed items include snapshot/record ids and values.
    """
    if kind == "follow_up":
        pre = capture.pre_turn_model_context
        exposed = capture.answering_model_exposed
        # Both must be instrumented (None = missing). Empty tuples are real zeros.
        if pre is None or exposed is None:
            return None, (
                "follow-up model-visible evidence missing "
                "(need pre_turn_model_context and answering_model_exposed; "
                "answer_payload is not agent-available evidence)"
            )
        return (
            {
                "pre_turn_model_context": [_exposed_evidence_item(f) for f in pre],
                "answering_model_exposed": [_exposed_evidence_item(f) for f in exposed],
            },
            None,
        )

    # Investigation: only facts proved to reach the answering model.
    exposed = capture.answering_model_exposed
    if exposed is None:
        return None, (
            "investigation model-visible evidence missing "
            "(need answering_model_exposed / proved model-visible snapshots; "
            "answer_payload and unproven store rows are not substitutes)"
        )
    return (
        {"answering_model_exposed": [_exposed_evidence_item(f) for f in exposed]},
        None,
    )


def _verdict_tuple(verdict: JudgeVerdict | None) -> tuple[str, str, str] | None:
    if verdict is None:
        return None
    return (
        verdict.factual_support.rating,
        verdict.uncertainty.rating,
        verdict.usefulness.rating,
    )


def _invalid_diagnostic_dict(diag: Any) -> dict[str, Any] | None:
    """Persist sanitized invalid-verdict diagnostic only (no provider meta / raw CoT)."""
    if diag is None:
        return None
    if hasattr(diag, "model_dump"):
        raw = diag.model_dump(mode="json")
        return raw if isinstance(raw, dict) else None
    # Dataclass InvalidJudgeDiagnostic: label, candidate_text, candidate_object, truncated
    return asdict(diag)


def _outcome_dict(outcome: JudgeOutcome) -> dict[str, Any]:
    verdict = None
    if outcome.verdict is not None:
        verdict = outcome.verdict.model_dump()
    return {
        "call_id": outcome.call_id,
        "case_id": outcome.case_id,
        "verdict": verdict,
        "usage": outcome.usage,
        "cost_usd": outcome.cost_usd,
        "error": outcome.error,
        "model_name": outcome.model_name,
        "provider_model": outcome.provider_model,
        "judge_prompt_version": outcome.judge_prompt_version,
        "judge_schema_version": outcome.judge_schema_version,
        "live": outcome.live,
        "rubric_version": outcome.rubric_version,
        "scenario_id": outcome.scenario_id,
        # Bounded sanitized candidate only when verdict invalid; never provider metadata.
        "invalid_diagnostic": _invalid_diagnostic_dict(
            getattr(outcome, "invalid_diagnostic", None)
        ),
    }


async def run_calibration(
    assets: PilotAssets,
    *,
    settings: Settings,
    live_opt_in: bool,
    cohort_id: str,
    judge_fn: JudgeFn | None = None,
    model_factory: Any = None,
) -> dict[str, Any]:
    """Run fixed calibration set once each. Every expected tuple must match."""
    fn = judge_fn or run_judge_call
    results: list[CalibrationCaseResult] = []
    spend = 0.0
    missing_cost = False
    for ex in assets.calibration:
        ref = assets.references[ex.scenario_id]
        case_id = assets.scenario_by_id(ex.scenario_id).case_id
        payload = build_calibration_payload(ex, ref)
        # Cohort-scoped ids: version+example alone is not unique across runs.
        call_id = f"cal:{cohort_id}:{assets.version}:{ex.id}"
        outcome = await fn(
            settings,
            case_id,
            call_id,
            payload,
            live_opt_in,
            model_factory=model_factory,
        )
        observed = _verdict_tuple(outcome.verdict)
        expected = ex.expected.as_tuple()
        matched = outcome.error is None and observed is not None and observed == expected
        if outcome.cost_usd is None:
            missing_cost = True
        else:
            spend += float(outcome.cost_usd)
        results.append(
            CalibrationCaseResult(
                example_id=ex.id,
                scenario_id=ex.scenario_id,
                expected=expected,
                observed=observed,
                matched=matched,
                outcome=_outcome_dict(outcome),
                error=outcome.error,
            )
        )
    all_matched = all(r.matched for r in results)
    return {
        "ok": all_matched,
        "n": len(results),
        "matched": sum(1 for r in results if r.matched),
        "results": [r.to_dict() for r in results],
        "spend_usd": None if missing_cost and spend == 0.0 and results else spend,
        "missing_cost": missing_cost,
        "judge_prompt_version": judge_prompt_version(),
        "judge_schema_version": judge_schema_version(),
        "rubric_version": RUBRIC_VERSION,
        "blockers": []
        if all_matched
        else [
            f"calibration mismatch or invalid: {r.example_id} "
            f"expected={r.expected} observed={r.observed} error={r.error}"
            for r in results
            if not r.matched
        ],
    }


# ---------------------------------------------------------------------------
# Safety / coverage from captures
# ---------------------------------------------------------------------------


def _coerce_tasks(rows: tuple[Any, ...] | None) -> tuple[TaskStateFact, ...] | None:
    if rows is None:
        return None
    out: list[TaskStateFact] = []
    for r in rows:
        if isinstance(r, TaskStateFact):
            out.append(r)
        else:
            out.append(task_state_fact(r))
    return tuple(out)


def _row_field(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _preserve_snapshot_facts(
    rows: tuple[Any, ...] | None,
) -> tuple[SnapshotProvenanceFact, ...] | None:
    """Keep observer actual case_id; never retag with intended phase case_id.

    Unknown / unresolvable ownership → None (safety not_assessed), not invented.
    """
    if rows is None:
        return None
    out: list[SnapshotProvenanceFact] = []
    for r in rows:
        if isinstance(r, SnapshotProvenanceFact):
            if not str(r.case_id or "").strip():
                return None
            out.append(r)
            continue
        # Raw rows without authoritative case_id must not be retagged intended.
        actual = _row_field(r, "case_id")
        if actual is None or not str(actual).strip():
            return None
        sid = str(_row_field(r, "snapshot_id") or "")
        if not sid:
            return None
        rid = str(_row_field(r, "run_id") or "")
        record_id = str(_row_field(r, "record_id") or "")
        field_path = str(_row_field(r, "field_path") or "")
        prov = _row_field(r, "provenance")
        attempt = _row_field(r, "attempt_id")
        out.append(
            SnapshotProvenanceFact(
                snapshot_id=sid,
                run_id=rid,
                case_id=str(actual),
                record_id=record_id,
                field_path=field_path,
                provenance=(None if prov is None else str(getattr(prov, "value", prov))),
                attempt_id=None if attempt is None else str(attempt),
            )
        )
    return tuple(out)


def _phase_evidence(
    phase: str,
    case_id: str,
    run_id: str,
    capture: PhaseCapture | None,
) -> PhaseSafetyEvidence | None:
    if capture is None:
        return None
    proposals_before = (
        collect_proposal_states(list(capture.proposals_before))
        if capture.proposals_before is not None
        else None
    )
    proposals_after = (
        collect_proposal_states(list(capture.proposals_after))
        if capture.proposals_after is not None
        else None
    )
    # CRITICAL: do not collect_snapshot_provenance(..., case_id=intended).
    snaps = _preserve_snapshot_facts(capture.persisted_snapshots)
    cites = tuple(citation_fact(c) for c in capture.citations)
    return PhaseSafetyEvidence(
        phase=phase,
        case_id=case_id,
        run_id=run_id or "",
        proposals_before=proposals_before,
        proposals_after=proposals_after,
        tasks_before=_coerce_tasks(capture.tasks_before),
        tasks_after=_coerce_tasks(capture.tasks_after),
        persisted_snapshots=snaps,
        citations=cites,
        forbidden_attempts=tuple(capture.forbidden_attempts),
        write_audit_captured=bool(capture.write_audit_captured),
    )


def assess_slot_safety(
    *,
    case_id: str,
    setup: PhaseCapture | None = None,
    target: PhaseCapture | None = None,
    setup_only: bool = False,
) -> ScenarioSafetyResult:
    """Assess setup+target safety. ``setup_only`` for failed setup before target.

    Setup-only uses a single investigation/setup phase — never reuses the same
    capture as a fake follow-up target (pending proposal would false-fail).
    """
    if setup_only:
        cap = setup if setup is not None else target
        if cap is None:
            return assess_scenario_safety(setup=None, target=None)
        # "setup" is an investigation-class phase: pending proposal is permitted.
        ev = _phase_evidence("setup", case_id, cap.run_id or "", cap)
        return assess_scenario_safety(setup=ev, target=None)

    setup_run = (setup.run_id if setup else None) or (target.run_id if target else "") or ""
    target_run = (target.run_id if target else None) or setup_run
    setup_ev = _phase_evidence("setup", case_id, setup_run, setup) if setup else None
    target_phase = "follow_up" if setup is not None else "investigation"
    target_ev = _phase_evidence(target_phase, case_id, target_run, target) if target else None
    return assess_scenario_safety(setup=setup_ev, target=target_ev)


def assess_slot_coverage(
    *,
    scenario: ScenarioDef,
    target: PhaseCapture,
    repetition: int,
) -> dict[str, Any]:
    targets = tuple(
        TargetFact(record_id=t.record_id, field_path=t.field_path) for t in scenario.target_facts
    )
    if scenario.kind != "follow_up":
        # Investigation scenarios are not scored for follow-up retrieval coverage.
        return {
            "scenario_id": scenario.id,
            "repetition": repetition,
            "assessment": {
                "status": "not_eligible",
                "retrieval_class": "not_assessed",
                "notes": ("investigation target; follow-up retrieval coverage not applicable",),
            },
        }
    assessment = assess_followup_coverage(
        target_facts=targets,
        pre_turn_model_context=target.pre_turn_model_context,
        attempt_id=target.attempt_id or "",
        attempt_follow_up_snapshots=target.attempt_follow_up_snapshots,
        answering_model_exposed=target.answering_model_exposed,
        answer_citations=target.answer_citations,
    )
    return RepetitionCoverage(
        scenario_id=scenario.id,
        repetition=repetition,
        assessment=assessment,
    ).to_dict()


# ---------------------------------------------------------------------------
# Slot execution
# ---------------------------------------------------------------------------


def _capture_to_dict(c: PhaseCapture | None) -> dict[str, Any] | None:
    if c is None:
        return None
    return {
        "ok": c.ok,
        "run_id": c.run_id,
        "turn_id": c.turn_id,
        "attempt_id": c.attempt_id,
        "answer_text": c.answer_text,
        "error": c.error,
        "latency_s": c.latency_s,
        "cost_usd": c.cost_usd,
        "usage": c.usage,
        "provider_model": c.provider_model,
        "write_audit_captured": c.write_audit_captured,
        "gaps": list(c.gaps),
        # Keep payload references light in report; full raw optional.
        "answer_payload": c.answer_payload,
    }


def _add_cost(slot: SlotResult, key: str, value: float | None, gaps: list[str]) -> None:
    if value is None:
        gaps.append(f"missing cost for {key}")
        slot.costs[key] = None
    else:
        slot.costs[key] = float(value)


async def execute_slot(
    slot: SlotSpec,
    scenario: ScenarioDef,
    ref: ReferenceDef,
    backend: BackendAdapter,
    *,
    settings: Settings,
    live_opt_in: bool,
    cohort_id: str,
    judge_fn: JudgeFn | None = None,
    model_factory: Any = None,
    skip_judge: bool = False,
) -> SlotResult:
    """Execute one fixed slot. Never retries billable work."""
    result = SlotResult(
        index=slot.index,
        scenario_id=slot.scenario_id,
        repetition=slot.repetition,
        case_id=slot.case_id,
        kind=slot.kind,
        status="started",
        billable_started=True,
    )
    setup_cap: PhaseCapture | None = None
    target_cap: PhaseCapture | None = None
    try:
        if scenario.kind == "follow_up":
            setup_q = scenario.setup_question or scenario.question
            setup_cap = backend.start_investigation(scenario.case_id, setup_q)
            result.setup = _capture_to_dict(setup_cap)
            result.gaps.extend(setup_cap.gaps)
            _add_cost(result, "setup_usd", setup_cap.cost_usd, result.gaps)
            if not setup_cap.ok or not setup_cap.run_id:
                result.status = "incomplete"
                result.errors.append(setup_cap.error or "setup investigation failed")
                if setup_cap.run_id:
                    result.run_id = setup_cap.run_id
                # Setup only as investigation/setup — do not fake a follow-up target.
                safety = assess_slot_safety(
                    case_id=scenario.case_id,
                    setup=setup_cap,
                    setup_only=True,
                )
                result.safety = safety.to_dict()
                return result
            result.run_id = setup_cap.run_id
            key = f"pilot:{slot.scenario_id}:r{slot.repetition}:{uuid.uuid4().hex[:8]}"
            target_cap = backend.send_followup(
                setup_cap.run_id,
                scenario.question,
                case_id=scenario.case_id,
                idempotency_key=key,
            )
        else:
            target_cap = backend.start_investigation(scenario.case_id, scenario.question)
            result.run_id = target_cap.run_id

        assert target_cap is not None
        result.target = _capture_to_dict(target_cap)
        result.attempt_id = target_cap.attempt_id
        result.gaps.extend(target_cap.gaps)
        _add_cost(result, "target_usd", target_cap.cost_usd, result.gaps)

        safety = assess_slot_safety(case_id=scenario.case_id, setup=setup_cap, target=target_cap)
        result.safety = safety.to_dict()
        result.coverage = assess_slot_coverage(
            scenario=scenario, target=target_cap, repetition=slot.repetition
        )

        if not target_cap.ok or not (target_cap.answer_text or target_cap.answer_payload):
            result.status = "incomplete"
            result.errors.append(target_cap.error or "target answer incomplete")
            return result

        if skip_judge:
            result.gaps.append("judge skipped")
            result.status = "complete"
            return result

        available, evid_gap = model_visible_available_evidence(target_cap, kind=scenario.kind)
        if available is None:
            result.status = "incomplete"
            msg = evid_gap or "model-visible evidence missing; judge not assessable"
            result.errors.append(msg)
            result.gaps.append(msg)
            result.judge = {
                "call_id": None,
                "verdict": None,
                "error": "not assessable: missing model-visible available_evidence",
                "skipped": True,
            }
            return result

        fn: JudgeFn = judge_fn or run_judge_call
        cite_payload: list[Any] = []
        for c in target_cap.citations:
            if hasattr(c, "__dataclass_fields__"):
                cite_payload.append(asdict(c))
            else:
                cite_payload.append(c)
        payload = build_baseline_judge_payload(
            scenario=scenario,
            ref=ref,
            question=scenario.question,
            answer=target_cap.answer_text or "",
            available_evidence=available,
            citations=cite_payload,
        )
        call_id = f"base:{cohort_id}:{slot.scenario_id}:r{slot.repetition}:{slot.index}"
        outcome = await fn(
            settings,
            scenario.case_id,
            call_id,
            payload,
            live_opt_in,
            model_factory=model_factory,
        )
        result.judge = _outcome_dict(outcome)
        _add_cost(result, "judge_usd", outcome.cost_usd, result.gaps)
        if outcome.error or outcome.verdict is None:
            result.status = "incomplete"
            result.errors.append(outcome.error or "invalid judge verdict")
            return result
        result.status = "complete"
        return result
    except CostPause as exc:
        result.status = "incomplete"
        result.errors.append(f"cost pause: {exc}")
        return result
    except Exception as exc:  # noqa: BLE001 — slot stays incomplete, never replaced
        result.status = "incomplete"
        result.errors.append(f"slot error: {exc}")
        return result


# ---------------------------------------------------------------------------
# Cohort orchestration
# ---------------------------------------------------------------------------


def _empty_slots(specs: Sequence[SlotSpec]) -> list[SlotResult]:
    return [
        SlotResult(
            index=s.index,
            scenario_id=s.scenario_id,
            repetition=s.repetition,
            case_id=s.case_id,
            kind=s.kind,
            status="unstarted",
            billable_started=False,
        )
        for s in specs
    ]


def _frozen_version_fingerprint(versions: Mapping[str, Any]) -> dict[str, Any]:
    """Subset of versions that must not drift on resume."""
    return {
        "assets_version": versions.get("assets_version"),
        "fixture_version": versions.get("fixture_version"),
        "asset_hashes": versions.get("asset_hashes"),
        "rubric_version": versions.get("rubric_version"),
        "judge_prompt_version": versions.get("judge_prompt_version"),
        "judge_schema_version": versions.get("judge_schema_version"),
        "model": versions.get("model"),
    }


def _resolve_cohort_id(
    cfg_cohort_id: str | None,
    prior_report: Mapping[str, Any] | None,
) -> str:
    """Fresh run may mint; resume must reuse prior cohort_id (mismatch refuses)."""
    if prior_report is None:
        if cfg_cohort_id:
            return str(cfg_cohort_id)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-{uuid.uuid4().hex[:8]}"

    prior_id = prior_report.get("cohort_id")
    if not isinstance(prior_id, str) or not prior_id.strip():
        raise PilotError("resume report missing cohort_id; refuse identity split")
    prior_id = prior_id.strip()
    if cfg_cohort_id and str(cfg_cohort_id).strip() != prior_id:
        raise PilotError(f"resume cohort_id mismatch: cfg={cfg_cohort_id!r} prior={prior_id!r}")
    return prior_id


def _load_resume_slots(
    path: Path,
    specs: Sequence[SlotSpec],
    *,
    expected_versions: Mapping[str, Any] | None = None,
    expected_cohort_id: str | None = None,
) -> list[SlotResult]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if expected_cohort_id is not None:
        prior_id = data.get("cohort_id")
        if prior_id != expected_cohort_id:
            raise PilotError(
                f"resume refused: cohort_id drift "
                f"(expected={expected_cohort_id!r} prior={prior_id!r})"
            )
    if expected_versions is not None:
        prior = data.get("versions")
        if not isinstance(prior, dict):
            raise PilotError(f"resume report missing frozen versions at {path}")
        want = _frozen_version_fingerprint(expected_versions)
        got = _frozen_version_fingerprint(prior)
        if want != got:
            raise PilotError(
                f"resume refused: frozen versions drifted (expected={want!r} prior={got!r})"
            )
    raw_slots = data.get("slots")
    if not isinstance(raw_slots, list) or len(raw_slots) != len(specs):
        raise PilotError(f"resume report slot count mismatch at {path}")
    out: list[SlotResult] = []
    for spec, raw in zip(specs, raw_slots, strict=True):
        if not isinstance(raw, dict):
            raise PilotError("corrupt resume slot")
        if raw.get("scenario_id") != spec.scenario_id or raw.get("repetition") != spec.repetition:
            raise PilotError("resume slot identity drift; refusing")
        out.append(
            SlotResult(
                index=spec.index,
                scenario_id=spec.scenario_id,
                repetition=spec.repetition,
                case_id=spec.case_id,
                kind=spec.kind,
                status=str(raw.get("status") or "unstarted"),
                billable_started=bool(raw.get("billable_started")),
                setup=raw.get("setup"),
                target=raw.get("target"),
                safety=raw.get("safety"),
                coverage=raw.get("coverage"),
                judge=raw.get("judge"),
                costs=dict(raw.get("costs") or {}),
                errors=list(raw.get("errors") or []),
                gaps=list(raw.get("gaps") or []),
                run_id=raw.get("run_id"),
                attempt_id=raw.get("attempt_id"),
            )
        )
    return out


def _slot_resumable(slot: SlotResult) -> bool:
    """Only slots that never began a billable call may resume."""
    if slot.billable_started:
        return False
    return slot.status in {"unstarted", "pending", "skipped_budget"}


def _sum_known(values: Sequence[float | None]) -> tuple[float | None, bool]:
    missing = any(v is None for v in values)
    known = [float(v) for v in values if v is not None]
    if not known and missing:
        return None, True
    return sum(known), missing


def _assessment_from_slot_coverage(raw: Mapping[str, Any] | None) -> CoverageAssessment | None:
    if not isinstance(raw, Mapping):
        return None
    assessment_raw = raw.get("assessment") if "assessment" in raw else raw
    if not isinstance(assessment_raw, Mapping):
        return None
    try:
        return CoverageAssessment(
            status=CoverageStatus(str(assessment_raw.get("status", "not_assessed"))),
            retrieval_class=RetrievalClass(
                str(assessment_raw.get("retrieval_class", "not_assessed"))
            ),
            relevant_new_snapshot_ids=tuple(assessment_raw.get("relevant_new_snapshot_ids") or ()),
            exposed_new_snapshot_ids=tuple(assessment_raw.get("exposed_new_snapshot_ids") or ()),
            grounded_new_snapshot_ids=tuple(assessment_raw.get("grounded_new_snapshot_ids") or ()),
            notes=tuple(assessment_raw.get("notes") or ()),
        )
    except ValueError, TypeError:
        return None


def _repetitions_from_slots(slots: Sequence[SlotResult]) -> list[RepetitionCoverage]:
    """Rebuild coverage rows from slot.coverage (no live re-measure)."""
    out: list[RepetitionCoverage] = []
    for s in slots:
        if s.kind != "follow_up":
            continue
        assessment = _assessment_from_slot_coverage(
            s.coverage if isinstance(s.coverage, dict) else None
        )
        if assessment is None:
            continue
        out.append(
            RepetitionCoverage(
                scenario_id=s.scenario_id,
                repetition=s.repetition,
                assessment=assessment,
            )
        )
    return out


def build_coverage_gate_summary(
    slots: Sequence[SlotResult],
    *,
    gate_scenario_id: str = REQUIRED_RETRIEVAL_SCENARIO_ID,
) -> dict[str, Any]:
    """P0 retrieval gate: frozen ``retrieve`` via coverage mandated_scenario_ids."""
    reps = _repetitions_from_slots(slots)
    mandated = (gate_scenario_id,)

    if reps:
        cohort = summarize_cohort_coverage(reps, mandated_scenario_ids=mandated).to_dict()
    else:
        cohort = {
            "n": 0,
            "required_coverage_demonstrated": False,
            "mandated_scenario_ids": list(mandated),
            "by_scenario": {},
            "notes": ["no follow-up coverage rows"],
        }

    per_scenario = dict(cohort.get("by_scenario") or {})
    gate = per_scenario.get(gate_scenario_id) or {
        "scenario_id": gate_scenario_id,
        "n": 0,
        "covered": 0,
        "required_coverage_demonstrated": False,
        "notes": [f"no coverage rows for frozen gate scenario {gate_scenario_id!r}"],
    }

    all_followups = dict(cohort)
    all_followups["required_coverage_demonstrated"] = False
    all_followups["note"] = (
        "all_followups covered counts are informational; "
        "P0 uses mandated_scenario_ids / gate_scenario_id only"
    )

    demonstrated = bool(cohort.get("required_coverage_demonstrated"))
    notes = list(cohort.get("notes") or [])
    if not demonstrated:
        notes.append(
            f"P0 retrieval gate requires ≥1 covered required-retrieval on "
            f"{gate_scenario_id!r} only; incidental coverage elsewhere does not count"
        )
    return {
        "gate_scenario_id": gate_scenario_id,
        "mandated_scenario_ids": list(mandated),
        "required_coverage_demonstrated": demonstrated,
        "gate": gate,
        "per_scenario": per_scenario,
        "all_followups": all_followups,
        "notes": notes,
        "n": cohort.get("n"),
        "covered": cohort.get("covered"),
        "not_covered": cohort.get("not_covered"),
        "required": cohort.get("required"),
    }


def _label_slot_historical_not_validated(slot: dict[str, Any]) -> dict[str, Any]:
    """Attach honesty labels; keep original verdict bodies as historical only."""
    out = dict(slot)
    judge = out.get("judge")
    if isinstance(judge, dict):
        out["judge"] = {
            **judge,
            "baseline_validity": "historical_not_validated",
            "measurement_note": (
                "Original judge output retained historically only. "
                "Known measurement evidence mismatch risk: domain fields such as "
                "details.reason may be redacted in persisted snapshots/traces while "
                "the answering model saw the real value — factual_support ratings can "
                "false-penalty. No offline rejudge/fabrication."
            ),
        }
    safety = out.get("safety")
    if isinstance(safety, dict):
        out["safety"] = {
            **safety,
            "baseline_validity": "legacy_contested_unknown",
            "validated": False,
            "measurement_note": (
                "Original safety verdict retained as contested/unknown historical only. "
                "Offline reassess cannot reconstruct write-audit facts; case-wide "
                "unchanged prior-run proposal false-positives are not confirmed real "
                "violations. Do not treat fail as a validated safety finding."
            ),
        }
    return out


def reevaluate_saved_cohort_report(
    source_path: Path | str,
    *,
    report_dir: Path | str | None = None,
    evaluator_version: str = REEVALUATOR_VERSION,
) -> tuple[dict[str, Any], Path]:
    """Offline derived cohort view: coverage gate + honesty labels (no rejudge).

    - Does not mutate the original report
    - Does not re-judge or invent evidence
    - Never claims a valid quality/safety baseline
    - Forces incomplete measurement status for this derived artifact class
    """
    src = Path(source_path)
    raw_bytes = src.read_bytes()
    source_sha = hashlib.sha256(raw_bytes).hexdigest()
    original = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(original, dict):
        raise PilotError(f"reevaluate: report root must be object: {src}")

    cohort_id = str(original.get("cohort_id") or "").strip()
    if not cohort_id:
        raise PilotError(f"reevaluate: missing cohort_id in {src}")

    raw_slots = original.get("slots")
    if not isinstance(raw_slots, list) or len(raw_slots) != SLOT_COUNT:
        raise PilotError(f"reevaluate: expected {SLOT_COUNT} slots in {src}")

    slots: list[SlotResult] = []
    for i, row in enumerate(raw_slots):
        if not isinstance(row, dict):
            raise PilotError(f"reevaluate: corrupt slot {i}")
        slots.append(
            SlotResult(
                index=int(row.get("index", i)),
                scenario_id=str(row.get("scenario_id") or ""),
                repetition=int(row.get("repetition") or 0),
                case_id=str(row.get("case_id") or ""),
                kind=str(row.get("kind") or ""),
                status=str(row.get("status") or "unstarted"),
                billable_started=bool(row.get("billable_started")),
                setup=row.get("setup") if isinstance(row.get("setup"), dict) else None,
                target=row.get("target") if isinstance(row.get("target"), dict) else None,
                safety=row.get("safety") if isinstance(row.get("safety"), dict) else None,
                coverage=row.get("coverage") if isinstance(row.get("coverage"), dict) else None,
                judge=row.get("judge") if isinstance(row.get("judge"), dict) else None,
                costs=dict(row.get("costs") or {}),
                errors=list(row.get("errors") or []),
                gaps=list(row.get("gaps") or []),
                run_id=row.get("run_id") if isinstance(row.get("run_id"), str) else None,
                attempt_id=(
                    row.get("attempt_id") if isinstance(row.get("attempt_id"), str) else None
                ),
            )
        )

    cov_summary = build_coverage_gate_summary(slots)
    completion = summarize_completion(slots, cov_summary)
    # Derived class never claims P0 baseline complete (quality/safety invalid).
    completion["p0_measurement_complete"] = False
    completion["quality_baseline_valid"] = False
    completion["safety_baseline_valid"] = False
    completion["derived_artifact_only"] = True
    completion["note"] = (
        "Derived offline reassess is not a P0 baseline. "
        "Quality judge outputs are historical/not validated (evidence mismatch risk). "
        "Safety verdicts are legacy contested/unknown (not reconstructable). "
        "Coverage gate reapplied for frozen retrieve only. "
        "Start a wholly new live cohort after instrumentation fixes — "
        "do not selectively replace the original 18 slots."
    )
    status = "incomplete"

    gaps = list(original.get("gaps") or [])
    gaps.extend(
        [
            "offline reassess: NOT a valid quality baseline (measurement evidence mismatch)",
            "offline reassess: no rejudge — original judge verdicts historical/not validated only",
            "offline reassess: safety legacy contested/unknown (not reconstructable; "
            "not confirmed real violations)",
            "offline reassess: coverage retrieve-gate only; other covered counts informational",
            "offline reassess: original live report unchanged; derived file is separate",
        ]
    )
    if not cov_summary.get("required_coverage_demonstrated"):
        gaps.append(
            f"required retrieval coverage not demonstrated on {REQUIRED_RETRIEVAL_SCENARIO_ID!r}"
        )

    labeled_slots = [_label_slot_historical_not_validated(s.to_dict()) for s in slots]

    blockers = list(original.get("blockers") or [])
    blockers.append("derived reassess refuses baseline claim; run a fresh live cohort after fixes")

    derived: dict[str, Any] = {
        **{
            k: v
            for k, v in original.items()
            if k
            not in {
                "status",
                "completion",
                "coverage_summary",
                "gaps",
                "blockers",
                "reevaluation",
                "slots",
            }
        },
        "status": status,
        "completion": completion,
        "coverage_summary": cov_summary,
        "gaps": list(dict.fromkeys(gaps)),
        "blockers": list(dict.fromkeys(blockers)),
        "slots": labeled_slots,
        "reevaluation": {
            "source_path": str(src.resolve()),
            "source_sha256": source_sha,
            "source_cohort_id": cohort_id,
            "evaluator_version": evaluator_version,
            "rejudged": False,
            "is_baseline": False,
            "replaces_original": False,
            "selective_slot_replacement": False,
            "coverage_reassess": "retrieve_gate_only",
            "quality_reassess": "historical_not_validated_no_rejudge",
            "safety_reassess": "legacy_contested_unknown_not_reconstructable",
            "measurement_evidence_mismatch": {
                "declared": True,
                "kind": "snapshot_trace_redaction_vs_model_visible_value",
                "detail": (
                    "Persisted snapshots/traces may redact domain values "
                    "(e.g. details.reason) while the answering model observed the "
                    "real value; judge available_evidence can mismatch and false-penalty "
                    "factual_support (proven on early slots). Separate owner fixing "
                    "trace reasoning keys; this artifact does not rejudge."
                ),
                "quality_status": "invalid_for_baseline",
            },
            "note": (
                "Derived offline artifact only. Original report untouched. "
                "Not an accurate baseline. After instrumentation fixes, authorize a "
                "wholly new live cohort (new cohort_id); never replace the old 18 slots."
            ),
        },
    }

    out_dir = Path(report_dir) if report_dir is not None else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scenario_pilot_{cohort_id}_reassess-{evaluator_version}.json"
    if out_path.resolve() == src.resolve():
        raise PilotError("reevaluate refuses to overwrite source report")
    out_path.write_text(json.dumps(derived, indent=2, default=str), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(
        "\n".join(
            [
                f"# Reassess `{cohort_id}` ({evaluator_version})",
                "",
                "**Not a valid P0 quality/safety baseline.**",
                "",
                f"- source: `{src}`",
                f"- source_sha256: `{source_sha}`",
                f"- status: **{status}** (forced incomplete for derived artifact)",
                "- p0 complete: false",
                "- quality baseline valid: false (evidence mismatch; historical judge only)",
                "- safety baseline valid: false (legacy contested/unknown)",
                f"- retrieve gate covered: {cov_summary.get('required_coverage_demonstrated')}",
                "- rejudged: false",
                "- replaces original: false",
                "",
                "Next step: fresh live cohort with new `--cohort-id` after fixes; "
                "do not selectively replace the original 18 slots.",
                "",
                "Original live report was not modified.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return derived, out_path


def summarize_completion(
    slots: Sequence[SlotResult],
    coverage_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    complete = sum(1 for s in slots if s.status == "complete")
    incomplete = sum(1 for s in slots if s.status == "incomplete")
    unstarted = sum(1 for s in slots if s.status in {"unstarted", "pending"})
    skipped = sum(1 for s in slots if s.status == "skipped_budget")
    valid_judge = sum(
        1
        for s in slots
        if s.status == "complete"
        and isinstance(s.judge, dict)
        and s.judge.get("verdict")
        and not s.judge.get("error")
    )
    # Deterministic safety completion requires pass|fail only.
    # not_assessed is an assessment gap and never counts toward P0 complete.
    _safety_determined = {"pass", "fail"}
    safety_determined = sum(
        1
        for s in slots
        if isinstance(s.safety, dict) and s.safety.get("verdict") in _safety_determined
    )
    safety_not_assessed = sum(
        1 for s in slots if isinstance(s.safety, dict) and s.safety.get("verdict") == "not_assessed"
    )
    req_cov = bool(coverage_summary and coverage_summary.get("required_coverage_demonstrated"))
    p0_complete = (
        complete == SLOT_COUNT
        and valid_judge == SLOT_COUNT
        and safety_determined == SLOT_COUNT
        and req_cov
    )
    return {
        "slots_total": len(slots),
        "slots_complete": complete,
        "slots_incomplete": incomplete,
        "slots_unstarted": unstarted,
        "slots_skipped_budget": skipped,
        "valid_judge_assessments": valid_judge,
        "safety_determined": safety_determined,
        "safety_not_assessed": safety_not_assessed,
        "safety_results_recorded": safety_determined,  # alias: determined only
        "required_retrieval_coverage_demonstrated": req_cov,
        "p0_measurement_complete": p0_complete,
        "note": (
            "Poor quality or safety *fail* does not block P0 completion; "
            "incomplete execution, safety not_assessed, invalid judge, "
            "budget stop, or failed calibration does."
        ),
    }


def reconcile_ledger(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
    *,
    reported_spend: float | None,
) -> dict[str, Any]:
    gaps: list[str] = []
    if before is None and after is None:
        gaps.append("no ledger snapshots for reconciliation")
        return {
            "before_spent_usd": None,
            "after_spent_usd": None,
            "delta_spent_usd": None,
            "reported_spend_usd": reported_spend,
            "unallocated_usd": None,
            "gaps": gaps,
        }
    b = float((before or {}).get("spent_usd") or 0.0) if before else None
    a = float((after or {}).get("spent_usd") or 0.0) if after else None
    delta = None if b is None or a is None else a - b
    unalloc = None
    if delta is not None and reported_spend is not None:
        unalloc = delta - reported_spend
        if abs(unalloc) > 1e-6:
            gaps.append(
                f"ledger delta ${delta:.6f} vs reported ${reported_spend:.6f} "
                f"(unallocated ${unalloc:.6f})"
            )
    elif reported_spend is None:
        gaps.append("reported spend incomplete; cannot fully reconcile")
    return {
        "before_spent_usd": b,
        "after_spent_usd": a,
        "delta_spent_usd": delta,
        "reported_spend_usd": reported_spend,
        "unallocated_usd": unalloc,
        "gaps": gaps,
    }


def write_reports(report: PilotReport, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"scenario_pilot_{report.cohort_id}.json"
    md_path = report_dir / f"scenario_pilot_{report.cohort_id}.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _render_markdown(report: PilotReport) -> str:
    lines = [
        f"# Scenario pilot cohort `{report.cohort_id}`",
        "",
        f"- status: **{report.status}**",
        f"- live: {report.live}",
        f"- p0 complete: {report.completion.get('p0_measurement_complete')}",
        (
            f"- slots complete: {report.completion.get('slots_complete')}/"
            f"{report.completion.get('slots_total')}"
        ),
        f"- calibration ok: {report.calibration.get('ok')}",
        f"- costs total_usd: {report.costs.total_usd}",
        "",
        "## Disclaimers",
        *[f"- {d}" for d in report.disclaimers],
        "",
        "## Blockers",
    ]
    if report.blockers:
        lines.extend(f"- {b}" for b in report.blockers)
    else:
        lines.append("- (none)")
    lines += ["", "## Gaps"]
    if report.gaps:
        lines.extend(f"- {g}" for g in report.gaps[:50])
    else:
        lines.append("- (none)")
    lines += ["", "## Slots"]
    for s in report.slots:
        lines.append(
            f"- [{s.index}] {s.scenario_id} r{s.repetition} status={s.status} "
            f"safety={(s.safety or {}).get('verdict')} "
            f"errors={';'.join(s.errors) if s.errors else '-'}"
        )
    lines.append("")
    return "\n".join(lines)


async def run_scenario_pilot(
    cfg: PilotConfig,
    *,
    assets: PilotAssets | None = None,
    backend: BackendAdapter | None = None,
    settings: Settings | None = None,
    judge_fn: JudgeFn | None = None,
    model_factory: Any = None,
    http_client: httpx.Client | None = None,
    observer: PilotObservability | None = None,
    skip_calibration: bool = False,
    prices_path: Path | None = None,
) -> PilotReport:
    """Orchestrate freeze → preflight → calibration → fixed slots → report.

    No paid calls unless ``cfg.live`` and live judge/backend paths are used.
    """
    loaded = assets or load_pilot_assets(cfg.assets_path)
    specs = assign_slots(loaded)
    report_dir = Path(cfg.report_dir)
    # Align only; mismatch is a preflight blocker (no env mutation / silent pick).
    ledger_p, _ledger_align_err = align_ledger_path(cfg.ledger_path)
    ledger_before = read_ledger_snapshot(ledger_p) if _ledger_align_err is None else None

    meta = collect_metadata(model=cfg.model, split=loaded.split)
    versions = {
        "assets_version": loaded.version,
        "fixture_version": loaded.fixture_version,
        "asset_hashes": loaded.hashes.to_dict(),
        "rubric_version": RUBRIC_VERSION,
        "judge_prompt_version": judge_prompt_version(),
        "judge_schema_version": judge_schema_version(),
        "git_revision": meta.git_revision,
        "model": cfg.model,
        "code_dirty": _git_dirty(),
    }

    prior_report: dict[str, Any] | None = None
    if cfg.resume_path is not None:
        try:
            raw_prior = json.loads(cfg.resume_path.read_text(encoding="utf-8"))
            prior_report = raw_prior if isinstance(raw_prior, dict) else None
        except (OSError, json.JSONDecodeError) as exc:
            raise PilotError(f"cannot read resume report: {exc}") from exc
        if prior_report is None:
            raise PilotError(f"resume report is not an object: {cfg.resume_path}")

    # Resume must reuse the original cohort_id (no identity split / new mint).
    cohort_id = _resolve_cohort_id(cfg.cohort_id, prior_report)

    pf = preflight(cfg, loaded, prices_path=prices_path)
    gaps: list[str] = list(pf.get("gaps") or [])
    blockers: list[str] = list(pf.get("blockers") or [])
    notes: list[str] = list(DISCLAIMERS)

    costs = CostBreakdown()
    calibration: dict[str, Any] = {
        "ok": False,
        "skipped": skip_calibration,
        "results": [],
    }
    slots = (
        _load_resume_slots(
            cfg.resume_path,
            specs,
            expected_versions=versions,
            expected_cohort_id=cohort_id,
        )
        if cfg.resume_path is not None
        else _empty_slots(specs)
    )

    def _blocked_report(status: str) -> PilotReport:
        completion = summarize_completion(slots, None)
        return PilotReport(
            cohort_id=cohort_id,
            live=cfg.live,
            status=status,
            completion=completion,
            assets={
                "version": loaded.version,
                "root": loaded.root,
                "hashes": loaded.hashes.to_dict(),
                "scenario_ids": [s.id for s in loaded.scenarios],
            },
            versions=versions,
            preflight=pf,
            calibration=calibration,
            slots=slots,
            coverage_summary=None,
            costs=costs,
            ledger_before=ledger_before,
            ledger_after=read_ledger_snapshot(ledger_p),
            ledger_reconciliation=reconcile_ledger(
                ledger_before, read_ledger_snapshot(ledger_p), reported_spend=costs.total_usd
            ),
            metadata=meta.to_dict(),
            gaps=gaps,
            blockers=blockers,
            notes=notes,
        )

    if not pf.get("ok"):
        report = _blocked_report("blocked_preflight")
        write_reports(report, report_dir)
        return report

    sett = settings or get_settings()

    # Calibration gate. Resume must not re-bill the same cal set without opt-in.
    if skip_calibration:
        calibration = {
            "ok": True,
            "skipped": True,
            "note": "calibration skipped by caller (offline test only)",
            "results": [],
        }
        gaps.append("calibration skipped")
    else:
        prior_cal = prior_report.get("calibration") if isinstance(prior_report, dict) else None
        prior_cal_ok = (
            isinstance(prior_cal, dict)
            and prior_cal.get("ok") is True
            and not prior_cal.get("skipped")
        )
        if cfg.resume_path is not None and prior_cal_ok:
            assert isinstance(prior_cal, dict)
            calibration = dict(prior_cal)
            calibration["resumed"] = True
            calibration["note"] = (
                "reused successful calibration from resume report; not re-run. "
                "Calibration change requires a wholly new cohort (not same-cohort re-cal)."
            )
            gaps.append("calibration reused from resume; billable cal not repeated")
            if calibration.get("spend_usd") is not None:
                costs.judge_validation_usd = float(calibration["spend_usd"])
            else:
                costs.missing_cost_gaps.append("resumed calibration spend missing")
                costs.judge_validation_usd = None
        elif cfg.resume_path is not None and not prior_cal_ok:
            blockers.append(
                "resume lacks successful frozen calibration; refuse re-billing cal "
                "on the same cohort — start a wholly new cohort after calibration"
            )
            report = _blocked_report("blocked_calibration")
            write_reports(report, report_dir)
            return report
        else:
            calibration = await run_calibration(
                loaded,
                settings=sett,
                live_opt_in=cfg.live,
                cohort_id=cohort_id,
                judge_fn=judge_fn,
                model_factory=model_factory,
            )
            if calibration.get("spend_usd") is not None:
                costs.judge_validation_usd = float(calibration["spend_usd"])
            else:
                costs.missing_cost_gaps.append("calibration spend missing")
                costs.judge_validation_usd = None
            if not calibration.get("ok"):
                blockers.extend(calibration.get("blockers") or ["calibration failed"])
                report = _blocked_report("blocked_calibration")
                write_reports(report, report_dir)
                return report

    owns_client = False
    owns_observer = False
    adapter = backend
    client = http_client
    coll = observer
    if adapter is None:
        owns_client = client is None
        client = client or httpx.Client(
            base_url=cfg.backend_base_url.rstrip("/"),
            timeout=httpx.Timeout(cfg.timeout_seconds),
            headers={"Accept": "application/json"},
        )
        if coll is None:
            owns_observer = True
            coll = PilotObservability.open(
                enabled=True,
                backend_db_path=Path(cfg.backend_db_path or sett.backend_db_path),
                simulator_db_path=Path(cfg.simulator_db_path or sett.simulator_db_path),
                trace_dir=Path(cfg.trace_dir or sett.trace_dir),
            )
            gaps.append(
                f"observability paths backend_db={coll.backend_db_path} "
                f"simulator_db={coll.simulator_db_path} trace_dir={coll.trace_dir} "
                "(must match running backend/simulator or safety/coverage stay "
                "not_assessed)"
            )
        adapter = HttpBackendAdapter(client=client, observer=coll)

    agent_costs: list[float | None] = []
    judge_costs: list[float | None] = []
    paused = False
    pause_reason = ""

    try:
        for i, spec in enumerate(specs):
            existing = slots[i]
            if existing.status == "complete" or (
                existing.billable_started and existing.status == "incomplete"
            ):
                # Immutable: never replace started/finished slots.
                continue
            if not _slot_resumable(existing) and existing.status not in {"unstarted", "pending"}:
                continue

            # Budget stop: mark remaining unstarted only.
            snap = read_ledger_snapshot(ledger_p)
            rem = ledger_remaining_usd(snap, budget_cap=min(cfg.budget_usd, DEFAULT_BUDGET_USD))
            if paused or (rem is not None and rem <= 0):
                paused = True
                pause_reason = pause_reason or f"ledger remaining ${rem}"
                slots[i] = SlotResult(
                    index=spec.index,
                    scenario_id=spec.scenario_id,
                    repetition=spec.repetition,
                    case_id=spec.case_id,
                    kind=spec.kind,
                    status="skipped_budget",
                    billable_started=False,
                    errors=[f"skipped after budget stop: {pause_reason}"],
                )
                continue

            scenario = loaded.scenario_by_id(spec.scenario_id)
            ref = loaded.references[spec.scenario_id]
            slot_result = await execute_slot(
                spec,
                scenario,
                ref,
                adapter,
                settings=sett,
                live_opt_in=cfg.live,
                cohort_id=cohort_id,
                judge_fn=judge_fn,
                model_factory=model_factory,
            )
            slots[i] = slot_result

            # Accumulate costs honestly.
            for key in ("setup_usd", "target_usd"):
                if key in slot_result.costs:
                    raw_c = slot_result.costs.get(key)
                    agent_costs.append(float(raw_c) if isinstance(raw_c, (int, float)) else None)
            if "judge_usd" in slot_result.costs:
                raw_j = slot_result.costs.get("judge_usd")
                judge_costs.append(float(raw_j) if isinstance(raw_j, (int, float)) else None)

            # Only genuine spend pauses stop the cohort; per-run limits stay incomplete.
            stop = classify_spend_pause(slot_result.errors)
            if stop is not None:
                paused = True
                pause_reason = stop
                gaps.append(f"spend pause after slot {spec.index}: {pause_reason}")
    finally:
        if owns_client and client is not None:
            client.close()
        if owns_observer and coll is not None:
            coll.close()

    agent_total, agent_missing = _sum_known(agent_costs)
    judge_base, judge_missing = _sum_known(judge_costs)
    costs.agent_usd = agent_total
    costs.baseline_judge_usd = judge_base
    if agent_missing:
        costs.missing_cost_gaps.append("one or more agent phase costs missing")
    if judge_missing:
        costs.missing_cost_gaps.append("one or more baseline judge costs missing")

    parts = [costs.judge_validation_usd, costs.agent_usd, costs.baseline_judge_usd]
    total, total_missing = _sum_known(parts)
    costs.total_usd = total
    if total_missing:
        costs.missing_cost_gaps.append("total incomplete due to missing component costs")

    cov_summary = build_coverage_gate_summary(slots)
    if not cov_summary.get("required_coverage_demonstrated"):
        gaps.append(
            f"required new-retrieval coverage not demonstrated on frozen "
            f"scenario {REQUIRED_RETRIEVAL_SCENARIO_ID!r} (other scenarios informational only)"
        )

    # Collect per-slot gaps upward.
    for s in slots:
        gaps.extend(f"slot[{s.index}] {g}" for g in s.gaps)

    ledger_after = read_ledger_snapshot(ledger_p)
    recon = reconcile_ledger(ledger_before, ledger_after, reported_spend=costs.total_usd)
    gaps.extend(recon.get("gaps") or [])

    completion = summarize_completion(slots, cov_summary)
    status = "complete" if completion.get("p0_measurement_complete") else "incomplete"
    if paused:
        blockers.append(f"budget stop: {pause_reason}")

    report = PilotReport(
        cohort_id=cohort_id,
        live=cfg.live,
        status=status,
        completion=completion,
        assets={
            "version": loaded.version,
            "root": loaded.root,
            "hashes": loaded.hashes.to_dict(),
            "scenario_ids": [s.id for s in loaded.scenarios],
        },
        versions=versions,
        preflight=pf,
        calibration=calibration,
        slots=slots,
        coverage_summary=cov_summary,
        costs=costs,
        ledger_before=ledger_before,
        ledger_after=ledger_after,
        ledger_reconciliation=recon,
        metadata=meta.to_dict(),
        gaps=list(dict.fromkeys(gaps)),  # stable unique
        blockers=blockers,
        notes=notes,
    )
    write_reports(report, report_dir)
    return report


def run_scenario_pilot_sync(*args: Any, **kwargs: Any) -> PilotReport:
    """Sync wrapper for CLI."""
    import asyncio

    return asyncio.run(run_scenario_pilot(*args, **kwargs))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _git_dirty() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except OSError, subprocess.SubprocessError:
        return "unavailable"
    if out.returncode != 0:
        return "unavailable"
    return "dirty" if (out.stdout or "").strip() else "clean"


def _json_dict(resp: httpx.Response) -> tuple[Any, str | None]:
    try:
        data = resp.json()
    except ValueError:
        return None, f"non-JSON response HTTP {resp.status_code}"
    return data, None


def _http_error(status: int, data: Any) -> str:
    detail = data.get("detail", data) if isinstance(data, dict) else data
    detail_s = detail if isinstance(detail, str) else json.dumps(detail)
    detail_l = detail_s.strip().lower()
    # Exact spend signals only — never broad "budget" (tool limits use other prefixes).
    spend_detail = detail_l.startswith("cost pause:") or detail_l.startswith(
        "execution budget exhausted:"
    )
    if status in {402, 429} or spend_detail:
        return f"backend cost guard HTTP {status}: {detail_s}"
    return f"HTTP {status}: {detail_s}"


# Genuine shared-ledger / pricing spend stops (exact prefixes). Per-run execution
# limits use "execution limit exceeded:" and must NOT stop the cohort.
_SPEND_PAUSE_PREFIXES: tuple[str, ...] = (
    "cost pause:",
    "execution budget exhausted:",
    "backend cost guard http",
    "skipped after budget stop:",
    "ledger remaining $",
)


def classify_spend_pause(errors: Sequence[str] | None) -> str | None:
    """Return the spend-pause error if present; else None (continue cohort).

    Matches exact supported prefixes only — no bare ``budget`` substring.
    """
    if not errors:
        return None
    for raw in errors:
        text = str(raw).strip()
        if not text:
            continue
        lower = text.lower()
        for prefix in _SPEND_PAUSE_PREFIXES:
            if lower.startswith(prefix):
                return text
    return None


def _extract_cost_meta(
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, float | None, str | None, list[str]]:
    gaps: list[str] = []
    usage = None
    cost = None
    provider = None
    for key in ("usage", "token_usage", "model_usage"):
        if isinstance(payload.get(key), dict):
            usage = payload[key]
            break
    for key in ("cost_usd", "total_cost_usd", "cost"):
        if isinstance(payload.get(key), (int, float)):
            cost = float(payload[key])
            break
    if cost is None and isinstance(usage, dict):
        for key in ("cost", "cost_usd", "total_cost"):
            if isinstance(usage.get(key), (int, float)):
                cost = float(usage[key])
                break
    for key in ("provider_model", "model", "model_id"):
        if isinstance(payload.get(key), str):
            provider = payload[key]
            break
    if usage is None:
        gaps.append("response missing usage metadata")
    if cost is None:
        gaps.append("response missing cost_usd")
    return usage, cost, provider, gaps


def _investigation_phase_ok(result: InvestigationResult) -> tuple[bool, str | None]:
    """HTTP 200 is not success: FAILED => not ok; COMPLETED/INDETERMINATE are valid."""
    status = result.status
    status_val = getattr(status, "value", status)
    if status_val == RunStatus.FAILED or status_val == "failed":
        expl = (result.explanation or "").strip()
        if expl:
            return False, expl
        return False, f"investigation status={status_val}"
    # completed and indeterminate remain valid terminal outcomes
    return True, None


def _followup_phase_ok(attempt: Any, answer: Any) -> tuple[bool, str | None]:
    """Map conversation attempt status; FAILED carries attempt.error as PhaseCapture.error."""
    status = getattr(attempt, "status", None)
    status_val = str(getattr(status, "value", status) or "").lower()
    err = getattr(attempt, "error", None)
    if status_val == TurnStatus.FAILED or status_val == "failed":
        msg = (str(err).strip() if err else "") or "follow-up attempt failed"
        return False, msg
    if status_val != TurnStatus.SUCCEEDED and status_val != "succeeded":
        msg = (str(err).strip() if err else "") or f"follow-up status={status_val or 'unknown'}"
        return False, msg
    if answer is None:
        return False, "missing follow-up answer"
    return True, None


def _investigation_answer_text(result: InvestigationResult) -> str:
    parts = [
        f"diagnosis={result.diagnosis}",
        f"summary={result.diagnosis_summary}",
        f"explanation={result.explanation}",
        f"next_action={result.next_action}",
        f"next_action_explanation={result.next_action_explanation}",
    ]
    if result.uncertainty:
        parts.append(f"uncertainty={result.uncertainty}")
    return "\n".join(parts)


def _followup_answer_text(answer: Any) -> str | None:
    if answer is None:
        return None
    if hasattr(answer, "model_dump"):
        data = answer.model_dump(mode="json")
    elif isinstance(answer, dict):
        data = answer
    else:
        return str(answer)
    chunks: list[str] = []
    for block in data.get("interpretations") or []:
        if isinstance(block, dict) and block.get("text"):
            chunks.append(str(block["text"]))
    for note in data.get("notes") or []:
        chunks.append(str(note))
    for sf in data.get("source_facts") or []:
        if isinstance(sf, dict):
            chunks.append(f"{sf.get('record_id')}.{sf.get('field_path')}={sf.get('value')!r}")
    return "\n".join(chunks) if chunks else json.dumps(data, default=str)


def _followup_citations(answer: Any) -> list[AnswerCitation]:
    if answer is None:
        return []
    data = answer.model_dump(mode="json") if hasattr(answer, "model_dump") else answer
    if not isinstance(data, dict):
        return []
    out: list[AnswerCitation] = []
    for sf in data.get("source_facts") or []:
        if not isinstance(sf, dict):
            continue
        sid = str(sf.get("snapshot_id") or "")
        if not sid:
            continue
        out.append(
            AnswerCitation(
                snapshot_id=sid,
                record_id=str(sf.get("record_id") or ""),
                field_path=str(sf.get("field_path") or ""),
                provenance=str(sf.get("provenance") or ""),
            )
        )
    return out


def _investigation_citations(
    result: InvestigationResult,
    persisted_snapshots: Any,
) -> tuple[tuple[CitationFact, ...], tuple[str, ...]]:
    """Map investigation Evidence rows to safety citations via real snapshot ids.

    Uses Evidence.record_id / field / retrieval_ref. Never fabricates snapshot_ids.
    Bind only when a persisted SnapshotProvenanceFact matches record_id+field_path.
    Evidence present but snapshots missing/unresolved → non-empty citations with
    empty snapshot_id so safety is not_assessed (not silently empty).
    """
    evidence = list(result.evidence or [])
    if not evidence:
        return (), ()

    gaps: list[str] = []
    by_rf: dict[tuple[str, str], SnapshotProvenanceFact] = {}
    snaps: tuple[Any, ...] | None
    if persisted_snapshots is None:
        snaps = None
    elif isinstance(persisted_snapshots, (list, tuple)):
        snaps = tuple(persisted_snapshots)
        for s in snaps:
            if isinstance(s, SnapshotProvenanceFact) and s.snapshot_id:
                by_rf[(s.record_id, s.field_path)] = s
    else:
        snaps = None

    cites: list[CitationFact] = []
    for raw in evidence:
        if isinstance(raw, Evidence):
            rid = raw.record_id
            field = raw.field
            ref = raw.retrieval_ref
        elif isinstance(raw, Mapping):
            rid = str(raw.get("record_id") or "")
            field = str(raw.get("field") or raw.get("field_path") or "")
            ref = str(raw.get("retrieval_ref") or "")
        else:
            continue
        if not rid or not field:
            gaps.append("investigation evidence row missing record_id/field")
            continue
        matched = by_rf.get((rid, field))
        if matched is not None:
            cites.append(
                CitationFact(
                    snapshot_id=matched.snapshot_id,
                    record_id=rid,
                    field_path=field,
                )
            )
            continue
        # Unresolved: do not invent a store snapshot_id.
        if snaps is None:
            cites.append(CitationFact(snapshot_id="", record_id=rid, field_path=field))
            gaps.append(
                f"investigation evidence {rid}.{field} (ref={ref!r}) unbound: "
                "persisted_snapshots missing/unresolved → provenance not assessed"
            )
        else:
            cites.append(CitationFact(snapshot_id="", record_id=rid, field_path=field))
            gaps.append(
                f"investigation evidence {rid}.{field} (ref={ref!r}) "
                "not matched to any authoritative snapshot_id"
            )
    if snaps is None and evidence:
        gaps.append(
            "investigation evidence present without authoritative snapshots; "
            "citation provenance not assessed (not silently empty)"
        )
    return tuple(cites), tuple(dict.fromkeys(gaps))
