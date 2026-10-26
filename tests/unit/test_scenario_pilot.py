"""Unit tests for scenario pilot orchestration (no paid calls)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from casepilot.backend.db import BackendStore
from casepilot.backend.evaluation_judge import (
    DimensionAssessment,
    InvalidJudgeDiagnostic,
    JudgeOutcome,
    JudgePayload,
    JudgeRating,
    JudgeVerdict,
    judge_prompt_version,
    judge_schema_version,
)
from casepilot.backend.pilot_observability import PilotObservability
from casepilot.backend.trace import TraceWriter
from casepilot.evaluation.__main__ import build_parser, main
from casepilot.evaluation.cost import ALLOWED_LIVE_MODEL, DEFAULT_BUDGET_USD, CostLedger
from casepilot.evaluation.pilot_assets import load_pilot_assets
from casepilot.evaluation.pilot_coverage import (
    AnswerCitation,
    AttemptSnapshot,
    ExposedFact,
    RetrievalClass,
)
from casepilot.evaluation.pilot_safety import (
    ForbiddenAttemptFact,
    SnapshotProvenanceFact,
)
from casepilot.evaluation.scenario_pilot import (
    DISCLAIMERS,
    REPETITIONS,
    REQUIRED_RETRIEVAL_SCENARIO_ID,
    SLOT_COUNT,
    HttpBackendAdapter,
    JudgeFn,
    PhaseCapture,
    PilotConfig,
    PilotError,
    SlotResult,
    _investigation_citations,
    _outcome_dict,
    _resolve_cohort_id,
    align_ledger_path,
    assess_slot_coverage,
    assess_slot_safety,
    assign_slots,
    build_calibration_payload,
    build_coverage_gate_summary,
    classify_spend_pause,
    conservative_budget_estimate,
    execute_slot,
    model_visible_available_evidence,
    preflight,
    read_ledger_snapshot,
    reconcile_ledger,
    reevaluate_saved_cohort_report,
    resolve_ledger_path,
    run_calibration,
    run_scenario_pilot,
    summarize_completion,
)
from casepilot.shared.config import Settings, get_settings
from casepilot.shared.contracts import (
    Diagnosis,
    Evidence,
    EvidenceProvenance,
    InvestigationResult,
    NextAction,
    RunStatus,
)
from casepilot.simulator.db import apply_schema, connect

ROOT = Path(__file__).resolve().parents[2]
PRICES = ROOT / "evals" / "pricing" / "verified_prices.json"


@pytest.fixture(autouse=True)
def _bind_canonical_ledger_to_tmp(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Align CASEPILOT_COST_LEDGER_PATH with common tmp_path/ledger.json test path."""
    if "tmp_path" not in request.fixturenames:
        return
    tmp_path = request.getfixturevalue("tmp_path")
    monkeypatch.setenv("CASEPILOT_COST_LEDGER_PATH", str(tmp_path / "ledger.json"))


def _settings(tmp_path: Path) -> Settings:
    return get_settings(
        **{
            "_env_file": None,
            "simulator_service_token": "t-sim",
            "operator_token": "t-op",
            "backend_db_path": tmp_path / "b.sqlite3",
            "trace_dir": tmp_path / "traces",
            "simulator_base_url": "http://simulator.test",
            "openrouter_api_key": None,
            "investigation_timeout_seconds": 5.0,
            "max_tool_calls": 1,
            "max_model_turns": 1,
        }
    )


def _verdict(fs: str, un: str, us: str) -> JudgeVerdict:
    def dim(r: str) -> DimensionAssessment:
        return DimensionAssessment(
            rating=cast(JudgeRating, r),
            rationale=f"r-{r}",
            source_ids=["EVT-001-A"],
        )

    return JudgeVerdict(
        factual_support=dim(fs),
        uncertainty=dim(un),
        usefulness=dim(us),
        material_unsupported_claims=[],
    )


def _outcome(
    call_id: str,
    case_id: str,
    *,
    verdict: JudgeVerdict | None = None,
    error: str | None = None,
    cost: float | None = 0.0,
) -> JudgeOutcome:
    return JudgeOutcome(
        call_id=call_id,
        case_id=case_id,
        verdict=verdict,
        usage={"total_tokens": 1} if cost is not None else None,
        cost_usd=cost,
        error=error,
        model_name="scripted",
        provider_model=None,
        judge_prompt_version=judge_prompt_version(),
        judge_schema_version=judge_schema_version(),
        live=False,
        rubric_version="p0-v1",
    )


@dataclass
class FakeBackend:
    """Deterministic backend double; never hits network."""

    calls: list[tuple[str, ...]] | None = None
    fail_setup_on: set[str] | None = None
    fail_target_on: set[str] | None = None
    cost_usd: float | None = 0.01
    include_write_audit: bool = True
    include_coverage_proof: bool = False
    include_model_visible: bool = True
    forbidden: tuple[ForbiddenAttemptFact, ...] = ()
    _n: int = 0

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []
        if self.fail_setup_on is None:
            self.fail_setup_on = set()
        if self.fail_target_on is None:
            self.fail_target_on = set()

    def start_investigation(self, case_id: str, objective: str) -> PhaseCapture:
        assert self.calls is not None and self.fail_setup_on is not None
        self.calls.append(("invest", case_id))
        self._n += 1
        run_id = f"run-{case_id}-{self._n}"
        if case_id in self.fail_setup_on:
            return PhaseCapture(ok=False, error="setup boom", run_id=run_id, cost_usd=self.cost_usd)
        props = (
            (
                {
                    "proposal_id": f"p-{run_id}",
                    "run_id": run_id,
                    "case_id": case_id,
                    "action": "create_follow_up_task",
                    "status": "pending",
                    "payload_digest": "d1",
                    "task_id": None,
                },
            )
            if self.include_write_audit
            else None
        )
        model_vis = () if self.include_model_visible else None
        return PhaseCapture(
            ok=True,
            run_id=run_id,
            answer_text=f"investigation answer for {case_id}: {objective}",
            answer_payload={"diagnosis_summary": "ok", "case_id": case_id},
            cost_usd=self.cost_usd,
            usage={"cost_usd": self.cost_usd} if self.cost_usd is not None else None,
            provider_model=ALLOWED_LIVE_MODEL,
            proposals_before=() if self.include_write_audit else None,
            proposals_after=props,
            write_audit_captured=self.include_write_audit,
            forbidden_attempts=self.forbidden,
            answering_model_exposed=model_vis,
        )

    def send_followup(
        self,
        run_id: str,
        question: str,
        *,
        case_id: str,
        idempotency_key: str,
    ) -> PhaseCapture:
        assert self.calls is not None and self.fail_target_on is not None
        self.calls.append(("followup", run_id, case_id))
        if case_id in self.fail_target_on or run_id in self.fail_target_on:
            return PhaseCapture(
                ok=False, run_id=run_id, error="followup boom", cost_usd=self.cost_usd
            )
        attempt_id = f"att-{idempotency_key[-8:]}"
        cites: tuple[AnswerCitation, ...] | None = None
        snaps: tuple[AttemptSnapshot, ...] | None = None
        exposed: tuple[ExposedFact, ...] | None = None
        pre: tuple[ExposedFact, ...] | None = None
        if self.include_coverage_proof:
            # Minimal proof chain for retrieve-like facts — still optional here.
            pre = ()  # empty pre-turn → retrieval required if targets exist
            snaps = (
                AttemptSnapshot(
                    snapshot_id="snap-new-1",
                    record_id="EVT-011-A",
                    field_path="details.error_code",
                    provenance="follow_up",
                    attempt_id=attempt_id,
                    retrieval_ref="ref-1",
                    value="E_RULE",
                ),
            )
            exposed = (
                ExposedFact(
                    record_id="EVT-011-A",
                    field_path="details.error_code",
                    snapshot_id="snap-new-1",
                    provenance="follow_up",
                    attempt_id=attempt_id,
                ),
            )
            cites = (
                AnswerCitation(
                    snapshot_id="snap-new-1",
                    record_id="EVT-011-A",
                    field_path="details.error_code",
                    provenance="follow_up",
                ),
            )
        elif self.include_model_visible:
            # Instrumented empty context — judge can run; coverage may be not_assessed/optional
            pre = ()
            exposed = ()
        return PhaseCapture(
            ok=True,
            run_id=run_id,
            attempt_id=attempt_id,
            turn_id=f"turn-{attempt_id}",
            answer_text=f"follow-up answer: {question}",
            answer_payload={"notes": [question]},
            cost_usd=self.cost_usd,
            usage={"cost_usd": self.cost_usd} if self.cost_usd is not None else None,
            proposals_before=() if self.include_write_audit else None,
            proposals_after=() if self.include_write_audit else None,
            write_audit_captured=self.include_write_audit,
            pre_turn_model_context=pre,
            attempt_follow_up_snapshots=snaps,
            answering_model_exposed=exposed,
            answer_citations=cites,
            citations=cites or (),
            gaps=(
                ()
                if (self.include_coverage_proof or self.include_model_visible)
                else ("pre_turn_model_context missing: needs runtime trace instrumentation",)
            ),
        )


def _matching_judge() -> JudgeFn:
    assets = load_pilot_assets()
    expected = {ex.id: ex.expected.as_tuple() for ex in assets.calibration}

    async def judge_fn(
        settings: Settings,
        case_id: str,
        call_id: str,
        payload: JudgePayload | dict[str, Any],
        live_opt_in: bool,
        model_factory: Any = None,
    ) -> JudgeOutcome:
        if call_id.startswith("cal:"):
            ex_id = call_id.split(":")[-1]
            fs, un, us = expected[ex_id]
            return _outcome(call_id, case_id, verdict=_verdict(fs, un, us), cost=0.001)
        return _outcome(call_id, case_id, verdict=_verdict("meets", "meets", "meets"), cost=0.002)

    return judge_fn


# ---------------------------------------------------------------------------
# Unit: freeze / slots / preflight
# ---------------------------------------------------------------------------


def test_assign_slots_immutable_18() -> None:
    assets = load_pilot_assets()
    slots = assign_slots(assets)
    assert len(slots) == SLOT_COUNT
    assert [s.scenario_id for s in slots[:3]] == ["straightforward"] * 3
    assert [s.repetition for s in slots[:3]] == [1, 2, 3]
    assert slots[-1].scenario_id == "bypass"
    assert slots[-1].repetition == REPETITIONS
    # fixed order, no duplicates of (scenario, rep)
    keys = {(s.scenario_id, s.repetition) for s in slots}
    assert len(keys) == SLOT_COUNT


def test_preflight_requires_live_opt_in(tmp_path: Path) -> None:
    assets = load_pilot_assets()
    cfg = PilotConfig(live=False, report_dir=tmp_path, ledger_path=None)
    pf = preflight(cfg, assets, prices_path=PRICES)
    assert pf["ok"] is False
    assert any("live opt-in" in b for b in pf["blockers"])


def test_preflight_ledger_exhausted(tmp_path: Path) -> None:
    assets = load_pilot_assets()
    ledger_path = tmp_path / "ledger.json"
    ledger = CostLedger(
        path=ledger_path,
        budget_usd=DEFAULT_BUDGET_USD,
        prices={},
        reserve_usd=0.01,
    )
    # Force spent full budget via raw write
    snap = ledger.snapshot()
    snap["spent_usd"] = DEFAULT_BUDGET_USD
    snap["reserved_usd"] = 0.0
    snap["pricing_reliable"] = True
    ledger_path.write_text(json.dumps(snap), encoding="utf-8")

    cfg = PilotConfig(live=True, report_dir=tmp_path, ledger_path=ledger_path)
    pf = preflight(cfg, assets, prices_path=PRICES)
    assert pf["ok"] is False
    assert any("remaining" in b for b in pf["blockers"])


def test_preflight_pricing_unreliable(tmp_path: Path) -> None:
    assets = load_pilot_assets()
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": False,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    cfg = PilotConfig(live=True, report_dir=tmp_path, ledger_path=ledger_path)
    pf = preflight(cfg, assets, prices_path=PRICES)
    assert pf["ok"] is False
    assert any("pricing_reliable" in b for b in pf["blockers"])


def test_conservative_estimate_positive() -> None:
    est = conservative_budget_estimate(prices_path=PRICES)
    assert est["conservative_estimate_usd"] > 0
    assert est["model"] == ALLOWED_LIVE_MODEL
    assert "not guarantee" in est["note"]


def test_resolve_ledger_path_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom_ledger.json"
    monkeypatch.setenv("CASEPILOT_COST_LEDGER_PATH", str(custom))
    assert resolve_ledger_path(None) == custom.resolve()
    assert resolve_ledger_path(custom) == custom.resolve()
    with pytest.raises(PilotError, match="differs from canonical"):
        resolve_ledger_path(tmp_path / "override.json")


def test_read_ledger_snapshot_readonly(tmp_path: Path) -> None:
    path = tmp_path / "nope.json"
    assert read_ledger_snapshot(path) is None
    assert not path.exists()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calibration_exact_tuples(tmp_path: Path) -> None:
    assets = load_pilot_assets()
    result = await run_calibration(
        assets,
        settings=_settings(tmp_path),
        live_opt_in=False,
        cohort_id="cal-test",
        judge_fn=_matching_judge(),
    )
    assert result["ok"] is True
    assert result["matched"] == 8
    for row in result["results"]:
        assert row["matched"] is True
        assert (
            row["observed"]["factual_support"],
            row["observed"]["uncertainty"],
            row["observed"]["usefulness"],
        ) == (
            row["expected"]["factual_support"],
            row["expected"]["uncertainty"],
            row["expected"]["usefulness"],
        )


@pytest.mark.asyncio
async def test_calibration_mismatch_blocks(tmp_path: Path) -> None:
    assets = load_pilot_assets()

    async def bad_judge(
        settings: Settings,
        case_id: str,
        call_id: str,
        payload: Any,
        live_opt_in: bool,
        model_factory: Any = None,
    ) -> JudgeOutcome:
        return _outcome(call_id, case_id, verdict=_verdict("meets", "meets", "meets"), cost=0.0)

    result = await run_calibration(
        assets,
        settings=_settings(tmp_path),
        live_opt_in=False,
        cohort_id="cal-bad",
        judge_fn=bad_judge,
    )
    assert result["ok"] is False
    assert result["blockers"]


def test_calibration_payload_uses_reference_not_agent_leak() -> None:
    assets = load_pilot_assets()
    ex = assets.calibration[0]
    ref = assets.references[ex.scenario_id]
    payload = build_calibration_payload(ex, ref)
    dumped = payload.model_dump()
    assert "reference" in dumped
    assert dumped["answer"] == ex.answer
    assert "system_prompt" not in json.dumps(dumped)


# ---------------------------------------------------------------------------
# Safety / coverage honesty
# ---------------------------------------------------------------------------


def test_safety_missing_audit_not_assessed() -> None:
    target = PhaseCapture(
        ok=True,
        run_id="r1",
        answer_text="x",
        write_audit_captured=False,
        proposals_before=None,
        proposals_after=None,
    )
    result = assess_slot_safety(case_id="CASE-001", setup=None, target=target)
    assert result.verdict == "not_assessed"


def test_safety_detects_forbidden_attempt() -> None:
    target = PhaseCapture(
        ok=True,
        run_id="r1",
        answer_text="x",
        write_audit_captured=True,
        proposals_before=(),
        proposals_after=(),
        forbidden_attempts=(
            ForbiddenAttemptFact(kind="approve_proposal", detail="nope", blocked=True),
        ),
    )
    result = assess_slot_safety(case_id="CASE-008", setup=None, target=target)
    assert result.verdict == "fail"


def test_coverage_missing_context_not_assessed() -> None:
    assets = load_pilot_assets()
    scen = assets.scenario_by_id("retrieve")
    target = PhaseCapture(
        ok=True,
        run_id="r1",
        attempt_id="a1",
        answer_text="x",
        pre_turn_model_context=None,  # honest missing
    )
    cov = assess_slot_coverage(scenario=scen, target=target, repetition=1)
    assert cov["assessment"]["status"] == "not_assessed"
    assert cov["assessment"]["retrieval_class"] == RetrievalClass.NOT_ASSESSED.value


# ---------------------------------------------------------------------------
# Cohort orchestration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pilot_offline_fake_backend(tmp_path: Path) -> None:
    assets = load_pilot_assets()
    ledger_path = tmp_path / "ledger.json"
    # Healthy empty-ish ledger snapshot for preflight remaining check.
    ledger_path.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": True,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    cfg = PilotConfig(
        live=True,  # opt-in flag; spend is faked via judge_fn/backend
        report_dir=tmp_path / "reports",
        ledger_path=ledger_path,
        cohort_id="test-cohort",
        budget_usd=5.0,
    )
    backend = FakeBackend(include_write_audit=True, include_coverage_proof=True)
    report = await run_scenario_pilot(
        cfg,
        assets=assets,
        backend=backend,
        settings=_settings(tmp_path),
        judge_fn=_matching_judge(),
        prices_path=PRICES,
    )
    assert report.cohort_id == "test-cohort"
    assert len(report.slots) == SLOT_COUNT
    assert report.calibration["ok"] is True
    assert all(s.status in {"complete", "incomplete"} for s in report.slots)
    # Investigation + follow-up setups should have been called; no approve routes.
    assert backend.calls is not None
    assert any(c[0] == "invest" for c in backend.calls)
    assert any(c[0] == "followup" for c in backend.calls)
    assert report.costs.total_usd is not None
    assert report.costs.total_usd > 0
    json_path = tmp_path / "reports" / "scenario_pilot_test-cohort.json"
    md_path = tmp_path / "reports" / "scenario_pilot_test-cohort.md"
    assert json_path.is_file()
    assert md_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["versions"]["asset_hashes"]["scenarios_sha256"]
    assert set(DISCLAIMERS).issubset(set(data["disclaimers"]))
    # Ledger not mutated by pilot
    snap = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert float(snap["spent_usd"]) == 0.0


@pytest.mark.asyncio
async def test_run_pilot_blocked_without_live(tmp_path: Path) -> None:
    cfg = PilotConfig(live=False, report_dir=tmp_path / "r", cohort_id="nolive")
    report = await run_scenario_pilot(
        cfg,
        assets=load_pilot_assets(),
        backend=FakeBackend(),
        settings=_settings(tmp_path),
        judge_fn=_matching_judge(),
        prices_path=PRICES,
    )
    assert report.status == "blocked_preflight"
    assert report.blockers


@pytest.mark.asyncio
async def test_run_pilot_calibration_failure_blocks_baseline(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": True,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    async def bad_judge(
        settings: Settings,
        case_id: str,
        call_id: str,
        payload: Any,
        live_opt_in: bool,
        model_factory: Any = None,
    ) -> JudgeOutcome:
        return _outcome(
            call_id, case_id, verdict=_verdict("does not meet", "does not meet", "does not meet")
        )

    backend = FakeBackend()
    report = await run_scenario_pilot(
        PilotConfig(
            live=True,
            report_dir=tmp_path / "r",
            ledger_path=ledger_path,
            cohort_id="cal-fail",
        ),
        assets=load_pilot_assets(),
        backend=backend,
        settings=_settings(tmp_path),
        judge_fn=bad_judge,
        prices_path=PRICES,
    )
    assert report.status == "blocked_calibration"
    assert backend.calls == []  # no baseline HTTP


@pytest.mark.asyncio
async def test_failed_slot_not_replaced(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": True,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    # Fail investigation for CASE-001 (straightforward)
    backend = FakeBackend(fail_setup_on={"CASE-001"})
    report = await run_scenario_pilot(
        PilotConfig(
            live=True,
            report_dir=tmp_path / "r",
            ledger_path=ledger_path,
            cohort_id="no-replace",
        ),
        assets=load_pilot_assets(),
        backend=backend,
        settings=_settings(tmp_path),
        judge_fn=_matching_judge(),
        prices_path=PRICES,
    )
    straight = [s for s in report.slots if s.scenario_id == "straightforward"]
    assert len(straight) == 3
    assert all(s.status == "incomplete" for s in straight)
    assert all(s.billable_started for s in straight)
    # Still exactly 18 slots
    assert len(report.slots) == SLOT_COUNT


@pytest.mark.asyncio
async def test_resume_only_unstarted(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": True,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    assets = load_pilot_assets()
    # First run: fail CASE-001 only partially by stopping after writing a hand-crafted resume
    specs = assign_slots(assets)
    prior_slots = []
    for i, s in enumerate(specs):
        if i == 0:
            prior_slots.append(
                {
                    "index": 0,
                    "scenario_id": s.scenario_id,
                    "repetition": s.repetition,
                    "case_id": s.case_id,
                    "kind": s.kind,
                    "status": "incomplete",
                    "billable_started": True,
                    "errors": ["prior fail"],
                    "costs": {},
                    "gaps": [],
                }
            )
        else:
            prior_slots.append(
                {
                    "index": i,
                    "scenario_id": s.scenario_id,
                    "repetition": s.repetition,
                    "case_id": s.case_id,
                    "kind": s.kind,
                    "status": "unstarted",
                    "billable_started": False,
                    "errors": [],
                    "costs": {},
                    "gaps": [],
                }
            )
    resume_path = tmp_path / "prior.json"
    # Frozen versions must match current assets/judge fingerprint.
    frozen_versions = {
        "assets_version": assets.version,
        "fixture_version": assets.fixture_version,
        "asset_hashes": assets.hashes.to_dict(),
        "rubric_version": "p0-v1",
        "judge_prompt_version": judge_prompt_version(),
        "judge_schema_version": judge_schema_version(),
        "model": ALLOWED_LIVE_MODEL,
    }
    resume_path.write_text(
        json.dumps(
            {
                "cohort_id": "resume",
                "slots": prior_slots,
                "versions": frozen_versions,
                "calibration": {
                    "ok": True,
                    "skipped": False,
                    "spend_usd": 0.01,
                    "results": [],
                    "matched": 8,
                    "n": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    backend = FakeBackend()
    report = await run_scenario_pilot(
        PilotConfig(
            live=True,
            report_dir=tmp_path / "r",
            ledger_path=ledger_path,
            cohort_id="resume",
            resume_path=resume_path,
            model=ALLOWED_LIVE_MODEL,
        ),
        assets=assets,
        backend=backend,
        settings=_settings(tmp_path),
        judge_fn=_matching_judge(),
        prices_path=PRICES,
    )
    assert report.slots[0].status == "incomplete"
    assert report.slots[0].errors == ["prior fail"]
    # Slot 0 not re-executed
    assert report.slots[1].status == "complete"


@pytest.mark.asyncio
async def test_missing_cost_reported_not_zero(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": True,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    async def judge_no_cost(
        settings: Settings,
        case_id: str,
        call_id: str,
        payload: Any,
        live_opt_in: bool,
        model_factory: Any = None,
    ) -> JudgeOutcome:
        assets = load_pilot_assets()
        if call_id.startswith("cal:"):
            ex_id = call_id.split(":")[-1]
            exp = next(e for e in assets.calibration if e.id == ex_id).expected.as_tuple()
            return _outcome(call_id, case_id, verdict=_verdict(*exp), cost=None)
        return _outcome(call_id, case_id, verdict=_verdict("meets", "meets", "meets"), cost=None)

    report = await run_scenario_pilot(
        PilotConfig(
            live=True,
            report_dir=tmp_path / "r",
            ledger_path=ledger_path,
            cohort_id="cost-gap",
        ),
        assets=load_pilot_assets(),
        backend=FakeBackend(cost_usd=None),
        settings=_settings(tmp_path),
        judge_fn=judge_no_cost,
        prices_path=PRICES,
        skip_calibration=False,
    )
    assert report.costs.missing_cost_gaps
    # Must not silently claim zero total when components missing
    assert report.costs.total_usd is None or report.costs.missing_cost_gaps


def test_reconcile_ledger_unallocated() -> None:
    before = {"spent_usd": 1.0}
    after = {"spent_usd": 1.5}
    recon = reconcile_ledger(before, after, reported_spend=0.2)
    assert recon["delta_spent_usd"] == pytest.approx(0.5)
    assert recon["unallocated_usd"] == pytest.approx(0.3)
    assert recon["gaps"]


def test_summarize_completion_poor_quality_can_complete() -> None:
    # Safety fail is a measured finding; still allows P0 completion if assessments exist.
    from casepilot.evaluation.scenario_pilot import SlotResult

    real = [
        SlotResult(
            index=i,
            scenario_id="s",
            repetition=1,
            case_id="c",
            kind="investigation",
            status="complete",
            judge={"verdict": {"a": 1}, "error": None},
            safety={"verdict": "fail"},
        )
        for i in range(SLOT_COUNT)
    ]
    comp = summarize_completion(real, {"required_coverage_demonstrated": True})
    assert comp["p0_measurement_complete"] is True


def test_summarize_completion_not_assessed_blocks_p0() -> None:
    """Regression: valid judge + coverage must not complete when safety is not_assessed."""
    from casepilot.evaluation.scenario_pilot import SlotResult

    real = [
        SlotResult(
            index=i,
            scenario_id="s",
            repetition=1,
            case_id="c",
            kind="investigation",
            status="complete",
            judge={"verdict": {"a": 1}, "error": None},
            safety={"verdict": "not_assessed"},
        )
        for i in range(SLOT_COUNT)
    ]
    comp = summarize_completion(real, {"required_coverage_demonstrated": True})
    assert comp["p0_measurement_complete"] is False
    assert comp["safety_determined"] == 0
    assert comp["safety_not_assessed"] == SLOT_COUNT


# ---------------------------------------------------------------------------
# HTTP adapter isolation
# ---------------------------------------------------------------------------


def test_http_adapter_investigation_no_labels_no_proposal_http(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(f"{request.method} {request.url.path}")
        body = json.loads(request.content.decode()) if request.content else {}
        if request.method == "POST" and request.url.path.endswith("/v1/investigations"):
            captured.append(body)
            return httpx.Response(
                200,
                json={
                    "run_id": "r1",
                    "result": {
                        "run_id": "r1",
                        "case_id": body["case_id"],
                        "status": "completed",
                        "diagnosis": "missing_document",
                        "diagnosis_summary": "missing poi",
                        "explanation": "x",
                        "evidence": [],
                        "next_action": "request_missing_document",
                        "next_action_explanation": "ask",
                        "approval_required": False,
                    },
                    "cost_usd": 0.01,
                },
            )
        return httpx.Response(500, json={"detail": f"unexpected {request.url.path}"})

    store = BackendStore(tmp_path / "backend.sqlite3")
    sim = tmp_path / "simulator.sqlite3"
    conn = connect(sim)
    apply_schema(conn)
    conn.execute(
        "INSERT INTO cases (case_id, status, symptom, application_fields, "
        "required_documents, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("CASE-001", "open", "x", "{}", "[]", datetime.now(UTC).isoformat()),
    )
    conn.commit()
    conn.close()
    store.insert_run(
        run_id="r1",
        case_id="CASE-001",
        objective="why stalled?",
        status=RunStatus.COMPLETED,
        result=None,
        error=None,
        completed_at=datetime.now(UTC),
    )
    obs = PilotObservability.open(
        enabled=True,
        backend_db_path=tmp_path / "backend.sqlite3",
        simulator_db_path=sim,
        trace_dir=tmp_path / "traces",
        store=store,
    )
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    adapter = HttpBackendAdapter(client=client, observer=obs)
    cap = adapter.start_investigation("CASE-001", "why stalled?")
    client.close()
    assert cap.ok
    assert captured[0] == {"case_id": "CASE-001", "objective": "why stalled?"}
    assert "reference" not in captured[0]
    assert not any("/proposals" in m for m in methods)
    assert cap.write_audit_captured is True
    assert cap.proposals_before is not None
    assert cap.tasks_before is not None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_scenario_pilot_requires_pilot_live() -> None:
    code = main(["--scenario-pilot", "--report-dir", "/tmp/x"])
    assert code == 2


def test_cli_parser_has_pilot_flags() -> None:
    p = build_parser()
    args = p.parse_args(["--scenario-pilot", "--pilot-live", "--cohort-id", "c1", "--json"])
    assert args.scenario_pilot is True
    assert args.pilot_live is True
    assert args.cohort_id == "c1"


@pytest.mark.asyncio
async def test_execute_slot_followup_setup_retained(tmp_path: Path) -> None:
    assets = load_pilot_assets()
    scen = assets.scenario_by_id("retrieve")
    ref = assets.references["retrieve"]
    specs = [s for s in assign_slots(assets) if s.scenario_id == "retrieve"]
    backend = FakeBackend()
    slot = await execute_slot(
        specs[0],
        scen,
        ref,
        backend,
        settings=_settings(tmp_path),
        live_opt_in=False,
        cohort_id="test-cohort",
        judge_fn=_matching_judge(),
    )
    assert slot.setup is not None
    assert slot.setup["ok"] is True
    assert slot.target is not None
    assert "setup_usd" in slot.costs
    assert slot.judge is not None
    assert str(slot.judge.get("call_id") or "").startswith("base:test-cohort:")


def test_setup_failure_pending_proposal_permitted() -> None:
    """Setup-only safety must not treat investigation pending proposal as follow-up fail."""
    cap = PhaseCapture(
        ok=False,
        run_id="run-setup-1",
        error="setup boom",
        write_audit_captured=True,
        proposals_before=(),
        proposals_after=(
            {
                "proposal_id": "p1",
                "run_id": "run-setup-1",
                "case_id": "CASE-011",
                "action": "create_follow_up_task",
                "status": "pending",
                "payload_digest": "d",
                "task_id": None,
            },
        ),
        tasks_before=(),
        tasks_after=(),
    )
    # Bug regression path: same capture as setup+target follow_up would false-fail.
    bad = assess_slot_safety(case_id="CASE-011", setup=cap, target=cap)
    good = assess_slot_safety(case_id="CASE-011", setup=cap, setup_only=True)
    assert good.verdict == "pass"
    assert good.target is None
    assert good.setup is not None
    assert good.setup.phase == "setup"
    # Documenting the false-fail mode we must not use on setup-only:
    assert bad.verdict == "fail"


def test_model_visible_evidence_not_answer_payload() -> None:
    cap = PhaseCapture(
        ok=True,
        run_id="r1",
        answer_text="secret answer",
        answer_payload={"diagnosis_summary": "should-not-be-evidence"},
        answering_model_exposed=None,
        pre_turn_model_context=None,
    )
    evid, gap = model_visible_available_evidence(cap, kind="investigation")
    assert evid is None
    assert gap is not None
    assert "answer_payload" in (gap or "")

    # Follow-up requires both instrumented fields
    evid2, gap2 = model_visible_available_evidence(cap, kind="follow_up")
    assert evid2 is None
    assert gap2 is not None

    ok_cap = PhaseCapture(
        ok=True,
        run_id="r1",
        answer_payload={"nope": True},
        pre_turn_model_context=(),
        answering_model_exposed=(ExposedFact(record_id="EVT-1", field_path="x", snapshot_id="s1"),),
    )
    evid3, gap3 = model_visible_available_evidence(ok_cap, kind="follow_up")
    assert gap3 is None
    assert evid3 is not None
    assert "pre_turn_model_context" in evid3
    assert "answering_model_exposed" in evid3
    assert "diagnosis_summary" not in str(evid3)
    assert "nope" not in str(evid3)


@pytest.mark.asyncio
async def test_missing_model_visible_blocks_judge(tmp_path: Path) -> None:
    assets = load_pilot_assets()
    scen = assets.scenario_by_id("straightforward")
    ref = assets.references["straightforward"]
    specs = [s for s in assign_slots(assets) if s.scenario_id == "straightforward"]
    backend = FakeBackend(include_model_visible=False)
    slot = await execute_slot(
        specs[0],
        scen,
        ref,
        backend,
        settings=_settings(tmp_path),
        live_opt_in=False,
        cohort_id="no-evid",
        judge_fn=_matching_judge(),
    )
    assert slot.status == "incomplete"
    assert slot.judge is not None
    assert slot.judge.get("skipped") is True
    assert any("model-visible" in e for e in slot.errors)


@pytest.mark.asyncio
async def test_calibration_call_id_includes_cohort(tmp_path: Path) -> None:
    seen: list[str] = []

    async def capture_judge(
        settings: Settings,
        case_id: str,
        call_id: str,
        payload: Any,
        live_opt_in: bool,
        model_factory: Any = None,
    ) -> JudgeOutcome:
        seen.append(call_id)
        assets = load_pilot_assets()
        if call_id.startswith("cal:"):
            ex_id = call_id.split(":")[-1]
            exp = next(e for e in assets.calibration if e.id == ex_id).expected.as_tuple()
            return _outcome(call_id, case_id, verdict=_verdict(*exp), cost=0.0)
        return _outcome(call_id, case_id, verdict=_verdict("meets", "meets", "meets"), cost=0.0)

    from casepilot.evaluation.scenario_pilot import run_calibration

    result = await run_calibration(
        load_pilot_assets(),
        settings=_settings(tmp_path),
        live_opt_in=False,
        cohort_id="cohort-xyz",
        judge_fn=capture_judge,
    )
    assert result["ok"] is True
    assert all(c.startswith("cal:cohort-xyz:") for c in seen)


def test_preserve_actual_snapshot_case_id_no_retag() -> None:
    """Runner must not retag observer SnapshotProvenanceFact with intended case_id."""
    from casepilot.evaluation.scenario_pilot import _phase_evidence, _preserve_snapshot_facts

    actual = SnapshotProvenanceFact(
        snapshot_id="snap-x",
        run_id="run-1",
        case_id="CASE-099",  # actual source, not intended CASE-008
        record_id="EVT-CROSS",
        field_path="details.error_code",
        provenance="original_run",
    )
    preserved = _preserve_snapshot_facts((actual,))
    assert preserved is not None
    assert preserved[0].case_id == "CASE-099"

    cap = PhaseCapture(
        ok=True,
        run_id="run-1",
        write_audit_captured=True,
        proposals_before=(),
        proposals_after=(),
        tasks_before=(),
        tasks_after=(),
        persisted_snapshots=(actual,),
        citations=(
            {"snapshot_id": "snap-x", "field_path": "details.error_code", "record_id": "EVT-CROSS"},
        ),
        answering_model_exposed=(),
    )
    ev = _phase_evidence("investigation", "CASE-008", "run-1", cap)
    assert ev is not None
    assert ev.persisted_snapshots is not None
    assert ev.persisted_snapshots[0].case_id == "CASE-099"
    safety = assess_slot_safety(case_id="CASE-008", target=cap)
    assert safety.verdict == "fail"  # cross-case citation


def test_unknown_snapshot_ownership_not_assessed() -> None:
    from casepilot.evaluation.scenario_pilot import _preserve_snapshot_facts

    # Mapping without case_id must not be retagged intended → None
    raw = {"snapshot_id": "s1", "run_id": "r", "record_id": "x", "field_path": "y"}
    assert _preserve_snapshot_facts((raw,)) is None


def test_judge_evidence_includes_ids_and_values() -> None:
    cap = PhaseCapture(
        ok=True,
        answering_model_exposed=(
            ExposedFact(
                record_id="EVT-1",
                field_path="details.code",
                snapshot_id="snap-1",
                value="E42",
                provenance="follow_up",
                attempt_id="a1",
            ),
        ),
        pre_turn_model_context=(),
    )
    evid, gap = model_visible_available_evidence(cap, kind="follow_up")
    assert gap is None
    assert evid is not None
    item = evid["answering_model_exposed"][0]
    assert item["snapshot_id"] == "snap-1"
    assert item["record_id"] == "EVT-1"
    assert item["value"] == "E42"


def test_http_adapter_observer_cost_and_exposure(tmp_path: Path) -> None:
    """Real observer + mock HTTP: cost/provider from end trace; exposure for judge."""
    store = BackendStore(tmp_path / "backend.sqlite3")
    sim = tmp_path / "simulator.sqlite3"
    conn = connect(sim)
    apply_schema(conn)
    conn.execute(
        "INSERT INTO cases (case_id, status, symptom, application_fields, "
        "required_documents, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("CASE-001", "open", "x", "{}", "[]", datetime.now(UTC).isoformat()),
    )
    conn.execute(
        "INSERT INTO events (event_id, case_id, event_type, timestamp, "
        "message, details) VALUES (?, ?, ?, ?, ?, ?)",
        ("EVT-001-A", "CASE-001", "error", datetime.now(UTC).isoformat(), "m", "{}"),
    )
    conn.commit()
    conn.close()
    store.insert_run(
        run_id="r-obs-1",
        case_id="CASE-001",
        objective="why",
        status=RunStatus.COMPLETED,
        result=None,
        error=None,
        completed_at=datetime.now(UTC),
    )
    snap = store.insert_evidence_snapshot(
        run_id="r-obs-1",
        source_system="events",
        record_id="EVT-001-A",
        field_path="details.missing",
        value=["proof_of_income"],
        retrieval_ref="ref/1",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    tw = TraceWriter(tmp_path / "traces", "r-obs-1")
    tw.emit(
        "tool",
        name="get_events",
        inputs={},
        record_ids=["EVT-001-A"],
        facts=[
            {
                "snapshot_id": snap.snapshot_id,
                "record_id": "EVT-001-A",
                "field_path": "details.missing",
                "value": ["proof_of_income"],
                "provenance": "original_run",
            }
        ],
    )
    tw.emit(
        "run_end",
        status="completed",
        cost_usd=0.042,
        usage={"total_tokens": 12},
        provider_model=ALLOWED_LIVE_MODEL,
        exposure_instrumented=True,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        # Only investigation HTTP — no proposals list.
        assert "/proposals" not in str(request.url.path)
        body = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "run_id": "r-obs-1",
                "result": {
                    "run_id": "r-obs-1",
                    "case_id": body["case_id"],
                    "status": "completed",
                    "diagnosis": "missing_document",
                    "diagnosis_summary": "missing poi",
                    "explanation": "x",
                    "evidence": [],
                    "next_action": "request_missing_document",
                    "next_action_explanation": "ask",
                    "approval_required": False,
                },
                # deliberately no cost_usd on HTTP
            },
        )

    obs = PilotObservability.open(
        enabled=True,
        backend_db_path=tmp_path / "backend.sqlite3",
        simulator_db_path=sim,
        trace_dir=tmp_path / "traces",
        store=store,
    )
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    adapter = HttpBackendAdapter(client=client, observer=obs)
    cap = adapter.start_investigation("CASE-001", "why stalled?")
    client.close()

    assert cap.ok
    assert cap.cost_usd == pytest.approx(0.042)
    assert cap.provider_model == ALLOWED_LIVE_MODEL
    assert cap.answering_model_exposed is not None
    assert any(f.record_id == "EVT-001-A" for f in cap.answering_model_exposed)
    assert cap.persisted_snapshots is not None
    assert all(isinstance(s, SnapshotProvenanceFact) for s in cap.persisted_snapshots)
    assert all(s.case_id == "CASE-001" for s in cap.persisted_snapshots)

    evid, gap = model_visible_available_evidence(cap, kind="investigation")
    assert gap is None
    assert evid is not None
    assert evid["answering_model_exposed"][0]["value"] == ["proof_of_income"]
    assert evid["answering_model_exposed"][0]["snapshot_id"]


@pytest.mark.asyncio
async def test_resume_without_prior_cal_blocks_rebill(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": True,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    assets = load_pilot_assets()
    specs = assign_slots(assets)
    prior_slots = [
        {
            "index": i,
            "scenario_id": s.scenario_id,
            "repetition": s.repetition,
            "case_id": s.case_id,
            "kind": s.kind,
            "status": "unstarted",
            "billable_started": False,
            "errors": [],
            "costs": {},
            "gaps": [],
        }
        for i, s in enumerate(specs)
    ]
    frozen_versions = {
        "assets_version": assets.version,
        "fixture_version": assets.fixture_version,
        "asset_hashes": assets.hashes.to_dict(),
        "rubric_version": "p0-v1",
        "judge_prompt_version": judge_prompt_version(),
        "judge_schema_version": judge_schema_version(),
        "model": ALLOWED_LIVE_MODEL,
    }
    resume_path = tmp_path / "prior.json"
    resume_path.write_text(
        json.dumps(
            {
                "cohort_id": "no-recal",
                "slots": prior_slots,
                "versions": frozen_versions,
                "calibration": {"ok": False, "skipped": False, "results": []},
            }
        ),
        encoding="utf-8",
    )
    cal_calls = {"n": 0}

    async def counting_judge(
        settings: Settings,
        case_id: str,
        call_id: str,
        payload: Any,
        live_opt_in: bool,
        model_factory: Any = None,
    ) -> JudgeOutcome:
        cal_calls["n"] += 1
        return _outcome(call_id, case_id, verdict=_verdict("meets", "meets", "meets"), cost=0.0)

    report = await run_scenario_pilot(
        PilotConfig(
            live=True,
            report_dir=tmp_path / "r",
            ledger_path=ledger_path,
            cohort_id="no-recal",
            resume_path=resume_path,
            model=ALLOWED_LIVE_MODEL,
        ),
        assets=assets,
        backend=FakeBackend(),
        settings=_settings(tmp_path),
        judge_fn=counting_judge,
        prices_path=PRICES,
    )
    assert report.status == "blocked_calibration"
    assert cal_calls["n"] == 0


def test_resolve_cohort_id_resume_reuses_prior() -> None:
    assert _resolve_cohort_id(None, {"cohort_id": "prior-c"}) == "prior-c"
    assert _resolve_cohort_id("prior-c", {"cohort_id": "prior-c"}) == "prior-c"


def test_resolve_cohort_id_mismatch_refuses() -> None:
    with pytest.raises(Exception) as ei:
        _resolve_cohort_id("new-id", {"cohort_id": "prior-c"})
    assert "mismatch" in str(ei.value).lower()


def test_resolve_cohort_id_resume_missing_id_refuses() -> None:
    with pytest.raises(Exception) as ei:
        _resolve_cohort_id(None, {"slots": []})
    assert "missing cohort_id" in str(ei.value).lower()


@pytest.mark.asyncio
async def test_resume_without_cli_cohort_reuses_prior_id(tmp_path: Path) -> None:
    """Regression: resume must not mint a new cohort_id while loading prior slots."""
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": True,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    assets = load_pilot_assets()
    specs = assign_slots(assets)
    prior_slots = [
        {
            "index": i,
            "scenario_id": s.scenario_id,
            "repetition": s.repetition,
            "case_id": s.case_id,
            "kind": s.kind,
            "status": "unstarted",
            "billable_started": False,
            "errors": [],
            "costs": {},
            "gaps": [],
        }
        for i, s in enumerate(specs)
    ]
    frozen_versions = {
        "assets_version": assets.version,
        "fixture_version": assets.fixture_version,
        "asset_hashes": assets.hashes.to_dict(),
        "rubric_version": "p0-v1",
        "judge_prompt_version": judge_prompt_version(),
        "judge_schema_version": judge_schema_version(),
        "model": ALLOWED_LIVE_MODEL,
    }
    resume_path = tmp_path / "prior.json"
    resume_path.write_text(
        json.dumps(
            {
                "cohort_id": "original-cohort-id",
                "slots": prior_slots,
                "versions": frozen_versions,
                "calibration": {
                    "ok": True,
                    "skipped": False,
                    "spend_usd": 0.0,
                    "results": [],
                    "matched": 8,
                    "n": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    report = await run_scenario_pilot(
        PilotConfig(
            live=True,
            report_dir=tmp_path / "r",
            ledger_path=ledger_path,
            cohort_id=None,  # CLI omitted --cohort-id
            resume_path=resume_path,
            model=ALLOWED_LIVE_MODEL,
        ),
        assets=assets,
        backend=FakeBackend(),
        settings=_settings(tmp_path),
        judge_fn=_matching_judge(),
        prices_path=PRICES,
    )
    assert report.cohort_id == "original-cohort-id"


@pytest.mark.asyncio
async def test_resume_explicit_cohort_mismatch_refuses(tmp_path: Path) -> None:
    assets = load_pilot_assets()
    specs = assign_slots(assets)
    prior_slots = [
        {
            "index": i,
            "scenario_id": s.scenario_id,
            "repetition": s.repetition,
            "case_id": s.case_id,
            "kind": s.kind,
            "status": "unstarted",
            "billable_started": False,
            "errors": [],
            "costs": {},
            "gaps": [],
        }
        for i, s in enumerate(specs)
    ]
    frozen_versions = {
        "assets_version": assets.version,
        "fixture_version": assets.fixture_version,
        "asset_hashes": assets.hashes.to_dict(),
        "rubric_version": "p0-v1",
        "judge_prompt_version": judge_prompt_version(),
        "judge_schema_version": judge_schema_version(),
        "model": ALLOWED_LIVE_MODEL,
    }
    resume_path = tmp_path / "prior.json"
    resume_path.write_text(
        json.dumps(
            {
                "cohort_id": "A",
                "slots": prior_slots,
                "versions": frozen_versions,
                "calibration": {"ok": True, "skipped": False, "results": [], "spend_usd": 0},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(Exception) as ei:
        await run_scenario_pilot(
            PilotConfig(
                live=True,
                report_dir=tmp_path / "r",
                ledger_path=tmp_path / "ledger.json",
                cohort_id="B",
                resume_path=resume_path,
                model=ALLOWED_LIVE_MODEL,
            ),
            assets=assets,
            backend=FakeBackend(),
            settings=_settings(tmp_path),
            judge_fn=_matching_judge(),
            prices_path=PRICES,
            skip_calibration=True,
        )
    assert "mismatch" in str(ei.value).lower()


def test_investigation_citations_bind_real_snapshot_ids() -> None:
    snaps = (
        SnapshotProvenanceFact(
            snapshot_id="snap-real-1",
            run_id="r1",
            case_id="CASE-001",
            record_id="EVT-001-A",
            field_path="details.missing",
            provenance="original_run",
        ),
    )
    result = InvestigationResult(
        run_id="r1",
        case_id="CASE-001",
        status=RunStatus.COMPLETED,
        diagnosis=Diagnosis.MISSING_DOCUMENT,
        diagnosis_summary="x",
        explanation="x",
        evidence=[
            Evidence(
                source_system="events",
                record_id="EVT-001-A",
                field="details.missing",
                fact="poi",
                retrieval_ref="ref/1",
            )
        ],
        next_action=NextAction.REQUEST_MISSING_DOCUMENT,
        next_action_explanation="ask",
    )
    cites, gaps = _investigation_citations(result, snaps)
    assert len(cites) == 1
    assert cites[0].snapshot_id == "snap-real-1"
    assert cites[0].record_id == "EVT-001-A"
    assert cites[0].field_path == "details.missing"
    assert gaps == ()


def test_investigation_citations_unresolved_not_silent_empty() -> None:
    result = InvestigationResult(
        run_id="r1",
        case_id="CASE-001",
        status=RunStatus.COMPLETED,
        diagnosis=Diagnosis.MISSING_DOCUMENT,
        diagnosis_summary="x",
        explanation="x",
        evidence=[
            Evidence(
                source_system="events",
                record_id="EVT-001-A",
                field="details.missing",
                fact="poi",
                retrieval_ref="ref/1",
            )
        ],
        next_action=NextAction.REQUEST_MISSING_DOCUMENT,
        next_action_explanation="ask",
    )
    cites, gaps = _investigation_citations(result, None)
    assert len(cites) == 1
    assert cites[0].snapshot_id == ""  # unresolved marker, not a fabricated store id
    assert any("not assessed" in g for g in gaps)
    # Safety: citations present + snaps None => not_assessed
    cap = PhaseCapture(
        ok=True,
        run_id="r1",
        write_audit_captured=True,
        proposals_before=(),
        proposals_after=(),
        tasks_before=(),
        tasks_after=(),
        persisted_snapshots=None,
        citations=cites,
    )
    safety = assess_slot_safety(case_id="CASE-001", target=cap)
    assert safety.verdict == "not_assessed"


def test_http_investigation_extracts_evidence_citations(tmp_path: Path) -> None:
    store = BackendStore(tmp_path / "backend.sqlite3")
    sim = tmp_path / "simulator.sqlite3"
    conn = connect(sim)
    apply_schema(conn)
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO cases (case_id, status, symptom, application_fields, "
        "required_documents, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("CASE-001", "open", "x", "{}", "[]", now),
    )
    conn.execute(
        "INSERT INTO events (event_id, case_id, event_type, timestamp, "
        "message, details) VALUES (?, ?, ?, ?, ?, ?)",
        ("EVT-001-A", "CASE-001", "error", now, "m", "{}"),
    )
    conn.commit()
    conn.close()
    store.insert_run(
        run_id="r-ev-1",
        case_id="CASE-001",
        objective="why",
        status=RunStatus.COMPLETED,
        result=None,
        error=None,
        completed_at=datetime.now(UTC),
    )
    snap = store.insert_evidence_snapshot(
        run_id="r-ev-1",
        source_system="events",
        record_id="EVT-001-A",
        field_path="details.missing",
        value=["proof_of_income"],
        retrieval_ref="ref/1",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    TraceWriter(tmp_path / "traces", "r-ev-1").emit(
        "run_end",
        status="completed",
        cost_usd=0.01,
        usage={},
        provider_model=ALLOWED_LIVE_MODEL,
        exposure_instrumented=True,
    )
    TraceWriter(tmp_path / "traces", "r-ev-1").emit(
        "tool",
        name="get_events",
        facts=[
            {
                "snapshot_id": snap.snapshot_id,
                "record_id": "EVT-001-A",
                "field_path": "details.missing",
                "value": ["proof_of_income"],
            }
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "run_id": "r-ev-1",
                "result": {
                    "run_id": "r-ev-1",
                    "case_id": body["case_id"],
                    "status": "completed",
                    "diagnosis": "missing_document",
                    "diagnosis_summary": "missing poi",
                    "explanation": "x",
                    "evidence": [
                        {
                            "source_system": "events",
                            "record_id": "EVT-001-A",
                            "field": "details.missing",
                            "fact": "proof_of_income missing",
                            "retrieval_ref": "ref/1",
                        }
                    ],
                    "next_action": "request_missing_document",
                    "next_action_explanation": "ask",
                    "approval_required": False,
                },
            },
        )

    obs = PilotObservability.open(
        enabled=True,
        backend_db_path=tmp_path / "backend.sqlite3",
        simulator_db_path=sim,
        trace_dir=tmp_path / "traces",
        store=store,
    )
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    adapter = HttpBackendAdapter(client=client, observer=obs)
    cap = adapter.start_investigation("CASE-001", "why")
    client.close()
    assert cap.citations
    assert any(
        getattr(c, "snapshot_id", None) == snap.snapshot_id
        or (isinstance(c, dict) and c.get("snapshot_id") == snap.snapshot_id)
        for c in cap.citations
    )


def test_classify_spend_pause_exact_prefixes_only() -> None:
    assert classify_spend_pause(["cost pause: ledger empty"]) is not None
    assert classify_spend_pause(["execution budget exhausted: reserve failed"]) is not None
    assert classify_spend_pause(["backend cost guard HTTP 402: x"]) is not None
    # Per-run tool/model limits must NOT stop the cohort
    assert classify_spend_pause(["execution limit exceeded: max tool calls"]) is None
    assert classify_spend_pause(["agent error: tool budget for case"]) is None
    assert classify_spend_pause(["HTTP 500: recursion budget"]) is None
    assert classify_spend_pause([]) is None
    assert classify_spend_pause(None) is None


@pytest.mark.asyncio
async def test_orchestrator_tool_limit_continues_later_slots(tmp_path: Path) -> None:
    """Per-scenario execution limit => incomplete slot, later slots still run."""
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": True,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    assets = load_pilot_assets()
    # Fail only CASE-001 (straightforward) with tool-limit wording that used to false-stop.
    backend = FakeBackend(fail_setup_on={"CASE-001"})
    orig = backend.start_investigation

    def start_with_limit(case_id: str, objective: str) -> PhaseCapture:
        cap = orig(case_id, objective)
        if case_id == "CASE-001":
            return PhaseCapture(
                ok=False,
                run_id=cap.run_id,
                error="execution limit exceeded: max_tool_calls",
                cost_usd=0.0,
                write_audit_captured=True,
                proposals_before=(),
                proposals_after=(),
                tasks_before=(),
                tasks_after=(),
                answering_model_exposed=(),
            )
        return cap

    backend.start_investigation = start_with_limit  # type: ignore[method-assign]
    report = await run_scenario_pilot(
        PilotConfig(
            live=True,
            report_dir=tmp_path / "r",
            ledger_path=ledger_path,
            cohort_id="tool-limit-continue",
            model=ALLOWED_LIVE_MODEL,
        ),
        assets=assets,
        backend=backend,
        settings=_settings(tmp_path),
        judge_fn=_matching_judge(),
        prices_path=PRICES,
    )
    straight = [s for s in report.slots if s.scenario_id == "straightforward"]
    assert all(s.status == "incomplete" for s in straight)
    # Later scenarios still executed (not skipped_budget)
    later = [s for s in report.slots if s.scenario_id == "conflict"]
    assert later
    assert all(s.status != "skipped_budget" for s in later)
    assert any(s.status == "complete" for s in later)
    assert report.completion.get("p0_measurement_complete") is not True
    assert not any("spend pause" in g for g in report.gaps)


@pytest.mark.asyncio
async def test_orchestrator_cost_pause_stops_later_slots(tmp_path: Path) -> None:
    """Genuine CostPause / spend signal stops cohort; remaining slots skipped."""
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": True,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    assets = load_pilot_assets()
    backend = FakeBackend()
    orig = backend.start_investigation
    calls = {"n": 0}

    def start_then_pause(case_id: str, objective: str) -> PhaseCapture:
        calls["n"] += 1
        if calls["n"] == 1:
            return PhaseCapture(
                ok=False,
                error="cost pause: shared ledger remaining $0.0000",
                cost_usd=None,
            )
        return orig(case_id, objective)

    backend.start_investigation = start_then_pause  # type: ignore[method-assign]
    report = await run_scenario_pilot(
        PilotConfig(
            live=True,
            report_dir=tmp_path / "r",
            ledger_path=ledger_path,
            cohort_id="cost-pause-stop",
            model=ALLOWED_LIVE_MODEL,
        ),
        assets=assets,
        backend=backend,
        settings=_settings(tmp_path),
        judge_fn=_matching_judge(),
        prices_path=PRICES,
    )
    assert report.slots[0].status == "incomplete"
    assert any("cost pause:" in e for e in report.slots[0].errors)
    # Subsequent slots skipped, not executed
    assert any(s.status == "skipped_budget" for s in report.slots[1:])
    assert calls["n"] == 1
    assert any("spend pause" in g for g in report.gaps)


@pytest.mark.asyncio
async def test_orchestrator_execution_budget_exhausted_stops(tmp_path: Path) -> None:
    """Runtime genuine spend prefix execution budget exhausted: stops cohort."""
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": True,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeBackend()
    orig = backend.start_investigation
    n = {"c": 0}

    def once(case_id: str, objective: str) -> PhaseCapture:
        n["c"] += 1
        if n["c"] == 1:
            return PhaseCapture(
                ok=False,
                error="execution budget exhausted: pricing unreliable",
            )
        return orig(case_id, objective)

    backend.start_investigation = once  # type: ignore[method-assign]
    report = await run_scenario_pilot(
        PilotConfig(
            live=True,
            report_dir=tmp_path / "r",
            ledger_path=ledger_path,
            cohort_id="exec-budget-stop",
            model=ALLOWED_LIVE_MODEL,
        ),
        assets=load_pilot_assets(),
        backend=backend,
        settings=_settings(tmp_path),
        judge_fn=_matching_judge(),
        prices_path=PRICES,
    )
    assert n["c"] == 1
    assert any(s.status == "skipped_budget" for s in report.slots[1:])


def _inv_result_json(
    *,
    case_id: str,
    run_id: str,
    status: str,
    explanation: str = "ok",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "case_id": case_id,
        "status": status,
        "diagnosis": "indeterminate" if status != "completed" else "missing_document",
        "diagnosis_summary": "s",
        "explanation": explanation,
        "evidence": [],
        "next_action": "escalate_to_human",
        "next_action_explanation": "n",
        "approval_required": False,
    }


def test_http_adapter_failed_status_not_ok() -> None:
    """HTTP 200 + InvestigationResult.status=failed => PhaseCapture.ok False + explanation."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "run_id": "r-fail",
                "result": _inv_result_json(
                    case_id=body["case_id"],
                    run_id="r-fail",
                    status="failed",
                    explanation="execution limit exceeded: max_tool_calls",
                ),
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    adapter = HttpBackendAdapter(client=client, observer=None)
    cap = adapter.start_investigation("CASE-001", "why")
    client.close()
    assert cap.ok is False
    assert cap.error == "execution limit exceeded: max_tool_calls"
    assert cap.run_id == "r-fail"


def test_http_adapter_indeterminate_still_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "run_id": "r-indet",
                "result": _inv_result_json(
                    case_id=body["case_id"],
                    run_id="r-indet",
                    status="indeterminate",
                    explanation="insufficient evidence",
                ),
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    adapter = HttpBackendAdapter(client=client, observer=None)
    cap = adapter.start_investigation("CASE-001", "why")
    client.close()
    assert cap.ok is True
    assert cap.error is None


def test_http_adapter_followup_failed_maps_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "created": True,
                "turn": {
                    "turn_id": "t1",
                    "run_id": "r1",
                    "question": "q",
                    "status": "failed",
                    "created_at": "2020-01-01T00:00:00Z",
                    "updated_at": "2020-01-01T00:00:00Z",
                    "attempts": [],
                },
                "attempt": {
                    "attempt_id": "a1",
                    "turn_id": "t1",
                    "request_key": "k",
                    "previous_attempt_id": None,
                    "status": "failed",
                    "answer": None,
                    "error": "execution limit exceeded: max_model_turns",
                    "created_at": "2020-01-01T00:00:00Z",
                    "completed_at": "2020-01-01T00:00:01Z",
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    adapter = HttpBackendAdapter(client=client, observer=None)
    cap = adapter.send_followup("r1", "q?", case_id="CASE-011", idempotency_key="k1")
    client.close()
    assert cap.ok is False
    assert cap.error == "execution limit exceeded: max_model_turns"


def _ledger(tmp_path: Path) -> Path:
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": True,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_orchestrator_http200_failed_limit_continues(tmp_path: Path) -> None:
    """Real HttpBackendAdapter: HTTP200 failed runtime result incomplete; later slots run."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/v1/investigations"):
            body = json.loads(request.content.decode())
            cid = body["case_id"]
            calls.append(cid)
            run_id = f"run-{cid}-{len(calls)}"
            if cid == "CASE-001":
                return httpx.Response(
                    200,
                    json={
                        "run_id": run_id,
                        "result": _inv_result_json(
                            case_id=cid,
                            run_id=run_id,
                            status="failed",
                            explanation="execution limit exceeded: max_tool_calls",
                        ),
                    },
                )
            return httpx.Response(
                200,
                json={
                    "run_id": run_id,
                    "result": _inv_result_json(
                        case_id=cid, run_id=run_id, status="completed", explanation="ok"
                    ),
                },
            )
        if "conversation/messages" in path:
            return httpx.Response(
                200,
                json={
                    "created": True,
                    "turn": {
                        "turn_id": "t1",
                        "run_id": "r",
                        "question": "q",
                        "status": "succeeded",
                        "created_at": "2020-01-01T00:00:00Z",
                        "updated_at": "2020-01-01T00:00:00Z",
                        "attempts": [],
                    },
                    "attempt": {
                        "attempt_id": "a1",
                        "turn_id": "t1",
                        "request_key": "k",
                        "previous_attempt_id": None,
                        "status": "succeeded",
                        "answer": {
                            "source_facts": [],
                            "interpretations": [{"text": "ok", "cites": [], "kind": "inference"}],
                            "notes": [],
                        },
                        "error": None,
                        "created_at": "2020-01-01T00:00:00Z",
                        "completed_at": "2020-01-01T00:00:01Z",
                    },
                },
            )
        return httpx.Response(404, json={"detail": f"unhandled {path}"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    adapter = HttpBackendAdapter(client=client, observer=None)
    report = await run_scenario_pilot(
        PilotConfig(
            live=True,
            report_dir=tmp_path / "r",
            ledger_path=_ledger(tmp_path),
            cohort_id="http-fail-continue",
            model=ALLOWED_LIVE_MODEL,
        ),
        assets=load_pilot_assets(),
        backend=adapter,
        settings=_settings(tmp_path),
        judge_fn=_matching_judge(),
        prices_path=PRICES,
    )
    client.close()
    straight = [s for s in report.slots if s.scenario_id == "straightforward"]
    assert straight and all(s.status == "incomplete" for s in straight)
    assert any("execution limit exceeded" in e for s in straight for e in s.errors)
    later = [s for s in report.slots if s.scenario_id == "conflict"]
    assert later
    # No observer in this test → judge may stay incomplete; must still run, not skip.
    assert all(s.status != "skipped_budget" for s in later)
    assert any(s.billable_started for s in later)
    assert not any(s.status == "skipped_budget" for s in report.slots)
    assert "CASE-016" in calls  # conflict case still hit


@pytest.mark.asyncio
async def test_orchestrator_http200_cost_pause_stops(tmp_path: Path) -> None:
    """Real HttpBackendAdapter: failed result with cost pause: stops later slots."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/v1/investigations"):
            calls["n"] += 1
            body = json.loads(request.content.decode())
            run_id = f"run-{calls['n']}"
            if calls["n"] == 1:
                return httpx.Response(
                    200,
                    json={
                        "run_id": run_id,
                        "result": _inv_result_json(
                            case_id=body["case_id"],
                            run_id=run_id,
                            status="failed",
                            explanation="cost pause: shared ledger remaining $0",
                        ),
                    },
                )
            return httpx.Response(
                200,
                json={
                    "run_id": run_id,
                    "result": _inv_result_json(
                        case_id=body["case_id"],
                        run_id=run_id,
                        status="completed",
                    ),
                },
            )
        return httpx.Response(500, json={"detail": "should not reach"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    adapter = HttpBackendAdapter(client=client, observer=None)
    report = await run_scenario_pilot(
        PilotConfig(
            live=True,
            report_dir=tmp_path / "r",
            ledger_path=_ledger(tmp_path),
            cohort_id="http-cost-stop",
            model=ALLOWED_LIVE_MODEL,
        ),
        assets=load_pilot_assets(),
        backend=adapter,
        settings=_settings(tmp_path),
        judge_fn=_matching_judge(),
        prices_path=PRICES,
    )
    client.close()
    assert report.slots[0].status == "incomplete"
    assert any("cost pause:" in e for e in report.slots[0].errors)
    assert calls["n"] == 1
    assert any(s.status == "skipped_budget" for s in report.slots[1:])


def _slot_cov(
    *,
    scenario_id: str,
    repetition: int,
    status: str,
    retrieval_class: str = "retrieval_required",
    kind: str = "follow_up",
    case_id: str = "CASE-X",
    index: int = 0,
) -> Any:

    return SlotResult(
        index=index,
        scenario_id=scenario_id,
        repetition=repetition,
        case_id=case_id,
        kind=kind,
        status="complete",
        billable_started=True,
        judge={"verdict": {"a": 1}, "error": None},
        safety={"verdict": "pass"},
        coverage={
            "scenario_id": scenario_id,
            "repetition": repetition,
            "assessment": {
                "status": status,
                "retrieval_class": retrieval_class,
                "notes": (),
            },
        },
    )


def test_coverage_gate_only_retrieve_scenario() -> None:
    """Incidental coverage on insufficient must not satisfy P0 retrieve gate."""
    slots: list[Any] = []
    # 3 retrieve: all not_covered
    for r in (1, 2, 3):
        slots.append(
            _slot_cov(
                scenario_id="retrieve",
                repetition=r,
                status="not_covered",
                index=len(slots),
                case_id="CASE-011",
            )
        )
    # insufficient r2 covered (the live6 false-complete pattern)
    for r in (1, 2, 3):
        slots.append(
            _slot_cov(
                scenario_id="insufficient",
                repetition=r,
                status="covered" if r == 2 else "not_covered",
                index=len(slots),
                case_id="CASE-013",
            )
        )
    summary = build_coverage_gate_summary(slots)
    assert summary["gate_scenario_id"] == REQUIRED_RETRIEVAL_SCENARIO_ID
    assert summary["required_coverage_demonstrated"] is False
    assert summary["per_scenario"]["insufficient"]["required_coverage_demonstrated"] is True
    assert summary["all_followups"]["required_coverage_demonstrated"] is False


def test_false_complete_regression_other_scenario_coverage_only() -> None:
    """Full 18 complete + judge + safety, but only non-retrieve covered → P0 incomplete."""
    from casepilot.evaluation.scenario_pilot import SlotResult

    ids = [
        ("straightforward", "investigation", "CASE-001"),
        ("conflict", "investigation", "CASE-016"),
        ("retrieve", "follow_up", "CASE-011"),
        ("insufficient", "follow_up", "CASE-013"),
        ("premise", "follow_up", "CASE-002"),
        ("bypass", "follow_up", "CASE-008"),
    ]
    slots: list[SlotResult] = []
    idx = 0
    for sid, kind, cid in ids:
        for rep in (1, 2, 3):
            cov = None
            if kind == "follow_up":
                # only insufficient r2 covered; retrieve never covered
                st = "covered" if sid == "insufficient" and rep == 2 else "not_covered"
                cov = {
                    "scenario_id": sid,
                    "repetition": rep,
                    "assessment": {
                        "status": st,
                        "retrieval_class": "retrieval_required",
                        "notes": (),
                    },
                }
            slots.append(
                SlotResult(
                    index=idx,
                    scenario_id=sid,
                    repetition=rep,
                    case_id=cid,
                    kind=kind,
                    status="complete",
                    billable_started=True,
                    judge={"verdict": {"a": 1}, "error": None},
                    safety={"verdict": "pass"},
                    coverage=cov,
                )
            )
            idx += 1
    assert len(slots) == SLOT_COUNT
    cov = build_coverage_gate_summary(slots)
    comp = summarize_completion(slots, cov)
    assert cov["required_coverage_demonstrated"] is False
    assert comp["p0_measurement_complete"] is False


def test_coverage_gate_retrieve_covered_satisfies() -> None:
    slots = [
        _slot_cov(
            scenario_id="retrieve",
            repetition=r,
            status="covered" if r == 1 else "not_covered",
            index=r - 1,
            case_id="CASE-011",
        )
        for r in (1, 2, 3)
    ]
    # noise: other scenario covered
    slots.append(
        _slot_cov(
            scenario_id="insufficient",
            repetition=1,
            status="covered",
            index=3,
            case_id="CASE-013",
        )
    )
    summary = build_coverage_gate_summary(slots)
    assert summary["required_coverage_demonstrated"] is True


def test_reevaluate_saved_report_retrieve_gate(tmp_path: Path) -> None:

    ids = [
        ("straightforward", "investigation", "CASE-001"),
        ("conflict", "investigation", "CASE-016"),
        ("retrieve", "follow_up", "CASE-011"),
        ("insufficient", "follow_up", "CASE-013"),
        ("premise", "follow_up", "CASE-002"),
        ("bypass", "follow_up", "CASE-008"),
    ]
    slots = []
    idx = 0
    for sid, kind, cid in ids:
        for rep in (1, 2, 3):
            cov = None
            if kind == "follow_up":
                st = "covered" if sid == "insufficient" and rep == 2 else "not_covered"
                cov = {
                    "scenario_id": sid,
                    "repetition": rep,
                    "assessment": {
                        "status": st,
                        "retrieval_class": "retrieval_required",
                        "notes": (),
                    },
                }
            slots.append(
                {
                    "index": idx,
                    "scenario_id": sid,
                    "repetition": rep,
                    "case_id": cid,
                    "kind": kind,
                    "status": "complete",
                    "billable_started": True,
                    "judge": {"verdict": {"a": 1}, "error": None},
                    "safety": {"verdict": "pass"},
                    "coverage": cov,
                    "costs": {},
                    "errors": [],
                    "gaps": [],
                }
            )
            idx += 1
    src = tmp_path / "scenario_pilot_fake6.json"
    src.write_text(
        json.dumps(
            {
                "cohort_id": "fake6",
                "status": "complete",
                "live": True,
                "completion": {"p0_measurement_complete": True},
                "coverage_summary": {"required_coverage_demonstrated": True},
                "slots": slots,
                "gaps": [],
                "blockers": [],
                "judge": "preserved",
            }
        ),
        encoding="utf-8",
    )
    derived, out = reevaluate_saved_cohort_report(src, report_dir=tmp_path)
    assert out != src
    assert out.is_file()
    assert derived["status"] == "incomplete"
    assert derived["completion"]["p0_measurement_complete"] is False
    assert derived["completion"]["quality_baseline_valid"] is False
    assert derived["completion"]["safety_baseline_valid"] is False
    assert derived["coverage_summary"]["gate_scenario_id"] == "retrieve"
    assert derived["reevaluation"]["rejudged"] is False
    assert derived["reevaluation"]["is_baseline"] is False
    assert derived["reevaluation"]["replaces_original"] is False
    assert derived["reevaluation"]["measurement_evidence_mismatch"]["declared"] is True
    assert derived["reevaluation"]["quality_reassess"] == "historical_not_validated_no_rejudge"
    assert derived["reevaluation"]["safety_reassess"] == (
        "legacy_contested_unknown_not_reconstructable"
    )
    assert derived["reevaluation"]["source_sha256"]
    # slot labels
    j0 = derived["slots"][0]["judge"]
    assert j0["baseline_validity"] == "historical_not_validated"
    assert derived["slots"][0]["safety"]["baseline_validity"] == "legacy_contested_unknown"
    assert derived["slots"][0]["safety"]["validated"] is False
    # original untouched
    orig = json.loads(src.read_text())
    assert orig["status"] == "complete"
    assert orig["completion"]["p0_measurement_complete"] is True
    assert "baseline_validity" not in (orig["slots"][0].get("judge") or {})


def test_align_ledger_path_mismatch_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "canonical_ledger.json"
    other = tmp_path / "other_ledger.json"
    monkeypatch.setenv("CASEPILOT_COST_LEDGER_PATH", str(canonical))
    path, err = align_ledger_path(other)
    assert err is not None
    assert "differs from canonical" in err
    assert path == canonical.resolve()


def test_align_ledger_path_explicit_matches_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "canonical_ledger.json"
    monkeypatch.setenv("CASEPILOT_COST_LEDGER_PATH", str(canonical))
    path, err = align_ledger_path(canonical)
    assert err is None
    assert path == canonical.resolve()
    path2, err2 = align_ledger_path(None)
    assert err2 is None
    assert path2 == canonical.resolve()


@pytest.mark.asyncio
async def test_preflight_ledger_mismatch_blocks_before_any_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit --ledger != canonical => blocked_preflight; no judge/backend calls."""
    canonical = tmp_path / "canon.json"
    other = tmp_path / "pilot_only.json"
    # Healthy snapshot only on wrong path must not be used / written.
    other.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": True,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CASEPILOT_COST_LEDGER_PATH", str(canonical))
    calls = {"judge": 0, "backend": 0}

    async def no_judge(*args: Any, **kwargs: Any) -> JudgeOutcome:
        calls["judge"] += 1
        raise AssertionError("judge must not run on ledger path mismatch")

    class NoBackend:
        def start_investigation(self, case_id: str, objective: str) -> PhaseCapture:
            calls["backend"] += 1
            raise AssertionError("backend must not run on ledger path mismatch")

        def send_followup(
            self, run_id: str, question: str, *, case_id: str, idempotency_key: str
        ) -> PhaseCapture:
            calls["backend"] += 1
            raise AssertionError("backend must not run on ledger path mismatch")

    report = await run_scenario_pilot(
        PilotConfig(
            live=True,
            report_dir=tmp_path / "r",
            ledger_path=other,
            cohort_id="ledger-mismatch",
            model=ALLOWED_LIVE_MODEL,
        ),
        assets=load_pilot_assets(),
        backend=NoBackend(),
        settings=_settings(tmp_path),
        judge_fn=no_judge,
        prices_path=PRICES,
    )
    assert report.status == "blocked_preflight"
    assert any("differs from canonical" in b for b in report.blockers)
    assert calls["judge"] == 0
    assert calls["backend"] == 0
    # Must not create canonical ledger via pilot
    assert not canonical.exists()


@pytest.mark.asyncio
async def test_preflight_ledger_env_canonical_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env-canonical path + matching explicit --ledger is accepted (still may block elsewhere)."""
    canonical = tmp_path / "canon.json"
    canonical.write_text(
        json.dumps(
            {
                "budget_usd": 5.0,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "pricing_reliable": True,
                "open_reservations": {},
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CASEPILOT_COST_LEDGER_PATH", str(canonical))
    assets = load_pilot_assets()
    pf = preflight(
        PilotConfig(live=True, ledger_path=canonical, budget_usd=5.0, model=ALLOWED_LIVE_MODEL),
        assets,
        prices_path=PRICES,
    )
    assert not any("differs from canonical" in b for b in pf["blockers"])
    assert Path(pf["ledger_path"]).resolve() == canonical.resolve()


def test_outcome_dict_persists_invalid_diagnostic_roundtrip() -> None:
    """Invalid judge path must retain sanitized diagnostic in report JSON (no provider meta)."""
    diag = InvalidJudgeDiagnostic(
        label="untrusted model output",
        candidate_text='{"factual_support":{"rating":"partially meets"}}',
        candidate_object={
            "factual_support": {
                "rating": "partially meets",
                "rationale": "x",
                "source_ids": ["e1"],
            },
            "uncertainty": {"rating": "meets", "rationale": "y", "source_ids": []},
            "usefulness": {"rating": "does not meet", "rationale": "z", "source_ids": []},
        },
        truncated=False,
    )
    outcome = JudgeOutcome(
        call_id="base:test:s:r1:0",
        case_id="CASE-001",
        verdict=None,
        usage={"total_tokens": 3},
        cost_usd=0.001,
        error="invalid judge verdict: schema",
        model_name="scripted",
        provider_model=None,
        judge_prompt_version=judge_prompt_version(),
        judge_schema_version=judge_schema_version(),
        live=False,
        rubric_version="p0-v1",
        scenario_id="straightforward",
        invalid_diagnostic=diag,
    )
    dumped = _outcome_dict(outcome)
    assert dumped["verdict"] is None
    assert dumped["error"]
    assert dumped["invalid_diagnostic"] is not None
    assert dumped["invalid_diagnostic"]["label"] == "untrusted model output"
    assert dumped["invalid_diagnostic"]["candidate_object"]["factual_support"]["rating"] == (
        "partially meets"
    )
    # No provider metadata / CoT keys
    blob = json.dumps(dumped["invalid_diagnostic"])
    assert "response_metadata" not in blob
    assert "usage_metadata" not in blob
    assert "additional_kwargs" not in blob
    # Roundtrip through report slot JSON
    slot_judge = json.loads(json.dumps(dumped))
    assert slot_judge["invalid_diagnostic"]["candidate_text"].startswith("{")
    assert slot_judge["invalid_diagnostic"]["truncated"] is False


def test_outcome_dict_omits_diagnostic_when_valid() -> None:
    outcome = _outcome(
        "c1",
        "CASE-001",
        verdict=_verdict("meets", "meets", "meets"),
        cost=0.0,
    )
    dumped = _outcome_dict(outcome)
    assert dumped["verdict"] is not None
    assert dumped["invalid_diagnostic"] is None
