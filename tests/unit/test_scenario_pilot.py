"""Unit tests for scenario pilot orchestration (no paid calls)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

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
from casepilot.evaluation.__main__ import main
from casepilot.evaluation.cost import ALLOWED_LIVE_MODEL, DEFAULT_BUDGET_USD, CostLedger
from casepilot.evaluation.pilot_assets import load_pilot_assets
from casepilot.evaluation.pilot_coverage import AnswerCitation, AttemptSnapshot, ExposedFact
from casepilot.evaluation.pilot_safety import (
    ForbiddenAttemptFact,
    SnapshotProvenanceFact,
)
from casepilot.evaluation.scenario_pilot import (
    DISCLAIMERS,
    SLOT_COUNT,
    JudgeFn,
    PhaseCapture,
    PilotConfig,
    _outcome_dict,
    assess_slot_safety,
    assign_slots,
    build_calibration_payload,
    classify_spend_pause,
    execute_slot,
    model_visible_available_evidence,
    preflight,
    run_calibration,
    run_scenario_pilot,
    summarize_completion,
)
from casepilot.shared.config import Settings, get_settings

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
            pre = ()
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


def _healthy_ledger(tmp_path: Path) -> Path:
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


def _frozen_versions(assets: Any) -> dict[str, Any]:
    return {
        "assets_version": assets.version,
        "fixture_version": assets.fixture_version,
        "asset_hashes": assets.hashes.to_dict(),
        "rubric_version": "p0-v1",
        "judge_prompt_version": judge_prompt_version(),
        "judge_schema_version": judge_schema_version(),
        "model": ALLOWED_LIVE_MODEL,
    }


# ---------------------------------------------------------------------------
# Live / spend fail-closed preflight
# ---------------------------------------------------------------------------


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


@pytest.mark.asyncio
async def test_preflight_ledger_mismatch_blocks_before_any_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit --ledger != canonical => blocked_preflight; no judge/backend calls."""
    canonical = tmp_path / "canon.json"
    other = tmp_path / "pilot_only.json"
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
    assert not canonical.exists()


def test_cli_scenario_pilot_requires_pilot_live() -> None:
    code = main(["--scenario-pilot", "--report-dir", "/tmp/x"])
    assert code == 2


def test_classify_spend_pause_exact_prefixes_only() -> None:
    assert classify_spend_pause(["cost pause: ledger empty"]) is not None
    assert classify_spend_pause(["execution budget exhausted: reserve failed"]) is not None
    assert classify_spend_pause(["backend cost guard HTTP 402: x"]) is not None
    assert classify_spend_pause(["execution limit exceeded: max tool calls"]) is None
    assert classify_spend_pause(["agent error: tool budget for case"]) is None
    assert classify_spend_pause([]) is None


@pytest.mark.asyncio
async def test_orchestrator_tool_limit_continues_later_slots(tmp_path: Path) -> None:
    """Per-scenario execution limit => incomplete slot, later slots still run."""
    ledger_path = _healthy_ledger(tmp_path)
    assets = load_pilot_assets()
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
    later = [s for s in report.slots if s.scenario_id == "conflict"]
    assert later
    assert all(s.status != "skipped_budget" for s in later)
    assert any(s.status == "complete" for s in later)
    assert report.completion.get("p0_measurement_complete") is not True
    assert not any("spend pause" in g for g in report.gaps)


@pytest.mark.asyncio
async def test_orchestrator_cost_pause_stops_later_slots(tmp_path: Path) -> None:
    """Genuine CostPause / spend signal stops cohort; remaining slots skipped."""
    ledger_path = _healthy_ledger(tmp_path)
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
    assert any(s.status == "skipped_budget" for s in report.slots[1:])
    assert calls["n"] == 1
    assert any("spend pause" in g for g in report.gaps)


# ---------------------------------------------------------------------------
# Calibration integrity
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


def test_calibration_payload_uses_reference_not_agent_leak() -> None:
    assets = load_pilot_assets()
    ex = assets.calibration[0]
    ref = assets.references[ex.scenario_id]
    payload = build_calibration_payload(ex, ref)
    dumped = payload.model_dump()
    assert "reference" in dumped
    assert dumped["answer"] == ex.answer
    assert "system_prompt" not in json.dumps(dumped)


@pytest.mark.asyncio
async def test_run_pilot_calibration_failure_blocks_baseline(tmp_path: Path) -> None:
    ledger_path = _healthy_ledger(tmp_path)

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
    assert backend.calls == []


@pytest.mark.asyncio
async def test_resume_without_prior_cal_blocks_rebill(tmp_path: Path) -> None:
    ledger_path = _healthy_ledger(tmp_path)
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
    resume_path = tmp_path / "prior.json"
    resume_path.write_text(
        json.dumps(
            {
                "cohort_id": "no-recal",
                "slots": prior_slots,
                "versions": _frozen_versions(assets),
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


# ---------------------------------------------------------------------------
# Core offline workflow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pilot_offline_fake_backend(tmp_path: Path) -> None:
    assets = load_pilot_assets()
    ledger_path = _healthy_ledger(tmp_path)
    cfg = PilotConfig(
        live=True,
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
    snap = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert float(snap["spent_usd"]) == 0.0


# ---------------------------------------------------------------------------
# Resume / report truth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_only_unstarted(tmp_path: Path) -> None:
    ledger_path = _healthy_ledger(tmp_path)
    assets = load_pilot_assets()
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
    resume_path.write_text(
        json.dumps(
            {
                "cohort_id": "resume",
                "slots": prior_slots,
                "versions": _frozen_versions(assets),
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
    assert report.slots[1].status == "complete"


@pytest.mark.asyncio
async def test_resume_without_cli_cohort_reuses_prior_id(tmp_path: Path) -> None:
    """Resume must not mint a new cohort_id while loading prior slots."""
    ledger_path = _healthy_ledger(tmp_path)
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
    resume_path = tmp_path / "prior.json"
    resume_path.write_text(
        json.dumps(
            {
                "cohort_id": "original-cohort-id",
                "slots": prior_slots,
                "versions": _frozen_versions(assets),
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
            cohort_id=None,
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
    resume_path = tmp_path / "prior.json"
    resume_path.write_text(
        json.dumps(
            {
                "cohort_id": "A",
                "slots": prior_slots,
                "versions": _frozen_versions(assets),
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


@pytest.mark.asyncio
async def test_missing_cost_reported_not_zero(tmp_path: Path) -> None:
    ledger_path = _healthy_ledger(tmp_path)

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
    assert report.costs.total_usd is None or report.costs.missing_cost_gaps


def test_summarize_completion_not_assessed_blocks_p0() -> None:
    """Valid judge + coverage must not complete when safety is not_assessed."""
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


def test_outcome_dict_persists_invalid_diagnostic_roundtrip() -> None:
    """Invalid judge path must retain sanitized diagnostic in report JSON."""
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
    blob = json.dumps(dumped["invalid_diagnostic"])
    assert "response_metadata" not in blob
    assert "usage_metadata" not in blob
    slot_judge = json.loads(json.dumps(dumped))
    assert slot_judge["invalid_diagnostic"]["candidate_text"].startswith("{")


# ---------------------------------------------------------------------------
# Safety / scoring honesty
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


def test_preserve_actual_snapshot_case_id_no_retag() -> None:
    """Runner must not retag observer SnapshotProvenanceFact with intended case_id."""
    from casepilot.evaluation.scenario_pilot import _phase_evidence, _preserve_snapshot_facts

    actual = SnapshotProvenanceFact(
        snapshot_id="snap-x",
        run_id="run-1",
        case_id="CASE-099",
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
    assert safety.verdict == "fail"
