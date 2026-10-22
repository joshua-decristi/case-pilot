"""Unit tests for the evaluation harness (no paid model calls)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from casepilot.evaluation.cost import (
    ALLOWED_LIVE_MODEL,
    DEFAULT_BUDGET_USD,
    CostLedger,
    CostPause,
    ModelPrice,
    RunTokenBudget,
    compute_max_reserve_usd,
    cost_from_usage,
    default_run_token_budget,
    get_conservative_reserve_usd,
    get_model_price,
    load_verified_prices,
    validate_live_model,
)
from casepilot.evaluation.grader import (
    FIXTURE_ACTION_TO_NEXT,
    grade_case,
    grade_results,
    load_ground_truth,
)
from casepilot.shared.contracts import (
    Diagnosis,
    Evidence,
    InvestigationResult,
    NextAction,
    RunStatus,
)

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "evals" / "ground_truth" / "labels.json"


def _vprice(
    inp: float = 1.0,
    out: float = 1.0,
    *,
    model: str = ALLOWED_LIVE_MODEL,
) -> ModelPrice:
    """Test helper: explicitly verified prices (never rely on guessed module defaults)."""
    return ModelPrice(
        model=model,
        input_usd_per_mtok=inp,
        output_usd_per_mtok=out,
        source="unit-test",
        verified_at="test",
        verified=True,
    )


def _result(
    case_id: str = "CASE-001",
    *,
    diagnosis: Diagnosis = Diagnosis.MISSING_DOCUMENT,
    status: RunStatus = RunStatus.COMPLETED,
    next_action: NextAction = NextAction.REQUEST_MISSING_DOCUMENT,
    evidence_ids: list[str] | None = None,
    retrieval_refs: list[str] | None = None,
    uncertainty: str | None = None,
    unresolved: list[str] | None = None,
) -> InvestigationResult:
    ids = evidence_ids if evidence_ids is not None else ["CASE-001", "EVT-001-A"]
    refs = retrieval_refs if retrieval_refs is not None else ["get_events"] * len(ids)
    if len(refs) < len(ids):
        refs = list(refs) + ["tool"] * (len(ids) - len(refs))
    evidence = [
        Evidence(
            source_system="simulator",
            record_id=rid,
            field="status",
            fact="observed",
            retrieval_ref=refs[i],
        )
        for i, rid in enumerate(ids)
    ]
    return InvestigationResult(
        run_id="run-test",
        case_id=case_id,
        status=status,
        diagnosis=diagnosis,
        diagnosis_summary="summary",
        explanation="explanation",
        evidence=evidence,
        uncertainty=uncertainty,
        unresolved_questions=unresolved or [],
        next_action=next_action,
        next_action_explanation="because",
        approval_required=next_action == NextAction.CREATE_FOLLOW_UP_TASK,
    )


# ---------------------------------------------------------------------------
# Ground truth + action mapping
# ---------------------------------------------------------------------------


def test_load_ground_truth_split_and_mapping() -> None:
    gt = load_ground_truth(LABELS)
    assert gt.version == "1.0"
    assert len(gt.cases) == 20
    assert len(gt.evaluation_ids) == 8
    assert len(gt.development_ids) == 12
    assert set(gt.evaluation_ids).isdisjoint(gt.development_ids)
    assert FIXTURE_ACTION_TO_NEXT["request_document"] == NextAction.REQUEST_MISSING_DOCUMENT
    assert FIXTURE_ACTION_TO_NEXT["reconcile_data"] == NextAction.CORRECT_DATA_MISMATCH
    assert FIXTURE_ACTION_TO_NEXT["investigate_processing"] == NextAction.RETRY_PROCESSING
    assert FIXTURE_ACTION_TO_NEXT["escalate"] == NextAction.ESCALATE_TO_HUMAN
    assert FIXTURE_ACTION_TO_NEXT["no_action"] == NextAction.NO_ACTION_REQUIRED
    assert NextAction.CREATE_FOLLOW_UP_TASK not in FIXTURE_ACTION_TO_NEXT.values()


def test_grade_category_action_evidence_happy_path() -> None:
    gt = load_ground_truth(LABELS)
    label = gt.cases["CASE-001"]
    result = _result(
        "CASE-001",
        evidence_ids=["CASE-001", "EVT-001-A"],
        next_action=NextAction.REQUEST_MISSING_DOCUMENT,
    )
    g = grade_case(label, result)
    assert g.category_ok is True
    assert g.evidence_ok is True
    assert g.action_ok is True
    assert g.action_needs_manual is False
    assert g.uncertainty_ok is None
    assert g.failed is False


def test_evidence_requires_record_id_not_ref_token_bypass() -> None:
    """IDs stuffed only into retrieval_ref path must NOT satisfy evidence."""
    gt = load_ground_truth(LABELS)
    label = gt.cases["CASE-011"]
    # alternative needs EVT-011-A as record_id — ref path alone is insufficient
    result = _result(
        "CASE-011",
        diagnosis=Diagnosis.PROCESSING_FAILURE,
        next_action=NextAction.RETRY_PROCESSING,
        evidence_ids=["CASE-011"],
        retrieval_refs=["get_events/EVT-011-A#1"],
    )
    assert grade_case(label, result).evidence_ok is False

    ok = _result(
        "CASE-011",
        diagnosis=Diagnosis.PROCESSING_FAILURE,
        next_action=NextAction.RETRY_PROCESSING,
        evidence_ids=["EVT-011-A"],
        retrieval_refs=["get_events"],
    )
    assert grade_case(label, ok).evidence_ok is True


def test_evidence_requires_nonempty_retrieval_ref() -> None:
    gt = load_ground_truth(LABELS)
    label = gt.cases["CASE-001"]
    result = _result(
        "CASE-001",
        evidence_ids=["CASE-001", "EVT-001-A"],
        retrieval_refs=["", ""],
    )
    assert grade_case(label, result).evidence_ok is False


def test_evidence_alternative_any_of() -> None:
    gt = load_ground_truth(LABELS)
    label = gt.cases["CASE-001"]
    result = _result("CASE-001", evidence_ids=["CASE-001", "EVT-001-B"])
    assert grade_case(label, result).evidence_ok is True
    bad = _result("CASE-001", evidence_ids=["CASE-001"])
    assert grade_case(label, bad).evidence_ok is False


def test_create_follow_up_task_not_auto_correct() -> None:
    gt = load_ground_truth(LABELS)
    label = gt.cases["CASE-001"]
    result = _result(
        "CASE-001",
        next_action=NextAction.CREATE_FOLLOW_UP_TASK,
        evidence_ids=["CASE-001", "EVT-001-A"],
    )
    g = grade_case(label, result)
    assert g.action_ok is None
    assert g.action_needs_manual is True
    assert any("manual" in n.lower() for n in g.notes)


def test_uncertainty_required_on_indeterminate() -> None:
    gt = load_ground_truth(LABELS)
    label = gt.cases["CASE-016"]
    assert label.require_uncertainty is True
    missing = _result(
        "CASE-016",
        diagnosis=Diagnosis.INDETERMINATE,
        next_action=NextAction.ESCALATE_TO_HUMAN,
        evidence_ids=["DOC-016-B", "DOC-016-C"],
        uncertainty=None,
    )
    assert grade_case(label, missing).uncertainty_ok is False
    present = _result(
        "CASE-016",
        diagnosis=Diagnosis.INDETERMINATE,
        next_action=NextAction.ESCALATE_TO_HUMAN,
        evidence_ids=["DOC-016-B", "DOC-016-C"],
        uncertainty="vendor vs store conflict",
    )
    assert grade_case(label, present).uncertainty_ok is True


def test_failures_count_in_denominator_as_incorrect() -> None:
    gt = load_ground_truth(LABELS)
    failed = _result(
        "CASE-001",
        status=RunStatus.FAILED,
        diagnosis=Diagnosis.INDETERMINATE,
        next_action=NextAction.ESCALATE_TO_HUMAN,
        evidence_ids=[],
    )
    good = _result(
        "CASE-019",
        diagnosis=Diagnosis.NO_ISSUE,
        next_action=NextAction.NO_ACTION_REQUIRED,
        evidence_ids=["CASE-019", "EVT-019-A", "EVT-019-B"],
    )
    sc = grade_results(
        gt,
        {"CASE-001": failed, "CASE-019": good, "CASE-004": None},
        errors={"CASE-004": "timeout"},
    )
    assert sc.n == 3
    assert sc.failures >= 2
    assert sc.category_correct == 1
    assert sc.category_accuracy == pytest.approx(1 / 3)
    assert sc.evidence_correct == 1
    assert sc.action_auto_scored >= 2
    assert any("denominator" in c.lower() or "Failure" in c for c in sc.caveats)


def test_wrong_action_mapped() -> None:
    gt = load_ground_truth(LABELS)
    label = gt.cases["CASE-019"]
    result = _result(
        "CASE-019",
        diagnosis=Diagnosis.NO_ISSUE,
        next_action=NextAction.REQUEST_MISSING_DOCUMENT,
        evidence_ids=["CASE-019", "EVT-019-A", "EVT-019-B"],
    )
    g = grade_case(label, result)
    assert g.category_ok is True
    assert g.action_ok is False


def test_no_entailment_caveat_in_scorecard() -> None:
    gt = load_ground_truth(LABELS)
    sc = grade_results(gt, {"CASE-001": _result()})
    assert any("entailment" in c.lower() for c in sc.caveats)


# ---------------------------------------------------------------------------
# Cost ledger (backend integration API)
# ---------------------------------------------------------------------------


def test_pricing_from_verified_unit_prices_not_flat_fee() -> None:
    price = _vprice(0.5, 2.0)
    tokens = RunTokenBudget(max_input_tokens=100_000, max_output_tokens=10_000)
    amt = compute_max_reserve_usd(price, tokens)
    expected = (100_000 / 1e6) * 0.5 + (10_000 / 1e6) * 2.0
    assert amt == pytest.approx(expected)
    assert cost_from_usage(price, input_tokens=1000, output_tokens=500) == pytest.approx(
        (1000 / 1e6) * 0.5 + (500 / 1e6) * 2.0
    )
    assert cost_from_usage(price, input_tokens=None, output_tokens=1) is None
    unverified = ModelPrice(
        ALLOWED_LIVE_MODEL, 1.0, 1.0, verified=False, source="x", verified_at=""
    )
    with pytest.raises(CostPause, match="unverified"):
        compute_max_reserve_usd(unverified, tokens)


def test_fail_closed_without_verified_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "nope.json"
    monkeypatch.setenv("CASEPILOT_VERIFIED_PRICES_PATH", str(missing))
    assert load_verified_prices() == {}
    with pytest.raises(CostPause, match="no verified unit price"):
        get_model_price(ALLOWED_LIVE_MODEL)
    with pytest.raises(CostPause, match="not allowed"):
        get_model_price("openai/gpt-4o")
    with pytest.raises(CostPause, match="no verified"):
        get_model_price(ALLOWED_LIVE_MODEL, prices={})
    with pytest.raises(CostPause):
        validate_live_model("other/model")


def test_load_verified_prices_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "prices.json"
    p.write_text(
        json.dumps(
            {
                "models": {
                    ALLOWED_LIVE_MODEL: {
                        "input_usd_per_mtok": 0.1,
                        "output_usd_per_mtok": 0.2,
                        "source": "https://openrouter.ai/example",
                        "verified_at": "2026-01-01",
                        "verified": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CASEPILOT_VERIFIED_PRICES_PATH", str(p))
    table = load_verified_prices()
    assert ALLOWED_LIVE_MODEL in table
    price = get_model_price(ALLOWED_LIVE_MODEL)
    assert price.input_usd_per_mtok == 0.1
    # verified=false ignored
    p.write_text(
        json.dumps(
            {
                "models": {
                    ALLOWED_LIVE_MODEL: {
                        "input_usd_per_mtok": 9.0,
                        "output_usd_per_mtok": 9.0,
                        "verified": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert load_verified_prices() == {}


def test_default_verified_prices_file_loads() -> None:
    """Committed verified_prices.json is the enforced-cap source (no .env needed)."""
    table = load_verified_prices()
    assert ALLOWED_LIVE_MODEL in table
    price = get_model_price(ALLOWED_LIVE_MODEL)
    assert price.verified is True
    assert price.kind == "enforced_routing_cap"
    assert price.input_usd_per_mtok == pytest.approx(0.23)
    assert price.output_usd_per_mtok == pytest.approx(2.0)
    assert price.list_input_usd_per_mtok == pytest.approx(0.225)
    assert price.list_output_usd_per_mtok == pytest.approx(2.0)
    assert "endpoints" in price.source
    prov = price.openrouter_provider()
    assert prov["require_parameters"] is True
    assert prov["max_price"]["prompt"] == pytest.approx(0.23)
    assert prov["max_price"]["completion"] == pytest.approx(2.0)
    # Import / proxy does not require .env
    from casepilot.evaluation.cost import CONSERVATIVE_RESERVE_USD

    assert float(CONSERVATIVE_RESERVE_USD) > 0


def test_cost_ledger_default_init_without_env(tmp_path: Path) -> None:
    """CostLedger constructs from committed verified file; no .env required."""
    ledger = CostLedger(path=tmp_path / "led.json", budget_usd=5.0)
    assert ALLOWED_LIVE_MODEL in (ledger.prices or {})
    res = ledger.create_reservation(
        key="k",
        amount_usd=0.01,
        model=ALLOWED_LIVE_MODEL,
        case_id="C",
    )
    ledger.release_reservation(res.id, detail="test cleanup")


def test_cost_ledger_compat_reserve_settle_release(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    price = _vprice(1.0, 1.0)
    amount = compute_max_reserve_usd(price, RunTokenBudget(50_000, 50_000))  # $0.10
    ledger = CostLedger(
        path=path,
        budget_usd=0.15,
        prices={ALLOWED_LIVE_MODEL: price},
        reserve_usd=amount,
    )

    r1 = ledger.reserve("C1")
    assert r1 == pytest.approx(0.10)
    spent = ledger.settle("C1", r1, actual_usd=None)
    assert spent == pytest.approx(0.10)
    snap = ledger.snapshot()
    assert snap["spent_usd"] == pytest.approx(0.10)
    assert snap["open_reservations"] == {}

    with pytest.raises(CostPause, match="exceed"):
        ledger.reserve("C2")

    ledger2 = CostLedger(
        path=tmp_path / "l2.json",
        budget_usd=1.0,
        prices={ALLOWED_LIVE_MODEL: price},
        reserve_usd=0.25,
    )
    r = ledger2.reserve("k")
    ledger2.release("k", r, detail="failed pre-call")
    assert ledger2.snapshot()["reserved_usd"] == pytest.approx(0.0)
    assert ledger2.snapshot()["spent_usd"] == pytest.approx(0.0)


def test_create_reservation_primary_api(tmp_path: Path) -> None:
    price = _vprice(2.0, 2.0)
    amount = compute_max_reserve_usd(price, RunTokenBudget(1_000, 1_000))
    ledger = CostLedger(
        path=tmp_path / "e.json",
        budget_usd=5.0,
        prices={ALLOWED_LIVE_MODEL: price},
    )
    res = ledger.create_reservation(
        key="run-x", amount_usd=amount, model=ALLOWED_LIVE_MODEL, case_id="C"
    )
    spent = ledger.settle_reservation(
        res.id,
        actual_usd=amount / 2,
        usage={"x": 1},
        provider_model=ALLOWED_LIVE_MODEL,
    )
    assert spent == pytest.approx(amount / 2)
    assert ledger.snapshot()["pricing_reliable"] is True


def test_actual_over_reserve_records_full_actual_and_pauses(tmp_path: Path) -> None:
    price = _vprice(1.0, 1.0)
    ledger = CostLedger(
        path=tmp_path / "over.json",
        budget_usd=5.0,
        prices={ALLOWED_LIVE_MODEL: price},
        reserve_usd=0.05,
    )
    res = ledger.create_reservation(
        key="run", amount_usd=0.05, model=ALLOWED_LIVE_MODEL, case_id="C"
    )
    spent = ledger.settle_reservation(res.id, actual_usd=0.20, usage={"cost": 0.20})
    assert spent == pytest.approx(0.20)  # truthful, not clamped to 0.05
    snap = ledger.snapshot()
    assert snap["spent_usd"] == pytest.approx(0.20)
    assert snap["pricing_reliable"] is False
    with pytest.raises(CostPause, match="unreliable"):
        ledger.create_reservation(key="next", amount_usd=0.01, model=ALLOWED_LIVE_MODEL)


def test_recover_stale_reservations_settles_full_no_refund(tmp_path: Path) -> None:
    price = _vprice(1.0, 1.0)
    path = tmp_path / "stale.json"
    ledger = CostLedger(
        path=path,
        budget_usd=5.0,
        prices={ALLOWED_LIVE_MODEL: price},
        reserve_usd=0.12,
    )
    r1 = ledger.create_reservation(key="a", amount_usd=0.12, model=ALLOWED_LIVE_MODEL, case_id="C1")
    r2 = ledger.create_reservation(key="b", amount_usd=0.12, model=ALLOWED_LIVE_MODEL, case_id="C2")
    assert len(ledger.open_reservations()) == 2

    with pytest.raises(CostPause, match="confirm"):
        ledger.recover_stale_reservations(operator="ops", reason="crash", confirm=False)

    recovered = ledger.recover_stale_reservations(
        operator="ops", reason="process crash mid-call", confirm=True
    )
    assert len(recovered) == 2
    assert {r["reservation_id"] for r in recovered} == {r1.id, r2.id}
    snap = ledger.snapshot()
    assert snap["open_reservations"] == {}
    assert snap["spent_usd"] == pytest.approx(0.24)  # full reserves, not released
    assert snap["reserved_usd"] == pytest.approx(0.0)
    assert snap["pricing_reliable"] is False
    kinds = [e["kind"] for e in snap["entries"]]
    assert "stale_recover" in kinds
    assert any(e.get("operator") == "ops" for e in snap["entries"] if e["kind"] == "stale_recover")
    # no further reserves
    with pytest.raises(CostPause, match="unreliable"):
        ledger.create_reservation(key="z", amount_usd=0.01, model=ALLOWED_LIVE_MODEL)


def test_cost_ledger_rejects_budget_over_global(tmp_path: Path) -> None:
    assert DEFAULT_BUDGET_USD == 5.0
    with pytest.raises(CostPause, match="envelope"):
        CostLedger(
            path=tmp_path / "x.json",
            budget_usd=DEFAULT_BUDGET_USD + 0.01,
            prices={},
            reserve_usd=0.01,
        )


def test_conservative_reserve_requires_verified_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CASEPILOT_VERIFIED_PRICES_PATH", str(tmp_path / "missing-prices.json"))
    with pytest.raises(CostPause, match="verified"):
        get_conservative_reserve_usd()
    p = tmp_path / "v.json"
    p.write_text(
        json.dumps(
            {
                "models": {
                    ALLOWED_LIVE_MODEL: {
                        "input_usd_per_mtok": 1.0,
                        "output_usd_per_mtok": 1.0,
                        "verified": True,
                        "source": "test",
                        "verified_at": "t",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CASEPILOT_VERIFIED_PRICES_PATH", str(p))
    expected = compute_max_reserve_usd(
        get_model_price(ALLOWED_LIVE_MODEL), default_run_token_budget()
    )
    assert get_conservative_reserve_usd() == pytest.approx(expected)


def test_cost_ledger_file_lock_serializes(tmp_path: Path) -> None:
    path = tmp_path / "shared.json"
    price = _vprice(10.0, 10.0)
    amount = 0.01
    errors: list[str] = []
    ok = 0
    lock = threading.Lock()

    def worker(i: int) -> None:
        nonlocal ok
        ledger = CostLedger(
            path=path,
            budget_usd=0.05,
            prices={ALLOWED_LIVE_MODEL: price},
            reserve_usd=amount,
        )
        try:
            r = ledger.reserve(f"k{i}", amount=amount)
            ledger.settle(f"k{i}", r, actual_usd=amount)
            with lock:
                ok += 1
        except CostPause as exc:
            with lock:
                errors.append(str(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ok == 5
    assert len(errors) == 7
    assert CostLedger(
        path=path, budget_usd=0.05, prices={ALLOWED_LIVE_MODEL: price}, reserve_usd=amount
    ).snapshot()["spent_usd"] == pytest.approx(0.05)
