"""Unit tests for deterministic evaluation harness (no paid model calls)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
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
from casepilot.evaluation.runner import LiveEvalConfig, run_live, run_offline_report
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
# Public grading truth + failure denominator
# ---------------------------------------------------------------------------


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


def test_scorecard_denominator_caveats_and_serialization() -> None:
    """Failures in denom, entailment caveat, and JSON-serializable to_dict."""
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
    assert any("entailment" in c.lower() for c in sc.caveats)
    assert "category_accuracy" in json.dumps(sc.to_dict())


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


# ---------------------------------------------------------------------------
# Verified pricing + actual ledger reserve/settle/release/concurrency
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
    from casepilot.evaluation.cost import CONSERVATIVE_RESERVE_USD

    assert float(CONSERVATIVE_RESERVE_USD) > 0


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


# ---------------------------------------------------------------------------
# Live runner gates + spend (mocked HTTP — zero paid calls; no eval-side reserve)
# ---------------------------------------------------------------------------


def test_run_live_refuses_without_flag() -> None:
    with pytest.raises(RuntimeError, match="live"):
        run_live(LiveEvalConfig(live=False, max_cases=1))


def test_run_live_refuses_budget_over_five() -> None:
    assert DEFAULT_BUDGET_USD == 5.0
    with pytest.raises(CostPause, match="envelope"):
        run_live(LiveEvalConfig(live=True, max_cases=1, budget_usd=DEFAULT_BUDGET_USD + 0.01))


def test_run_live_refuses_non_glm_model() -> None:
    assert ALLOWED_LIVE_MODEL == "z-ai/glm-5.3-flash"
    with pytest.raises(CostPause, match="not allowed"):
        run_live(LiveEvalConfig(live=True, max_cases=1, model="anthropic/claude-sonnet-4"))


def test_run_live_http_mock_no_labels_no_eval_reserve(tmp_path: Path) -> None:
    gt = load_ground_truth(LABELS)
    case_id = gt.evaluation_ids[0]
    inv = _result(
        case_id,
        diagnosis=Diagnosis(gt.cases[case_id].expected_categories[0]),
        next_action=FIXTURE_ACTION_TO_NEXT[gt.cases[case_id].acceptable_actions[0]],
        evidence_ids=list(gt.cases[case_id].evidence_alternatives[0]),
        uncertainty="maybe" if gt.cases[case_id].require_uncertainty else None,
    )
    captured: list[dict] = []
    ledger_path = tmp_path / "reports" / "cost_ledger.json"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured.append(body)
        assert set(body) <= {"case_id", "objective"}
        return httpx.Response(
            200,
            json={
                "run_id": inv.run_id,
                "result": inv.model_dump(mode="json"),
                "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001},
                "cost_usd": 0.001,
                "provider_model": ALLOWED_LIVE_MODEL,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    cfg = LiveEvalConfig(
        live=True,
        split="evaluation",
        max_cases=1,
        budget_usd=5.0,
        report_dir=tmp_path / "reports",
        ledger_path=ledger_path,
    )
    report = run_live(cfg, http_client=client, gt=gt)
    client.close()
    assert len(captured) == 1
    assert report.live is True
    assert report.scorecard.n == 1
    assert report.session_spend_usd == pytest.approx(0.001)
    assert report.case_results[0].provider_model == ALLOWED_LIVE_MODEL
    assert report.case_results[0].usage is not None
    assert not ledger_path.exists() or (
        json.loads(ledger_path.read_text()).get("entries", []) == []
        if ledger_path.exists()
        else True
    )
    if ledger_path.exists():
        snap = json.loads(ledger_path.read_text())
        assert float(snap.get("spent_usd") or 0) == 0.0
        assert float(snap.get("reserved_usd") or 0) == 0.0


def test_run_live_session_threshold_from_backend_costs(tmp_path: Path) -> None:
    gt = load_ground_truth(LABELS)
    n = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        body = json.loads(request.content.decode())
        inv = _result(body["case_id"])
        return httpx.Response(
            200,
            json={
                "run_id": f"r{n}",
                "result": inv.model_dump(mode="json"),
                "cost_usd": 0.08,
                "usage": {"cost_usd": 0.08},
                "provider_model": ALLOWED_LIVE_MODEL,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    cfg = LiveEvalConfig(
        live=True,
        split="evaluation",
        max_cases=5,
        budget_usd=0.15,  # two cases at 0.08 then pause
        report_dir=tmp_path / "r",
        ledger_path=tmp_path / "r" / "ledger.json",
    )
    report = run_live(cfg, http_client=client, gt=gt)
    client.close()
    assert report.scorecard.n == 5
    assert report.session_spend_usd >= 0.08
    skipped = [c for c in report.case_results if c.error and "threshold" in (c.error or "")]
    assert skipped or any("threshold" in g for g in report.gaps)


def test_run_live_backend_cost_guard_pauses(tmp_path: Path) -> None:
    gt = load_ground_truth(LABELS)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(402, json={"detail": "cost pause: budget exceeded"})
        return httpx.Response(500, json={"detail": "should not be called"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    report = run_live(
        LiveEvalConfig(
            live=True,
            split="evaluation",
            max_cases=3,
            report_dir=tmp_path / "r",
            ledger_path=tmp_path / "r" / "ledger.json",
        ),
        http_client=client,
        gt=gt,
    )
    client.close()
    assert calls["n"] == 1
    assert any("cost guard" in (c.error or "") for c in report.case_results)


def test_run_live_pauses_when_global_ledger_exhausted(tmp_path: Path) -> None:
    gt = load_ground_truth(LABELS)
    ledger_path = tmp_path / "ledger.json"
    ledger = CostLedger(
        path=ledger_path,
        budget_usd=0.10,
        prices={ALLOWED_LIVE_MODEL: _vprice()},
        reserve_usd=0.10,
    )
    amt = ledger.reserve("prior")
    ledger.settle("prior", amt, actual_usd=amt)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call backend when ledger empty")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t")
    with pytest.raises(CostPause, match="remaining"):
        run_live(
            LiveEvalConfig(
                live=True,
                max_cases=2,
                split="evaluation",
                report_dir=tmp_path / "r",
                ledger_path=ledger_path,
                budget_usd=5.0,
            ),
            http_client=client,
            gt=gt,
        )
    client.close()


def test_offline_report(tmp_path: Path) -> None:
    """Public offline path: results JSON → report; failures/missing in denominator."""
    results = {
        "cases": [
            {"case_id": "CASE-001", "result": _result("CASE-001").model_dump(mode="json")},
            {"case_id": "CASE-019", "result": None, "error": "timeout"},
        ]
    }
    path = tmp_path / "in.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    report = run_offline_report(path, labels_path=LABELS, report_dir=tmp_path / "out")
    assert report.live is False
    assert report.scorecard.n == 2
    assert report.scorecard.failures >= 1
