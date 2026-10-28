"""Eval harness coverage-gate contract tests (applied redesign).

Fixture-driven public-boundary checks for the P0 retrieve coverage gate.
Historically applied map: nine original gate nodes → these contract tests
(see per-test maps). Does not replace run_scenario_pilot orchestration.

Assessment honesty (classify/assess status paths, not_assessed serialization of
raw assessments) stays with test_pilot_coverage assessment tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from casepilot.evaluation.pilot_coverage import (
    CoverageAssessment,
    CoverageStatus,
    RepetitionCoverage,
    RetrievalClass,
    summarize_cohort_coverage,
)
from casepilot.evaluation.scenario_pilot import (
    REQUIRED_RETRIEVAL_SCENARIO_ID,
    SLOT_COUNT,
    SlotResult,
    build_coverage_gate_summary,
    reevaluate_saved_cohort_report,
    summarize_completion,
)

# Pilot scenario layout (frozen six × 3 reps).
_COHORT_IDS: list[tuple[str, str, str]] = [
    ("straightforward", "investigation", "CASE-001"),
    ("conflict", "investigation", "CASE-016"),
    ("retrieve", "follow_up", "CASE-011"),
    ("insufficient", "follow_up", "CASE-013"),
    ("premise", "follow_up", "CASE-002"),
    ("bypass", "follow_up", "CASE-008"),
]


def _assessment(
    status: str,
    *,
    retrieval_class: str = "retrieval_required",
) -> CoverageAssessment:
    return CoverageAssessment(
        status=CoverageStatus(status),
        retrieval_class=RetrievalClass(retrieval_class),
    )


def _rep(
    scenario_id: str,
    repetition: int,
    status: str,
    *,
    retrieval_class: str = "retrieval_required",
) -> RepetitionCoverage:
    return RepetitionCoverage(
        scenario_id,
        repetition,
        _assessment(status, retrieval_class=retrieval_class),
    )


def _slot(
    *,
    scenario_id: str,
    repetition: int,
    kind: str,
    case_id: str,
    index: int,
    coverage_status: str | None = None,
) -> SlotResult:
    cov = None
    if kind == "follow_up" and coverage_status is not None:
        cov = {
            "scenario_id": scenario_id,
            "repetition": repetition,
            "assessment": {
                "status": coverage_status,
                "retrieval_class": "retrieval_required",
                "notes": (),
            },
        }
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
        coverage=cov,
    )


def _live6_false_complete_slots() -> list[SlotResult]:
    """18 complete slots; only insufficient r2 covered (retrieve never)."""
    slots: list[SlotResult] = []
    idx = 0
    for sid, kind, cid in _COHORT_IDS:
        for rep in (1, 2, 3):
            st: str | None = None
            if kind == "follow_up":
                st = "covered" if sid == "insufficient" and rep == 2 else "not_covered"
            slots.append(
                _slot(
                    scenario_id=sid,
                    repetition=rep,
                    kind=kind,
                    case_id=cid,
                    index=idx,
                    coverage_status=st,
                )
            )
            idx += 1
    assert len(slots) == SLOT_COUNT
    return slots


def _slots_to_report_rows(slots: list[SlotResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in slots:
        rows.append(
            {
                "index": s.index,
                "scenario_id": s.scenario_id,
                "repetition": s.repetition,
                "case_id": s.case_id,
                "kind": s.kind,
                "status": s.status,
                "billable_started": s.billable_started,
                "judge": s.judge,
                "safety": s.safety,
                "coverage": s.coverage,
                "costs": {},
                "errors": [],
                "gaps": [],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Cohort mandated-gate API (summarize_cohort_coverage)
# ---------------------------------------------------------------------------


def test_cohort_mandated_gate_tallies_fail_closed_and_optional_mix() -> None:
    """Mandated cohort gate: tallies, fail-closed, only-retrieve, optional mix.

    Prospective full replace:
    - test_pilot_coverage::test_cohort_requires_at_least_one_covered
    - test_pilot_coverage::test_cohort_demonstrated_when_mandated_scenario_has_one_covered
    - test_pilot_coverage::test_cohort_without_mandated_ids_never_defaults_gate_true
    - test_pilot_coverage::test_retrieve_zero_of_three_incomplete_even_if_only_retrieve_in_cohort
    """
    # optional + not_covered (not_eligible): tallies and incomplete note
    mixed = summarize_cohort_coverage(
        [
            _rep("retrieve", 1, "not_eligible", retrieval_class="retrieval_optional"),
            _rep("retrieve", 2, "not_covered"),
        ],
        mandated_scenario_ids=("retrieve",),
    )
    assert mixed.n == 2
    assert mixed.optional == 1
    assert mixed.required == 1
    assert mixed.covered == 0
    assert mixed.not_covered == 1
    assert mixed.not_eligible == 1
    assert mixed.required_coverage_demonstrated is False
    assert mixed.by_scenario["retrieve"].covered == 0
    assert any("P0 retrieval-coverage incomplete" in n for n in mixed.notes)

    # only-retrieve 0/3 still incomplete (no other scenarios in cohort)
    only_retrieve = summarize_cohort_coverage(
        [_rep("retrieve", i, "not_covered") for i in (1, 2, 3)],
        mandated_scenario_ids=("retrieve",),
    )
    assert only_retrieve.by_scenario["retrieve"].covered == 0
    assert only_retrieve.by_scenario["retrieve"].n == 3
    assert only_retrieve.required_coverage_demonstrated is False

    # empty mandate never defaults true from global covered
    no_mandate = summarize_cohort_coverage([_rep("retrieve", 1, "covered")])
    assert no_mandate.covered == 1
    assert no_mandate.required_coverage_demonstrated is False
    assert no_mandate.mandated_scenario_ids == ()
    assert any("mandated_scenario_ids not provided" in n for n in no_mandate.notes)

    # missing mandated id fail-closed
    missing = summarize_cohort_coverage(
        [_rep("insufficient", 1, "covered")],
        mandated_scenario_ids=("retrieve",),
    )
    assert missing.covered == 1
    assert missing.required_coverage_demonstrated is False
    assert missing.by_scenario["retrieve"].n == 0
    assert any("missing from cohort" in n for n in missing.notes)

    # one covered + optional mix satisfies; to_dict preserves gate fields
    demonstrated = summarize_cohort_coverage(
        [
            _rep("retrieve", 1, "covered"),
            _rep("retrieve", 2, "not_eligible", retrieval_class="retrieval_optional"),
            _rep("retrieve", 3, "not_eligible", retrieval_class="retrieval_optional"),
        ],
        mandated_scenario_ids=("retrieve",),
    )
    assert demonstrated.covered == 1
    assert demonstrated.optional == 2
    assert demonstrated.required_coverage_demonstrated is True
    assert demonstrated.mandated_scenario_ids == ("retrieve",)
    assert demonstrated.by_scenario["retrieve"].covered == 1
    assert demonstrated.by_scenario["retrieve"].required_coverage_demonstrated is True
    assert demonstrated.notes == []
    d = demonstrated.to_dict()
    assert d["required_coverage_demonstrated"] is True
    assert d["by_scenario"]["retrieve"]["covered"] == 1
    assert d["repetitions"][0]["assessment"]["status"] == "covered"


# ---------------------------------------------------------------------------
# Slot gate via build_coverage_gate_summary
# ---------------------------------------------------------------------------


def test_p0_retrieve_gate_rejects_incidental_insufficient_coverage() -> None:
    """Incidental insufficient covered must not open the frozen retrieve gate.

    Prospective full replace:
    - test_scenario_pilot::test_coverage_gate_only_retrieve_scenario
    - test_pilot_coverage::test_incidental_insufficient_covered_does_not_satisfy_retrieve_gate
    """
    slots = [
        _slot(
            scenario_id="retrieve",
            repetition=r,
            kind="follow_up",
            case_id="CASE-011",
            index=r - 1,
            coverage_status="not_covered",
        )
        for r in (1, 2, 3)
    ]
    slots.extend(
        _slot(
            scenario_id="insufficient",
            repetition=r,
            kind="follow_up",
            case_id="CASE-013",
            index=3 + r - 1,
            coverage_status="covered" if r == 2 else "not_covered",
        )
        for r in (1, 2, 3)
    )

    summary = build_coverage_gate_summary(slots)

    assert summary["gate_scenario_id"] == REQUIRED_RETRIEVAL_SCENARIO_ID
    assert summary["mandated_scenario_ids"] == [REQUIRED_RETRIEVAL_SCENARIO_ID]
    assert summary["required_coverage_demonstrated"] is False
    assert summary["per_scenario"]["retrieve"]["covered"] == 0
    assert summary["per_scenario"]["retrieve"]["n"] == 3
    assert summary["per_scenario"]["insufficient"]["covered"] == 1
    assert summary["per_scenario"]["insufficient"]["required_coverage_demonstrated"] is True
    # Informational global/all_followups must not act as the P0 gate.
    assert summary["covered"] == 1
    assert summary["all_followups"]["required_coverage_demonstrated"] is False
    assert any("retrieve" in n and "0/3 covered" in n for n in summary["notes"])


def test_p0_retrieve_gate_accepts_one_retrieve_covered() -> None:
    """≥1 covered on retrieve satisfies gate despite other-scenario noise.

    Prospective full replace:
    - test_scenario_pilot::test_coverage_gate_retrieve_covered_satisfies
    """
    slots = [
        _slot(
            scenario_id="retrieve",
            repetition=r,
            kind="follow_up",
            case_id="CASE-011",
            index=r - 1,
            coverage_status="covered" if r == 1 else "not_covered",
        )
        for r in (1, 2, 3)
    ]
    slots.append(
        _slot(
            scenario_id="insufficient",
            repetition=1,
            kind="follow_up",
            case_id="CASE-013",
            index=3,
            coverage_status="covered",
        )
    )

    summary = build_coverage_gate_summary(slots)

    assert summary["required_coverage_demonstrated"] is True
    assert summary["gate_scenario_id"] == "retrieve"
    assert summary["per_scenario"]["retrieve"]["covered"] == 1
    assert summary["per_scenario"]["retrieve"]["required_coverage_demonstrated"] is True


# ---------------------------------------------------------------------------
# False-complete regression (gate + completion composition)
# ---------------------------------------------------------------------------


def test_false_complete_full_cohort_p0_stays_incomplete() -> None:
    """Full 18 complete+judge+safety with only non-retrieve covered → P0 incomplete.

    Prospective full replace:
    - test_scenario_pilot::test_false_complete_regression_other_scenario_coverage_only

    Composition only: not run_scenario_pilot end-to-end.
    """
    slots = _live6_false_complete_slots()
    cov = build_coverage_gate_summary(slots)
    comp = summarize_completion(slots, cov)

    assert cov["required_coverage_demonstrated"] is False
    assert cov["gate_scenario_id"] == "retrieve"
    assert cov["per_scenario"]["retrieve"]["covered"] == 0
    assert cov["per_scenario"]["insufficient"]["covered"] == 1
    assert comp["slots_complete"] == SLOT_COUNT
    assert comp["valid_judge_assessments"] == SLOT_COUNT
    assert comp["safety_determined"] == SLOT_COUNT
    assert comp["required_retrieval_coverage_demonstrated"] is False
    assert comp["p0_measurement_complete"] is False


# ---------------------------------------------------------------------------
# Saved-report boundary
# ---------------------------------------------------------------------------


def test_reevaluate_saved_report_reapplies_retrieve_gate(tmp_path: Path) -> None:
    """Offline reassess reapplies retrieve gate; original report stays untouched.

    Prospective full replace:
    - test_scenario_pilot::test_reevaluate_saved_report_retrieve_gate
    """
    slots = _live6_false_complete_slots()
    src = tmp_path / "scenario_pilot_fake6.json"
    src.write_text(
        json.dumps(
            {
                "cohort_id": "fake6",
                "status": "complete",
                "live": True,
                "completion": {"p0_measurement_complete": True},
                "coverage_summary": {"required_coverage_demonstrated": True},
                "slots": _slots_to_report_rows(slots),
                "gaps": [],
                "blockers": [],
                "judge": "preserved",
            }
        ),
        encoding="utf-8",
    )

    derived, out = reevaluate_saved_cohort_report(src, report_dir=tmp_path)

    assert out != src and out.is_file()
    assert derived["status"] == "incomplete"
    assert derived["completion"]["p0_measurement_complete"] is False
    assert derived["completion"]["quality_baseline_valid"] is False
    assert derived["completion"]["safety_baseline_valid"] is False
    assert derived["coverage_summary"]["gate_scenario_id"] == "retrieve"
    assert derived["coverage_summary"]["required_coverage_demonstrated"] is False
    assert derived["reevaluation"]["rejudged"] is False
    assert derived["reevaluation"]["is_baseline"] is False
    assert derived["reevaluation"]["replaces_original"] is False
    assert derived["reevaluation"]["measurement_evidence_mismatch"]["declared"] is True
    assert derived["reevaluation"]["quality_reassess"] == "historical_not_validated_no_rejudge"
    assert derived["reevaluation"]["safety_reassess"] == (
        "legacy_contested_unknown_not_reconstructable"
    )
    assert derived["reevaluation"]["source_sha256"]
    assert derived["slots"][0]["judge"]["baseline_validity"] == "historical_not_validated"
    assert derived["slots"][0]["safety"]["baseline_validity"] == "legacy_contested_unknown"
    assert derived["slots"][0]["safety"]["validated"] is False

    orig = json.loads(src.read_text())
    assert orig["status"] == "complete"
    assert orig["completion"]["p0_measurement_complete"] is True
    assert "baseline_validity" not in (orig["slots"][0].get("judge") or {})
