"""Offline unit tests for follow-up retrieval coverage (pure functions)."""

from __future__ import annotations

from casepilot.evaluation.pilot_coverage import (
    AnswerCitation,
    AttemptSnapshot,
    CoverageStatus,
    ExposedFact,
    RepetitionCoverage,
    RetrievalClass,
    TargetFact,
    assess_followup_coverage,
    assess_required_coverage,
    classify_retrieval,
    fact_key,
    summarize_cohort_coverage,
)

TARGET = TargetFact(record_id="EVT-99", field_path="details.error_code")
OTHER = TargetFact(record_id="DOC-1", field_path="fields.amount")


def _exposed(record_id: str, field_path: str, *, snapshot_id: str = "", **kw) -> ExposedFact:
    return ExposedFact(
        record_id=record_id,
        field_path=field_path,
        snapshot_id=snapshot_id,
        **kw,
    )


def _snap(
    snapshot_id: str,
    record_id: str,
    field_path: str,
    *,
    attempt_id: str = "att-1",
    provenance: str = "follow_up",
    retrieval_ref: str = "ref/1",
) -> AttemptSnapshot:
    return AttemptSnapshot(
        snapshot_id=snapshot_id,
        record_id=record_id,
        field_path=field_path,
        provenance=provenance,
        attempt_id=attempt_id,
        retrieval_ref=retrieval_ref,
    )


def _cite(snapshot_id: str, record_id: str, field_path: str) -> AnswerCitation:
    return AnswerCitation(
        snapshot_id=snapshot_id,
        record_id=record_id,
        field_path=field_path,
    )


# ---------------------------------------------------------------------------
# Classification from actual pre-turn model context
# ---------------------------------------------------------------------------


def test_fact_key_strips():
    assert fact_key("  R1 ", " a.b ") == ("R1", "a.b")


def test_missing_pre_turn_context_not_assessed():
    c = classify_retrieval([TARGET], None)
    assert c.retrieval_class is RetrievalClass.NOT_ASSESSED
    assert c.present_targets == ()
    assert c.absent_targets == ()
    assert any("pre_turn_model_context missing" in n for n in c.notes)


def test_empty_pre_turn_with_targets_is_required():
    c = classify_retrieval([TARGET], [])
    assert c.retrieval_class is RetrievalClass.REQUIRED
    assert c.absent_targets == (("EVT-99", "details.error_code"),)
    assert c.present_targets == ()


def test_all_targets_present_is_optional():
    ctx = [_exposed("EVT-99", "details.error_code", snapshot_id="s-old")]
    c = classify_retrieval([TARGET], ctx)
    assert c.retrieval_class is RetrievalClass.OPTIONAL
    assert c.absent_targets == ()
    assert c.present_targets == (("EVT-99", "details.error_code"),)


def test_partial_targets_absent_is_required():
    ctx = [_exposed("DOC-1", "fields.amount")]
    c = classify_retrieval([TARGET, OTHER], ctx)
    assert c.retrieval_class is RetrievalClass.REQUIRED
    assert ("EVT-99", "details.error_code") in c.absent_targets
    assert ("DOC-1", "fields.amount") in c.present_targets


def test_no_targets_is_optional():
    c = classify_retrieval([], [_exposed("X", "y")])
    assert c.retrieval_class is RetrievalClass.OPTIONAL


def test_classification_ignores_answer_alone_contract():
    """Pre-turn context is the sole availability signal — not answer text."""
    # Even with rich "answer-like" material elsewhere, empty context → required.
    c = classify_retrieval([TARGET], pre_turn_model_context=[])
    assert c.retrieval_class is RetrievalClass.REQUIRED


# ---------------------------------------------------------------------------
# Required coverage proof chain
# ---------------------------------------------------------------------------


def test_required_coverage_full_chain_covered():
    snap = _snap("snap-new", "EVT-99", "details.error_code")
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[snap],
        answering_model_exposed=[
            _exposed("EVT-99", "details.error_code", snapshot_id="snap-new", provenance="follow_up")
        ],
        answer_citations=[_cite("snap-new", "EVT-99", "details.error_code")],
    )
    assert a.status is CoverageStatus.COVERED
    assert a.retrieval_class is RetrievalClass.REQUIRED
    assert a.relevant_new_snapshot_ids == ("snap-new",)
    assert a.exposed_new_snapshot_ids == ("snap-new",)
    assert a.grounded_new_snapshot_ids == ("snap-new",)


def test_redundant_fetch_of_present_target_does_not_cover():
    """Fetch matching a non-absent target cannot satisfy required coverage."""
    snap = _snap("snap-redun", "DOC-1", "fields.amount")  # already present target
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[snap],
        answering_model_exposed=[_exposed("DOC-1", "fields.amount", snapshot_id="snap-redun")],
        answer_citations=[_cite("snap-redun", "DOC-1", "fields.amount")],
    )
    assert a.status is CoverageStatus.NOT_COVERED
    assert a.relevant_new_snapshot_ids == ()


def test_wrong_attempt_id_snapshot_ignored():
    snap = _snap("snap-new", "EVT-99", "details.error_code", attempt_id="other-att")
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[snap],
        answering_model_exposed=[_exposed("EVT-99", "details.error_code", snapshot_id="snap-new")],
        answer_citations=[_cite("snap-new", "EVT-99", "details.error_code")],
    )
    assert a.status is CoverageStatus.NOT_COVERED


def test_original_run_provenance_does_not_count_as_new():
    snap = _snap(
        "snap-old",
        "EVT-99",
        "details.error_code",
        provenance="original_run",
    )
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[snap],
        answering_model_exposed=[_exposed("EVT-99", "details.error_code", snapshot_id="snap-old")],
        answer_citations=[_cite("snap-old", "EVT-99", "details.error_code")],
    )
    assert a.status is CoverageStatus.NOT_COVERED


def test_empty_retrieval_ref_not_observable():
    snap = _snap(
        "snap-new",
        "EVT-99",
        "details.error_code",
        retrieval_ref="",
    )
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[snap],
        answering_model_exposed=[_exposed("EVT-99", "details.error_code", snapshot_id="snap-new")],
        answer_citations=[_cite("snap-new", "EVT-99", "details.error_code")],
    )
    assert a.status is CoverageStatus.NOT_COVERED


def test_snapshot_in_store_but_not_exposed_not_covered():
    snap = _snap("snap-new", "EVT-99", "details.error_code")
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[snap],
        answering_model_exposed=[],  # known empty exposure
        answer_citations=[_cite("snap-new", "EVT-99", "details.error_code")],
    )
    assert a.status is CoverageStatus.NOT_COVERED
    assert a.relevant_new_snapshot_ids == ("snap-new",)
    assert a.exposed_new_snapshot_ids == ()
    assert any("did not reach answering model" in n for n in a.notes)


def test_exposed_but_uncited_not_covered():
    snap = _snap("snap-new", "EVT-99", "details.error_code")
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[snap],
        answering_model_exposed=[_exposed("EVT-99", "details.error_code", snapshot_id="snap-new")],
        answer_citations=[],  # known empty citations
    )
    assert a.status is CoverageStatus.NOT_COVERED
    assert a.exposed_new_snapshot_ids == ("snap-new",)
    assert a.grounded_new_snapshot_ids == ()


def test_citation_field_mismatch_not_grounded():
    snap = _snap("snap-new", "EVT-99", "details.error_code")
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[snap],
        answering_model_exposed=[_exposed("EVT-99", "details.error_code", snapshot_id="snap-new")],
        answer_citations=[_cite("snap-new", "EVT-99", "details.other")],
    )
    assert a.status is CoverageStatus.NOT_COVERED
    assert a.grounded_new_snapshot_ids == ()


def test_citation_of_unexposed_snapshot_not_grounded():
    """Cite a relevant snap that never reached the model → not covered."""
    snap_a = _snap("snap-a", "EVT-99", "details.error_code")
    snap_b = _snap("snap-b", "EVT-99", "details.error_code")
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[snap_a, snap_b],
        answering_model_exposed=[_exposed("EVT-99", "details.error_code", snapshot_id="snap-a")],
        # cites only the unexposed one
        answer_citations=[_cite("snap-b", "EVT-99", "details.error_code")],
    )
    assert a.status is CoverageStatus.NOT_COVERED
    assert a.exposed_new_snapshot_ids == ("snap-a",)
    assert a.grounded_new_snapshot_ids == ()


# ---------------------------------------------------------------------------
# Missing instrumentation → not assessed (never covered)
# ---------------------------------------------------------------------------


def test_missing_snapshots_not_assessed():
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=None,
        answering_model_exposed=[],
        answer_citations=[],
    )
    assert a.status is CoverageStatus.NOT_ASSESSED


def test_missing_model_exposure_not_assessed():
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[_snap("s", "EVT-99", "details.error_code")],
        answering_model_exposed=None,
        answer_citations=[],
    )
    assert a.status is CoverageStatus.NOT_ASSESSED
    assert any("answering_model_exposed missing" in n for n in a.notes)


def test_missing_citations_capture_not_assessed():
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[_snap("s", "EVT-99", "details.error_code")],
        answering_model_exposed=[_exposed("EVT-99", "details.error_code", snapshot_id="s")],
        answer_citations=None,
    )
    assert a.status is CoverageStatus.NOT_ASSESSED


def test_missing_attempt_id_not_assessed():
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="",
        attempt_follow_up_snapshots=[],
        answering_model_exposed=[],
        answer_citations=[],
    )
    assert a.status is CoverageStatus.NOT_ASSESSED


def test_no_absent_targets_not_eligible():
    a = assess_required_coverage(
        absent_targets=[],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[],
        answering_model_exposed=[],
        answer_citations=[],
    )
    assert a.status is CoverageStatus.NOT_ELIGIBLE


# ---------------------------------------------------------------------------
# End-to-end assess_followup_coverage
# ---------------------------------------------------------------------------


def test_e2e_optional_not_eligible_for_required_count():
    a = assess_followup_coverage(
        target_facts=[TARGET],
        pre_turn_model_context=[_exposed("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[],
        answering_model_exposed=[],
        answer_citations=[],
    )
    assert a.retrieval_class is RetrievalClass.OPTIONAL
    assert a.status is CoverageStatus.NOT_ELIGIBLE


def test_e2e_missing_pre_turn_not_assessed_even_with_post_proof():
    """Post-turn success cannot backfill missing pre-turn classification proof."""
    snap = _snap("snap-new", "EVT-99", "details.error_code")
    a = assess_followup_coverage(
        target_facts=[TARGET],
        pre_turn_model_context=None,
        attempt_id="att-1",
        attempt_follow_up_snapshots=[snap],
        answering_model_exposed=[_exposed("EVT-99", "details.error_code", snapshot_id="snap-new")],
        answer_citations=[_cite("snap-new", "EVT-99", "details.error_code")],
    )
    assert a.status is CoverageStatus.NOT_ASSESSED
    assert a.retrieval_class is RetrievalClass.NOT_ASSESSED


def test_e2e_required_and_covered():
    a = assess_followup_coverage(
        target_facts=[TARGET],
        pre_turn_model_context=[],  # absent
        attempt_id="att-1",
        attempt_follow_up_snapshots=[_snap("snap-new", "EVT-99", "details.error_code")],
        answering_model_exposed=[_exposed("EVT-99", "details.error_code", snapshot_id="snap-new")],
        answer_citations=[_cite("snap-new", "EVT-99", "details.error_code")],
    )
    assert a.status is CoverageStatus.COVERED


def test_exposure_by_fact_key_without_snapshot_id_still_counts():
    """If exposure omits snapshot_id, matching record_id+field_path still proves exposure."""
    snap = _snap("snap-new", "EVT-99", "details.error_code")
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[snap],
        answering_model_exposed=[_exposed("EVT-99", "details.error_code")],  # no snapshot_id
        answer_citations=[_cite("snap-new", "EVT-99", "details.error_code")],
    )
    assert a.status is CoverageStatus.COVERED
    assert a.exposed_new_snapshot_ids == ("snap-new",)


# ---------------------------------------------------------------------------
# Cohort aggregation
# ---------------------------------------------------------------------------


def _covered_assessment():
    return assess_followup_coverage(
        target_facts=[TARGET],
        pre_turn_model_context=[],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[_snap("s1", "EVT-99", "details.error_code")],
        answering_model_exposed=[_exposed("EVT-99", "details.error_code", snapshot_id="s1")],
        answer_citations=[_cite("s1", "EVT-99", "details.error_code")],
    )


def _not_covered_assessment(*, attempt_id: str = "att-2"):
    return assess_followup_coverage(
        target_facts=[TARGET],
        pre_turn_model_context=[],
        attempt_id=attempt_id,
        attempt_follow_up_snapshots=[
            _snap("s2", "EVT-99", "details.error_code", attempt_id=attempt_id)
        ],
        answering_model_exposed=[],
        answer_citations=[],
    )


def _optional_assessment():
    return assess_followup_coverage(
        target_facts=[TARGET],
        pre_turn_model_context=[_exposed("EVT-99", "details.error_code")],
    )


def test_cohort_requires_at_least_one_covered():
    reps = [
        RepetitionCoverage("retrieve", 1, _optional_assessment()),
        RepetitionCoverage("retrieve", 2, _not_covered_assessment()),
    ]
    summary = summarize_cohort_coverage(reps, mandated_scenario_ids=("retrieve",))
    assert summary.n == 2
    assert summary.optional == 1
    assert summary.required == 1
    assert summary.covered == 0
    assert summary.not_covered == 1
    assert summary.not_eligible == 1
    assert summary.required_coverage_demonstrated is False
    assert summary.by_scenario["retrieve"].covered == 0
    assert any("P0 retrieval-coverage incomplete" in n for n in summary.notes)


def test_cohort_demonstrated_when_mandated_scenario_has_one_covered():
    summary = summarize_cohort_coverage(
        [
            RepetitionCoverage("retrieve", 1, _covered_assessment()),
            RepetitionCoverage("retrieve", 2, _optional_assessment()),
            RepetitionCoverage("retrieve", 3, _optional_assessment()),
        ],
        mandated_scenario_ids=("retrieve",),
    )
    assert summary.covered == 1
    assert summary.required_coverage_demonstrated is True
    assert summary.mandated_scenario_ids == ("retrieve",)
    assert summary.by_scenario["retrieve"].covered == 1
    assert summary.by_scenario["retrieve"].required_coverage_demonstrated is True
    assert summary.notes == []
    d = summary.to_dict()
    assert d["required_coverage_demonstrated"] is True
    assert d["by_scenario"]["retrieve"]["covered"] == 1
    assert d["repetitions"][0]["assessment"]["status"] == "covered"


def test_cohort_without_mandated_ids_never_defaults_gate_true():
    """Back-compat call site: global covered must not open the P0 gate."""
    summary = summarize_cohort_coverage([RepetitionCoverage("retrieve", 1, _covered_assessment())])
    assert summary.covered == 1
    assert summary.required_coverage_demonstrated is False
    assert summary.mandated_scenario_ids == ()
    assert any("mandated_scenario_ids not provided" in n for n in summary.notes)


def test_incidental_insufficient_covered_does_not_satisfy_retrieve_gate():
    """Live cohort6 false-complete: insufficient r2 covered while retrieve 0/3."""
    reps = [
        RepetitionCoverage("retrieve", 1, _not_covered_assessment(attempt_id="r1")),
        RepetitionCoverage("retrieve", 2, _not_covered_assessment(attempt_id="r2")),
        RepetitionCoverage("retrieve", 3, _not_covered_assessment(attempt_id="r3")),
        RepetitionCoverage("insufficient", 1, _optional_assessment()),
        RepetitionCoverage("insufficient", 2, _covered_assessment()),
        RepetitionCoverage("insufficient", 3, _optional_assessment()),
    ]
    summary = summarize_cohort_coverage(reps, mandated_scenario_ids=("retrieve",))
    assert summary.covered == 1  # global descriptive only
    assert summary.by_scenario["retrieve"].covered == 0
    assert summary.by_scenario["retrieve"].n == 3
    assert summary.by_scenario["insufficient"].covered == 1
    assert summary.required_coverage_demonstrated is False
    assert any("retrieve" in n and "0/3 covered" in n for n in summary.notes)


def test_retrieve_zero_of_three_incomplete_even_if_only_retrieve_in_cohort():
    reps = [
        RepetitionCoverage("retrieve", i, _not_covered_assessment(attempt_id=f"a{i}"))
        for i in (1, 2, 3)
    ]
    summary = summarize_cohort_coverage(reps, mandated_scenario_ids=("retrieve",))
    assert summary.by_scenario["retrieve"].covered == 0
    assert summary.required_coverage_demonstrated is False


def test_to_dict_serializable_on_assessment():
    a = assess_followup_coverage(target_facts=[TARGET], pre_turn_model_context=None)
    d = a.to_dict()
    assert d["status"] == "not_assessed"
    assert d["retrieval_class"] == "not_assessed"
