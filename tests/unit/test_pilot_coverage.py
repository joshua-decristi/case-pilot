"""Offline unit tests for follow-up retrieval coverage (pure functions)."""

from __future__ import annotations

from casepilot.evaluation.pilot_coverage import (
    AnswerCitation,
    AttemptSnapshot,
    CoverageStatus,
    ExposedFact,
    RetrievalClass,
    TargetFact,
    assess_followup_coverage,
    assess_required_coverage,
    classify_retrieval,
)

TARGET = TargetFact(record_id="EVT-99", field_path="details.error_code")


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
# Classification honesty
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Grounding / false-coverage chain
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
    snap = _snap("snap-redun", "DOC-1", "fields.amount")
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[snap],
        answering_model_exposed=[_exposed("DOC-1", "fields.amount", snapshot_id="snap-redun")],
        answer_citations=[_cite("snap-redun", "DOC-1", "fields.amount")],
    )
    assert a.status is CoverageStatus.NOT_COVERED
    assert a.relevant_new_snapshot_ids == ()


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


def test_snapshot_in_store_but_not_exposed_not_covered():
    snap = _snap("snap-new", "EVT-99", "details.error_code")
    a = assess_required_coverage(
        absent_targets=[("EVT-99", "details.error_code")],
        attempt_id="att-1",
        attempt_follow_up_snapshots=[snap],
        answering_model_exposed=[],
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
        answer_citations=[],
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
