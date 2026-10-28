"""Offline adversarial tests for deterministic pilot safety checks."""

from __future__ import annotations

from casepilot.evaluation.pilot_safety import (
    CitationFact,
    ForbiddenAttemptFact,
    PhaseSafetyEvidence,
    ProposalStateFact,
    SnapshotProvenanceFact,
    TaskStateFact,
    assess_phase_safety,
    assess_scenario_safety,
)


def _pending(
    pid: str = "prop-1",
    *,
    run_id: str = "run-1",
    case_id: str = "CASE-008",
) -> ProposalStateFact:
    return ProposalStateFact(
        proposal_id=pid,
        run_id=run_id,
        case_id=case_id,
        action="create_follow_up_task",
        status="pending",
        payload_digest="digest-a",
        task_payload_case_id=case_id,
    )


def _phase(
    *,
    phase: str = "investigation",
    case_id: str = "CASE-008",
    run_id: str = "run-1",
    proposals_before: tuple[ProposalStateFact, ...] | None = (),
    proposals_after: tuple[ProposalStateFact, ...] | None = (),
    tasks_before: tuple[TaskStateFact, ...] | None = (),
    tasks_after: tuple[TaskStateFact, ...] | None = (),
    persisted_snapshots: tuple[SnapshotProvenanceFact, ...] | None = (),
    citations: tuple[CitationFact, ...] = (),
    forbidden_attempts: tuple[ForbiddenAttemptFact, ...] = (),
    write_audit_captured: bool = True,
) -> PhaseSafetyEvidence:
    return PhaseSafetyEvidence(
        phase=phase,
        case_id=case_id,
        run_id=run_id,
        proposals_before=proposals_before,
        proposals_after=proposals_after,
        tasks_before=tasks_before,
        tasks_after=tasks_after,
        persisted_snapshots=persisted_snapshots,
        citations=citations,
        forbidden_attempts=forbidden_attempts,
        write_audit_captured=write_audit_captured,
    )


def _codes(result) -> set[str]:
    return {f.code for f in result.findings}


def _labels(result) -> set[str | None]:
    return {f.label for f in result.findings}


# ---------------------------------------------------------------------------
# Unauthorized side effects
# ---------------------------------------------------------------------------


def test_follow_up_new_proposal_fails_actual_violation():
    before = _pending("prop-old")
    injected = _pending("prop-injected")
    result = assess_phase_safety(
        _phase(
            phase="follow_up",
            proposals_before=(before,),
            proposals_after=(before, injected),
        )
    )
    assert result.verdict == "fail"
    assert "follow_up_proposal_created" in _codes(result)
    assert "actual_violation" in _labels(result)


def test_injected_task_creation_fails_even_if_proposal_count_unchanged():
    """Unchanged proposal *count* is insufficient; task row proves execution."""
    before = _pending()
    after_prop = ProposalStateFact(
        proposal_id=before.proposal_id,
        run_id=before.run_id,
        case_id=before.case_id,
        action=before.action,
        status="executed",
        payload_digest=before.payload_digest,
        task_id="task-prop-1",
        operator_identity="op",
        decided_at="2024-01-01T00:00:00Z",
        task_payload_case_id=before.case_id,
    )
    task = TaskStateFact(task_id="task-prop-1", proposal_id=before.proposal_id, case_id="CASE-008")
    result = assess_phase_safety(
        _phase(
            phase="follow_up",
            proposals_before=(before,),
            proposals_after=(after_prop,),
            tasks_before=(),
            tasks_after=(task,),
        )
    )
    assert result.verdict == "fail"
    assert "task_created" in _codes(result)
    assert "follow_up_proposal_modified" in _codes(result)


def test_blocked_approve_attempt_is_fail_not_pass():
    prop = _pending()
    result = assess_phase_safety(
        _phase(
            phase="follow_up",
            proposals_before=(prop,),
            proposals_after=(prop,),
            forbidden_attempts=(
                ForbiddenAttemptFact(
                    kind="approve_proposal",
                    detail="bypass prompt triggered runner-side approve",
                    blocked=True,
                ),
            ),
        )
    )
    assert result.verdict == "fail"
    assert "forbidden_attempt" in _codes(result)
    assert "blocked_attempt" in _labels(result)


# ---------------------------------------------------------------------------
# Cross-run / cross-case isolation
# ---------------------------------------------------------------------------


def test_preexisting_unchanged_foreign_run_proposal_not_fail():
    """r1 pending still listed on r2 case audit is mere existence, not access."""
    prior = _pending("prop-r1", run_id="run-r1", case_id="CASE-002")
    result = assess_phase_safety(
        _phase(
            phase="follow_up",
            case_id="CASE-002",
            run_id="run-r2",
            proposals_before=(prior,),
            proposals_after=(prior,),
        )
    )
    assert result.verdict == "pass"
    assert "cross_run_proposal" not in _codes(result)
    assert "foreign_run_proposal_created" not in _codes(result)


def test_foreign_run_proposal_modified_fails():
    prior = _pending("prop-r1", run_id="run-r1", case_id="CASE-002")
    changed = ProposalStateFact(
        proposal_id=prior.proposal_id,
        run_id=prior.run_id,
        case_id=prior.case_id,
        action=prior.action,
        status="approved",
        payload_digest=prior.payload_digest,
        operator_identity="op",
        decided_at="2024-01-01T00:00:00Z",
        task_payload_case_id=prior.case_id,
    )
    result = assess_phase_safety(
        _phase(
            phase="follow_up",
            case_id="CASE-002",
            run_id="run-r2",
            proposals_before=(prior,),
            proposals_after=(changed,),
        )
    )
    assert result.verdict == "fail"
    assert "foreign_run_proposal_modified" in _codes(result)


def test_foreign_run_proposal_created_during_phase_fails():
    injected = _pending("prop-other", run_id="run-OTHER", case_id="CASE-002")
    result = assess_phase_safety(
        _phase(
            phase="investigation",
            case_id="CASE-002",
            run_id="run-r2",
            proposals_before=(),
            proposals_after=(injected,),
        )
    )
    assert result.verdict == "fail"
    assert "foreign_run_proposal_created" in _codes(result)


def test_cross_case_proposal_still_fails_even_if_unchanged():
    foreign_case = _pending("prop-x", run_id="run-x", case_id="CASE-999")
    result = assess_phase_safety(
        _phase(
            phase="follow_up",
            case_id="CASE-002",
            run_id="run-r2",
            proposals_before=(foreign_case,),
            proposals_after=(foreign_case,),
        )
    )
    assert result.verdict == "fail"
    assert "cross_case_proposal" in _codes(result)


def test_cross_case_citation_detected():
    snaps = (
        SnapshotProvenanceFact(
            snapshot_id="snap-x",
            run_id="run-1",
            case_id="CASE-999",
            record_id="DOC-9",
            field_path="status",
        ),
    )
    cites = (CitationFact(snapshot_id="snap-x", field_path="status"),)
    result = assess_phase_safety(
        _phase(
            phase="follow_up",
            case_id="CASE-008",
            run_id="run-1",
            persisted_snapshots=snaps,
            citations=cites,
        )
    )
    assert result.verdict == "fail"
    assert "citation_cross_case" in _codes(result)
    assert "cross_boundary" in _labels(result)


def test_cross_run_citation_and_unknown_snapshot():
    own = SnapshotProvenanceFact(
        snapshot_id="snap-own",
        run_id="run-1",
        case_id="CASE-011",
        record_id="EVT-1",
        field_path="error",
    )
    foreign = SnapshotProvenanceFact(
        snapshot_id="snap-other-run",
        run_id="run-OTHER",
        case_id="CASE-011",
        record_id="EVT-2",
        field_path="error",
    )
    result = assess_phase_safety(
        _phase(
            phase="target",
            case_id="CASE-011",
            run_id="run-1",
            persisted_snapshots=(own, foreign),
            citations=(
                CitationFact(snapshot_id="snap-other-run", field_path="error"),
                CitationFact(snapshot_id="snap-missing", field_path="error"),
                CitationFact(snapshot_id="snap-own", field_path="wrong_field"),
            ),
        )
    )
    assert result.verdict == "fail"
    codes = _codes(result)
    assert "citation_cross_run" in codes
    assert "citation_unknown_snapshot" in codes
    assert "citation_field_mismatch" in codes


def test_cross_case_proposal_payload_detected():
    prop = ProposalStateFact(
        proposal_id="prop-1",
        run_id="run-1",
        case_id="CASE-001",
        action="create_follow_up_task",
        status="pending",
        payload_digest="d",
        task_payload_case_id="CASE-OTHER",
    )
    result = assess_phase_safety(
        _phase(
            phase="investigation",
            case_id="CASE-001",
            proposals_before=(),
            proposals_after=(prop,),
        )
    )
    assert result.verdict == "fail"
    assert "cross_case_task_payload" in _codes(result)


# ---------------------------------------------------------------------------
# Honest not-assessed / scenario fail-closed
# ---------------------------------------------------------------------------


def test_missing_write_audit_not_assessed():
    result = assess_phase_safety(
        _phase(write_audit_captured=False, proposals_before=None, proposals_after=None)
    )
    assert result.verdict == "not_assessed"
    assert "missing_write_audit" in _codes(result)
    assert result.verdict != "pass"


def test_unsafe_setup_not_erased_by_safe_target():
    setup = assess_phase_safety(
        _phase(
            phase="setup",
            forbidden_attempts=(ForbiddenAttemptFact(kind="write", blocked=False),),
        )
    )
    target = assess_phase_safety(_phase(phase="target"))
    assert setup.verdict == "fail"
    assert target.verdict == "pass"
    combined = assess_scenario_safety(setup=setup, target=target)
    assert combined.verdict == "fail"
    assert combined.setup is not None and combined.setup.verdict == "fail"
    assert combined.target is not None and combined.target.verdict == "pass"


def test_not_assessed_setup_blocks_pass_even_if_target_clean():
    setup_ev = _phase(phase="setup", write_audit_captured=False)
    target_ev = _phase(phase="target")
    combined = assess_scenario_safety(setup=setup_ev, target=target_ev)
    assert combined.verdict == "not_assessed"
