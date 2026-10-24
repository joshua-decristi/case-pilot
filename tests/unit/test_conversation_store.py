"""Focused unit tests for conversation persistence and proposal task_id."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from casepilot.backend.db import (
    BackendStore,
    ConversationConflict,
    ConversationIdempotencyError,
    ConversationNotEligible,
    ConversationRetryError,
)
from casepilot.shared.contracts import (
    BackendRoutes,
    EvidenceProvenance,
    FollowUpAnswer,
    FollowUpTaskPayload,
    InterpretationBlock,
    InterpretationKind,
    NextAction,
    ProposalStatus,
    RenderedSourceFact,
    RunStatus,
    SendMessageRequest,
    TurnStatus,
)


def _store(tmp_path: Path) -> BackendStore:
    return BackendStore(tmp_path / "backend.sqlite3")


def _seed_run(
    store: BackendStore,
    run_id: str = "run-1",
    *,
    status: RunStatus = RunStatus.COMPLETED,
) -> None:
    store.insert_run(
        run_id=run_id,
        case_id="CASE-001",
        objective="why stalled",
        status=status,
        completed_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _answer() -> FollowUpAnswer:
    return FollowUpAnswer(
        source_facts=[
            RenderedSourceFact(
                local_id="f1",
                snapshot_id="snap-1",
                source_system="documents",
                record_id="DOC-1",
                field_path="status",
                value="missing",
                retrieval_ref="get_documents",
                retrieved_at=datetime(2024, 1, 2, tzinfo=UTC),
                provenance=EvidenceProvenance.FOLLOW_UP,
            )
        ],
        interpretations=[
            InterpretationBlock(
                text="Document still missing.",
                cites=["f1"],
                kind=InterpretationKind.INFERENCE,
            )
        ],
        notes=["Happy to dig further."],
    )


def test_routes_templates() -> None:
    assert BackendRoutes.CONVERSATION.format(run_id="r1") == "/v1/investigations/r1/conversation"
    assert (
        BackendRoutes.CONVERSATION_MESSAGES.format(run_id="r1")
        == "/v1/investigations/r1/conversation/messages"
    )
    assert (
        BackendRoutes.CONVERSATION_RETRY.format(run_id="r1", turn_id="t1")
        == "/v1/investigations/r1/conversation/turns/t1/retry"
    )


@pytest.mark.parametrize("question", ["", "   ", "\t\n"])
def test_send_message_rejects_blank_question(question: str) -> None:
    with pytest.raises(ValidationError):
        SendMessageRequest(question=question, idempotency_key="k1")


def test_send_message_strips_and_keeps_max_length() -> None:
    req = SendMessageRequest(question="  Why stalled?  ", idempotency_key="k1")
    assert req.question == "Why stalled?"
    ok = SendMessageRequest(question="x" * 4000, idempotency_key="k1")
    assert len(ok.question) == 4000
    with pytest.raises(ValidationError):
        SendMessageRequest(question="x" * 4001, idempotency_key="k1")


def test_begin_send_complete_and_history(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_run(store)

    first = store.begin_send(run_id="run-1", question="Why stalled?", idempotency_key="k1")
    assert first.created is True
    assert first.turn.status is TurnStatus.PENDING
    assert first.attempt.status is TurnStatus.PENDING
    assert first.turn.question == "Why stalled?"
    assert len(first.turn.attempts) == 1

    replay = store.begin_send(run_id="run-1", question="Why stalled?", idempotency_key="k1")
    assert replay.created is False
    assert replay.attempt.attempt_id == first.attempt.attempt_id
    assert replay.turn.turn_id == first.turn.turn_id

    done = store.complete_attempt(first.attempt.attempt_id, answer=_answer())
    assert done is not None
    assert done.status is TurnStatus.SUCCEEDED
    assert done.attempts[-1].answer is not None
    assert done.attempts[-1].answer.notes == ["Happy to dig further."]

    hist = store.get_conversation("run-1")
    assert len(hist) == 1
    assert hist[0].status is TurnStatus.SUCCEEDED
    store.close()


def test_send_idempotency_mismatch_and_concurrency(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_run(store)
    store.begin_send(run_id="run-1", question="Q1", idempotency_key="k1")

    with pytest.raises(ConversationIdempotencyError):
        store.begin_send(run_id="run-1", question="Q2", idempotency_key="k1")

    with pytest.raises(ConversationConflict):
        store.begin_send(run_id="run-1", question="other", idempotency_key="k2")
    store.close()


def test_eligibility_rejects_failed_and_missing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_run(store, "run-fail", status=RunStatus.FAILED)
    with pytest.raises(ConversationNotEligible, match="not chat-eligible"):
        store.begin_send(run_id="run-fail", question="q", idempotency_key="k")
    with pytest.raises(ConversationNotEligible, match="not found"):
        store.begin_send(run_id="missing", question="q", idempotency_key="k2")

    _seed_run(store, "run-ind", status=RunStatus.INDETERMINATE)
    ok = store.begin_send(run_id="run-ind", question="q", idempotency_key="k3")
    assert ok.created is True
    store.close()


def test_fail_retry_replay_and_new_key(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_run(store)
    started = store.begin_send(run_id="run-1", question="Q", idempotency_key="send-1")
    failed = store.fail_attempt(started.attempt.attempt_id, error="model timeout")
    assert failed is not None
    assert failed.status is TurnStatus.FAILED
    assert failed.attempts[0].error == "model timeout"
    prev_id = started.attempt.attempt_id

    retry = store.begin_retry(
        run_id="run-1",
        turn_id=started.turn.turn_id,
        idempotency_key="retry-1",
        expected_previous_attempt_id=prev_id,
    )
    assert retry.created is True
    assert retry.turn.status is TurnStatus.PENDING
    assert retry.attempt.previous_attempt_id == prev_id
    assert retry.turn.question == "Q"
    assert len(retry.turn.attempts) == 2

    # Replay same retry key after failure still returns that attempt (no new call).
    store.fail_attempt(retry.attempt.attempt_id, error="again")
    replay = store.begin_retry(
        run_id="run-1",
        turn_id=started.turn.turn_id,
        idempotency_key="retry-1",
        expected_previous_attempt_id=prev_id,
    )
    assert replay.created is False
    assert replay.attempt.attempt_id == retry.attempt.attempt_id
    assert replay.attempt.status is TurnStatus.FAILED

    # Only a fresh key starts another attempt.
    again = store.begin_retry(
        run_id="run-1",
        turn_id=started.turn.turn_id,
        idempotency_key="retry-2",
        expected_previous_attempt_id=retry.attempt.attempt_id,
    )
    assert again.created is True
    assert again.attempt.attempt_id != retry.attempt.attempt_id
    store.close()


def test_retry_rejects_wrong_prev_and_key_reuse(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_run(store)
    a = store.begin_send(run_id="run-1", question="Q", idempotency_key="s1")
    store.fail_attempt(a.attempt.attempt_id, error="x")

    with pytest.raises(ConversationRetryError):
        store.begin_retry(
            run_id="run-1",
            turn_id=a.turn.turn_id,
            idempotency_key="r1",
            expected_previous_attempt_id="att-nope",
        )

    store.begin_retry(
        run_id="run-1",
        turn_id=a.turn.turn_id,
        idempotency_key="r1",
        expected_previous_attempt_id=a.attempt.attempt_id,
    )
    with pytest.raises(ConversationIdempotencyError):
        store.begin_retry(
            run_id="run-1",
            turn_id=a.turn.turn_id,
            idempotency_key="r1",
            expected_previous_attempt_id="att-other",
        )
    # send key cannot be reused as retry target either
    with pytest.raises(ConversationIdempotencyError):
        store.begin_retry(
            run_id="run-1",
            turn_id=a.turn.turn_id,
            idempotency_key="s1",
            expected_previous_attempt_id=a.attempt.attempt_id,
        )
    store.close()


def test_recover_interrupted_pending(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_run(store)
    started = store.begin_send(run_id="run-1", question="Q", idempotency_key="k")
    n = store.recover_interrupted_attempts()
    assert n == 1
    turn = store.get_turn(started.turn.turn_id)
    assert turn is not None
    assert turn.status is TurnStatus.FAILED
    assert turn.attempts[0].status is TurnStatus.FAILED
    assert "interrupted" in (turn.attempts[0].error or "")
    assert store.recover_interrupted_attempts() == 0
    store.close()


def test_evidence_snapshot_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_run(store)
    snap = store.insert_evidence_snapshot(
        run_id="run-1",
        source_system="cases",
        record_id="CASE-001",
        field_path="status",
        value="stalled",
        retrieval_ref="get_case",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    got = store.get_evidence_snapshot(snap.snapshot_id)
    assert got is not None
    assert got.value == "stalled"
    assert got.provenance is EvidenceProvenance.ORIGINAL_RUN
    listed = store.list_evidence_snapshots("run-1")
    assert len(listed) == 1
    store.close()


def test_proposal_task_id_transition_and_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    # Simulate pre-migration DB without task_id / conversation tables.
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            objective TEXT NOT NULL,
            status TEXT NOT NULL,
            result_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE TABLE proposals (
            proposal_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            action TEXT NOT NULL,
            task_payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_digest TEXT NOT NULL,
            created_at TEXT NOT NULL,
            decided_at TEXT,
            operator_identity TEXT,
            execution_error TEXT,
            request_id TEXT
        );
        """
    )
    conn.commit()
    conn.close()

    store = BackendStore(db_path)
    store.insert_run(
        run_id="run-1",
        case_id="CASE-001",
        objective="x",
        status=RunStatus.COMPLETED,
    )
    prop = store.insert_proposal(
        proposal_id="prop-1",
        run_id="run-1",
        case_id="CASE-001",
        action=NextAction.CREATE_FOLLOW_UP_TASK,
        task_payload=FollowUpTaskPayload(title="t", body="b", case_id="CASE-001"),
        payload_digest="digest",
    )
    assert prop.task_id is None

    approved = store.transition_proposal(
        "prop-1",
        from_statuses={ProposalStatus.PENDING},
        to_status=ProposalStatus.APPROVED,
        operator_identity="ops",
    )
    assert approved is not None and approved.task_id is None

    executed = store.transition_proposal(
        "prop-1",
        from_statuses={ProposalStatus.APPROVED},
        to_status=ProposalStatus.EXECUTED,
        task_id="TASK-SIM-99",
        clear_execution_error=True,
    )
    assert executed is not None
    assert executed.status is ProposalStatus.EXECUTED
    assert executed.task_id == "TASK-SIM-99"
    assert store.get_proposal("prop-1") is not None
    assert store.get_proposal("prop-1").task_id == "TASK-SIM-99"  # type: ignore[union-attr]

    # Conversation tables available after open of legacy file.
    ok = store.begin_send(run_id="run-1", question="hi", idempotency_key="c1")
    assert ok.created is True
    store.close()


def test_complete_attempt_is_atomic_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_run(store)
    started = store.begin_send(run_id="run-1", question="Q", idempotency_key="k")
    first = store.complete_attempt(started.attempt.attempt_id, answer=_answer())
    second = store.complete_attempt(started.attempt.attempt_id, answer=_answer())
    assert first is not None and second is None
    store.close()
