"""Backend SQLite: investigation runs, proposals, and run-scoped conversation."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from casepilot.shared.contracts import (
    ConversationAttempt,
    ConversationTurn,
    EvidenceProvenance,
    EvidenceSnapshot,
    FollowUpAnswer,
    FollowUpTaskPayload,
    InvestigationResult,
    NextAction,
    Proposal,
    ProposalStatus,
    RunRecord,
    RunStatus,
    RunSummary,
    TurnStatus,
)

CHAT_ELIGIBLE_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.INDETERMINATE})


def _now() -> datetime:
    return datetime.now(UTC)


def _dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class ConversationConflict(Exception):
    """Another turn/attempt is already in flight for this run."""


class ConversationIdempotencyError(Exception):
    """Request key reused with different content or target."""


class ConversationNotEligible(Exception):
    """Run missing or not eligible for chat."""


class ConversationRetryError(Exception):
    """Retry preconditions failed (status, previous attempt, etc.)."""


@dataclass(frozen=True)
class BeginAttemptResult:
    turn: ConversationTurn
    attempt: ConversationAttempt
    created: bool


class BackendStore:
    """Thread-safe thin SQLite wrapper for runs, proposals, and conversation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS proposals (
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
                    request_id TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS conversation_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    turn_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    request_key TEXT NOT NULL UNIQUE,
                    previous_attempt_id TEXT,
                    status TEXT NOT NULL,
                    answer_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (turn_id) REFERENCES conversation_turns(turn_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS evidence_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    attempt_id TEXT,
                    source_system TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    field_path TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    retrieval_ref TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_conv_turns_run
                    ON conversation_turns(run_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_conv_attempts_run_status
                    ON conversation_attempts(run_id, status);
                CREATE INDEX IF NOT EXISTS idx_conv_attempts_turn
                    ON conversation_attempts(turn_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_snapshots_run
                    ON evidence_snapshots(run_id, retrieved_at);
                """
            )
            cols = {row[1] for row in self._conn.execute("PRAGMA table_info(proposals)").fetchall()}
            if "task_id" not in cols:
                self._conn.execute("ALTER TABLE proposals ADD COLUMN task_id TEXT")
            self._conn.commit()

    # ----- runs -----

    def insert_run(
        self,
        *,
        run_id: str,
        case_id: str,
        objective: str,
        status: RunStatus,
        result: InvestigationResult | None = None,
        error: str | None = None,
        created_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> RunRecord:
        created = created_at or _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO runs (
                    run_id, case_id, objective, status, result_json, error,
                    created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    case_id,
                    objective,
                    status.value,
                    result.model_dump_json() if result else None,
                    error,
                    _dt(created),
                    _dt(completed_at),
                ),
            )
            self._conn.commit()
        return RunRecord(
            run_id=run_id,
            case_id=case_id,
            objective=objective,
            status=status,
            result=result,
            error=error,
            created_at=created,
            completed_at=completed_at,
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        result = None
        if row["result_json"]:
            result = InvestigationResult.model_validate_json(row["result_json"])
        return RunRecord(
            run_id=row["run_id"],
            case_id=row["case_id"],
            objective=row["objective"],
            status=RunStatus(row["status"]),
            result=result,
            error=row["error"],
            created_at=_parse_dt(row["created_at"]) or _now(),
            completed_at=_parse_dt(row["completed_at"]),
        )

    def update_run_result(
        self,
        run_id: str,
        *,
        result: InvestigationResult,
        status: RunStatus | None = None,
        error: str | None = None,
        completed_at: datetime | None = None,
    ) -> RunRecord | None:
        """Patch stored run result (e.g. attach proposal_id after insert)."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            new_status = status.value if status is not None else row["status"]
            new_completed = _dt(completed_at) if completed_at is not None else row["completed_at"]
            self._conn.execute(
                """
                UPDATE runs SET result_json = ?, status = ?, error = ?, completed_at = ?
                WHERE run_id = ?
                """,
                (
                    result.model_dump_json(),
                    new_status,
                    error if error is not None else row["error"],
                    new_completed,
                    run_id,
                ),
            )
            self._conn.commit()
            updated = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        assert updated is not None
        return self._row_to_run(updated)

    def list_runs_for_case(self, case_id: str) -> list[RunSummary]:
        """Persisted investigation history for a case, newest first."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT run_id, case_id, status, created_at, completed_at
                FROM runs
                WHERE case_id = ?
                ORDER BY created_at DESC, run_id DESC
                """,
                (case_id,),
            ).fetchall()
        return [
            RunSummary(
                run_id=row["run_id"],
                case_id=row["case_id"],
                status=RunStatus(row["status"]),
                created_at=_parse_dt(row["created_at"]) or _now(),
                completed_at=_parse_dt(row["completed_at"]),
            )
            for row in rows
        ]

    # ----- proposals -----

    def insert_proposal(
        self,
        *,
        proposal_id: str,
        run_id: str,
        case_id: str,
        action: NextAction,
        task_payload: FollowUpTaskPayload,
        payload_digest: str,
        created_at: datetime | None = None,
    ) -> Proposal:
        created = created_at or _now()
        payload_json = task_payload.model_dump_json()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO proposals (
                    proposal_id, run_id, case_id, action, task_payload_json,
                    status, payload_digest, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal_id,
                    run_id,
                    case_id,
                    action.value,
                    payload_json,
                    ProposalStatus.PENDING.value,
                    payload_digest,
                    _dt(created),
                ),
            )
            self._conn.commit()
        return Proposal(
            proposal_id=proposal_id,
            run_id=run_id,
            case_id=case_id,
            action=action,
            task_payload=task_payload,
            status=ProposalStatus.PENDING,
            payload_digest=payload_digest,
            created_at=created,
        )

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM proposals WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_proposal(row)

    def get_proposal_by_run(self, run_id: str) -> Proposal | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM proposals WHERE run_id = ? ORDER BY created_at LIMIT 1",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_proposal(row)

    def list_proposals_for_case(self, case_id: str) -> list[Proposal]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM proposals WHERE case_id = ? ORDER BY created_at",
                (case_id,),
            ).fetchall()
        return [self._row_to_proposal(r) for r in rows]

    def case_action_flags(self) -> dict[str, tuple[bool, bool]]:
        """One-pass aggregate: case_id -> (pending_review, needs_execution).

        Flags are independent: pending vs approved/execution_failed.
        """
        pending = ProposalStatus.PENDING.value
        needs = (ProposalStatus.APPROVED.value, ProposalStatus.EXECUTION_FAILED.value)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT case_id,
                       MAX(CASE WHEN status = ? THEN 1 ELSE 0 END) AS pending_review,
                       MAX(CASE WHEN status IN (?, ?) THEN 1 ELSE 0 END) AS needs_execution
                FROM proposals
                GROUP BY case_id
                """,
                (pending, needs[0], needs[1]),
            ).fetchall()
        return {
            row["case_id"]: (bool(row["pending_review"]), bool(row["needs_execution"]))
            for row in rows
        }

    def _row_to_proposal(self, row: sqlite3.Row) -> Proposal:
        keys = row.keys()
        task_id = row["task_id"] if "task_id" in keys else None
        return Proposal(
            proposal_id=row["proposal_id"],
            run_id=row["run_id"],
            case_id=row["case_id"],
            action=NextAction(row["action"]),
            task_payload=FollowUpTaskPayload.model_validate_json(row["task_payload_json"]),
            status=ProposalStatus(row["status"]),
            payload_digest=row["payload_digest"],
            created_at=_parse_dt(row["created_at"]) or _now(),
            decided_at=_parse_dt(row["decided_at"]),
            operator_identity=row["operator_identity"],
            execution_error=row["execution_error"],
            task_id=task_id,
        )

    def transition_proposal(
        self,
        proposal_id: str,
        *,
        from_statuses: set[ProposalStatus],
        to_status: ProposalStatus,
        operator_identity: str | None = None,
        execution_error: str | None = None,
        clear_execution_error: bool = False,
        request_id: str | None = None,
        decided_at: datetime | None = None,
        task_id: str | None = None,
    ) -> Proposal | None:
        """Atomic status transition. Returns updated proposal or None if no match.

        Pass task_id when moving to EXECUTED after simulator confirms identity.
        """
        decided = decided_at or _now()
        allowed = tuple(s.value for s in from_statuses)
        placeholders = ",".join("?" * len(allowed))
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM proposals WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
            if row is None or row["status"] not in allowed:
                return None
            if clear_execution_error:
                new_error = execution_error
            elif execution_error is not None:
                new_error = execution_error
            else:
                new_error = row["execution_error"]
            new_operator = operator_identity or row["operator_identity"]
            new_request = request_id or row["request_id"]
            decided_value = row["decided_at"] or _dt(decided)
            keys = row.keys()
            prev_task = row["task_id"] if "task_id" in keys else None
            new_task = task_id if task_id is not None else prev_task
            cur = self._conn.execute(
                f"""
                UPDATE proposals SET
                    status = ?,
                    operator_identity = ?,
                    execution_error = ?,
                    request_id = ?,
                    decided_at = ?,
                    task_id = ?
                WHERE proposal_id = ? AND status IN ({placeholders})
                """,
                (
                    to_status.value,
                    new_operator,
                    new_error,
                    new_request,
                    decided_value,
                    new_task,
                    proposal_id,
                    *allowed,
                ),
            )
            if cur.rowcount != 1:
                self._conn.rollback()
                return None
            self._conn.commit()
            updated = self._conn.execute(
                "SELECT * FROM proposals WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
        assert updated is not None
        return self._row_to_proposal(updated)

    def dump_proposal_raw(self, proposal_id: str) -> dict[str, Any] | None:
        """Debug helper for tests."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM proposals WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
        return dict(row) if row else None

    # ----- conversation -----

    def get_conversation(self, run_id: str) -> list[ConversationTurn]:
        """Full transcript for a run, oldest turn first."""
        with self._lock:
            return self._load_turns_unlocked(run_id)

    def get_turn(self, turn_id: str) -> ConversationTurn | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM conversation_turns WHERE turn_id = ?", (turn_id,)
            ).fetchone()
            if row is None:
                return None
            return self._turn_from_row_unlocked(row)

    def begin_send(
        self,
        *,
        run_id: str,
        question: str,
        idempotency_key: str,
    ) -> BeginAttemptResult:
        """Start a send attempt. Idempotent on request key; one in-flight per run."""
        with self._lock:
            existing = self._attempt_by_key_unlocked(idempotency_key)
            if existing is not None:
                turn = self._require_turn_unlocked(existing["turn_id"])
                if turn["run_id"] != run_id or turn["question"] != question:
                    raise ConversationIdempotencyError(
                        "idempotency key reused with different content or target"
                    )
                if existing["previous_attempt_id"] is not None:
                    raise ConversationIdempotencyError(
                        "idempotency key reused with different content or target"
                    )
                return BeginAttemptResult(
                    turn=self._turn_from_row_unlocked(turn),
                    attempt=self._row_to_attempt(existing),
                    created=False,
                )

            self._assert_chat_eligible_unlocked(run_id)
            self._assert_no_pending_unlocked(run_id)

            now = _now()
            turn_id = _new_id("turn")
            attempt_id = _new_id("att")
            self._conn.execute(
                """
                INSERT INTO conversation_turns (
                    turn_id, run_id, question, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    run_id,
                    question,
                    TurnStatus.PENDING.value,
                    _dt(now),
                    _dt(now),
                ),
            )
            self._conn.execute(
                """
                INSERT INTO conversation_attempts (
                    attempt_id, turn_id, run_id, request_key, previous_attempt_id,
                    status, answer_json, error, created_at, completed_at
                ) VALUES (?, ?, ?, ?, NULL, ?, NULL, NULL, ?, NULL)
                """,
                (
                    attempt_id,
                    turn_id,
                    run_id,
                    idempotency_key,
                    TurnStatus.PENDING.value,
                    _dt(now),
                ),
            )
            self._conn.commit()
            turn_row = self._conn.execute(
                "SELECT * FROM conversation_turns WHERE turn_id = ?", (turn_id,)
            ).fetchone()
            att_row = self._conn.execute(
                "SELECT * FROM conversation_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            assert turn_row is not None and att_row is not None
            return BeginAttemptResult(
                turn=self._turn_from_row_unlocked(turn_row),
                attempt=self._row_to_attempt(att_row),
                created=True,
            )

    def begin_retry(
        self,
        *,
        run_id: str,
        turn_id: str,
        idempotency_key: str,
        expected_previous_attempt_id: str,
    ) -> BeginAttemptResult:
        """Start a retry attempt on a failed turn. Key replay returns same attempt."""
        with self._lock:
            existing = self._attempt_by_key_unlocked(idempotency_key)
            if existing is not None:
                if (
                    existing["turn_id"] != turn_id
                    or existing["run_id"] != run_id
                    or existing["previous_attempt_id"] != expected_previous_attempt_id
                ):
                    raise ConversationIdempotencyError(
                        "idempotency key reused with different content or target"
                    )
                turn = self._require_turn_unlocked(turn_id)
                return BeginAttemptResult(
                    turn=self._turn_from_row_unlocked(turn),
                    attempt=self._row_to_attempt(existing),
                    created=False,
                )

            self._assert_chat_eligible_unlocked(run_id)
            turn = self._require_turn_unlocked(turn_id)
            if turn["run_id"] != run_id:
                raise ConversationRetryError("turn does not belong to run")
            if turn["status"] != TurnStatus.FAILED.value:
                raise ConversationRetryError("turn is not failed")

            prev = self._conn.execute(
                "SELECT * FROM conversation_attempts WHERE attempt_id = ?",
                (expected_previous_attempt_id,),
            ).fetchone()
            if prev is None or prev["turn_id"] != turn_id:
                raise ConversationRetryError("expected previous attempt not found on turn")
            if prev["status"] != TurnStatus.FAILED.value:
                raise ConversationRetryError("expected previous attempt is not failed")

            latest = self._conn.execute(
                """
                SELECT * FROM conversation_attempts
                WHERE turn_id = ?
                ORDER BY created_at DESC, attempt_id DESC
                LIMIT 1
                """,
                (turn_id,),
            ).fetchone()
            if latest is None or latest["attempt_id"] != expected_previous_attempt_id:
                raise ConversationRetryError("expected previous attempt is not latest")

            self._assert_no_pending_unlocked(run_id)

            now = _now()
            attempt_id = _new_id("att")
            self._conn.execute(
                """
                INSERT INTO conversation_attempts (
                    attempt_id, turn_id, run_id, request_key, previous_attempt_id,
                    status, answer_json, error, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL)
                """,
                (
                    attempt_id,
                    turn_id,
                    run_id,
                    idempotency_key,
                    expected_previous_attempt_id,
                    TurnStatus.PENDING.value,
                    _dt(now),
                ),
            )
            self._conn.execute(
                """
                UPDATE conversation_turns
                SET status = ?, updated_at = ?
                WHERE turn_id = ?
                """,
                (TurnStatus.PENDING.value, _dt(now), turn_id),
            )
            self._conn.commit()
            turn_row = self._conn.execute(
                "SELECT * FROM conversation_turns WHERE turn_id = ?", (turn_id,)
            ).fetchone()
            att_row = self._conn.execute(
                "SELECT * FROM conversation_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            assert turn_row is not None and att_row is not None
            return BeginAttemptResult(
                turn=self._turn_from_row_unlocked(turn_row),
                attempt=self._row_to_attempt(att_row),
                created=True,
            )

    def complete_attempt(
        self,
        attempt_id: str,
        *,
        answer: FollowUpAnswer,
        completed_at: datetime | None = None,
    ) -> ConversationTurn | None:
        """Atomic pending → succeeded for attempt and its turn."""
        done = completed_at or _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM conversation_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if row is None or row["status"] != TurnStatus.PENDING.value:
                return None
            cur = self._conn.execute(
                """
                UPDATE conversation_attempts
                SET status = ?, answer_json = ?, error = NULL, completed_at = ?
                WHERE attempt_id = ? AND status = ?
                """,
                (
                    TurnStatus.SUCCEEDED.value,
                    answer.model_dump_json(),
                    _dt(done),
                    attempt_id,
                    TurnStatus.PENDING.value,
                ),
            )
            if cur.rowcount != 1:
                self._conn.rollback()
                return None
            self._conn.execute(
                """
                UPDATE conversation_turns
                SET status = ?, updated_at = ?
                WHERE turn_id = ?
                """,
                (TurnStatus.SUCCEEDED.value, _dt(done), row["turn_id"]),
            )
            self._conn.commit()
            turn_row = self._conn.execute(
                "SELECT * FROM conversation_turns WHERE turn_id = ?",
                (row["turn_id"],),
            ).fetchone()
            assert turn_row is not None
            return self._turn_from_row_unlocked(turn_row)

    def fail_attempt(
        self,
        attempt_id: str,
        *,
        error: str,
        completed_at: datetime | None = None,
    ) -> ConversationTurn | None:
        """Atomic pending → failed for attempt and its turn."""
        done = completed_at or _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM conversation_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if row is None or row["status"] != TurnStatus.PENDING.value:
                return None
            cur = self._conn.execute(
                """
                UPDATE conversation_attempts
                SET status = ?, error = ?, completed_at = ?
                WHERE attempt_id = ? AND status = ?
                """,
                (
                    TurnStatus.FAILED.value,
                    error,
                    _dt(done),
                    attempt_id,
                    TurnStatus.PENDING.value,
                ),
            )
            if cur.rowcount != 1:
                self._conn.rollback()
                return None
            self._conn.execute(
                """
                UPDATE conversation_turns
                SET status = ?, updated_at = ?
                WHERE turn_id = ?
                """,
                (TurnStatus.FAILED.value, _dt(done), row["turn_id"]),
            )
            self._conn.commit()
            turn_row = self._conn.execute(
                "SELECT * FROM conversation_turns WHERE turn_id = ?",
                (row["turn_id"],),
            ).fetchone()
            assert turn_row is not None
            return self._turn_from_row_unlocked(turn_row)

    def recover_interrupted_attempts(
        self,
        *,
        error: str = "interrupted by restart",
    ) -> int:
        """Mark all pending attempts (and their turns) failed. Call on startup."""
        done = _now()
        with self._lock:
            pending = self._conn.execute(
                """
                SELECT attempt_id, turn_id FROM conversation_attempts
                WHERE status = ?
                """,
                (TurnStatus.PENDING.value,),
            ).fetchall()
            if not pending:
                return 0
            for row in pending:
                self._conn.execute(
                    """
                    UPDATE conversation_attempts
                    SET status = ?, error = ?, completed_at = ?
                    WHERE attempt_id = ? AND status = ?
                    """,
                    (
                        TurnStatus.FAILED.value,
                        error,
                        _dt(done),
                        row["attempt_id"],
                        TurnStatus.PENDING.value,
                    ),
                )
                self._conn.execute(
                    """
                    UPDATE conversation_turns
                    SET status = ?, updated_at = ?
                    WHERE turn_id = ?
                    """,
                    (TurnStatus.FAILED.value, _dt(done), row["turn_id"]),
                )
            self._conn.commit()
            return len(pending)

    def insert_evidence_snapshot(
        self,
        *,
        run_id: str,
        source_system: str,
        record_id: str,
        field_path: str,
        value: Any,
        retrieval_ref: str,
        provenance: EvidenceProvenance,
        attempt_id: str | None = None,
        snapshot_id: str | None = None,
        retrieved_at: datetime | None = None,
    ) -> EvidenceSnapshot:
        sid = snapshot_id or _new_id("snap")
        when = retrieved_at or _now()
        with self._lock:
            run = self._conn.execute(
                "SELECT run_id FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise ConversationNotEligible("run not found")
            self._conn.execute(
                """
                INSERT INTO evidence_snapshots (
                    snapshot_id, run_id, attempt_id, source_system, record_id,
                    field_path, value_json, retrieval_ref, retrieved_at, provenance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    run_id,
                    attempt_id,
                    source_system,
                    record_id,
                    field_path,
                    json.dumps(value),
                    retrieval_ref,
                    _dt(when),
                    provenance.value,
                ),
            )
            self._conn.commit()
        return EvidenceSnapshot(
            snapshot_id=sid,
            run_id=run_id,
            attempt_id=attempt_id,
            source_system=source_system,
            record_id=record_id,
            field_path=field_path,
            value=value,
            retrieval_ref=retrieval_ref,
            retrieved_at=when,
            provenance=provenance,
        )

    def get_evidence_snapshot(self, snapshot_id: str) -> EvidenceSnapshot | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM evidence_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_snapshot(row)

    def list_evidence_snapshots(self, run_id: str) -> list[EvidenceSnapshot]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM evidence_snapshots
                WHERE run_id = ?
                ORDER BY retrieved_at, snapshot_id
                """,
                (run_id,),
            ).fetchall()
        return [self._row_to_snapshot(r) for r in rows]

    # ----- conversation internals (caller holds lock) -----

    def _attempt_by_key_unlocked(self, request_key: str) -> sqlite3.Row | None:
        row = self._conn.execute(
            "SELECT * FROM conversation_attempts WHERE request_key = ?",
            (request_key,),
        ).fetchone()
        return cast(sqlite3.Row | None, row)

    def _require_turn_unlocked(self, turn_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM conversation_turns WHERE turn_id = ?", (turn_id,)
        ).fetchone()
        if row is None:
            raise ConversationRetryError("turn not found")
        return cast(sqlite3.Row, row)

    def _assert_chat_eligible_unlocked(self, run_id: str) -> None:
        row = self._conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ConversationNotEligible("run not found")
        status = RunStatus(row["status"])
        if status not in CHAT_ELIGIBLE_STATUSES:
            raise ConversationNotEligible(f"run status {status.value} is not chat-eligible")

    def _assert_no_pending_unlocked(self, run_id: str) -> None:
        row = self._conn.execute(
            """
            SELECT attempt_id FROM conversation_attempts
            WHERE run_id = ? AND status = ?
            LIMIT 1
            """,
            (run_id, TurnStatus.PENDING.value),
        ).fetchone()
        if row is not None:
            raise ConversationConflict("run already has an in-flight conversation attempt")

    def _load_turns_unlocked(self, run_id: str) -> list[ConversationTurn]:
        rows = self._conn.execute(
            """
            SELECT * FROM conversation_turns
            WHERE run_id = ?
            ORDER BY created_at, turn_id
            """,
            (run_id,),
        ).fetchall()
        return [self._turn_from_row_unlocked(r) for r in rows]

    def _turn_from_row_unlocked(self, row: sqlite3.Row) -> ConversationTurn:
        attempts = self._conn.execute(
            """
            SELECT * FROM conversation_attempts
            WHERE turn_id = ?
            ORDER BY created_at, attempt_id
            """,
            (row["turn_id"],),
        ).fetchall()
        return ConversationTurn(
            turn_id=row["turn_id"],
            run_id=row["run_id"],
            question=row["question"],
            status=TurnStatus(row["status"]),
            created_at=_parse_dt(row["created_at"]) or _now(),
            updated_at=_parse_dt(row["updated_at"]) or _now(),
            attempts=[self._row_to_attempt(a) for a in attempts],
        )

    def _row_to_attempt(self, row: sqlite3.Row) -> ConversationAttempt:
        answer = None
        if row["answer_json"]:
            answer = FollowUpAnswer.model_validate_json(row["answer_json"])
        return ConversationAttempt(
            attempt_id=row["attempt_id"],
            turn_id=row["turn_id"],
            request_key=row["request_key"],
            previous_attempt_id=row["previous_attempt_id"],
            status=TurnStatus(row["status"]),
            answer=answer,
            error=row["error"],
            created_at=_parse_dt(row["created_at"]) or _now(),
            completed_at=_parse_dt(row["completed_at"]),
        )

    def _row_to_snapshot(self, row: sqlite3.Row) -> EvidenceSnapshot:
        return EvidenceSnapshot(
            snapshot_id=row["snapshot_id"],
            run_id=row["run_id"],
            attempt_id=row["attempt_id"],
            source_system=row["source_system"],
            record_id=row["record_id"],
            field_path=row["field_path"],
            value=json.loads(row["value_json"]),
            retrieval_ref=row["retrieval_ref"],
            retrieved_at=_parse_dt(row["retrieved_at"]) or _now(),
            provenance=EvidenceProvenance(row["provenance"]),
        )
