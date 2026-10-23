"""Backend SQLite: investigation runs and action proposals."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from casepilot.shared.contracts import (
    FollowUpTaskPayload,
    InvestigationResult,
    NextAction,
    Proposal,
    ProposalStatus,
    RunRecord,
    RunStatus,
    RunSummary,
)


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


class BackendStore:
    """Thread-safe thin SQLite wrapper for runs and proposals."""

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
                """
            )
            self._conn.commit()

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
    ) -> Proposal | None:
        """Atomic status transition. Returns updated proposal or None if no match."""
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
            cur = self._conn.execute(
                f"""
                UPDATE proposals SET
                    status = ?,
                    operator_identity = ?,
                    execution_error = ?,
                    request_id = ?,
                    decided_at = ?
                WHERE proposal_id = ? AND status IN ({placeholders})
                """,
                (
                    to_status.value,
                    new_operator,
                    new_error,
                    new_request,
                    decided_value,
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
