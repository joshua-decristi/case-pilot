"""Read/write access to simulator SQLite records (contract-shaped)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from casepilot.shared import (
    CaseRecord,
    CreateFollowUpTaskResponse,
    DocumentRecord,
    FollowUpTaskPayload,
    ProcessingEvent,
)


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _require_dt(value: str) -> datetime:
    parsed = _parse_dt(value)
    assert parsed is not None
    return parsed


class ConflictError(Exception):
    """Idempotency key reused with a mismatched case or payload."""


class SimulatorStore:
    """Case-scoped queries and transactional follow-up task writes.

    Owns one connection for the request lifetime (see app dependency).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def case_exists(self, case_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 AS ok FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        return row is not None

    def get_case(self, case_id: str) -> CaseRecord | None:
        row = self._conn.execute(
            "SELECT * FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if row is None:
            return None
        return CaseRecord(
            case_id=row["case_id"],
            status=row["status"],
            symptom=row["symptom"],
            application_fields=json.loads(row["application_fields"]),
            required_documents=json.loads(row["required_documents"]),
            created_at=_require_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    def list_documents(self, case_id: str) -> list[DocumentRecord]:
        rows = self._conn.execute(
            "SELECT * FROM documents WHERE case_id = ? ORDER BY document_id",
            (case_id,),
        ).fetchall()
        return [
            DocumentRecord(
                document_id=row["document_id"],
                case_id=row["case_id"],
                doc_type=row["doc_type"],
                status=row["status"],
                received_at=_parse_dt(row["received_at"]),
                fields=json.loads(row["fields"]),
                content_summary=row["content_summary"],
            )
            for row in rows
        ]

    def list_events(self, case_id: str) -> list[ProcessingEvent]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE case_id = ? ORDER BY timestamp, event_id",
            (case_id,),
        ).fetchall()
        return [
            ProcessingEvent(
                event_id=row["event_id"],
                case_id=row["case_id"],
                event_type=row["event_type"],
                timestamp=_require_dt(row["timestamp"]),
                message=row["message"],
                details=json.loads(row["details"]),
            )
            for row in rows
        ]

    def create_follow_up_task(
        self,
        *,
        case_id: str,
        proposal_id: str,
        task: FollowUpTaskPayload,
    ) -> CreateFollowUpTaskResponse:
        """Insert or idempotently return an existing task.

        Spec: unique proposal_id; retry returns original only if case and
        payload also match, else conflict. Concurrent inserts race on the
        UNIQUE(proposal_id) constraint and re-check the stored row.
        """
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        task_id = f"task-{proposal_id}"

        existing = self._fetch_by_proposal(proposal_id)
        if existing is not None:
            return self._idempotent_or_conflict(existing, case_id=case_id, task=task)

        try:
            self._conn.execute(
                """
                INSERT INTO follow_up_tasks (
                    task_id, proposal_id, case_id, title, body, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (task_id, proposal_id, case_id, task.title, task.body, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            self._conn.rollback()
            raced = self._fetch_by_proposal(proposal_id)
            if raced is None:
                raise
            return self._idempotent_or_conflict(raced, case_id=case_id, task=task)
        except Exception:
            self._conn.rollback()
            raise

        return CreateFollowUpTaskResponse(
            task_id=task_id,
            proposal_id=proposal_id,
            case_id=case_id,
            created=True,
        )

    def _fetch_by_proposal(self, proposal_id: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT * FROM follow_up_tasks WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        return row

    def _idempotent_or_conflict(
        self,
        row: Any,
        *,
        case_id: str,
        task: FollowUpTaskPayload,
    ) -> CreateFollowUpTaskResponse:
        if not _payload_matches(row, case_id=case_id, task=task):
            raise ConflictError(
                "follow-up task exists for proposal_id with different case or payload"
            )
        return CreateFollowUpTaskResponse(
            task_id=row["task_id"],
            proposal_id=row["proposal_id"],
            case_id=row["case_id"],
            created=False,
        )


def _payload_matches(
    row: Any,
    *,
    case_id: str,
    task: FollowUpTaskPayload,
) -> bool:
    return bool(
        row["case_id"] == case_id
        and row["case_id"] == task.case_id
        and row["title"] == task.title
        and row["body"] == task.body
    )
