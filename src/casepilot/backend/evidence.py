"""Verify agent evidence cites records actually retrieved this run.

``Evidence.field`` must be an exact path into the tool JSON for that record:
top-level keys (``status``, ``content_summary``, ``required_documents``, …)
or dotted nested paths (``details.missing``, ``fields.name``,
``application_fields.applicant_name``). No slash multi-field strings, no
fabricated aliases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from casepilot.shared.contracts import (
    CaseRecord,
    DocumentRecord,
    Evidence,
    ProcessingEvent,
)


@dataclass
class RetrievedIndex:
    """Index of tool results for evidence verification."""

    case: CaseRecord | None = None
    documents: dict[str, DocumentRecord] = field(default_factory=dict)
    events: dict[str, ProcessingEvent] = field(default_factory=dict)
    tool_calls: int = 0
    raw_snapshots: list[dict[str, Any]] = field(default_factory=list)

    def note_case(self, record: CaseRecord) -> None:
        self.case = record
        self.raw_snapshots.append({"tool": "get_case", "record_id": record.case_id})

    def note_documents(self, docs: list[DocumentRecord]) -> None:
        for d in docs:
            self.documents[d.document_id] = d
            self.raw_snapshots.append({"tool": "get_documents", "record_id": d.document_id})

    def note_events(self, events: list[ProcessingEvent]) -> None:
        for e in events:
            self.events[e.event_id] = e
            self.raw_snapshots.append({"tool": "get_events", "record_id": e.event_id})


def validate_evidence(
    evidence: list[Evidence], index: RetrievedIndex
) -> tuple[list[Evidence], list[str]]:
    """Return (accepted evidence, rejection reasons)."""
    accepted: list[Evidence] = []
    errors: list[str] = []
    for ev in evidence:
        ok, reason = _check_one(ev, index)
        if ok:
            accepted.append(ev)
        else:
            errors.append(reason)
    return accepted, errors


def field_path_exists(path: str, record_data: dict[str, Any]) -> bool:
    """True if ``path`` is a single exact key path into ``record_data``.

    - ``status`` → top-level key
    - ``details.missing`` → nested key under ``details``
    - ``fields.name`` → nested key under ``fields``
    - ``application_fields.dob`` → nested under application_fields
    - ``status/content_summary`` → rejected (not a single path)
    """
    raw = path.strip()
    if not raw:
        return False
    # Multi-field separators are not schema paths.
    if "/" in raw or " " in raw or "," in raw:
        return False
    if raw.startswith(".") or raw.endswith("."):
        return False
    parts = raw.split(".")
    if any(not p or p.strip() != p for p in parts):
        return False
    cur: Any = record_data
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _record_payload(model: CaseRecord | DocumentRecord | ProcessingEvent) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _check_one(ev: Evidence, index: RetrievedIndex) -> tuple[bool, str]:
    rid = ev.record_id
    if index.case is not None and rid == index.case.case_id:
        data = _record_payload(index.case)
        if not field_path_exists(ev.field, data):
            return False, f"evidence field not found on case {rid}: {ev.field}"
        return True, ""
    if rid in index.documents:
        data = _record_payload(index.documents[rid])
        if not field_path_exists(ev.field, data):
            return False, f"evidence field not found on document {rid}: {ev.field}"
        return True, ""
    if rid in index.events:
        data = _record_payload(index.events[rid])
        if not field_path_exists(ev.field, data):
            return False, f"evidence field not found on event {rid}: {ev.field}"
        return True, ""
    return False, f"evidence record_id not retrieved this run: {rid}"
