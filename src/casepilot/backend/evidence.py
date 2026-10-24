"""Evidence path validation, field-level snapshot capture, and follow-up rendering.

Canonical types live in ``casepilot.shared.contracts``:
``EvidenceSnapshot``, ``SourceFactClaim``, ``RenderedSourceFact``,
``InterpretationBlock``, ``FollowUpAgentOutput``, ``FollowUpAnswer``,
``EvidenceProvenance``.

Persistence uses ``BackendStore``:
- ``insert_evidence_snapshot(...)`` — one field snapshot (run must already exist)
- ``list_evidence_snapshots(run_id)``
- ``get_evidence_snapshot(snapshot_id)``

Read tools accumulate redacted record captures in ``RetrievedIndex``; call
``persist_index_snapshots(store, index)`` after the run row exists (investigation
insert_run / follow-up attempt). Snapshots are field-level leaves.

Follow-up usage
---------------
1. ``idx = RetrievedIndex(context=RetrievalContext(run_id, case_id,
     provenance=EvidenceProvenance.FOLLOW_UP, attempt_id=...))``
2. ``tools = build_case_read_tools(...)`` or ``build_follow_up_read_tools(...)``
3. After tools: ``persist_index_snapshots(store, idx)``
4. ``snaps = store.list_evidence_snapshots(run_id)``
5. Model returns ``FollowUpAgentOutput``; server calls
   ``render_follow_up_answer(output, snaps)`` → ``FollowUpRenderResult``.
   Model ``notes`` are allowlisted codes only (``read_only``, ``no_proposal``, …);
   free-text notes are discarded. Prefer caller ``server_notes=[...]``.
   On ``ok is False``, show ``result.answer`` (SAFE_UNCERTAINTY only).
6. Legacy runs: ``attribute_run_evidence(result.evidence, snaps)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from casepilot.backend.trace import _redact
from casepilot.shared.contracts import (
    CaseRecord,
    DocumentRecord,
    Evidence,
    EvidenceProvenance,
    EvidenceSnapshot,
    FollowUpAgentOutput,
    FollowUpAnswer,
    InterpretationBlock,
    InterpretationKind,
    ProcessingEvent,
    RenderedSourceFact,
    SourceFactClaim,
)

SOURCE_CASES = "cases"
SOURCE_DOCUMENTS = "documents"
SOURCE_EVENTS = "events"

TOOL_GET_CASE = "get_case"
TOOL_GET_DOCUMENTS = "get_documents"
TOOL_GET_EVENTS = "get_events"

SAFE_UNCERTAINTY = (
    "Unable to present validated source facts for this answer. "
    "Retrieved evidence was missing, malformed, or could not be verified "
    "against server-owned snapshots."
)

LEGACY_DISCLOSURE = (
    "Historical investigation report only — not a server-captured evidence "
    "snapshot. Re-retrieve before treating claims as established facts."
)

# Server-owned operator notes only. Model ``FollowUpAgentOutput.notes`` may request
# these by exact code (or exact full string); free-text model notes are discarded.
NOTE_READ_ONLY = (
    "This chat is read-only. It cannot create tasks, approve proposals, or take "
    "write actions. Use existing proposal controls when a proposal is available."
)
NOTE_NO_PROPOSAL = (
    "No action proposal is attached to this run. Chat cannot create one; start a "
    "new investigation if a follow-up task may be warranted."
)
NOTE_BOUNDED_CONTEXT = (
    "Older conversation turns may fall outside the model context window and are "
    "not evidence by themselves."
)

SERVER_NOTES_BY_CODE: dict[str, str] = {
    "read_only": NOTE_READ_ONLY,
    "no_proposal": NOTE_NO_PROPOSAL,
    "bounded_context": NOTE_BOUNDED_CONTEXT,
    "safe_uncertainty": SAFE_UNCERTAINTY,
}
# Exact full-string match also accepted (stable codes preferred).
_SERVER_NOTE_TEXTS: frozenset[str] = frozenset(SERVER_NOTES_BY_CODE.values())


@dataclass(frozen=True)
class RetrievalContext:
    """Provenance for captures flushed via BackendStore."""

    run_id: str
    case_id: str
    provenance: EvidenceProvenance = EvidenceProvenance.ORIGINAL_RUN
    attempt_id: str | None = None


@dataclass(frozen=True)
class RecordCapture:
    """In-memory redacted record pending field-level snapshot persistence."""

    source_system: str
    record_id: str
    tool_name: str
    payload: dict[str, Any]
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class RetrievedIndex:
    """Tool-result index for investigation validation + pending snapshot captures."""

    context: RetrievalContext | None = None
    case: CaseRecord | None = None
    documents: dict[str, DocumentRecord] = field(default_factory=dict)
    events: dict[str, ProcessingEvent] = field(default_factory=dict)
    tool_calls: int = 0
    captures: list[RecordCapture] = field(default_factory=list)

    @property
    def raw_snapshots(self) -> list[dict[str, Any]]:
        """Light retrieval log (compat for traces/tests)."""
        return [
            {
                "tool": c.tool_name,
                "record_id": c.record_id,
                "source_system": c.source_system,
            }
            for c in self.captures
        ]

    def note_case(self, record: CaseRecord) -> None:
        self.case = record
        self._capture(SOURCE_CASES, record.case_id, TOOL_GET_CASE, _record_payload(record))

    def note_documents(self, docs: list[DocumentRecord]) -> None:
        for d in docs:
            self.documents[d.document_id] = d
            self._capture(SOURCE_DOCUMENTS, d.document_id, TOOL_GET_DOCUMENTS, _record_payload(d))

    def note_events(self, events: list[ProcessingEvent]) -> None:
        for e in events:
            self.events[e.event_id] = e
            self._capture(SOURCE_EVENTS, e.event_id, TOOL_GET_EVENTS, _record_payload(e))

    def _capture(
        self, source_system: str, record_id: str, tool_name: str, payload: dict[str, Any]
    ) -> None:
        self.captures.append(
            RecordCapture(
                source_system=source_system,
                record_id=record_id,
                tool_name=tool_name,
                payload=redact_evidence_payload(payload),
            )
        )


def redact_evidence_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = _redact(payload)
    return redacted if isinstance(redacted, dict) else {}


def iter_leaf_fields(data: dict[str, Any], *, prefix: str = "") -> Iterable[tuple[str, Any]]:
    """Yield (dotted_path, json_value) for leaf fields (dicts nested; lists leaf)."""
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from iter_leaf_fields(value, prefix=path)
        else:
            yield path, value


def persist_index_snapshots(store: Any, index: RetrievedIndex) -> list[EvidenceSnapshot]:
    """Flush pending captures as field-level ``EvidenceSnapshot`` rows.

    Requires the run row to already exist (FK). No-op if there is no context,
    no captures, or the run is missing.
    """
    ctx = index.context
    if ctx is None or not index.captures:
        return []
    get_run = getattr(store, "get_run", None)
    if callable(get_run) and get_run(ctx.run_id) is None:
        return []
    return persist_record_captures(
        store,
        index.captures,
        run_id=ctx.run_id,
        provenance=ctx.provenance,
        attempt_id=ctx.attempt_id,
    )


def persist_record_captures(
    store: Any,
    captures: Sequence[RecordCapture],
    *,
    run_id: str,
    provenance: EvidenceProvenance,
    attempt_id: str | None = None,
) -> list[EvidenceSnapshot]:
    """Insert one DB snapshot per leaf field via ``store.insert_evidence_snapshot``."""
    insert = store.insert_evidence_snapshot
    out: list[EvidenceSnapshot] = []
    for cap in captures:
        for field_path, value in iter_leaf_fields(cap.payload):
            ref = f"{cap.tool_name}:{cap.record_id}:{field_path}"
            snap = insert(
                run_id=run_id,
                source_system=cap.source_system,
                record_id=cap.record_id,
                field_path=field_path,
                value=value,
                retrieval_ref=ref,
                provenance=provenance,
                attempt_id=attempt_id,
                retrieved_at=cap.retrieved_at,
            )
            out.append(snap)
    return out


def snapshots_by_id(
    snapshots: Sequence[EvidenceSnapshot],
) -> dict[str, EvidenceSnapshot]:
    return {s.snapshot_id: s for s in snapshots}


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
    """True if ``path`` is a single exact key path into ``record_data``."""
    return _field_at(path, record_data) is not _MISSING


def field_value_at(path: str, record_data: dict[str, Any]) -> Any:
    val = _field_at(path, record_data)
    if val is _MISSING:
        raise KeyError(path)
    return val


class _Missing:
    pass


_MISSING = _Missing()


def _field_at(path: str, record_data: dict[str, Any]) -> Any:
    raw = path.strip()
    if not raw:
        return _MISSING
    if "/" in raw or " " in raw or "," in raw:
        return _MISSING
    if raw.startswith(".") or raw.endswith("."):
        return _MISSING
    parts = raw.split(".")
    if any(not p or p.strip() != p for p in parts):
        return _MISSING
    cur: Any = record_data
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


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


# ---------------------------------------------------------------------------
# Follow-up structured rendering (canonical contracts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FollowUpRenderResult:
    """Result of validating ``FollowUpAgentOutput`` against server snapshots."""

    ok: bool
    answer: FollowUpAnswer


def render_source_fact(
    claim: SourceFactClaim,
    catalog: Mapping[str, EvidenceSnapshot],
) -> RenderedSourceFact | None:
    """Render one claim from a server snapshot; None if invalid.

    Model supplies ``local_id`` + ``snapshot_id`` + ``field_path`` only.
    Value, times, system, and provenance always come from the snapshot row.
    ``field_path`` must equal the snapshot's stored path.
    """
    snap = catalog.get(claim.snapshot_id)
    if snap is None:
        return None
    if claim.field_path.strip() != snap.field_path:
        return None
    return RenderedSourceFact(
        local_id=claim.local_id,
        snapshot_id=snap.snapshot_id,
        source_system=snap.source_system,
        record_id=snap.record_id,
        field_path=snap.field_path,
        value=snap.value,
        retrieval_ref=snap.retrieval_ref,
        retrieved_at=snap.retrieved_at,
        provenance=snap.provenance,
    )


def resolve_server_notes(requested: Sequence[str] | None = None) -> list[str]:
    """Map model note codes / exact server strings to server-owned text; drop rest."""
    if not requested:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in requested:
        text = _server_note_text(raw)
        if text is None or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _server_note_text(raw: str) -> str | None:
    key = raw.strip()
    if not key:
        return None
    if key in SERVER_NOTES_BY_CODE:
        return SERVER_NOTES_BY_CODE[key]
    if key in _SERVER_NOTE_TEXTS:
        return key
    return None


def render_follow_up_answer(
    output: FollowUpAgentOutput,
    snapshots: Sequence[EvidenceSnapshot] | Mapping[str, EvidenceSnapshot],
    *,
    server_notes: Sequence[str] | None = None,
) -> FollowUpRenderResult:
    """Validate model output and build a server-owned ``FollowUpAnswer``.

    Model ``notes`` never pass through as prose. Only allowlisted server note
    codes (or identical server strings) and optional caller ``server_notes`` codes
    appear in ``FollowUpAnswer.notes``. On failure: ``ok=False`` + safe uncertainty.
    """
    catalog = snapshots if isinstance(snapshots, Mapping) else snapshots_by_id(snapshots)
    # Merge caller codes + model-requested codes; never free-text model notes.
    notes = resolve_server_notes([*(server_notes or ()), *output.notes])

    if not output.source_facts and not output.interpretations:
        if notes:
            return FollowUpRenderResult(ok=True, answer=FollowUpAnswer(notes=notes))
        return _safe_fail()

    rendered: list[RenderedSourceFact] = []
    local_ids: set[str] = set()

    for claim in output.source_facts:
        if not claim.local_id or claim.local_id in local_ids:
            return _safe_fail()
        fact = render_source_fact(claim, catalog)
        if fact is None:
            return _safe_fail()
        local_ids.add(claim.local_id)
        rendered.append(fact)

    interpretations: list[InterpretationBlock] = []
    for block in output.interpretations:
        if block.kind not in (InterpretationKind.INFERENCE, InterpretationKind.UNCERTAINTY):
            return _safe_fail()
        if not block.cites:
            return _safe_fail()
        if any(c not in local_ids for c in block.cites):
            return _safe_fail()
        interpretations.append(
            InterpretationBlock(text=block.text, cites=list(block.cites), kind=block.kind)
        )

    return FollowUpRenderResult(
        ok=True,
        answer=FollowUpAnswer(
            source_facts=rendered,
            interpretations=interpretations,
            notes=notes,
        ),
    )


def _safe_fail() -> FollowUpRenderResult:
    return FollowUpRenderResult(
        ok=False,
        answer=FollowUpAnswer(notes=[SAFE_UNCERTAINTY]),
    )


# ---------------------------------------------------------------------------
# Legacy / historical evidence attribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributedEvidence:
    """Honest attribution of investigation Evidence vs verified field snapshots."""

    evidence: Evidence
    is_verified_snapshot: bool
    disclosure: str | None
    matching_snapshot_id: str | None
    rendered_value: Any | None = None
    snapshot_retrieved_at: datetime | None = None
    provenance: EvidenceProvenance | None = None


def attribute_run_evidence(
    evidence: Sequence[Evidence],
    snapshots: Sequence[EvidenceSnapshot],
) -> list[AttributedEvidence]:
    """Map run Evidence items to field snapshots when possible.

    No matching snapshot → historical report; never invent retrieval timestamps.
    """
    by_key: dict[tuple[str, str], list[EvidenceSnapshot]] = {}
    for s in snapshots:
        by_key.setdefault((s.record_id, s.field_path), []).append(s)

    out: list[AttributedEvidence] = []
    for ev in evidence:
        candidates = by_key.get((ev.record_id, ev.field), [])
        match = _best_snapshot(ev, candidates)
        if match is None:
            out.append(
                AttributedEvidence(
                    evidence=ev,
                    is_verified_snapshot=False,
                    disclosure=LEGACY_DISCLOSURE,
                    matching_snapshot_id=None,
                )
            )
            continue
        out.append(
            AttributedEvidence(
                evidence=ev,
                is_verified_snapshot=True,
                disclosure=None,
                matching_snapshot_id=match.snapshot_id,
                rendered_value=match.value,
                snapshot_retrieved_at=match.retrieved_at,
                provenance=match.provenance,
            )
        )
    return out


def _best_snapshot(ev: Evidence, candidates: Sequence[EvidenceSnapshot]) -> EvidenceSnapshot | None:
    if not candidates:
        return None
    scored = sorted(
        candidates,
        key=lambda s: (
            0 if s.provenance == EvidenceProvenance.ORIGINAL_RUN else 1,
            s.retrieved_at,
            s.snapshot_id,
        ),
    )
    for snap in scored:
        if ev.source_system and snap.source_system != ev.source_system:
            continue
        return snap
    return None


def established_snapshots(
    snapshots: Sequence[EvidenceSnapshot],
    *,
    include_follow_up: bool = True,
) -> list[EvidenceSnapshot]:
    """Filter snapshots by provenance for grounding context."""
    if include_follow_up:
        return list(snapshots)
    return [s for s in snapshots if s.provenance == EvidenceProvenance.ORIGINAL_RUN]
