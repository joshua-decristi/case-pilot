"""Distinct tests for field-level evidence snapshots and follow-up rendering."""

from __future__ import annotations

import sys
import tempfile
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import httpx
import pytest

from casepilot.backend.agent import build_follow_up_read_tools, run_investigation
from casepilot.backend.client import SimulatorClient
from casepilot.backend.db import BackendStore
from casepilot.backend.evidence import (
    LEGACY_DISCLOSURE,
    NOTE_READ_ONLY,
    SAFE_UNCERTAINTY,
    RecordCapture,
    RetrievalContext,
    RetrievedIndex,
    attribute_run_evidence,
    established_snapshots,
    field_value_at,
    iter_leaf_fields,
    persist_index_snapshots,
    redact_evidence_payload,
    render_follow_up_answer,
    render_source_fact,
    resolve_server_notes,
    snapshots_by_id,
    validate_evidence,
)
from casepilot.backend.scripted import ScriptedChatModel
from casepilot.backend.trace import TraceWriter
from casepilot.shared.config import get_settings
from casepilot.shared.contracts import (
    CaseRecord,
    DocumentListResponse,
    EventListResponse,
    Evidence,
    EvidenceProvenance,
    EvidenceSnapshot,
    FollowUpAgentOutput,
    InterpretationBlock,
    InterpretationKind,
    RunStatus,
    SimulatorRoutes,
    SourceFactClaim,
)
from casepilot.simulator.db import connect
from casepilot.simulator.seed import seed
from casepilot.simulator.store import SimulatorStore

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from tests.helpers import _finding, _tool_then_finding  # noqa: E402

CASE_ID = "CASE-001"
TOKEN_SIM = "test-sim-token"
TOKEN_OP = "test-op-token"


@cache
def _case001_bundle() -> tuple[CaseRecord, DocumentListResponse, EventListResponse]:
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "sim.sqlite3"
        assert seed(db_path=db_path, reset=True) == 0
        conn = connect(db_path)
        try:
            store = SimulatorStore(conn)
            case = store.get_case(CASE_ID)
            assert case is not None
            docs = store.list_documents(CASE_ID)
            events = store.list_events(CASE_ID)
            return (
                case,
                DocumentListResponse(case_id=CASE_ID, documents=docs),
                EventListResponse(case_id=CASE_ID, events=events),
            )
        finally:
            conn.close()


def _settings(tmp_path: Path, **extra: object):
    base: dict[str, object] = {
        "_env_file": None,
        "simulator_service_token": TOKEN_SIM,
        "operator_token": TOKEN_OP,
        "backend_db_path": tmp_path / "backend.sqlite3",
        "trace_dir": tmp_path / "traces",
        "simulator_base_url": "http://simulator.test",
        "openrouter_api_key": None,
        "investigation_timeout_seconds": 30.0,
        "max_tool_calls": 12,
        "max_model_turns": 8,
    }
    base.update(extra)
    return get_settings(**base)


def _mock_simulator() -> httpx.AsyncClient:
    case, docs, events = _case001_bundle()

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == SimulatorRoutes.CASE.format(case_id=CASE_ID):
            return httpx.Response(200, json=case.model_dump(mode="json"))
        if path == SimulatorRoutes.DOCUMENTS.format(case_id=CASE_ID):
            return httpx.Response(200, json=docs.model_dump(mode="json"))
        if path == SimulatorRoutes.EVENTS.format(case_id=CASE_ID):
            return httpx.Response(200, json=events.model_dump(mode="json"))
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://simulator.test"
    )


def _seed_run(store: BackendStore, run_id: str = "run-1") -> None:
    from casepilot.shared.contracts import Diagnosis, InvestigationResult, NextAction

    store.insert_run(
        run_id=run_id,
        case_id=CASE_ID,
        objective="test",
        status=RunStatus.COMPLETED,
        result=InvestigationResult(
            run_id=run_id,
            case_id=CASE_ID,
            status=RunStatus.COMPLETED,
            diagnosis=Diagnosis.MISSING_DOCUMENT,
            diagnosis_summary="s",
            explanation="e",
            evidence=[],
            next_action=NextAction.NO_ACTION_REQUIRED,
            next_action_explanation="n",
        ),
    )


def test_note_case_captures_redacted_record():
    case = _case001_bundle()[0]
    idx = RetrievedIndex(
        context=RetrievalContext(
            run_id="run-a",
            case_id=CASE_ID,
            provenance=EvidenceProvenance.ORIGINAL_RUN,
        )
    )
    idx.note_case(case)
    assert len(idx.captures) == 1
    cap = idx.captures[0]
    assert cap.source_system == "cases"
    assert cap.record_id == CASE_ID
    assert cap.tool_name == "get_case"
    assert cap.payload["case_id"] == CASE_ID
    assert idx.raw_snapshots[0]["record_id"] == CASE_ID


def test_iter_leaf_fields_flattens_nested():
    leaves = dict(iter_leaf_fields({"status": "x", "details": {"missing": "y", "n": 1}}))
    assert leaves == {"status": "x", "details.missing": "y", "details.n": 1}


def test_snapshot_redacts_secrets():
    out = redact_evidence_payload(
        {"status": "open", "api_key": "super-secret", "nested": {"token": "t-1"}}
    )
    assert out["status"] == "open"
    assert out["api_key"] == "[REDACTED]"
    assert out["nested"]["token"] == "[REDACTED]"


def test_validate_evidence_still_accepts_fixture_paths():
    case, docs, events = _case001_bundle()
    idx = RetrievedIndex()
    idx.note_case(case)
    idx.note_documents(docs.documents)
    idx.note_events(events.events)
    ok, errs = validate_evidence(
        [
            Evidence(
                source_system="events",
                record_id="EVT-001-A",
                field="details.missing",
                fact="proof_of_income",
                retrieval_ref="get_events",
            )
        ],
        idx,
    )
    assert not errs and len(ok) == 1


def test_persist_index_snapshots_writes_field_rows(tmp_path: Path):
    store = BackendStore(tmp_path / "b.sqlite3")
    _seed_run(store, "run-p")
    idx = RetrievedIndex(
        context=RetrievalContext(
            run_id="run-p",
            case_id=CASE_ID,
            provenance=EvidenceProvenance.ORIGINAL_RUN,
        )
    )
    idx.captures.append(
        RecordCapture(
            source_system="documents",
            record_id="DOC-1",
            tool_name="get_documents",
            payload={"status": "missing", "fields": {"name": "A"}},
            retrieved_at=datetime(2024, 1, 2, tzinfo=UTC),
        )
    )
    written = persist_index_snapshots(store, idx)
    assert len(written) == 2
    paths = {s.field_path for s in written}
    assert paths == {"status", "fields.name"}
    listed = store.list_evidence_snapshots("run-p")
    assert len(listed) == 2
    status = next(s for s in listed if s.field_path == "status")
    assert status.value == "missing"
    assert status.provenance is EvidenceProvenance.ORIGINAL_RUN
    assert status.retrieval_ref == "get_documents:DOC-1:status"
    assert store.get_evidence_snapshot(status.snapshot_id) is not None
    store.close()


def test_persist_skips_when_run_missing(tmp_path: Path):
    store = BackendStore(tmp_path / "b.sqlite3")
    idx = RetrievedIndex(
        context=RetrievalContext(run_id="nope", case_id=CASE_ID),
        captures=[
            RecordCapture(
                source_system="cases",
                record_id=CASE_ID,
                tool_name="get_case",
                payload={"status": "x"},
            )
        ],
    )
    assert persist_index_snapshots(store, idx) == []
    store.close()


def test_render_source_fact_uses_snapshot_value_only():
    snap = EvidenceSnapshot(
        snapshot_id="snap-1",
        run_id="r",
        source_system="documents",
        record_id="DOC-X",
        field_path="status",
        value="missing",
        retrieval_ref="get_documents:DOC-X:status",
        retrieved_at=datetime(2024, 1, 2, tzinfo=UTC),
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    catalog = snapshots_by_id([snap])
    claim = SourceFactClaim(local_id="f1", snapshot_id="snap-1", field_path="status")
    rendered = render_source_fact(claim, catalog)
    assert rendered is not None
    assert rendered.value == "missing"
    assert rendered.local_id == "f1"
    assert rendered.provenance is EvidenceProvenance.ORIGINAL_RUN
    # wrong path for this snapshot id
    assert (
        render_source_fact(
            SourceFactClaim(local_id="f2", snapshot_id="snap-1", field_path="other"),
            catalog,
        )
        is None
    )


def test_follow_up_render_happy_path_model_prose_not_fact_value():
    snap = EvidenceSnapshot(
        snapshot_id="snap-1",
        run_id="r",
        attempt_id="a1",
        source_system="documents",
        record_id="DOC-X",
        field_path="status",
        value="missing",
        retrieval_ref="ref",
        retrieved_at=datetime(2024, 1, 2, tzinfo=UTC),
        provenance=EvidenceProvenance.FOLLOW_UP,
    )
    output = FollowUpAgentOutput(
        source_facts=[SourceFactClaim(local_id="s1", snapshot_id="snap-1", field_path="status")],
        interpretations=[
            InterpretationBlock(
                text="documents received",  # contradictory model prose
                cites=["s1"],
                kind=InterpretationKind.INFERENCE,
            )
        ],
    )
    result = render_follow_up_answer(output, [snap])
    assert result.ok
    assert result.answer.source_facts[0].value == "missing"
    assert result.answer.interpretations[0].kind is InterpretationKind.INFERENCE
    assert result.answer.interpretations[0].text == "documents received"


def test_follow_up_rejects_bad_citation_with_safe_uncertainty():
    snap = EvidenceSnapshot(
        snapshot_id="snap-1",
        run_id="r",
        source_system="cases",
        record_id=CASE_ID,
        field_path="status",
        value="stalled",
        retrieval_ref="ref",
        retrieved_at=datetime.now(UTC),
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    bad = FollowUpAgentOutput(
        source_facts=[SourceFactClaim(local_id="s1", snapshot_id="nope", field_path="status")]
    )
    out = render_follow_up_answer(bad, [snap])
    assert not out.ok
    assert out.answer.source_facts == []
    assert SAFE_UNCERTAINTY in out.answer.notes

    no_cites = FollowUpAgentOutput(
        source_facts=[SourceFactClaim(local_id="s1", snapshot_id="snap-1", field_path="status")],
        interpretations=[
            InterpretationBlock(text="guess", cites=[], kind=InterpretationKind.INFERENCE)
        ],
    )
    out2 = render_follow_up_answer(no_cites, [snap])
    assert not out2.ok


def test_notes_only_allowlisted_code_needs_no_citations():
    out = render_follow_up_answer(
        FollowUpAgentOutput(notes=["read_only"]),
        [],
    )
    assert out.ok
    assert out.answer.notes == [NOTE_READ_ONLY]
    assert out.answer.source_facts == []


def test_notes_only_factual_lie_rejected():
    """Unrestricted model prose in notes must not become operator-visible."""
    lie = "All required documents have been received."
    out = render_follow_up_answer(FollowUpAgentOutput(notes=[lie]), [])
    assert not out.ok
    assert out.answer.source_facts == []
    assert out.answer.interpretations == []
    assert out.answer.notes == [SAFE_UNCERTAINTY]
    assert lie not in out.answer.notes


def test_contradictory_model_notes_discarded_beside_valid_facts():
    snap = EvidenceSnapshot(
        snapshot_id="snap-1",
        run_id="r",
        source_system="documents",
        record_id="DOC-X",
        field_path="status",
        value="missing",
        retrieval_ref="ref",
        retrieved_at=datetime(2024, 1, 2, tzinfo=UTC),
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    lie = "Documents were received and the case is complete."
    output = FollowUpAgentOutput(
        source_facts=[SourceFactClaim(local_id="s1", snapshot_id="snap-1", field_path="status")],
        interpretations=[
            InterpretationBlock(
                text="status indicates missing",
                cites=["s1"],
                kind=InterpretationKind.INFERENCE,
            )
        ],
        notes=[lie],
    )
    result = render_follow_up_answer(output, [snap])
    assert result.ok
    assert result.answer.source_facts[0].value == "missing"
    assert result.answer.notes == []
    assert lie not in result.answer.notes


def test_server_notes_kwarg_and_code_resolution():
    assert resolve_server_notes(["read_only", "bogus", "read_only"]) == [NOTE_READ_ONLY]
    out = render_follow_up_answer(FollowUpAgentOutput(), [], server_notes=["read_only"])
    assert out.ok and out.answer.notes == [NOTE_READ_ONLY]


def test_legacy_evidence_discloses_missing_snapshot_backing():
    ev = Evidence(
        source_system="documents",
        record_id="DOC-001-A",
        field="status",
        fact="received",
        retrieval_ref="get_documents",
    )
    attributed = attribute_run_evidence([ev], snapshots=[])
    assert attributed[0].is_verified_snapshot is False
    assert attributed[0].disclosure == LEGACY_DISCLOSURE
    assert attributed[0].snapshot_retrieved_at is None


def test_legacy_evidence_links_when_snapshot_exists():
    snap = EvidenceSnapshot(
        snapshot_id="snap-1",
        run_id="r",
        source_system="documents",
        record_id="DOC-001-A",
        field_path="status",
        value="received",
        retrieval_ref="ref",
        retrieved_at=datetime(2024, 6, 1, tzinfo=UTC),
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    ev = Evidence(
        source_system="documents",
        record_id="DOC-001-A",
        field="status",
        fact="model wrote anything",
        retrieval_ref="get_documents",
    )
    attributed = attribute_run_evidence([ev], [snap])
    assert attributed[0].is_verified_snapshot is True
    assert attributed[0].rendered_value == "received"
    assert attributed[0].matching_snapshot_id == "snap-1"
    assert attributed[0].disclosure is None


def test_established_snapshots_origin_filter():
    a = EvidenceSnapshot(
        snapshot_id="a",
        run_id="r",
        source_system="cases",
        record_id=CASE_ID,
        field_path="status",
        value="a",
        retrieval_ref="r",
        retrieved_at=datetime.now(UTC),
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    b = EvidenceSnapshot(
        snapshot_id="b",
        run_id="r",
        source_system="cases",
        record_id=CASE_ID,
        field_path="status",
        value="b",
        retrieval_ref="r",
        retrieved_at=datetime.now(UTC),
        provenance=EvidenceProvenance.FOLLOW_UP,
        attempt_id="t",
    )
    assert established_snapshots([a, b], include_follow_up=False) == [a]


def test_follow_up_read_tools_set_provenance(tmp_path: Path):
    idx = RetrievedIndex()
    store = BackendStore(tmp_path / "fu.sqlite3")
    tools = build_follow_up_read_tools(
        client=object(),  # type: ignore[arg-type]
        case_id=CASE_ID,
        index=idx,
        trace=TraceWriter(tmp_path, "run-f"),
        run_id="run-f",
        store=store,
        attempt_id="attempt-9",
    )
    assert idx.context is not None
    assert idx.context.provenance is EvidenceProvenance.FOLLOW_UP
    assert idx.context.attempt_id == "attempt-9"
    assert len(tools) == 3
    store.close()


def test_field_value_at_nested():
    assert field_value_at("details.missing", {"details": {"missing": "x"}}) == "x"


@pytest.mark.asyncio
async def test_investigation_persists_snapshots_to_backend_store(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    finding = _finding()

    def factory(_s):
        return ScriptedChatModel(
            responses=_tool_then_finding(finding, tools=["get_case", "get_documents", "get_events"])
        )

    outcome = await run_investigation(
        settings=settings,
        store=store,
        client=client,
        case_id=CASE_ID,
        objective="Why stalled?",
        model_factory=factory,
    )
    assert outcome.result.status == RunStatus.COMPLETED
    snaps = store.list_evidence_snapshots(outcome.result.run_id)
    assert len(snaps) >= 3
    assert all(s.provenance is EvidenceProvenance.ORIGINAL_RUN for s in snaps)
    assert all(s.run_id == outcome.result.run_id for s in snaps)
    # field-level rows with server retrieval_ref
    assert any(s.field_path == "status" for s in snaps)
    case_status = next(s for s in snaps if s.source_system == "cases" and s.field_path == "status")
    rendered = render_source_fact(
        SourceFactClaim(
            local_id="c1",
            snapshot_id=case_status.snapshot_id,
            field_path="status",
        ),
        snapshots_by_id(snaps),
    )
    assert rendered is not None
    assert rendered.value == case_status.value
    store.close()
    await http.aclose()
