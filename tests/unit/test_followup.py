"""Focused unit tests for request-bound follow-up agent (scripted, no paid calls)."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage

from casepilot.backend.client import SimulatorClient
from casepilot.backend.db import BackendStore
from casepilot.backend.evidence import LEGACY_DISCLOSURE, NOTE_READ_ONLY, SAFE_UNCERTAINTY
from casepilot.backend.followup import (
    CONTEXT_TRUNCATION_NOTICE,
    build_user_message,
    established_snapshots_for_run,
    ground_follow_up_output,
    run_follow_up,
)
from casepilot.backend.hooks import TEST_MODE_ENV, clear_test_hooks
from casepilot.backend.scripted import ScriptedChatModel
from casepilot.shared.config import get_settings
from casepilot.shared.contracts import (
    CaseRecord,
    Diagnosis,
    DocumentListResponse,
    EventListResponse,
    Evidence,
    EvidenceProvenance,
    EvidenceSnapshot,
    FollowUpAgentOutput,
    FollowUpAnswer,
    InterpretationBlock,
    InterpretationKind,
    InvestigationResult,
    NextAction,
    RunStatus,
    SimulatorRoutes,
    SourceFactClaim,
    TurnStatus,
)
from casepilot.simulator.db import connect
from casepilot.simulator.seed import seed
from casepilot.simulator.store import SimulatorStore

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CASE_ID = "CASE-001"
TOKEN_SIM = "test-sim-token"
TOKEN_OP = "test-op-token"


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
            return (
                case,
                DocumentListResponse(case_id=CASE_ID, documents=store.list_documents(CASE_ID)),
                EventListResponse(case_id=CASE_ID, events=store.list_events(CASE_ID)),
            )
        finally:
            conn.close()


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


def _result(**overrides: object) -> InvestigationResult:
    base: dict[str, object] = dict(
        run_id="run-1",
        case_id=CASE_ID,
        status=RunStatus.COMPLETED,
        diagnosis=Diagnosis.MISSING_DOCUMENT,
        diagnosis_summary="missing doc",
        explanation="doc missing",
        evidence=[
            Evidence(
                source_system="documents",
                record_id="DOC-001-A",
                field="status",
                fact="missing",
                retrieval_ref="get_documents",
            )
        ],
        next_action=NextAction.ESCALATE_TO_HUMAN,
        next_action_explanation="escalate",
        approval_required=False,
    )
    base.update(overrides)
    return InvestigationResult(**base)  # type: ignore[arg-type]


def _seed_run(
    store: BackendStore,
    *,
    run_id: str = "run-1",
    status: RunStatus = RunStatus.COMPLETED,
    result: InvestigationResult | None = None,
) -> None:
    store.insert_run(
        run_id=run_id,
        case_id=CASE_ID,
        objective="why stalled",
        status=status,
        result=result if result is not None else _result(run_id=run_id, status=status),
        completed_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _seed_snapshot(
    store: BackendStore,
    *,
    run_id: str = "run-1",
    snapshot_id: str = "snap-doc-status",
    record_id: str = "DOC-001-A",
    field_path: str = "status",
    value: object = "missing",
    provenance: EvidenceProvenance = EvidenceProvenance.ORIGINAL_RUN,
) -> EvidenceSnapshot:
    return store.insert_evidence_snapshot(
        run_id=run_id,
        snapshot_id=snapshot_id,
        source_system="documents",
        record_id=record_id,
        field_path=field_path,
        value=value,
        retrieval_ref="srv:test",
        provenance=provenance,
        retrieved_at=datetime(2024, 1, 1, 12, tzinfo=UTC),
    )


def _follow_up_msgs(
    output: FollowUpAgentOutput, *, tools: list[str] | None = None
) -> list[AIMessage]:
    msgs: list[AIMessage] = []
    for i, name in enumerate(tools or []):
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{"name": name, "args": {}, "id": f"t{i}", "type": "tool_call"}],
            )
        )
    msgs.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "FollowUpAgentOutput",
                    "args": output.model_dump(mode="json"),
                    "id": "final",
                    "type": "tool_call",
                }
            ],
        )
    )
    return msgs


def test_ground_happy_path_exact_server_value():
    when = datetime(2024, 2, 1, tzinfo=UTC)
    catalog = {
        "snap-1": EvidenceSnapshot(
            snapshot_id="snap-1",
            run_id="run-1",
            source_system="documents",
            record_id="DOC-1",
            field_path="status",
            value="missing",
            retrieval_ref="srv:1",
            retrieved_at=when,
            provenance=EvidenceProvenance.ORIGINAL_RUN,
        )
    }
    out = ground_follow_up_output(
        FollowUpAgentOutput(
            source_facts=[
                SourceFactClaim(local_id="f1", snapshot_id="snap-1", field_path="status")
            ],
            interpretations=[
                InterpretationBlock(
                    text="documents received",  # contradictory prose — inference only
                    cites=["f1"],
                    kind=InterpretationKind.INFERENCE,
                )
            ],
        ),
        catalog,
    )
    assert out.source_facts[0].value == "missing"
    assert out.source_facts[0].retrieved_at == when
    assert out.interpretations[0].text == "documents received"
    assert out.interpretations[0].kind is InterpretationKind.INFERENCE


def test_ground_rejects_bad_citation_and_notes_only():
    catalog = {
        "snap-1": EvidenceSnapshot(
            snapshot_id="snap-1",
            run_id="r",
            source_system="cases",
            record_id=CASE_ID,
            field_path="status",
            value="stalled",
            retrieval_ref="srv:x",
            retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
            provenance=EvidenceProvenance.ORIGINAL_RUN,
        )
    }
    bad = ground_follow_up_output(
        FollowUpAgentOutput(
            source_facts=[SourceFactClaim(local_id="f1", snapshot_id="nope", field_path="status")]
        ),
        catalog,
    )
    assert bad.source_facts == []
    assert bad.notes == [SAFE_UNCERTAINTY]

    path_mismatch = ground_follow_up_output(
        FollowUpAgentOutput(
            source_facts=[
                SourceFactClaim(local_id="f1", snapshot_id="snap-1", field_path="symptom")
            ]
        ),
        catalog,
    )
    assert path_mismatch.notes == [SAFE_UNCERTAINTY]

    no_cites = ground_follow_up_output(
        FollowUpAgentOutput(
            source_facts=[
                SourceFactClaim(local_id="f1", snapshot_id="snap-1", field_path="status")
            ],
            interpretations=[
                InterpretationBlock(text="guess", cites=[], kind=InterpretationKind.INFERENCE)
            ],
        ),
        catalog,
    )
    assert no_cites.notes == [SAFE_UNCERTAINTY]

    # Free-text notes rejected (no allowlisted code).
    lie = ground_follow_up_output(
        FollowUpAgentOutput(notes=["All documents received."]),
        {},
    )
    assert lie.notes == [SAFE_UNCERTAINTY]
    assert lie.source_facts == []

    # Allowlisted code → server-owned boundary text.
    notes = ground_follow_up_output(
        FollowUpAgentOutput(notes=["read_only"]),
        {},
    )
    assert notes.notes == [NOTE_READ_ONLY]
    assert notes.source_facts == []


def test_legacy_and_verified_established(tmp_path: Path):
    store = BackendStore(tmp_path / "b.sqlite3")
    _seed_run(store)
    run = store.get_run("run-1")
    assert run is not None
    established, notes = established_snapshots_for_run(store, run)
    assert established == []
    assert LEGACY_DISCLOSURE in notes

    _seed_snapshot(store)
    run = store.get_run("run-1")
    assert run is not None
    established, notes = established_snapshots_for_run(store, run)
    assert any(s.snapshot_id == "snap-doc-status" for s in established)
    store.close()


def test_indeterminate_without_accepted_evidence_excludes_original_snaps(tmp_path: Path):
    store = BackendStore(tmp_path / "b.sqlite3")
    result = _result(
        status=RunStatus.INDETERMINATE,
        diagnosis=Diagnosis.INDETERMINATE,
        uncertainty="Rejected unsupported evidence: bad field",
        evidence=[],
    )
    _seed_run(store, status=RunStatus.INDETERMINATE, result=result)
    _seed_snapshot(store, provenance=EvidenceProvenance.ORIGINAL_RUN)
    run = store.get_run("run-1")
    assert run is not None
    established, notes = established_snapshots_for_run(store, run)
    assert established == []
    assert any("INDETERMINATE" in n for n in notes)
    assert LEGACY_DISCLOSURE in notes
    store.close()


def test_context_drops_oldest_turns(tmp_path: Path):
    store = BackendStore(tmp_path / "b.sqlite3")
    _seed_run(store)
    _seed_snapshot(store)
    run = store.get_run("run-1")
    assert run is not None

    answer = FollowUpAnswer(notes=["ok"])
    for i in range(6):
        begun = store.begin_send(
            run_id="run-1",
            question=f"q-{i}-" + ("x" * 200),
            idempotency_key=f"k-{i}",
        )
        store.complete_attempt(begun.attempt.attempt_id, answer=answer)

    current = store.begin_send(run_id="run-1", question="latest?", idempotency_key="k-cur")
    text, truncated = build_user_message(
        run=run,
        established=store.list_evidence_snapshots("run-1"),
        legacy_notes=[],
        turns=store.get_conversation("run-1"),
        turn_id=current.turn.turn_id,
        question="latest?",
        budget_tokens=3500,
    )
    assert truncated is True
    assert CONTEXT_TRUNCATION_NOTICE in text
    assert "latest?" in text
    store.close()


@pytest.mark.asyncio
async def test_run_follow_up_test_mode_no_script_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Follow-up default factory must not fall through to paid provider in test mode."""
    monkeypatch.setenv(TEST_MODE_ENV, "1")
    clear_test_hooks()
    settings = _settings(tmp_path, openrouter_api_key="sk-would-be-paid")
    store = BackendStore(settings.backend_db_path)
    _seed_run(store)
    _seed_snapshot(store, snapshot_id="snap-doc-status", value="missing")
    begun = store.begin_send(
        run_id="run-1", question="Is the document missing?", idempotency_key="send-fc"
    )
    http = _mock_simulator()
    await run_follow_up(
        run_id="run-1",
        turn_id=begun.turn.turn_id,
        attempt_id=begun.attempt.attempt_id,
        store=store,
        client=SimulatorClient(settings, client=http),
        settings=settings,
        # default_model_factory path
    )
    turn = store.get_turn(begun.turn.turn_id)
    assert turn is not None
    assert turn.status == TurnStatus.FAILED
    assert turn.attempts
    assert "refusing live paid provider fallback" in (turn.attempts[-1].error or "")
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_run_follow_up_succeeds_with_scripted_citation(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    _seed_run(store)
    _seed_snapshot(store, snapshot_id="snap-doc-status", value="missing")

    begun = store.begin_send(
        run_id="run-1", question="Is the document missing?", idempotency_key="send-1"
    )
    output = FollowUpAgentOutput(
        source_facts=[
            SourceFactClaim(local_id="f1", snapshot_id="snap-doc-status", field_path="status")
        ],
        interpretations=[
            InterpretationBlock(
                text="The cited status indicates a gap.",
                cites=["f1"],
                kind=InterpretationKind.INFERENCE,
            )
        ],
    )

    def factory(_s):
        return ScriptedChatModel(responses=_follow_up_msgs(output))

    http = _mock_simulator()
    await run_follow_up(
        run_id="run-1",
        turn_id=begun.turn.turn_id,
        attempt_id=begun.attempt.attempt_id,
        store=store,
        client=SimulatorClient(settings, client=http),
        settings=settings,
        model_factory=factory,
    )

    turn = store.get_turn(begun.turn.turn_id)
    assert turn is not None
    assert turn.status is TurnStatus.SUCCEEDED
    att = turn.attempts[-1]
    assert att.status is TurnStatus.SUCCEEDED
    assert att.answer is not None
    assert att.answer.source_facts[0].value == "missing"
    assert att.answer.source_facts[0].provenance is EvidenceProvenance.ORIGINAL_RUN
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_run_follow_up_bad_citation_safe_uncertainty(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    _seed_run(store)
    _seed_snapshot(store)

    begun = store.begin_send(run_id="run-1", question="?", idempotency_key="send-bad")
    output = FollowUpAgentOutput(
        source_facts=[SourceFactClaim(local_id="f1", snapshot_id="invented", field_path="status")],
        interpretations=[
            InterpretationBlock(text="sure thing", cites=["f1"], kind=InterpretationKind.INFERENCE)
        ],
    )

    def factory(_s):
        return ScriptedChatModel(responses=_follow_up_msgs(output))

    http = _mock_simulator()
    await run_follow_up(
        run_id="run-1",
        turn_id=begun.turn.turn_id,
        attempt_id=begun.attempt.attempt_id,
        store=store,
        client=SimulatorClient(settings, client=http),
        settings=settings,
        model_factory=factory,
    )
    turn = store.get_turn(begun.turn.turn_id)
    assert turn is not None and turn.status is TurnStatus.SUCCEEDED
    assert turn.attempts[-1].answer is not None
    assert turn.attempts[-1].answer.notes == [SAFE_UNCERTAINTY]
    assert turn.attempts[-1].answer.source_facts == []
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_run_follow_up_tool_retrieval_persists_follow_up_snapshots(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    _seed_run(store)

    begun = store.begin_send(run_id="run-1", question="refresh case?", idempotency_key="send-tools")
    output = FollowUpAgentOutput(notes=["read_only"])

    def factory(_s):
        return ScriptedChatModel(responses=_follow_up_msgs(output, tools=["get_case"]))

    http = _mock_simulator()
    await run_follow_up(
        run_id="run-1",
        turn_id=begun.turn.turn_id,
        attempt_id=begun.attempt.attempt_id,
        store=store,
        client=SimulatorClient(settings, client=http),
        settings=settings,
        model_factory=factory,
    )
    snaps = store.list_evidence_snapshots("run-1")
    assert snaps
    assert all(s.provenance is EvidenceProvenance.FOLLOW_UP for s in snaps)
    assert all(s.attempt_id == begun.attempt.attempt_id for s in snaps)
    turn = store.get_turn(begun.turn.turn_id)
    assert turn is not None and turn.status is TurnStatus.SUCCEEDED
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_run_follow_up_timeout_fails_attempt(tmp_path: Path):
    settings = _settings(tmp_path, investigation_timeout_seconds=0.01)
    store = BackendStore(settings.backend_db_path)
    _seed_run(store)
    begun = store.begin_send(run_id="run-1", question="?", idempotency_key="send-to")

    class SlowModel(ScriptedChatModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
            await asyncio.sleep(1.0)
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def factory(_s):
        return SlowModel(responses=_follow_up_msgs(FollowUpAgentOutput(notes=["read_only"])))

    http = _mock_simulator()
    await run_follow_up(
        run_id="run-1",
        turn_id=begun.turn.turn_id,
        attempt_id=begun.attempt.attempt_id,
        store=store,
        client=SimulatorClient(settings, client=http),
        settings=settings,
        model_factory=factory,
    )
    turn = store.get_turn(begun.turn.turn_id)
    assert turn is not None
    assert turn.status is TurnStatus.FAILED
    assert turn.attempts[-1].error is not None
    assert "deadline" in turn.attempts[-1].error
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_run_follow_up_cancel_fails_attempt(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    _seed_run(store)
    begun = store.begin_send(run_id="run-1", question="?", idempotency_key="cancel-1")

    class CancelModel(ScriptedChatModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
            raise asyncio.CancelledError()

    def factory(_s):
        return CancelModel(responses=[])

    http = _mock_simulator()
    # LangChain may absorb CancelledError; run_follow_up must still fail the attempt.
    try:
        await run_follow_up(
            run_id="run-1",
            turn_id=begun.turn.turn_id,
            attempt_id=begun.attempt.attempt_id,
            store=store,
            client=SimulatorClient(settings, client=http),
            settings=settings,
            model_factory=factory,
        )
    except asyncio.CancelledError:
        pass
    turn = store.get_turn(begun.turn.turn_id)
    assert turn is not None
    assert turn.status is TurnStatus.FAILED
    assert turn.attempts[-1].error is not None
    assert "cancel" in turn.attempts[-1].error
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_run_follow_up_skips_non_pending(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    _seed_run(store)
    begun = store.begin_send(run_id="run-1", question="?", idempotency_key="done-1")
    store.complete_attempt(begun.attempt.attempt_id, answer=FollowUpAnswer(notes=["already done"]))

    called = {"n": 0}

    def factory(_s):
        called["n"] += 1
        return ScriptedChatModel(responses=[])

    http = _mock_simulator()
    await run_follow_up(
        run_id="run-1",
        turn_id=begun.turn.turn_id,
        attempt_id=begun.attempt.attempt_id,
        store=store,
        client=SimulatorClient(settings, client=http),
        settings=settings,
        model_factory=factory,
    )
    assert called["n"] == 0
    turn = store.get_turn(begun.turn.turn_id)
    assert turn is not None and turn.status is TurnStatus.SUCCEEDED
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_run_follow_up_indeterminate_status_in_context(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    result = _result(
        status=RunStatus.INDETERMINATE,
        diagnosis=Diagnosis.INDETERMINATE,
        uncertainty="Rejected unsupported evidence: bad field",
        evidence=[],
    )
    _seed_run(store, status=RunStatus.INDETERMINATE, result=result)
    run = store.get_run("run-1")
    assert run is not None
    established, notes = established_snapshots_for_run(store, run)
    text, _ = build_user_message(
        run=run,
        established=established,
        legacy_notes=notes,
        turns=[],
        turn_id="t-x",
        question="why uncertain?",
    )
    assert "indeterminate" in text
    assert any("INDETERMINATE" in n for n in notes)

    begun = store.begin_send(run_id="run-1", question="why uncertain?", idempotency_key="ind-1")

    def factory(_s):
        return ScriptedChatModel(
            responses=_follow_up_msgs(FollowUpAgentOutput(notes=["read_only"]))
        )

    http = _mock_simulator()
    await run_follow_up(
        run_id="run-1",
        turn_id=begun.turn.turn_id,
        attempt_id=begun.attempt.attempt_id,
        store=store,
        client=SimulatorClient(settings, client=http),
        settings=settings,
        model_factory=factory,
    )
    turn = store.get_turn(begun.turn.turn_id)
    assert turn is not None and turn.status is TurnStatus.SUCCEEDED
    store.close()
    await http.aclose()
