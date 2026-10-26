"""Runtime follow_up_output + model_turn_batches diagnostics (no paid calls)."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from casepilot.backend.client import SimulatorClient
from casepilot.backend.db import BackendStore
from casepilot.backend.evidence import NOTE_READ_ONLY, SAFE_UNCERTAINTY, FollowUpRenderRejectCode
from casepilot.backend.followup import run_follow_up
from casepilot.backend.hooks import clear_test_hooks
from casepilot.backend.pilot_observability import (
    EVENT_FOLLOW_UP_OUTPUT,
    EVENT_MODEL_TURN_BATCHES,
    EVENT_TOOL_ATTEMPT_BATCH,
    EVENT_UNKNOWN_TOOL,
    UNTRUSTED_MODEL_OUTPUT_LABEL,
    capture_model_turn_batches,
    emit_follow_up_output,
    follow_up_output_refs,
    model_turn_batches,
    resolve_ai_messages,
)
from casepilot.backend.scripted import ScriptedChatModel
from casepilot.backend.trace import TraceWriter
from casepilot.shared.config import Settings, get_settings
from casepilot.shared.contracts import (
    CaseRecord,
    Diagnosis,
    DocumentListResponse,
    EventListResponse,
    Evidence,
    EvidenceProvenance,
    FollowUpAgentOutput,
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


def _settings(tmp_path: Path, **extra: object) -> Settings:
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


def _seed_run(store: BackendStore, *, run_id: str = "run-1") -> None:
    store.insert_run(
        run_id=run_id,
        case_id=CASE_ID,
        objective="why stalled",
        status=RunStatus.COMPLETED,
        result=_result(run_id=run_id),
        completed_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _seed_snapshot(store: BackendStore, *, run_id: str = "run-1") -> None:
    store.insert_evidence_snapshot(
        run_id=run_id,
        snapshot_id="snap-doc-status",
        source_system="documents",
        record_id="DOC-001-A",
        field_path="status",
        value="missing",
        retrieval_ref="srv:test",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
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


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ----- pure helpers ---------------------------------------------------------


def test_follow_up_output_refs_schema_only_no_free_text() -> None:
    prose = "SECRET CoT: DOC-SECRET has value leaked-xyz"
    out = FollowUpAgentOutput(
        source_facts=[SourceFactClaim(local_id="s1", snapshot_id="snap-1", field_path="status")],
        interpretations=[
            InterpretationBlock(
                text=prose,
                cites=["s1"],
                kind=InterpretationKind.INFERENCE,
            )
        ],
        notes=["read_only", prose, "no_proposal", NOTE_READ_ONLY],
    )
    refs = follow_up_output_refs(out)
    assert refs["source_facts"] == [
        {"local_id": "s1", "snapshot_id": "snap-1", "field_path": "status"}
    ]
    assert refs["interpretations"] == [{"kind": "inference", "cites": ["s1"]}]
    # Allowlist codes only; free text dropped; exact server string → code once.
    assert refs["notes"] == ["read_only", "no_proposal"]
    blob = json.dumps(refs)
    assert prose not in blob
    assert "leaked-xyz" not in blob
    assert "text" not in refs["interpretations"][0]
    assert "value" not in blob


def test_emit_follow_up_output_includes_reject_code(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "traces", "run:att")
    out = FollowUpAgentOutput(
        source_facts=[SourceFactClaim(local_id="s1", snapshot_id="nope", field_path="status")]
    )
    payload = emit_follow_up_output(
        trace.emit,
        out,
        ok=False,
        reject_code=FollowUpRenderRejectCode.UNKNOWN_SNAPSHOT.value,
        case_id=CASE_ID,
        run_id="run-1",
        attempt_id="att-1",
    )
    assert payload["label"] == UNTRUSTED_MODEL_OUTPUT_LABEL
    assert payload["ok"] is False
    assert payload["reject_code"] == "unknown_snapshot"
    assert payload["source_facts"][0]["snapshot_id"] == "nope"

    rows = _read_trace(trace.path)
    assert len(rows) == 1
    assert rows[0]["event"] == EVENT_FOLLOW_UP_OUTPUT
    assert rows[0]["reject_code"] == "unknown_snapshot"
    assert rows[0]["label"] == UNTRUSTED_MODEL_OUTPUT_LABEL


def test_model_turn_batches_all_turns_names_count_only() -> None:
    messages = [
        HumanMessage(content="q"),
        AIMessage(
            content="ignore-me",
            tool_calls=[
                {
                    "name": "get_case",
                    "args": {"secret": "sk-nope"},
                    "id": "t0",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_documents",
                    "args": {},
                    "id": "t1",
                    "type": "tool_call",
                },
                {
                    "name": "FollowUpAgentOutput",
                    "args": {"notes": ["read_only"], "reasoning": "hidden"},
                    "id": "t2",
                    "type": "tool_call",
                },
            ],
        ),
        # Structured-output retry attempt
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "FollowUpAgentOutput",
                    "args": {"notes": ["safe_uncertainty"]},
                    "id": "t3",
                    "type": "tool_call",
                }
            ],
        ),
    ]
    batches = model_turn_batches(messages)
    assert batches == [
        {"count": 1, "names": ["get_case"]},
        {"count": 2, "names": ["get_documents", "FollowUpAgentOutput"]},
        {"count": 1, "names": ["FollowUpAgentOutput"]},
    ]
    blob = json.dumps(batches)
    assert "sk-nope" not in blob
    assert "hidden" not in blob
    assert "safe_uncertainty" not in blob  # args never included


def test_capture_model_turn_batches_aggregate_event(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "traces", "run-b")
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_case", "args": {}, "id": "a", "type": "tool_call"},
                {"name": "evil_write", "args": {"x": 1}, "id": "b", "type": "tool_call"},
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "FollowUpAgentOutput",
                    "args": {},
                    "id": "c",
                    "type": "tool_call",
                }
            ],
        ),
    ]
    summary = capture_model_turn_batches(trace, messages, case_id=CASE_ID, run_id="run-b")
    assert summary["turn_count"] == 2
    rows = _read_trace(trace.path)
    assert len(rows) == 1
    assert rows[0]["event"] == EVENT_MODEL_TURN_BATCHES
    assert rows[0]["turn_count"] == 2
    assert rows[0]["batches"][0]["names"] == ["get_case", "evil_write"]
    # Helper must not emit unknown_tool (callers scan once separately).
    assert not any(r["event"] == EVENT_UNKNOWN_TOOL for r in rows)


def test_resolve_ai_messages_prefers_messages_then_tracker() -> None:
    class _T:
        captured_ai_messages = [AIMessage(content="from-tracker")]

    msgs = [HumanMessage(content="h"), AIMessage(content="from-msgs")]
    assert resolve_ai_messages(msgs, _T())[1].content == "from-msgs"
    assert resolve_ai_messages(None, _T())[0].content == "from-tracker"
    assert resolve_ai_messages(None, None) == []


# ----- runtime integration --------------------------------------------------


@pytest.mark.asyncio
async def test_run_follow_up_emits_output_and_turn_batches_success(tmp_path: Path) -> None:
    clear_test_hooks()
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    _seed_run(store)
    _seed_snapshot(store)
    begun = store.begin_send(run_id="run-1", question="status?", idempotency_key="diag-ok")
    http = _mock_simulator()

    output = FollowUpAgentOutput(
        source_facts=[
            SourceFactClaim(local_id="s1", snapshot_id="snap-doc-status", field_path="status")
        ],
        interpretations=[
            InterpretationBlock(
                text="status is missing — never emit this prose",
                cites=["s1"],
                kind=InterpretationKind.INFERENCE,
            )
        ],
        notes=["read_only"],
    )

    def factory(_s: object) -> ScriptedChatModel:
        return ScriptedChatModel(responses=_follow_up_msgs(output, tools=["get_case"]))

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

    path = settings.trace_dir / f"run-1:{begun.attempt.attempt_id}.jsonl"
    rows = _read_trace(path)
    events = [r["event"] for r in rows]
    assert EVENT_MODEL_TURN_BATCHES in events
    assert EVENT_FOLLOW_UP_OUTPUT in events
    assert events.count(EVENT_FOLLOW_UP_OUTPUT) == 1
    assert events.count(EVENT_MODEL_TURN_BATCHES) == 1

    batches = next(r for r in rows if r["event"] == EVENT_MODEL_TURN_BATCHES)
    names_flat = [n for b in batches["batches"] for n in b["names"]]
    assert "get_case" in names_flat
    assert "FollowUpAgentOutput" in names_flat

    out_ev = next(r for r in rows if r["event"] == EVENT_FOLLOW_UP_OUTPUT)
    assert out_ev["label"] == UNTRUSTED_MODEL_OUTPUT_LABEL
    assert out_ev["ok"] is True
    assert "reject_code" not in out_ev or out_ev.get("reject_code") is None
    assert out_ev["source_facts"] == [
        {
            "local_id": "s1",
            "snapshot_id": "snap-doc-status",
            "field_path": "status",
        }
    ]
    assert out_ev["interpretations"] == [{"kind": "inference", "cites": ["s1"]}]
    assert out_ev["notes"] == ["read_only"]
    blob = json.dumps(out_ev)
    assert "never emit this prose" not in blob
    assert "missing" not in out_ev["source_facts"][0]  # no values on claim refs

    # Output event must appear before follow_up_end
    assert events.index(EVENT_FOLLOW_UP_OUTPUT) < events.index("follow_up_end")

    store.close()
    await http.aclose()
    clear_test_hooks()


@pytest.mark.asyncio
async def test_run_follow_up_bad_citation_emits_reject_code(tmp_path: Path) -> None:
    clear_test_hooks()
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    _seed_run(store)
    _seed_snapshot(store)
    begun = store.begin_send(run_id="run-1", question="lie?", idempotency_key="diag-bad")
    http = _mock_simulator()

    output = FollowUpAgentOutput(
        source_facts=[
            SourceFactClaim(local_id="s1", snapshot_id="no-such-snap", field_path="status")
        ],
        interpretations=[
            InterpretationBlock(
                text="invented claim text",
                cites=["s1"],
                kind=InterpretationKind.INFERENCE,
            )
        ],
    )

    def factory(_s: object) -> ScriptedChatModel:
        return ScriptedChatModel(responses=_follow_up_msgs(output))

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
    ans = turn.attempts[-1].answer
    assert ans is not None
    assert ans.notes == [SAFE_UNCERTAINTY]
    assert ans.source_facts == []

    path = settings.trace_dir / f"run-1:{begun.attempt.attempt_id}.jsonl"
    out_ev = next(r for r in _read_trace(path) if r["event"] == EVENT_FOLLOW_UP_OUTPUT)
    assert out_ev["ok"] is False
    assert out_ev["reject_code"] == FollowUpRenderRejectCode.UNKNOWN_SNAPSHOT.value
    assert out_ev["source_facts"][0]["snapshot_id"] == "no-such-snap"
    assert "invented claim text" not in json.dumps(out_ev)

    store.close()
    await http.aclose()
    clear_test_hooks()


@pytest.mark.asyncio
async def test_run_follow_up_error_emits_turn_batches_no_dup_unknown(
    tmp_path: Path,
) -> None:
    clear_test_hooks()
    settings = _settings(tmp_path, max_tool_calls=1, max_model_turns=8)
    store = BackendStore(settings.backend_db_path)
    _seed_run(store)
    begun = store.begin_send(run_id="run-1", question="over?", idempotency_key="diag-ol")
    http = _mock_simulator()

    # First turn one allowed tool; second turn two tools → tool-call limit abort.
    def factory(_s: object) -> ScriptedChatModel:
        return ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_case",
                            "args": {},
                            "id": "t0",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_documents",
                            "args": {},
                            "id": "t1",
                            "type": "tool_call",
                        },
                        {
                            "name": "create_follow_up_task",
                            "args": {"title": "x"},
                            "id": "t2",
                            "type": "tool_call",
                        },
                    ],
                ),
            ]
        )

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
    assert turn is not None and turn.status is TurnStatus.FAILED

    path = settings.trace_dir / f"run-1:{begun.attempt.attempt_id}.jsonl"
    rows = _read_trace(path)
    events = [r["event"] for r in rows]

    assert EVENT_MODEL_TURN_BATCHES in events
    assert EVENT_TOOL_ATTEMPT_BATCH in events
    assert events.count(EVENT_MODEL_TURN_BATCHES) == 1
    assert events.count(EVENT_TOOL_ATTEMPT_BATCH) == 1

    batches = next(r for r in rows if r["event"] == EVENT_MODEL_TURN_BATCHES)
    assert batches["turn_count"] >= 1
    # Unknown tool once only (no duplicate records).
    unknown = [r for r in rows if r["event"] == EVENT_UNKNOWN_TOOL]
    assert len(unknown) == 1
    assert unknown[0]["name"] == "create_follow_up_task"
    # No follow_up_output without structured response.
    assert EVENT_FOLLOW_UP_OUTPUT not in events

    store.close()
    await http.aclose()
    clear_test_hooks()
