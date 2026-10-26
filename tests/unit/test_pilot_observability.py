"""Unit tests for backend-owned pilot observability collector (no paid calls)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from casepilot.backend.db import BackendStore
from casepilot.backend.pilot_observability import (
    EVENT_FORBIDDEN,
    EVENT_PRE_TURN,
    EVENT_TOOL,
    EVENT_TOOL_ATTEMPT_BATCH,
    EVENT_UNKNOWN_TOOL,
    EXEC_STATUS_NOT_EXECUTED,
    EXEC_STATUS_UNKNOWN,
    PilotObservability,
    allowed_tool_name_set,
    capture_leaf_exposure_facts,
    capture_tool_attempts_on_error,
    emit_forbidden_attempt,
    emit_pre_turn_context,
    emit_unknown_tool_attempt,
    proposal_facts_from_audit,
    scan_unknown_tool_calls,
    task_facts_from_audit,
    tool_exposure_facts,
)
from casepilot.backend.trace import TraceWriter
from casepilot.evaluation.pilot_coverage import (
    RetrievalClass,
    TargetFact,
    classify_retrieval,
)
from casepilot.evaluation.pilot_safety import (
    CitationFact,
    PhaseSafetyEvidence,
    assess_phase_safety,
)
from casepilot.shared.contracts import (
    EvidenceProvenance,
    FollowUpTaskPayload,
    NextAction,
    RunStatus,
)
from casepilot.simulator.db import apply_schema, connect


def _store(tmp_path: Path) -> BackendStore:
    return BackendStore(tmp_path / "backend.sqlite3")


def _sim_db(
    tmp_path: Path,
    *,
    cases: tuple[str, ...] = ("CASE-008",),
    documents: tuple[tuple[str, str], ...] = (),
    events: tuple[tuple[str, str], ...] = (),
) -> Path:
    """Seed simulator SQLite. documents/events are (record_id, case_id)."""
    path = tmp_path / "simulator.sqlite3"
    conn = connect(path)
    apply_schema(conn)
    now = datetime.now(UTC).isoformat()
    for case_id in cases:
        conn.execute(
            "INSERT INTO cases (case_id, status, symptom, application_fields, "
            "required_documents, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (case_id, "open", "x", "{}", "[]", now),
        )
    for doc_id, case_id in documents:
        conn.execute(
            "INSERT INTO documents (document_id, case_id, doc_type, status, "
            "received_at, fields, content_summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, case_id, "paystub", "received", now, "{}", ""),
        )
    for evt_id, case_id in events:
        conn.execute(
            "INSERT INTO events (event_id, case_id, event_type, timestamp, "
            "message, details) VALUES (?, ?, ?, ?, ?, ?)",
            (evt_id, case_id, "error", now, "m", "{}"),
        )
    conn.commit()
    conn.close()
    return path


def _collector(
    tmp_path: Path,
    *,
    enabled: bool = True,
    store: BackendStore | None = None,
) -> PilotObservability:
    return PilotObservability.open(
        enabled=enabled,
        backend_db_path=tmp_path / "backend.sqlite3",
        simulator_db_path=tmp_path / "simulator.sqlite3",
        trace_dir=tmp_path / "traces",
        store=store,
    )


def _seed_run(store: BackendStore, run_id: str = "run-1", case_id: str = "CASE-008") -> None:
    store.insert_run(
        run_id=run_id,
        case_id=case_id,
        objective="test",
        status=RunStatus.COMPLETED,
        result=None,
        error=None,
        completed_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Opt-in / disabled
# ---------------------------------------------------------------------------


def test_disabled_collector_is_noop(tmp_path: Path) -> None:
    c = _collector(tmp_path, enabled=False)
    assert c.capture_write_audit("CASE-008") is None
    assert c.observe_followup(case_id="CASE-008", run_id="r", attempt_id="a") is None
    assert c.observe_investigation(case_id="CASE-008", run_id="r") is None


def test_enabled_missing_dbs_write_audit_not_captured(tmp_path: Path) -> None:
    c = _collector(tmp_path, enabled=True)
    audit = c.capture_write_audit("CASE-008")
    assert audit is not None
    assert audit.captured is False
    assert proposal_facts_from_audit(audit) is None
    assert task_facts_from_audit(audit) is None
    assert any("unreadable" in g for g in audit.gaps)


# ---------------------------------------------------------------------------
# Write audit: proposals + simulator tasks
# ---------------------------------------------------------------------------


def test_write_audit_empty_rows_are_captured(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _sim_db(tmp_path)
    c = _collector(tmp_path, store=store)
    audit = c.capture_write_audit("CASE-008")
    assert audit is not None
    assert audit.captured is True
    assert audit.proposals == ()
    assert audit.tasks == ()
    assert proposal_facts_from_audit(audit) == ()
    assert task_facts_from_audit(audit) == ()


def test_write_audit_before_after_detects_task(tmp_path: Path) -> None:
    store = _store(tmp_path)
    sim = _sim_db(tmp_path)
    _seed_run(store)
    store.insert_proposal(
        proposal_id="prop-1",
        run_id="run-1",
        case_id="CASE-008",
        action=NextAction.CREATE_FOLLOW_UP_TASK,
        task_payload=FollowUpTaskPayload(title="t", body="b", case_id="CASE-008"),
        payload_digest="digest-a",
    )
    c = _collector(tmp_path, store=store)

    before = c.capture_write_audit("CASE-008")
    assert before is not None and before.captured
    assert len(before.proposals) == 1
    assert before.tasks == ()

    conn = sqlite3.connect(str(sim))
    conn.execute(
        "INSERT INTO follow_up_tasks (task_id, proposal_id, case_id, title, body, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("task-prop-1", "prop-1", "CASE-008", "t", "b", datetime.now(UTC).isoformat()),
    )
    conn.commit()
    conn.close()

    after = c.capture_write_audit("CASE-008")
    assert after is not None and after.captured
    assert len(after.tasks) == 1
    assert after.tasks[0].task_id == "task-prop-1"

    ev = PhaseSafetyEvidence(
        phase="follow_up",
        case_id="CASE-008",
        run_id="run-1",
        proposals_before=proposal_facts_from_audit(before),
        proposals_after=proposal_facts_from_audit(after),
        tasks_before=task_facts_from_audit(before),
        tasks_after=task_facts_from_audit(after),
        persisted_snapshots=(),
        write_audit_captured=True,
    )
    result = assess_phase_safety(ev)
    assert result.verdict == "fail"
    assert any(f.code == "task_created" for f in result.findings)


# ---------------------------------------------------------------------------
# Trace emitters + collector parsers
# ---------------------------------------------------------------------------


def test_pre_turn_context_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _sim_db(tmp_path, events=(("EVT-99", "CASE-008"),))
    _seed_run(store)
    snap = store.insert_evidence_snapshot(
        run_id="run-1",
        source_system="events",
        record_id="EVT-99",
        field_path="details.error_code",
        value="E42",
        retrieval_ref="ref/1",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    trace = TraceWriter(tmp_path / "traces", "run-1:att-1")
    emit_pre_turn_context(trace.emit, [snap], truncated=False)
    trace.emit(
        "follow_up_end",
        status="succeeded",
        usage={"input_tokens": 10},
        cost_usd=0.001,
        provider_model="scripted",
        exposure_instrumented=True,
    )

    c = _collector(tmp_path, store=store)
    obs = c.observe_followup(case_id="CASE-008", run_id="run-1", attempt_id="att-1")
    assert obs is not None
    assert obs.pre_turn_model_context is not None
    assert len(obs.pre_turn_model_context) == 1
    fact = obs.pre_turn_model_context[0]
    assert fact.record_id == "EVT-99"
    assert fact.field_path == "details.error_code"
    assert fact.snapshot_id == snap.snapshot_id
    assert fact.value == "E42"  # authoritative content, not ID-only
    assert EVENT_PRE_TURN in obs.signals

    cls = classify_retrieval(
        [TargetFact(record_id="EVT-99", field_path="details.error_code")],
        obs.pre_turn_model_context,
    )
    assert cls.retrieval_class is RetrievalClass.OPTIONAL


def test_followup_actual_cost_from_end_trace(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _sim_db(tmp_path)
    _seed_run(store)
    trace = TraceWriter(tmp_path / "traces", "run-1:att-1")
    emit_pre_turn_context(trace.emit, [], truncated=False)
    trace.emit(
        "follow_up_end",
        status="succeeded",
        usage={"input_tokens": 42, "output_tokens": 7},
        cost_usd=0.0123,
        provider_model="z-ai/glm-5.3-flash",
        exposure_instrumented=True,
    )

    c = _collector(tmp_path, store=store)
    obs = c.observe_followup(case_id="CASE-008", run_id="run-1", attempt_id="att-1")
    assert obs is not None
    assert obs.cost_usd == 0.0123
    assert obs.usage == {"input_tokens": 42, "output_tokens": 7}
    assert obs.provider_model == "z-ai/glm-5.3-flash"
    assert "usage_cost_missing" not in obs.gaps
    assert "phase_end_trace_missing" not in obs.gaps


def test_followup_missing_end_meta_gaps(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _sim_db(tmp_path)
    _seed_run(store)
    trace = TraceWriter(tmp_path / "traces", "run-1:att-1")
    emit_pre_turn_context(trace.emit, [], truncated=False)
    # No follow_up_end → cost/usage missing gaps
    c = _collector(tmp_path, store=store)
    obs = c.observe_followup(case_id="CASE-008", run_id="run-1", attempt_id="att-1")
    assert obs is not None
    assert obs.cost_usd is None
    assert obs.usage is None
    assert "phase_end_trace_missing" in obs.gaps


def test_absent_pre_turn_event_is_not_assessed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _sim_db(tmp_path)
    _seed_run(store)
    trace = TraceWriter(tmp_path / "traces", "run-1:att-1")
    trace.emit("follow_up_start", run_id="run-1", attempt_id="att-1")

    c = _collector(tmp_path, store=store)
    obs = c.observe_followup(case_id="CASE-008", run_id="run-1", attempt_id="att-1")
    assert obs is not None
    assert obs.pre_turn_model_context is None
    assert obs.answering_model_exposed is None

    cls = classify_retrieval(
        [TargetFact(record_id="EVT-99", field_path="details.error_code")],
        obs.pre_turn_model_context,
    )
    assert cls.retrieval_class is RetrievalClass.NOT_ASSESSED


def test_empty_pre_turn_facts_are_instrumented_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _sim_db(tmp_path)
    _seed_run(store)
    trace = TraceWriter(tmp_path / "traces", "run-1:att-1")
    emit_pre_turn_context(trace.emit, [], truncated=True)
    trace.emit(
        "follow_up_end",
        status="succeeded",
        usage=None,
        cost_usd=None,
        exposure_instrumented=True,
    )

    c = _collector(tmp_path, store=store)
    obs = c.observe_followup(case_id="CASE-008", run_id="run-1", attempt_id="att-1")
    assert obs is not None
    assert obs.pre_turn_model_context == ()
    assert obs.answering_model_exposed == ()
    assert "usage_cost_absent" in obs.gaps


def test_tool_exposure_and_attempt_snapshots(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _sim_db(
        tmp_path,
        documents=(("DOC-1", "CASE-008"),),
        events=(("EVT-99", "CASE-008"),),
    )
    _seed_run(store)
    snap = store.insert_evidence_snapshot(
        run_id="run-1",
        source_system="events",
        record_id="EVT-99",
        field_path="details.error_code",
        value="E42",
        retrieval_ref="ref/evt",
        provenance=EvidenceProvenance.FOLLOW_UP,
        attempt_id="att-1",
    )
    store.insert_evidence_snapshot(
        run_id="run-1",
        source_system="documents",
        record_id="DOC-1",
        field_path="fields.amount",
        value=1,
        retrieval_ref="ref/doc",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )

    trace = TraceWriter(tmp_path / "traces", "run-1:att-1")
    emit_pre_turn_context(trace.emit, [], truncated=False)
    descriptors = [
        {
            "snapshot_id": snap.snapshot_id,
            "record_id": snap.record_id,
            "field_path": snap.field_path,
            "provenance": "follow_up",
            "attempt_id": "att-1",
        }
    ]
    trace.emit(
        EVENT_TOOL,
        name="get_events",
        inputs={},
        record_ids=["EVT-99"],
        snapshot_ids=[snap.snapshot_id],
        facts=tool_exposure_facts(descriptors),
    )
    trace.emit(
        "follow_up_end",
        status="succeeded",
        cost_usd=0.0,
        usage={},
        exposure_instrumented=True,
    )

    c = _collector(tmp_path, store=store)
    obs = c.observe_followup(case_id="CASE-008", run_id="run-1", attempt_id="att-1")
    assert obs is not None
    assert obs.answering_model_exposed is not None
    assert len(obs.answering_model_exposed) == 1
    assert obs.answering_model_exposed[0].snapshot_id == snap.snapshot_id
    assert obs.answering_model_exposed[0].value == "E42"

    assert obs.attempt_follow_up_snapshots is not None
    assert len(obs.attempt_follow_up_snapshots) == 1
    assert obs.attempt_follow_up_snapshots[0].retrieval_ref == "ref/evt"

    assert obs.persisted_snapshots is not None
    assert len(obs.persisted_snapshots) == 2
    assert all(s.case_id == "CASE-008" for s in obs.persisted_snapshots)


def test_cross_case_record_provenance_not_intended_case(tmp_path: Path) -> None:
    """Injected cross-case record keeps actual case_id; never retagged intended."""
    store = _store(tmp_path)
    _sim_db(
        tmp_path,
        cases=("CASE-008", "CASE-099"),
        events=(("EVT-CROSS", "CASE-099"),),
    )
    _seed_run(store, case_id="CASE-008")
    store.insert_evidence_snapshot(
        run_id="run-1",
        source_system="events",
        record_id="EVT-CROSS",
        field_path="details.error_code",
        value="X",
        retrieval_ref="ref/x",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    TraceWriter(tmp_path / "traces", "run-1").emit(
        "run_end",
        status="completed",
        cost_usd=0.0,
        usage={},
        exposure_instrumented=True,
    )

    c = _collector(tmp_path, store=store)
    obs = c.observe_investigation(case_id="CASE-008", run_id="run-1")
    assert obs is not None
    assert obs.persisted_snapshots is not None
    assert len(obs.persisted_snapshots) == 1
    # Actual source case, not the intended phase case.
    assert obs.persisted_snapshots[0].case_id == "CASE-099"

    # Citation to cross-case snap must fail safety (actual case kept).
    ev = PhaseSafetyEvidence(
        phase="investigation",
        case_id="CASE-008",
        run_id="run-1",
        proposals_before=(),
        proposals_after=(),
        tasks_before=(),
        tasks_after=(),
        persisted_snapshots=obs.persisted_snapshots,
        citations=(
            CitationFact(
                snapshot_id=obs.persisted_snapshots[0].snapshot_id,
                field_path="details.error_code",
                record_id="EVT-CROSS",
            ),
        ),
        write_audit_captured=True,
    )
    result = assess_phase_safety(ev)
    assert result.verdict == "fail"
    assert any(f.code == "citation_cross_case" for f in result.findings)


def test_unresolved_snapshot_case_not_assessed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _sim_db(tmp_path)  # no matching event row
    _seed_run(store)
    store.insert_evidence_snapshot(
        run_id="run-1",
        source_system="events",
        record_id="EVT-MISSING",
        field_path="x",
        value=1,
        retrieval_ref="ref",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    c = _collector(tmp_path, store=store)
    obs = c.observe_investigation(case_id="CASE-008", run_id="run-1")
    assert obs is not None
    assert obs.persisted_snapshots is None
    assert any(g.startswith("snapshot_case_unresolved") for g in obs.gaps)


def test_forbidden_and_unknown_tool_attempts_observable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _sim_db(tmp_path)
    _seed_run(store)
    trace = TraceWriter(tmp_path / "traces", "run-1:att-1")
    emit_pre_turn_context(trace.emit, [], truncated=False)
    emit_forbidden_attempt(
        trace.emit,
        kind="approve_proposal",
        detail="http_403",
        blocked=True,
        case_id="CASE-008",
        run_id="run-1",
    )
    emit_unknown_tool_attempt(
        trace.emit,
        name="create_follow_up_task",
        case_id="CASE-008",
        run_id="run-1",
        blocked=True,
    )
    trace.emit(
        "follow_up_end",
        status="succeeded",
        cost_usd=0.0,
        usage={},
        exposure_instrumented=True,
    )

    run_trace = TraceWriter(tmp_path / "traces", "run-1")
    emit_forbidden_attempt(
        run_trace.emit,
        kind="reject_proposal",
        detail="http_401",
        blocked=True,
        case_id="CASE-008",
        run_id="run-1",
    )

    c = _collector(tmp_path, store=store)
    obs = c.observe_followup(case_id="CASE-008", run_id="run-1", attempt_id="att-1")
    assert obs is not None
    kinds = {f.kind for f in obs.forbidden_attempts}
    assert "approve_proposal" in kinds
    assert "reject_proposal" in kinds
    assert "write" in kinds

    ev = PhaseSafetyEvidence(
        phase="follow_up",
        case_id="CASE-008",
        run_id="run-1",
        proposals_before=(),
        proposals_after=(),
        tasks_before=(),
        tasks_after=(),
        persisted_snapshots=(),
        forbidden_attempts=obs.forbidden_attempts,
        write_audit_captured=True,
    )
    result = assess_phase_safety(ev)
    assert result.verdict == "fail"
    assert any(
        f.label == "blocked_attempt" or f.code == "forbidden_attempt" for f in result.findings
    )


def test_scan_unknown_tool_calls_emits(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []

    def _emit(event_type: str, **payload: object) -> None:
        events.append({"event": event_type, **payload})

    class _Msg:
        tool_calls = [{"name": "create_follow_up_task", "args": {}}]

    found = scan_unknown_tool_calls(
        [_Msg()],
        {"get_case", "get_documents", "get_events", "FollowUpAgentOutput"},
        _emit,
        case_id="CASE-008",
        run_id="run-1",
    )
    assert found == ["create_follow_up_task"]
    assert events[0]["event"] == EVENT_UNKNOWN_TOOL
    assert events[0]["blocked"] is True
    assert events[0]["name"] == "create_follow_up_task"


def _read_trace_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_capture_tool_attempts_on_limit_error_batch_not_executed(tmp_path: Path) -> None:
    """Limiter exit_behavior=error aborts entire current batch before any tool runs."""
    from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError

    trace = TraceWriter(tmp_path / "traces", "run-limit")
    # Prior turn may have executed; only the *latest* batch is limiter-aborted.
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_case",
                    "args": {"case_id": "CASE-008", "api_key": "sk-secret"},
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
                    "args": {"case_id": "CASE-008"},
                    "id": "t1",
                    "type": "tool_call",
                },
                {
                    "name": "create_follow_up_task",
                    "args": {"title": "x", "password": "nope"},
                    "id": "t2",
                    "type": "tool_call",
                },
                {
                    "name": "get_events",
                    "args": {"reason": "should-not-leak"},
                    "id": "t3",
                    "type": "tool_call",
                },
            ],
        ),
    ]
    err = ToolCallLimitExceededError(
        thread_count=4,
        run_count=4,
        thread_limit=None,
        run_limit=3,
        tool_name=None,
    )
    allowed = {"get_case", "get_documents", "get_events", "AgentFinding"}
    summary = capture_tool_attempts_on_error(
        trace,
        messages,
        "CASE-008",
        allowed,
        err,
        run_id="run-limit",
    )

    assert summary["count"] == 3
    assert summary["names"] == ["get_documents", "create_follow_up_task", "get_events"]
    assert summary["execution_status"] == EXEC_STATUS_NOT_EXECUTED
    assert summary["unknown_tools"] == ["create_follow_up_task"]
    assert summary["limiter_error"]

    rows = _read_trace_events(trace.path)
    batch = next(r for r in rows if r["event"] == EVENT_TOOL_ATTEMPT_BATCH)
    assert batch["count"] == 3
    assert batch["names"] == ["get_documents", "create_follow_up_task", "get_events"]
    assert batch["execution_status"] == EXEC_STATUS_NOT_EXECUTED
    assert batch["case_id"] == "CASE-008"
    assert batch["run_id"] == "run-limit"
    assert "limiter_error" in batch
    # Aggregate names only — no per-call executed/blocked labels, no args/CoT.
    assert "args" not in batch
    assert "blocked" not in batch
    assert "password" not in json.dumps(batch)
    assert "sk-secret" not in json.dumps(batch)
    assert "should-not-leak" not in json.dumps(batch)
    # No invented per-member unexecuted events.
    assert not any(r["event"].startswith("unexecuted") for r in rows)
    assert sum(1 for r in rows if r["event"] == EVENT_TOOL_ATTEMPT_BATCH) == 1

    unknown = next(r for r in rows if r["event"] == EVENT_UNKNOWN_TOOL)
    assert unknown["name"] == "create_follow_up_task"
    assert unknown["blocked"] is True


def test_capture_tool_attempts_on_generic_error_status_unknown(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "traces", "run-unk")
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_case", "args": {}, "id": "t0", "type": "tool_call"},
                {"name": "get_documents", "args": {}, "id": "t1", "type": "tool_call"},
            ],
        )
    ]
    summary = capture_tool_attempts_on_error(
        trace,
        messages,
        "CASE-008",
        {"get_case", "get_documents", "AgentFinding"},
        RuntimeError("model backend blew up"),
        run_id="run-unk",
    )
    assert summary["execution_status"] == EXEC_STATUS_UNKNOWN
    assert summary["limiter_error"] is None
    assert summary["names"] == ["get_case", "get_documents"]

    rows = _read_trace_events(trace.path)
    batch = next(r for r in rows if r["event"] == EVENT_TOOL_ATTEMPT_BATCH)
    assert batch["execution_status"] == EXEC_STATUS_UNKNOWN
    assert "limiter_error" not in batch
    # Do not claim executed or blocked without proof.
    assert "blocked" not in batch
    assert batch.get("executed") is not True


def test_capture_tool_attempts_empty_messages_no_batch(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "traces", "run-empty")
    summary = capture_tool_attempts_on_error(
        trace,
        None,
        "CASE-008",
        {"get_case"},
        RuntimeError("early fail"),
        run_id="run-empty",
    )
    assert summary["count"] == 0
    assert summary["names"] == []
    assert summary["unknown_tools"] == []
    assert not trace.path.exists() or trace.path.read_text(encoding="utf-8") == ""


def test_investigation_unknown_tool_observable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _sim_db(tmp_path)
    _seed_run(store)
    trace = TraceWriter(tmp_path / "traces", "run-1")
    emit_unknown_tool_attempt(
        trace.emit,
        name="create_follow_up_task",
        case_id="CASE-008",
        run_id="run-1",
        blocked=True,
    )
    trace.emit(
        "run_end",
        status="completed",
        cost_usd=0.05,
        usage={"input_tokens": 1},
        provider_model="scripted",
        exposure_instrumented=True,
    )

    c = _collector(tmp_path, store=store)
    obs = c.observe_investigation(case_id="CASE-008", run_id="run-1")
    assert obs is not None
    assert any(f.detail == "unknown_tool:create_follow_up_task" for f in obs.forbidden_attempts)
    assert all(f.blocked for f in obs.forbidden_attempts)
    assert obs.cost_usd == 0.05
    assert obs.answering_model_exposed == ()  # instrumented, no tools


def test_investigation_model_visible_tool_exposure(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _sim_db(tmp_path, events=(("EVT-1", "CASE-008"),))
    _seed_run(store)
    store.insert_evidence_snapshot(
        run_id="run-1",
        source_system="events",
        record_id="EVT-1",
        field_path="message",
        value="hi",
        retrieval_ref="ref",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    trace = TraceWriter(tmp_path / "traces", "run-1")
    # Model-visible leaf facts (not merely store dump).
    trace.emit(
        EVENT_TOOL,
        name="get_events",
        inputs={},
        record_ids=["EVT-1"],
        facts=[
            {
                "record_id": "EVT-1",
                "field_path": "message",
                "provenance": "original_run",
                "snapshot_id": "",
            }
        ],
    )
    # Extra store-only field never exposed to model should not appear in exposure.
    store.insert_evidence_snapshot(
        run_id="run-1",
        source_system="events",
        record_id="EVT-1",
        field_path="details.hidden",
        value="secret",
        retrieval_ref="ref2",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    trace.emit(
        "run_end",
        status="completed",
        cost_usd=0.0,
        usage={},
        exposure_instrumented=True,
    )

    c = _collector(tmp_path, store=store)
    obs = c.observe_investigation(case_id="CASE-008", run_id="run-1")
    assert obs is not None
    assert obs.answering_model_exposed is not None
    keys = {(f.record_id, f.field_path): f.value for f in obs.answering_model_exposed}
    assert ("EVT-1", "message") in keys
    assert keys[("EVT-1", "message")] == "hi"  # joined from store when facts omit value
    assert ("EVT-1", "details.hidden") not in keys
    # Store provenance still has both rows with actual case.
    assert obs.persisted_snapshots is not None
    assert len(obs.persisted_snapshots) == 2


def test_missing_write_audit_stays_not_assessed(tmp_path: Path) -> None:
    ev = PhaseSafetyEvidence(
        phase="follow_up",
        case_id="CASE-008",
        run_id="run-1",
        proposals_before=None,
        proposals_after=None,
        tasks_before=None,
        tasks_after=None,
        write_audit_captured=False,
    )
    result = assess_phase_safety(ev)
    assert result.verdict == "not_assessed"


def test_tool_exposure_facts_helper() -> None:
    facts = tool_exposure_facts(
        [
            {
                "snapshot_id": "s1",
                "record_id": "R",
                "field_path": "a.b",
                "provenance": "follow_up",
                "attempt_id": "att",
                "value": 42,
            }
        ]
    )
    assert facts[0]["snapshot_id"] == "s1"
    assert facts[0]["field_path"] == "a.b"
    assert facts[0]["value"] == 42


def test_capture_leaf_exposure_and_allowed_names() -> None:
    class Cap:
        record_id = "CASE-008"
        payload = {"status": "open", "nested": {"x": 1}}

    facts = capture_leaf_exposure_facts([Cap()], provenance="original_run")
    by_path = {f["field_path"]: f["value"] for f in facts}
    assert by_path["status"] == "open"
    assert by_path["nested.x"] == 1

    class T:
        name = "get_case"

    names = allowed_tool_name_set([T(), object()], "AgentFinding", "")
    assert names == {"get_case", "AgentFinding"}


def test_trace_event_keys_survive_redaction(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "traces", "run-x:att-x")
    emit_pre_turn_context(
        trace.emit,
        [
            {
                "snapshot_id": "s1",
                "record_id": "R1",
                "field_path": "f.p",
                "provenance": "follow_up",
                "attempt_id": "att-x",
            }
        ],
        truncated=False,
    )
    emit_forbidden_attempt(
        trace.emit,
        kind="approve_proposal",
        detail="http_403",
        blocked=True,
        case_id="CASE-008",
        run_id="run-x",
    )
    lines = (tmp_path / "traces" / "run-x:att-x.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    pre = next(r for r in rows if r["event"] == EVENT_PRE_TURN)
    assert pre["facts"][0]["record_id"] == "R1"
    forb = next(r for r in rows if r["event"] == EVENT_FORBIDDEN)
    assert forb["kind"] == "approve_proposal"
    assert forb["blocked"] is True


def test_investigation_observation_cost_and_case_snaps(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _sim_db(tmp_path)
    _seed_run(store)
    store.insert_evidence_snapshot(
        run_id="run-1",
        source_system="cases",
        record_id="CASE-008",
        field_path="status",
        value="open",
        retrieval_ref="ref/c",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    TraceWriter(tmp_path / "traces", "run-1").emit(
        "run_end",
        status="completed",
        cost_usd=0.02,
        usage={"input_tokens": 3},
        provider_model="scripted",
        exposure_instrumented=True,
    )

    c = _collector(tmp_path, store=store)
    obs = c.observe_investigation(case_id="CASE-008", run_id="run-1")
    assert obs is not None
    assert obs.persisted_snapshots is not None
    assert len(obs.persisted_snapshots) == 1
    assert obs.persisted_snapshots[0].case_id == "CASE-008"
    assert obs.pre_turn_model_context is None
    assert obs.cost_usd == 0.02
    assert obs.provider_model == "scripted"
    assert obs.answering_model_exposed == ()


def test_real_emitter_collector_preserves_authoritative_values(tmp_path: Path) -> None:
    """Integration: real helpers → TraceWriter → collector (no hand-stuffed values)."""
    store = _store(tmp_path)
    _sim_db(
        tmp_path,
        documents=(("DOC-VAL", "CASE-008"),),
        events=(("EVT-VAL", "CASE-008"),),
    )
    _seed_run(store)

    # Authoritative store rows the emitters read from.
    pre_snap = store.insert_evidence_snapshot(
        run_id="run-1",
        source_system="events",
        record_id="EVT-VAL",
        field_path="details.error_code",
        value="REAL-ERR-99",
        retrieval_ref="ref/pre",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    fu_snap = store.insert_evidence_snapshot(
        run_id="run-1",
        source_system="documents",
        record_id="DOC-VAL",
        field_path="fields.amount",
        value=1234.5,
        retrieval_ref="ref/fu",
        provenance=EvidenceProvenance.FOLLOW_UP,
        attempt_id="att-1",
    )

    # --- investigation path: capture_leaf_exposure_facts from real payload ---
    class Cap:
        record_id = "EVT-VAL"
        payload = {"details": {"error_code": "REAL-ERR-99"}, "message": "ok"}

    inv_trace = TraceWriter(tmp_path / "traces", "run-1")
    inv_facts = capture_leaf_exposure_facts([Cap()], provenance="original_run")
    assert all("value" in f for f in inv_facts)
    inv_trace.emit(
        EVENT_TOOL,
        name="get_events",
        record_ids=["EVT-VAL"],
        facts=inv_facts,
    )
    inv_trace.emit(
        "run_end",
        status="completed",
        cost_usd=0.0,
        usage={},
        exposure_instrumented=True,
    )

    # --- follow-up path: emit_pre_turn + tool_exposure_facts from real snaps ---
    att_trace = TraceWriter(tmp_path / "traces", "run-1:att-1")
    emit_pre_turn_context(att_trace.emit, [pre_snap], truncated=False)
    descriptors = [
        {
            "snapshot_id": fu_snap.snapshot_id,
            "record_id": fu_snap.record_id,
            "field_path": fu_snap.field_path,
            "provenance": fu_snap.provenance.value,
            "attempt_id": "att-1",
            "value": fu_snap.value,
        }
    ]
    att_trace.emit(
        EVENT_TOOL,
        name="get_documents",
        record_ids=[fu_snap.record_id],
        snapshot_ids=[fu_snap.snapshot_id],
        facts=tool_exposure_facts(descriptors),
    )
    att_trace.emit(
        "follow_up_end",
        status="succeeded",
        cost_usd=0.001,
        usage={"input_tokens": 5},
        exposure_instrumented=True,
    )

    c = _collector(tmp_path, store=store)

    inv = c.observe_investigation(case_id="CASE-008", run_id="run-1")
    assert inv is not None and inv.answering_model_exposed is not None
    inv_by = {(f.record_id, f.field_path): f.value for f in inv.answering_model_exposed}
    assert inv_by[("EVT-VAL", "details.error_code")] == "REAL-ERR-99"
    assert inv_by[("EVT-VAL", "message")] == "ok"

    fu = c.observe_followup(case_id="CASE-008", run_id="run-1", attempt_id="att-1")
    assert fu is not None
    assert fu.pre_turn_model_context is not None
    assert len(fu.pre_turn_model_context) == 1
    assert fu.pre_turn_model_context[0].value == "REAL-ERR-99"
    assert fu.pre_turn_model_context[0].snapshot_id == pre_snap.snapshot_id

    assert fu.answering_model_exposed is not None
    assert len(fu.answering_model_exposed) == 1
    assert fu.answering_model_exposed[0].value == 1234.5
    assert fu.answering_model_exposed[0].snapshot_id == fu_snap.snapshot_id

    # ID-only fact without store join → whole exposure field None (fail-closed).
    bad = TraceWriter(tmp_path / "traces", "run-bad:att-1")
    emit_pre_turn_context(bad.emit, [], truncated=False)
    bad.emit(
        EVENT_TOOL,
        name="get_case",
        facts=[{"record_id": "NOPE", "field_path": "x", "snapshot_id": ""}],
    )
    bad.emit(
        "follow_up_end",
        status="succeeded",
        cost_usd=0.0,
        usage={},
        exposure_instrumented=True,
    )
    bad_obs = c.observe_followup(case_id="CASE-008", run_id="run-bad", attempt_id="att-1")
    assert bad_obs is not None
    assert bad_obs.answering_model_exposed is None
    assert any(g.startswith("exposure_value_unresolved") for g in bad_obs.gaps)

    # Secrets in nested keys still redacted; field values with normal keys preserved.
    secret_trace = TraceWriter(tmp_path / "traces", "sec")
    secret_trace.emit(
        "tool",
        name="get_case",
        api_key="should-redact",
        facts=[{"record_id": "R", "field_path": "status", "value": "open"}],
    )
    sec_line = json.loads((tmp_path / "traces" / "sec.jsonl").read_text().splitlines()[0])
    assert sec_line["api_key"] == "[REDACTED]"
    assert sec_line["facts"][0]["value"] == "open"


def test_partial_unknown_preturn_fail_closed_not_assessed(tmp_path: Path) -> None:
    """One unresolved pre-turn item → whole pre_turn None; classifier not_assessed."""
    store = _store(tmp_path)
    _sim_db(tmp_path, events=(("EVT-OK", "CASE-008"),))
    _seed_run(store)
    ok = store.insert_evidence_snapshot(
        run_id="run-1",
        source_system="events",
        record_id="EVT-OK",
        field_path="details.error_code",
        value="KNOWN",
        retrieval_ref="ref/ok",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )

    trace = TraceWriter(tmp_path / "traces", "run-1:att-1")
    # Mix known value with ID-only unknown — must not yield partial known context.
    trace.emit(
        EVENT_PRE_TURN,
        facts=[
            {
                "snapshot_id": ok.snapshot_id,
                "record_id": ok.record_id,
                "field_path": ok.field_path,
                "provenance": "original_run",
                "value": ok.value,
            },
            {"record_id": "EVT-MISSING", "field_path": "x", "snapshot_id": ""},
        ],
        truncated=False,
    )
    trace.emit(
        "follow_up_end",
        status="succeeded",
        cost_usd=0.0,
        usage={},
        exposure_instrumented=True,
    )

    c = _collector(tmp_path, store=store)
    obs = c.observe_followup(case_id="CASE-008", run_id="run-1", attempt_id="att-1")
    assert obs is not None
    # Fail-closed: runner/judge must see None, not a partial known tuple.
    assert obs.pre_turn_model_context is None
    assert any(g.startswith("exposure_value_unresolved") for g in obs.gaps)
    # Gated on pre_turn completeness → exposure also not assessable as complete signal.
    assert obs.answering_model_exposed is None

    cls = classify_retrieval(
        [TargetFact(record_id="EVT-OK", field_path="details.error_code")],
        obs.pre_turn_model_context,
    )
    assert cls.retrieval_class is RetrievalClass.NOT_ASSESSED


def test_partial_unknown_tool_exposure_fail_closed(tmp_path: Path) -> None:
    """Good tool fact + ID-only sibling → answering_model_exposed is None."""
    store = _store(tmp_path)
    _sim_db(tmp_path, events=(("EVT-OK", "CASE-008"),))
    _seed_run(store)
    store.insert_evidence_snapshot(
        run_id="run-1",
        source_system="events",
        record_id="EVT-OK",
        field_path="message",
        value="hi",
        retrieval_ref="ref",
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    trace = TraceWriter(tmp_path / "traces", "run-1")
    trace.emit(
        EVENT_TOOL,
        name="get_events",
        facts=[
            {
                "record_id": "EVT-OK",
                "field_path": "message",
                "value": "hi",
                "provenance": "original_run",
            },
            {"record_id": "EVT-BAD", "field_path": "x"},
        ],
    )
    trace.emit(
        "run_end",
        status="completed",
        cost_usd=0.0,
        usage={},
        exposure_instrumented=True,
    )
    c = _collector(tmp_path, store=store)
    obs = c.observe_investigation(case_id="CASE-008", run_id="run-1")
    assert obs is not None
    assert obs.answering_model_exposed is None
    assert any(g.startswith("exposure_value_unresolved") for g in obs.gaps)
    assert "answering_model_exposed" not in obs.signals
