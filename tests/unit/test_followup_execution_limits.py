"""Follow-up runtime failure labels: per-run execution limits vs spend pause."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage

from casepilot.backend.client import SimulatorClient
from casepilot.backend.db import BackendStore
from casepilot.backend.followup import run_follow_up
from casepilot.backend.hooks import TEST_MODE_ENV, clear_test_hooks
from casepilot.backend.scripted import ScriptedChatModel
from casepilot.evaluation.cost import CostPause
from casepilot.shared.config import get_settings
from casepilot.shared.contracts import (
    CreateFollowUpTaskRequest,
    CreateFollowUpTaskResponse,
    Diagnosis,
    Evidence,
    EvidenceProvenance,
    InvestigationResult,
    NextAction,
    RunStatus,
    SimulatorRoutes,
    TurnStatus,
)

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


def _mock_simulator() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {TOKEN_SIM}":
            return httpx.Response(401, json={"detail": "unauthorized"})
        path = request.url.path
        if path == SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id=CASE_ID):
            body = CreateFollowUpTaskRequest.model_validate_json(request.content)
            return httpx.Response(
                200,
                json=CreateFollowUpTaskResponse(
                    task_id="TASK-1",
                    proposal_id=body.proposal_id,
                    case_id=CASE_ID,
                    created=True,
                ).model_dump(mode="json"),
            )
        return httpx.Response(404, json={"detail": "no route"})

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://simulator.test",
        headers={"Authorization": f"Bearer {TOKEN_SIM}"},
    )


@pytest.fixture(autouse=True)
def _clean_hooks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(TEST_MODE_ENV, "1")
    clear_test_hooks()
    yield
    clear_test_hooks()


def _tool_batch(n: int, *, name: str = "get_case") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": {}, "id": f"t{i}", "type": "tool_call"} for i in range(n)
        ],
    )


def _seed_chat_run(store: BackendStore, *, run_id: str = "run-1") -> None:
    store.insert_run(
        run_id=run_id,
        case_id=CASE_ID,
        objective="why stalled",
        status=RunStatus.COMPLETED,
        result=InvestigationResult(
            run_id=run_id,
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
        ),
        completed_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
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


@pytest.mark.asyncio
async def test_followup_reserve_cost_pause_persists_classified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    _seed_chat_run(store)
    begun = store.begin_send(run_id="run-1", question="?", idempotency_key="rsv-cp")
    http = _mock_simulator()

    def factory(_s):
        return ScriptedChatModel(responses=[_tool_batch(1)])

    def boom(**_kw):
        raise CostPause("ledger remaining $0.00; live calls paused")

    monkeypatch.setattr("casepilot.backend.followup._reserve_attempt", boom)

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
    assert turn is not None and turn.status == TurnStatus.FAILED
    err = turn.attempts[-1].error or ""
    assert err.startswith("cost pause:"), err

    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_followup_input_bound_is_execution_limit_not_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    _seed_chat_run(store)
    begun = store.begin_send(run_id="run-1", question="?", idempotency_key="ib-1")
    http = _mock_simulator()

    def factory(_s):
        return ScriptedChatModel(responses=[_tool_batch(1)])

    def boom(**_kw):
        raise CostPause("follow-up mandatory context exceeds input bound; refusing unbounded call")

    monkeypatch.setattr("casepilot.backend.followup.build_user_message", boom)

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
    assert turn is not None and turn.status == TurnStatus.FAILED
    err = turn.attempts[-1].error or ""
    assert err.startswith("execution limit exceeded:"), err
    assert not err.startswith("cost pause:")

    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_followup_tool_overlimit_is_execution_limit_not_spend(tmp_path: Path) -> None:
    """Real follow-up ToolCallLimit path labels execution limit, not cost pause."""
    settings = _settings(tmp_path, max_tool_calls=2, max_model_turns=8)
    store = BackendStore(settings.backend_db_path)
    _seed_chat_run(store)
    begun = store.begin_send(run_id="run-1", question="overlimit?", idempotency_key="fu-ol")
    http = _mock_simulator()

    def factory(_s):
        return ScriptedChatModel(responses=[_tool_batch(1) for _ in range(6)])

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
    assert turn is not None and turn.status == TurnStatus.FAILED
    err = turn.attempts[-1].error or ""
    assert err.startswith("execution limit exceeded:"), err
    assert not err.startswith("cost pause:")

    store.close()
    await http.aclose()
