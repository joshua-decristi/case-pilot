"""Investigation runtime failure labels: per-run execution limits vs spend pause."""

from __future__ import annotations

import tempfile
from functools import cache
from pathlib import Path

import httpx
import pytest
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from casepilot.backend.agent import _classify_runtime_failure, run_investigation
from casepilot.backend.client import SimulatorClient
from casepilot.backend.db import BackendStore
from casepilot.backend.hooks import TEST_MODE_ENV, clear_test_hooks
from casepilot.backend.scripted import ScriptedChatModel
from casepilot.evaluation.cost import CostPause
from casepilot.shared.config import get_settings
from casepilot.shared.contracts import (
    CaseRecord,
    CreateFollowUpTaskRequest,
    CreateFollowUpTaskResponse,
    DocumentListResponse,
    EventListResponse,
    RunStatus,
    SimulatorRoutes,
)
from casepilot.simulator.db import connect
from casepilot.simulator.seed import seed
from casepilot.simulator.store import SimulatorStore

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

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {TOKEN_SIM}":
            return httpx.Response(401, json={"detail": "unauthorized"})
        path = request.url.path
        if path == SimulatorRoutes.CASE.format(case_id=CASE_ID):
            return httpx.Response(200, json=case.model_dump(mode="json"))
        if path == SimulatorRoutes.DOCUMENTS.format(case_id=CASE_ID):
            return httpx.Response(200, json=docs.model_dump(mode="json"))
        if path == SimulatorRoutes.EVENTS.format(case_id=CASE_ID):
            return httpx.Response(200, json=events.model_dump(mode="json"))
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


# ---------------------------------------------------------------------------
# classification (type-first; no bare limit/budget/recursion substrings)
# Prefix contract matches scenario_pilot.classify_spend_pause (tested there).
# ---------------------------------------------------------------------------


def test_classify_tool_call_limit_is_execution_limit_not_spend() -> None:
    exc = ToolCallLimitExceededError(thread_count=14, run_count=14, thread_limit=None, run_limit=12)
    label = _classify_runtime_failure(exc)
    assert label.startswith("execution limit exceeded:")
    assert "execution budget exhausted:" not in label
    assert not label.startswith("cost pause:")


def test_classify_model_and_graph_limits_are_execution_limit() -> None:
    model_exc = ModelCallLimitExceededError(
        thread_count=8, run_count=8, thread_limit=None, run_limit=8
    )
    graph_exc = GraphRecursionError(
        "Recursion limit of 25 reached without hitting a stop condition."
    )
    for exc in (model_exc, graph_exc):
        label = _classify_runtime_failure(exc)
        assert label.startswith("execution limit exceeded:"), label
        assert not label.startswith("cost pause:")


def test_classify_cost_pause_matches_spend_pause_contract() -> None:
    # Message contains "budget" — must still be exact ``cost pause:`` spend stop.
    exc = CostPause("ledger remaining $0.00; budget exhausted for live calls")
    label = _classify_runtime_failure(exc)
    assert label.startswith("cost pause:"), label
    assert "execution budget exhausted:" not in label
    assert "execution limit exceeded:" not in label
    assert not label.startswith("cost/input guard:")


def test_classify_input_bound_is_execution_limit_not_spend() -> None:
    phrases = (
        "model input bound exceeded: estimated 9001 tokens (reservation assumes ≤8000/turn)",
        "follow-up mandatory context exceeds input bound; refusing unbounded call",
    )
    for text in phrases:
        label = _classify_runtime_failure(CostPause(text))
        assert label.startswith("execution limit exceeded:"), label
        assert not label.startswith("cost pause:")


def test_classify_generic_failure_is_agent_error_not_budget() -> None:
    label = _classify_runtime_failure(RuntimeError("tool budget for case blew up"))
    assert label.startswith("agent error:")
    assert "execution budget exhausted:" not in label
    assert not label.startswith("cost pause:")


# ---------------------------------------------------------------------------
# scripted over-limit batch through real agent middleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scripted_tool_overlimit_persists_execution_limit_not_cost_pause(
    tmp_path: Path,
) -> None:
    """Real ToolCallLimitMiddleware error → FAILED + execution limit; no CostPause."""
    settings = _settings(tmp_path, max_tool_calls=2, max_model_turns=8)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)

    responses = [_tool_batch(1) for _ in range(6)]

    def factory(_s):
        return ScriptedChatModel(responses=list(responses))

    outcome = await run_investigation(
        settings=settings,
        store=store,
        client=client,
        case_id=CASE_ID,
        objective="Why stalled?",
        model_factory=factory,
    )
    result = outcome.result
    assert result.status == RunStatus.FAILED
    err = result.explanation or ""
    assert err.startswith("execution limit exceeded:"), err
    assert "Tool call limit" in err or "tool call limit" in err.lower() or "run limit" in err
    assert "execution budget exhausted:" not in err
    assert not err.lower().startswith("cost pause:")

    stored = store.get_run(result.run_id)
    assert stored is not None
    assert stored.status == RunStatus.FAILED
    assert stored.error is not None
    assert stored.error.startswith("execution limit exceeded:")
    assert stored.error == err

    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_scripted_overlimit_batch_in_one_turn_persists_execution_limit(
    tmp_path: Path,
) -> None:
    """One AIMessage with more tool calls than run_limit still labels execution limit."""
    settings = _settings(tmp_path, max_tool_calls=2, max_model_turns=8)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)

    def factory(_s):
        return ScriptedChatModel(responses=[_tool_batch(4)])

    outcome = await run_investigation(
        settings=settings,
        store=store,
        client=client,
        case_id=CASE_ID,
        objective="batch overlimit",
        model_factory=factory,
    )
    result = outcome.result
    assert result.status == RunStatus.FAILED
    err = result.explanation or ""
    assert err.startswith("execution limit exceeded:"), err
    assert not err.startswith("cost pause:")
    stored = store.get_run(result.run_id)
    assert stored is not None and stored.error is not None
    assert stored.error.startswith("execution limit exceeded:")

    store.close()
    await http.aclose()


# ---------------------------------------------------------------------------
# pre-invoke CostPause paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_investigation_reserve_cost_pause_persists_classified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)

    def factory(_s):
        return ScriptedChatModel(responses=[_tool_batch(1)])

    def boom(**_kw):
        raise CostPause("ledger remaining $0.00; live calls paused")

    monkeypatch.setattr("casepilot.backend.agent.guard_live_call", boom)

    outcome = await run_investigation(
        settings=settings,
        store=store,
        client=client,
        case_id=CASE_ID,
        objective="reserve pause",
        model_factory=factory,
    )
    err = outcome.result.explanation or ""
    assert err.startswith("cost pause:"), err
    stored = store.get_run(outcome.result.run_id)
    assert stored is not None and stored.error == err

    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_investigation_factory_cost_pause_persists_classified(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)

    def factory(_s):
        raise CostPause("pricing marked unreliable; live calls paused")

    outcome = await run_investigation(
        settings=settings,
        store=store,
        client=client,
        case_id=CASE_ID,
        objective="factory pause",
        model_factory=factory,
    )
    err = outcome.result.explanation or ""
    assert err.startswith("cost pause:"), err

    store.close()
    await http.aclose()
