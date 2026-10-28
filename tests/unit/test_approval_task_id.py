"""Approval path: persist simulator task_id atomically; idempotent recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from casepilot.backend.agent import payload_digest
from casepilot.backend.app import create_app
from casepilot.backend.client import SimulatorClient
from casepilot.backend.db import BackendStore
from casepilot.backend.hooks import TEST_MODE_ENV, clear_test_hooks
from casepilot.shared.config import get_settings
from casepilot.shared.contracts import (
    BackendRoutes,
    CreateFollowUpTaskRequest,
    CreateFollowUpTaskResponse,
    FollowUpTaskPayload,
    NextAction,
    ProposalStatus,
    RunStatus,
    SimulatorRoutes,
)

CASE_ID = "CASE-001"
TOKEN_SIM = "test-sim-token"
TOKEN_OP = "test-op-token"
# Non-derived ID (not f"task-{proposal_id}") — acceptance criterion #8.
SIM_TASK_ID = "TASK-NONDERIVED-99"
SIM_TASK_REPLAY = "TASK-NONDERIVED-99"


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


def _mock_simulator(
    *,
    fail_write: bool = False,
    task_id: str = SIM_TASK_ID,
) -> tuple[httpx.AsyncClient, dict[str, Any]]:
    """Minimal sim: only follow-up-tasks needed after investigation is seeded."""
    tasks: dict[str, CreateFollowUpTaskRequest] = {}
    meta: dict[str, Any] = {"create_calls": 0, "tasks": tasks}

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {TOKEN_SIM}":
            return httpx.Response(401, json={"detail": "unauthorized"})
        path = request.url.path
        if path == SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id=CASE_ID):
            meta["create_calls"] += 1
            body = CreateFollowUpTaskRequest.model_validate_json(request.content)
            if fail_write:
                return httpx.Response(503, json={"detail": "down"})
            if body.proposal_id in tasks:
                prev = tasks[body.proposal_id]
                if prev.task != body.task:
                    return httpx.Response(409, json={"detail": "conflict"})
                return httpx.Response(
                    200,
                    json=CreateFollowUpTaskResponse(
                        task_id=SIM_TASK_REPLAY,
                        proposal_id=body.proposal_id,
                        case_id=CASE_ID,
                        created=False,
                    ).model_dump(mode="json"),
                )
            tasks[body.proposal_id] = body
            return httpx.Response(
                200,
                json=CreateFollowUpTaskResponse(
                    task_id=task_id,
                    proposal_id=body.proposal_id,
                    case_id=CASE_ID,
                    created=True,
                ).model_dump(mode="json"),
            )
        return httpx.Response(404, json={"detail": "no route"})

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        transport=transport,
        base_url="http://simulator.test",
        headers={"Authorization": f"Bearer {TOKEN_SIM}"},
    )
    return http, meta


def _seed_pending_proposal(store: BackendStore, proposal_id: str = "prop-1") -> str:
    store.insert_run(
        run_id="run-1",
        case_id=CASE_ID,
        objective="x",
        status=RunStatus.COMPLETED,
    )
    payload = FollowUpTaskPayload(
        title=f"Follow up missing docs for {CASE_ID}",
        body="Please request the missing documents.",
        case_id=CASE_ID,
    )
    store.insert_proposal(
        proposal_id=proposal_id,
        run_id="run-1",
        case_id=CASE_ID,
        action=NextAction.CREATE_FOLLOW_UP_TASK,
        task_payload=payload,
        payload_digest=payload_digest(payload),
    )
    return proposal_id


@pytest.fixture(autouse=True)
def _clean_hooks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(TEST_MODE_ENV, "1")
    clear_test_hooks()
    yield
    clear_test_hooks()


@pytest.mark.asyncio
async def test_approve_persists_nonderived_task_id(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    pid = _seed_pending_proposal(store)
    http, meta = _mock_simulator(task_id=SIM_TASK_ID)
    client = SimulatorClient(settings, client=http)
    app = create_app(settings, store=store, client=client)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        ok = await ac.post(
            BackendRoutes.APPROVE.format(proposal_id=pid),
            json={"operator_identity": "alice"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert ok.status_code == 200, ok.text
        prop = ok.json()["proposal"]
        assert prop["status"] == "executed"
        assert prop["task_id"] == SIM_TASK_ID
        assert prop["task_id"] != f"task-{pid}"

        got = await ac.get(BackendRoutes.PROPOSAL.format(proposal_id=pid))
        assert got.json()["task_id"] == SIM_TASK_ID
        assert meta["create_calls"] == 1

        again = await ac.post(
            BackendRoutes.APPROVE.format(proposal_id=pid),
            json={"operator_identity": "alice"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert again.status_code == 200
        assert again.json()["proposal"]["task_id"] == SIM_TASK_ID
        assert meta["create_calls"] == 1

    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_sim_success_persist_loss_recovers_same_task_id(tmp_path: Path):
    """Simulator committed; backend lost EXECUTED write → retry recovers same task_id."""
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    pid = _seed_pending_proposal(store)
    http, meta = _mock_simulator(task_id=SIM_TASK_ID)
    client = SimulatorClient(settings, client=http)
    app = create_app(settings, store=store, client=client)

    real_transition = store.transition_proposal
    drop_executed = {"active": True}

    def flaky_transition(proposal_id: str, **kwargs: Any):
        to_status = kwargs.get("to_status")
        if drop_executed["active"] and to_status == ProposalStatus.EXECUTED:
            drop_executed["active"] = False
            return None
        return real_transition(proposal_id, **kwargs)

    store.transition_proposal = flaky_transition  # type: ignore[method-assign]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        first = await ac.post(
            BackendRoutes.APPROVE.format(proposal_id=pid),
            json={"operator_identity": "ops"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert first.status_code == 200, first.text
        prop = first.json()["proposal"]
        assert prop["status"] == "approved"
        assert prop.get("task_id") is None
        assert meta["create_calls"] == 1

        store.transition_proposal = real_transition  # type: ignore[method-assign]
        retry = await ac.post(
            BackendRoutes.APPROVE.format(proposal_id=pid),
            json={"operator_identity": "ops"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert retry.status_code == 200, retry.text
        body = retry.json()["proposal"]
        assert body["status"] == "executed"
        assert body["task_id"] == SIM_TASK_ID
        assert meta["create_calls"] == 2
        assert store.get_proposal(pid).task_id == SIM_TASK_ID  # type: ignore[union-attr]

    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_execution_failed_then_retry_stores_task_id(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    pid = _seed_pending_proposal(store)

    http_fail, _ = _mock_simulator(fail_write=True)
    client_fail = SimulatorClient(settings, client=http_fail)
    app = create_app(settings, store=store, client=client_fail)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        fail = await ac.post(
            BackendRoutes.APPROVE.format(proposal_id=pid),
            json={"operator_identity": "ops"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert fail.json()["proposal"]["status"] == "execution_failed"
        assert fail.json()["proposal"].get("task_id") is None

    await http_fail.aclose()

    http_ok, meta = _mock_simulator(task_id=SIM_TASK_ID)
    app.state.backend.client = SimulatorClient(settings, client=http_ok)  # type: ignore[attr-defined]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        retry = await ac.post(
            BackendRoutes.APPROVE.format(proposal_id=pid),
            json={"operator_identity": "ops"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert retry.status_code == 200
        assert retry.json()["proposal"]["status"] == "executed"
        assert retry.json()["proposal"]["task_id"] == SIM_TASK_ID
        assert meta["create_calls"] == 1

    store.close()
    await http_ok.aclose()
