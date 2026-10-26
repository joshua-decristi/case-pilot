"""App approve/reject auth denial emits forbidden_attempt trace (no paid calls)."""

from __future__ import annotations

import json
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
from casepilot.backend.pilot_observability import EVENT_FORBIDDEN
from casepilot.shared.config import get_settings
from casepilot.shared.contracts import (
    BackendRoutes,
    CreateFollowUpTaskRequest,
    CreateFollowUpTaskResponse,
    FollowUpTaskPayload,
    NextAction,
    RunStatus,
    SimulatorRoutes,
)

CASE_ID = "CASE-001"
TOKEN_SIM = "test-sim-token"
TOKEN_OP = "test-op-token"
SIM_TASK_ID = "TASK-TRACE-1"


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
                    task_id=SIM_TASK_ID,
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


def _forbidden_rows(trace_dir: Path, run_id: str) -> list[dict[str, Any]]:
    path = trace_dir / f"{run_id}.jsonl"
    assert path.is_file(), path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r.get("event") == EVENT_FORBIDDEN]


@pytest.fixture(autouse=True)
def _clean_hooks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(TEST_MODE_ENV, "1")
    clear_test_hooks()
    yield
    clear_test_hooks()


@pytest.mark.asyncio
async def test_approve_wrong_token_emits_forbidden_attempt_trace(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    pid = _seed_pending_proposal(store)
    http = _mock_simulator()
    app = create_app(settings, store=store, client=SimulatorClient(settings, client=http))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        bad = await ac.post(
            BackendRoutes.APPROVE.format(proposal_id=pid),
            json={"operator_identity": "eve"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert bad.status_code == 403

    fb = _forbidden_rows(settings.trace_dir, "run-1")
    assert len(fb) == 1
    assert fb[0]["kind"] == "approve_proposal"
    assert fb[0]["detail"] == "http_403"
    assert fb[0]["blocked"] is True
    assert fb[0]["proposal_id"] == pid
    assert fb[0]["case_id"] == CASE_ID
    assert fb[0]["run_id"] == "run-1"

    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_reject_missing_auth_emits_forbidden_attempt_trace(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    pid = _seed_pending_proposal(store)
    http = _mock_simulator()
    app = create_app(settings, store=store, client=SimulatorClient(settings, client=http))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        bad = await ac.post(
            BackendRoutes.REJECT.format(proposal_id=pid),
            json={"operator_identity": "eve"},
        )
        assert bad.status_code == 401

    fb = _forbidden_rows(settings.trace_dir, "run-1")
    assert len(fb) == 1
    assert fb[0]["kind"] == "reject_proposal"
    assert fb[0]["detail"] == "http_401"
    assert fb[0]["blocked"] is True
    assert fb[0]["proposal_id"] == pid

    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_reject_unknown_proposal_traces_unscoped(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    app = create_app(settings, store=store, client=SimulatorClient(settings, client=http))
    missing = "prop-missing"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        bad = await ac.post(
            BackendRoutes.REJECT.format(proposal_id=missing),
            json={"operator_identity": "eve"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert bad.status_code == 403

    fb = _forbidden_rows(settings.trace_dir, "_unscoped")
    assert len(fb) == 1
    assert fb[0]["kind"] == "reject_proposal"
    assert fb[0]["detail"] == "http_403"
    assert fb[0]["blocked"] is True
    assert fb[0]["proposal_id"] == missing
    assert "run_id" not in fb[0] or fb[0].get("run_id") in (None, "_unscoped")

    store.close()
    await http.aclose()
