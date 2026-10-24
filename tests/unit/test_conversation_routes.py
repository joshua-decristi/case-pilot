"""API tests for conversation GET/send/retry (scripted follow-up, no paid calls)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from casepilot.backend.app import create_app
from casepilot.backend.client import SimulatorClient
from casepilot.backend.db import BackendStore
from casepilot.backend.hooks import TEST_MODE_ENV, clear_test_hooks
from casepilot.shared.config import get_settings
from casepilot.shared.contracts import (
    BackendRoutes,
    FollowUpAnswer,
    RunStatus,
    TurnStatus,
)

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
    }
    base.update(extra)
    return get_settings(**base)


def _seed_run(
    store: BackendStore,
    run_id: str = "run-1",
    *,
    status: RunStatus = RunStatus.COMPLETED,
) -> None:
    store.insert_run(
        run_id=run_id,
        case_id="CASE-001",
        objective="why stalled",
        status=status,
        completed_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _answer() -> FollowUpAnswer:
    return FollowUpAnswer(notes=["ok"])


async def _succeed_follow_up(**kwargs: object) -> None:
    store = kwargs["store"]
    assert isinstance(store, BackendStore)
    attempt_id = kwargs["attempt_id"]
    assert isinstance(attempt_id, str)
    store.complete_attempt(attempt_id, answer=_answer())


async def _fail_follow_up(**kwargs: object) -> None:
    store = kwargs["store"]
    assert isinstance(store, BackendStore)
    attempt_id = kwargs["attempt_id"]
    assert isinstance(attempt_id, str)
    store.fail_attempt(attempt_id, error="model blew up")


@pytest.fixture(autouse=True)
def _clean_hooks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(TEST_MODE_ENV, "1")
    clear_test_hooks()
    yield
    clear_test_hooks()


def _install_followup_module(monkeypatch: pytest.MonkeyPatch, impl: object) -> ModuleType:
    """Stub followup module so API tests do not depend on the agent slice."""
    import sys

    mod = ModuleType("casepilot.backend.followup")
    mod.run_follow_up = impl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "casepilot.backend.followup", mod)
    return mod


@pytest.fixture
def follow_up(monkeypatch: pytest.MonkeyPatch):
    mock = AsyncMock(side_effect=_succeed_follow_up)
    _install_followup_module(monkeypatch, mock)
    return mock


def _app(tmp_path: Path, store: BackendStore | None = None):
    settings = _settings(tmp_path)
    st = store or BackendStore(settings.backend_db_path)
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(404)),
        base_url="http://simulator.test",
    )
    client = SimulatorClient(settings, client=http)
    app = create_app(settings, store=st, client=client)
    return app, st, http


@pytest.mark.asyncio
async def test_get_conversation_empty_and_missing(tmp_path: Path, follow_up):
    app, store, http = _app(tmp_path)
    _seed_run(store)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        ok = await ac.get(BackendRoutes.CONVERSATION.format(run_id="run-1"))
        assert ok.status_code == 200
        assert ok.json() == {"run_id": "run-1", "turns": []}

        missing = await ac.get(BackendRoutes.CONVERSATION.format(run_id="nope"))
        assert missing.status_code == 404
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_send_success_replay_and_eligibility(tmp_path: Path, follow_up):
    app, store, http = _app(tmp_path)
    _seed_run(store)
    path = BackendRoutes.CONVERSATION_MESSAGES.format(run_id="run-1")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        first = await ac.post(path, json={"question": "Why stalled?", "idempotency_key": "k1"})
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["created"] is True
        assert body["turn"]["status"] == TurnStatus.SUCCEEDED.value
        assert body["attempt"]["status"] == TurnStatus.SUCCEEDED.value
        assert body["attempt"]["answer"]["notes"] == ["ok"]
        turn_id = body["turn"]["turn_id"]
        attempt_id = body["attempt"]["attempt_id"]
        assert follow_up.await_count == 1

        replay = await ac.post(path, json={"question": "Why stalled?", "idempotency_key": "k1"})
        assert replay.status_code == 200
        assert replay.json()["created"] is False
        assert replay.json()["attempt"]["attempt_id"] == attempt_id
        assert follow_up.await_count == 1  # no second model call

        mismatch = await ac.post(path, json={"question": "different?", "idempotency_key": "k1"})
        assert mismatch.status_code == 409

        hist = await ac.get(BackendRoutes.CONVERSATION.format(run_id="run-1"))
        assert len(hist.json()["turns"]) == 1
        assert hist.json()["turns"][0]["turn_id"] == turn_id

    # Ineligible run status
    _seed_run(store, run_id="run-fail", status=RunStatus.FAILED)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        bad = await ac.post(
            BackendRoutes.CONVERSATION_MESSAGES.format(run_id="run-fail"),
            json={"question": "x", "idempotency_key": "kf"},
        )
        assert bad.status_code == 409
        missing = await ac.post(
            BackendRoutes.CONVERSATION_MESSAGES.format(run_id="ghost"),
            json={"question": "x", "idempotency_key": "kg"},
        )
        assert missing.status_code == 404
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_send_conflict_and_failure_then_retry(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    _seed_run(store)
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(404)),
        base_url="http://simulator.test",
    )
    client = SimulatorClient(settings, client=http)

    mock = AsyncMock(side_effect=_fail_follow_up)
    _install_followup_module(monkeypatch, mock)
    app = create_app(settings, store=store, client=client)

    path = BackendRoutes.CONVERSATION_MESSAGES.format(run_id="run-1")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        failed = await ac.post(path, json={"question": "Q?", "idempotency_key": "send-1"})
        assert failed.status_code == 200
        body = failed.json()
        assert body["turn"]["status"] == TurnStatus.FAILED.value
        assert body["attempt"]["error"] == "model blew up"
        turn_id = body["turn"]["turn_id"]
        prev_id = body["attempt"]["attempt_id"]

        # Concurrent-style: leave a pending via store, then send conflicts
        pending = store.begin_send(run_id="run-1", question="other", idempotency_key="inflight")
        assert pending.created is True
        conflict = await ac.post(path, json={"question": "nope", "idempotency_key": "other-key"})
        assert conflict.status_code == 409
        store.fail_attempt(pending.attempt.attempt_id, error="clear")

    # Retry after failure of original turn (turn is still failed from first send;
    # the pending we failed was a different turn).
    # Re-fetch: first turn still failed.
    turn = store.get_turn(turn_id)
    assert turn is not None and turn.status is TurnStatus.FAILED

    mock.side_effect = _succeed_follow_up
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        retry_path = BackendRoutes.CONVERSATION_RETRY.format(run_id="run-1", turn_id=turn_id)
        ok = await ac.post(
            retry_path,
            json={
                "idempotency_key": "retry-1",
                "expected_previous_attempt_id": prev_id,
            },
        )
        assert ok.status_code == 200, ok.text
        assert ok.json()["created"] is True
        assert ok.json()["turn"]["status"] == TurnStatus.SUCCEEDED.value
        assert ok.json()["attempt"]["previous_attempt_id"] == prev_id
        new_attempt = ok.json()["attempt"]["attempt_id"]

        # Idempotent retry replay after success
        again = await ac.post(
            retry_path,
            json={
                "idempotency_key": "retry-1",
                "expected_previous_attempt_id": prev_id,
            },
        )
        assert again.status_code == 200
        assert again.json()["created"] is False
        assert again.json()["attempt"]["attempt_id"] == new_attempt

        # Key reuse with different target rejected
        bad_key = await ac.post(
            retry_path,
            json={
                "idempotency_key": "retry-1",
                "expected_previous_attempt_id": "not-the-prev",
            },
        )
        assert bad_key.status_code == 409

        # Bad previous attempt
        bad_prev = await ac.post(
            retry_path,
            json={
                "idempotency_key": "retry-2",
                "expected_previous_attempt_id": prev_id,  # no longer latest
            },
        )
        assert bad_prev.status_code == 409

        missing_turn = await ac.post(
            BackendRoutes.CONVERSATION_RETRY.format(run_id="run-1", turn_id="missing"),
            json={
                "idempotency_key": "retry-x",
                "expected_previous_attempt_id": prev_id,
            },
        )
        assert missing_turn.status_code == 404

    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_follow_up_exception_marks_failed(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    _seed_run(store)
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(404)),
        base_url="http://simulator.test",
    )
    client = SimulatorClient(settings, client=http)
    _install_followup_module(monkeypatch, AsyncMock(side_effect=RuntimeError("boom")))
    app = create_app(settings, store=store, client=client)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.post(
            BackendRoutes.CONVERSATION_MESSAGES.format(run_id="run-1"),
            json={"question": "x", "idempotency_key": "e1"},
        )
        assert resp.status_code == 200
        assert resp.json()["turn"]["status"] == TurnStatus.FAILED.value
        assert resp.json()["attempt"]["error"] == "follow-up failed"
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_startup_recovers_interrupted_pending(tmp_path: Path, follow_up):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    _seed_run(store)
    started = store.begin_send(run_id="run-1", question="hang", idempotency_key="p1")
    assert started.attempt.status is TurnStatus.PENDING
    store.close()

    # Re-open via create_app eager recovery
    store2 = BackendStore(settings.backend_db_path)
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(404)),
        base_url="http://simulator.test",
    )
    client = SimulatorClient(settings, client=http)
    create_app(settings, store=store2, client=client)
    turn = store2.get_turn(started.turn.turn_id)
    assert turn is not None
    assert turn.status is TurnStatus.FAILED
    assert turn.attempts[0].error == "interrupted by restart"
    store2.close()
    await http.aclose()
