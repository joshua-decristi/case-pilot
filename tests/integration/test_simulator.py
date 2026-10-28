"""Representative simulator integration: auth, concurrency, outage, seed safety.

CRUD/schema/idempotent happy-path units deleted; domain/boundary cover real CRUD.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from casepilot.shared import (
    CreateFollowUpTaskResponse,
    SimulatorRoutes,
    get_settings,
)
from casepilot.simulator.app import clear_outage, create_app
from casepilot.simulator.seed import seed

TOKEN = "test-simulator-service-token"
OPERATOR = "test-operator-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "simulator.sqlite3"
    rc = seed(db_path=path, reset=True)
    assert rc == 0
    return path


@pytest.fixture()
def client(db_path: Path) -> Iterator[TestClient]:
    settings = get_settings(
        _env_file=None,
        simulator_service_token=TOKEN,
        operator_token=OPERATOR,
        simulator_db_path=db_path,
    )
    app = create_app(settings=settings, db_path=db_path)
    with TestClient(app) as test_client:
        yield test_client


def test_read_requires_service_token(client: TestClient) -> None:
    path = SimulatorRoutes.CASE.format(case_id="CASE-001")
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(path, headers={"Authorization": "Token x"}).status_code == 401
    assert client.get(path, headers=AUTH).status_code == 200


def test_follow_up_requires_auth(client: TestClient) -> None:
    path = SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id="CASE-001")
    payload = {
        "proposal_id": "prop-1",
        "task": {
            "title": "Request proof of income",
            "body": "Proof of income is missing for CASE-001.",
            "case_id": "CASE-001",
        },
    }
    assert client.post(path, json=payload).status_code == 401
    assert client.post(path, headers=AUTH, json=payload).status_code == 200


def test_seed_safety_default_refuses_populated_explicit_reset_ok(tmp_path: Path) -> None:
    """Default/non-reset seed must not wipe data; explicit reset=True rebuilds."""
    path = tmp_path / "seed-safety.db"
    assert seed(db_path=path, reset=True) == 0, "explicit reset must succeed on empty/new db"
    assert seed(db_path=path, reset=False) == 1, "reset=False must refuse nonempty db"
    assert seed(db_path=path) == 1, "default seed (reset=False) must refuse nonempty db"
    assert seed(db_path=path, reset=True) == 0, "explicit reset must succeed on populated db"


def test_concurrent_identical_follow_up_writes(client: TestClient) -> None:
    path = SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id="CASE-001")
    payload = {
        "proposal_id": "prop-concurrent-same",
        "task": {
            "title": "Concurrent identical write",
            "body": "Same proposal and payload from many threads.",
            "case_id": "CASE-001",
        },
    }

    def _post() -> tuple[int, dict[str, object]]:
        resp = client.post(path, headers=AUTH, json=payload)
        return resp.status_code, resp.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _post(), range(16)))

    assert all(code == 200 for code, _ in results)
    bodies = [CreateFollowUpTaskResponse.model_validate(body) for _, body in results]
    task_ids = {b.task_id for b in bodies}
    assert len(task_ids) == 1
    assert sum(1 for b in bodies if b.created) == 1
    assert sum(1 for b in bodies if not b.created) == 15


def test_concurrent_conflicting_follow_up_writes(client: TestClient) -> None:
    path = SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id="CASE-001")
    proposal_id = "prop-concurrent-conflict"

    def _post(title: str) -> int:
        return client.post(
            path,
            headers=AUTH,
            json={
                "proposal_id": proposal_id,
                "task": {
                    "title": title,
                    "body": f"body for {title}",
                    "case_id": "CASE-001",
                },
            },
        ).status_code

    titles = [f"Title-{i}" for i in range(12)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_post, t) for t in titles]
        codes = [f.result() for f in as_completed(futures)]

    assert codes.count(200) == 1
    assert codes.count(409) == 11


def test_outage_routes_absent_without_test_mode(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CASEPILOT_TEST_MODE", raising=False)
    clear_outage()
    settings = get_settings(
        _env_file=None,
        simulator_service_token=TOKEN,
        operator_token=OPERATOR,
        simulator_db_path=db_path,
    )
    app = create_app(settings=settings, db_path=db_path)
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/v1/_test/outage" not in paths
    assert "/v1/_test/mode" not in paths


def test_reversible_simulator_outage_injection(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test-mode outage 503s domain routes and clears cleanly (approval failure inject)."""
    monkeypatch.setenv("CASEPILOT_TEST_MODE", "1")
    clear_outage()
    settings = get_settings(
        _env_file=None,
        simulator_service_token=TOKEN,
        operator_token=OPERATOR,
        simulator_db_path=db_path,
    )
    app = create_app(settings=settings, db_path=db_path)
    with TestClient(app) as client:
        mode = client.get("/v1/_test/mode")
        assert mode.status_code == 200
        assert mode.json()["outage"] == "0"

        case_path = SimulatorRoutes.CASE.format(case_id="CASE-001")
        assert client.get(case_path, headers=AUTH).status_code == 200

        on = client.put("/v1/_test/outage", json={"enabled": True})
        assert on.status_code == 200
        assert on.json()["outage"] == "1"
        assert client.get("/v1/_test/mode").json()["outage"] == "1"
        assert client.get(case_path, headers=AUTH).status_code == 503
        assert client.get(SimulatorRoutes.HEALTH).status_code == 200

        task_path = SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id="CASE-001")
        write = client.post(
            task_path,
            headers=AUTH,
            json={
                "proposal_id": "prop-outage",
                "task": {
                    "title": "t",
                    "body": "b",
                    "case_id": "CASE-001",
                },
            },
        )
        assert write.status_code == 503

        off = client.put("/v1/_test/outage", json={"enabled": False})
        assert off.status_code == 200
        assert off.json()["outage"] == "0"
        assert client.get(case_path, headers=AUTH).status_code == 200
    clear_outage()
