"""Integration tests for the enterprise simulator FastAPI app + SQLite."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from casepilot.shared import (
    CaseRecord,
    CreateFollowUpTaskRequest,
    CreateFollowUpTaskResponse,
    DocumentListResponse,
    EventListResponse,
    FollowUpTaskPayload,
    SimulatorRoutes,
    get_settings,
)
from casepilot.simulator.app import create_app
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


def test_health_unauthenticated(client: TestClient) -> None:
    resp = client.get(SimulatorRoutes.HEALTH)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_read_requires_service_token(client: TestClient) -> None:
    path = SimulatorRoutes.CASE.format(case_id="CASE-001")
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(path, headers={"Authorization": "Token x"}).status_code == 401


def test_get_case_ok_and_scoped(client: TestClient) -> None:
    resp = client.get(
        SimulatorRoutes.CASE.format(case_id="CASE-001"),
        headers=AUTH,
    )
    assert resp.status_code == 200
    case = CaseRecord.model_validate(resp.json())
    assert case.case_id == "CASE-001"
    assert case.status
    assert case.symptom
    assert "applicant_name" in case.application_fields
    assert isinstance(case.required_documents, list)
    assert case.required_documents


def test_get_case_missing_404(client: TestClient) -> None:
    resp = client.get(
        SimulatorRoutes.CASE.format(case_id="CASE-DOES-NOT-EXIST"),
        headers=AUTH,
    )
    assert resp.status_code == 404


def test_documents_case_scoped_and_contract(client: TestClient) -> None:
    resp = client.get(
        SimulatorRoutes.DOCUMENTS.format(case_id="CASE-001"),
        headers=AUTH,
    )
    assert resp.status_code == 200
    body = DocumentListResponse.model_validate(resp.json())
    assert body.case_id == "CASE-001"
    assert body.documents
    assert all(d.case_id == "CASE-001" for d in body.documents)
    assert all(d.document_id and d.doc_type and d.status for d in body.documents)

    missing = client.get(
        SimulatorRoutes.DOCUMENTS.format(case_id="CASE-MISSING"),
        headers=AUTH,
    )
    assert missing.status_code == 404


def test_events_case_scoped_and_contract(client: TestClient) -> None:
    resp = client.get(
        SimulatorRoutes.EVENTS.format(case_id="CASE-001"),
        headers=AUTH,
    )
    assert resp.status_code == 200
    body = EventListResponse.model_validate(resp.json())
    assert body.case_id == "CASE-001"
    assert body.events
    assert all(e.case_id == "CASE-001" for e in body.events)
    assert all(e.event_id and e.event_type and e.message is not None for e in body.events)

    missing = client.get(
        SimulatorRoutes.EVENTS.format(case_id="CASE-MISSING"),
        headers=AUTH,
    )
    assert missing.status_code == 404


def test_follow_up_requires_auth_and_valid_case(client: TestClient) -> None:
    path = SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id="CASE-001")
    payload = CreateFollowUpTaskRequest(
        proposal_id="prop-1",
        task=FollowUpTaskPayload(
            title="Request proof of income",
            body="Proof of income is missing for CASE-001.",
            case_id="CASE-001",
        ),
    ).model_dump()

    assert client.post(path, json=payload).status_code == 401

    bad_case = client.post(
        SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id="CASE-MISSING"),
        headers=AUTH,
        json={
            "proposal_id": "prop-x",
            "task": {
                "title": "t",
                "body": "b",
                "case_id": "CASE-MISSING",
            },
        },
    )
    assert bad_case.status_code == 404

    mismatch = client.post(
        path,
        headers=AUTH,
        json={
            "proposal_id": "prop-mismatch-path",
            "task": {"title": "t", "body": "b", "case_id": "CASE-002"},
        },
    )
    assert mismatch.status_code == 400


def test_follow_up_bounds_enforced(client: TestClient) -> None:
    path = SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id="CASE-001")
    resp = client.post(
        path,
        headers=AUTH,
        json={
            "proposal_id": "prop-bounds",
            "task": {
                "title": "x" * 201,
                "body": "ok",
                "case_id": "CASE-001",
            },
        },
    )
    assert resp.status_code == 422

    resp = client.post(
        path,
        headers=AUTH,
        json={
            "proposal_id": "prop-bounds-body",
            "task": {
                "title": "ok",
                "body": "y" * 4001,
                "case_id": "CASE-001",
            },
        },
    )
    assert resp.status_code == 422


def test_follow_up_create_idempotent_and_conflict(client: TestClient) -> None:
    path = SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id="CASE-001")
    request = CreateFollowUpTaskRequest(
        proposal_id="prop-idem-1",
        task=FollowUpTaskPayload(
            title="Chase missing APS",
            body="Request attending physician statement.",
            case_id="CASE-001",
        ),
    )

    first = client.post(path, headers=AUTH, json=request.model_dump())
    assert first.status_code == 200
    created = CreateFollowUpTaskResponse.model_validate(first.json())
    assert created.created is True
    assert created.proposal_id == "prop-idem-1"
    assert created.case_id == "CASE-001"
    assert created.task_id

    second = client.post(path, headers=AUTH, json=request.model_dump())
    assert second.status_code == 200
    replay = CreateFollowUpTaskResponse.model_validate(second.json())
    assert replay.created is False
    assert replay.task_id == created.task_id
    assert replay.proposal_id == created.proposal_id

    conflict = client.post(
        path,
        headers=AUTH,
        json={
            "proposal_id": "prop-idem-1",
            "task": {
                "title": "Different title",
                "body": "Different body",
                "case_id": "CASE-001",
            },
        },
    )
    assert conflict.status_code == 409

    # Same proposal id against a different case path also conflicts.
    other = client.post(
        SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id="CASE-002"),
        headers=AUTH,
        json={
            "proposal_id": "prop-idem-1",
            "task": {
                "title": "Chase missing APS",
                "body": "Request attending physician statement.",
                "case_id": "CASE-002",
            },
        },
    )
    assert other.status_code == 409


def test_seed_safety_default_refuses_populated_explicit_reset_ok(tmp_path: Path) -> None:
    """Default/non-reset seed must not wipe data; explicit reset=True rebuilds."""
    path = tmp_path / "seed-safety.db"
    assert seed(db_path=path, reset=True) == 0, "explicit reset must succeed on empty/new db"
    assert seed(db_path=path, reset=False) == 1, "reset=False must refuse nonempty db"
    assert seed(db_path=path) == 1, "default seed (reset=False) must refuse nonempty db"
    assert seed(db_path=path, reset=True) == 0, "explicit reset must succeed on populated db"


def test_seed_is_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a.db"
    b = tmp_path / "b.db"
    assert seed(db_path=a, reset=True) == 0
    assert seed(db_path=b, reset=True) == 0
    assert a.read_bytes() == b.read_bytes()


def test_app_startup_does_not_overwrite_seed(tmp_path: Path) -> None:
    path = tmp_path / "persist.db"
    assert seed(db_path=path, reset=True) == 0
    before = path.read_bytes()

    settings = get_settings(
        _env_file=None,
        simulator_service_token=TOKEN,
        operator_token=OPERATOR,
        simulator_db_path=path,
    )
    app = create_app(settings=settings, db_path=path)
    with TestClient(app) as test_client:
        resp = test_client.get(
            SimulatorRoutes.CASE.format(case_id="CASE-001"),
            headers=AUTH,
        )
        assert resp.status_code == 200

    # Startup must not reseed/wipe; follow-up table may be empty still — file
    # should still contain original case data (byte-identical if no writes).
    assert path.read_bytes() == before


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


def test_concurrent_mixed_reads_and_writes(client: TestClient) -> None:
    case_path = SimulatorRoutes.CASE.format(case_id="CASE-001")
    write_path = SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id="CASE-001")

    def _read() -> int:
        return client.get(case_path, headers=AUTH).status_code

    def _write(i: int) -> int:
        return client.post(
            write_path,
            headers=AUTH,
            json={
                "proposal_id": f"prop-mixed-{i}",
                "task": {
                    "title": f"Mixed write {i}",
                    "body": f"body {i}",
                    "case_id": "CASE-001",
                },
            },
        ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        read_futs = [pool.submit(_read) for _ in range(12)]
        write_futs = [pool.submit(_write, i) for i in range(8)]
        read_codes = [f.result() for f in read_futs]
        write_codes = [f.result() for f in write_futs]

    assert read_codes == [200] * 12
    assert write_codes == [200] * 8
