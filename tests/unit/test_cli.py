"""CLI security/timeout checks (parser and mock-flow command units deleted)."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from typing import Any

import httpx

from casepilot.cli import (
    CLIENT_TIMEOUT_SLACK_SECONDS,
    BackendClient,
    client_timeout_seconds,
    run,
)
from casepilot.shared import (
    BackendRoutes,
    FollowUpTaskPayload,
    NextAction,
    Proposal,
    ProposalDecisionResponse,
    ProposalStatus,
)
from casepilot.shared.config import Settings

OPERATOR_TOKEN = "test-operator-token-secret"
SERVICE_TOKEN = "test-simulator-token-secret"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "simulator_service_token": SERVICE_TOKEN,
        "operator_token": OPERATOR_TOKEN,
        "backend_base_url": "http://127.0.0.1:8000",
        "backend_host": "127.0.0.1",
        "backend_port": 8000,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _now() -> datetime:
    return datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def _proposal(**overrides: Any) -> Proposal:
    data: dict[str, Any] = {
        "proposal_id": "prop-1",
        "run_id": "run-1",
        "case_id": "CASE-001",
        "action": NextAction.CREATE_FOLLOW_UP_TASK,
        "task_payload": FollowUpTaskPayload(
            title="Request ID",
            body="Need ID scan",
            case_id="CASE-001",
        ),
        "status": ProposalStatus.PENDING,
        "payload_digest": "abc123",
        "created_at": _now(),
        "decided_at": None,
        "operator_identity": None,
        "execution_error": None,
    }
    data.update(overrides)
    return Proposal(**data)


def _mock_client(handler: Any) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://127.0.0.1:8000",
    )


def test_approve_sends_bearer_token_not_printed() -> None:
    decided = _proposal(
        status=ProposalStatus.EXECUTED,
        decided_at=_now(),
        operator_identity="local-operator",
    )
    response = ProposalDecisionResponse(proposal=decided)
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == BackendRoutes.APPROVE.format(proposal_id="prop-1")
        seen["authorization"] = request.headers.get("Authorization", "")
        return httpx.Response(200, json=response.model_dump(mode="json"))

    out = io.StringIO()
    err = io.StringIO()
    code = run(
        ["approve", "prop-1", "--json"],
        settings=_settings(),
        http_client=_mock_client(handler),
        stdout=out,
        stderr=err,
    )
    assert code == 0
    assert seen["authorization"] == f"Bearer {OPERATOR_TOKEN}"
    printed = out.getvalue() + err.getvalue()
    assert OPERATOR_TOKEN not in printed
    assert "Bearer" not in printed
    assert json.loads(out.getvalue())["proposal"]["status"] == "executed"


def test_client_timeout_exceeds_backend_budget() -> None:
    settings = _settings(investigation_timeout_seconds=60.0)
    assert client_timeout_seconds(settings) == 60.0 + CLIENT_TIMEOUT_SLACK_SECONDS
    assert client_timeout_seconds(settings) > settings.investigation_timeout_seconds

    with BackendClient(settings) as backend:
        timeout = backend._client.timeout
        assert timeout is not None
        assert float(timeout.read) == client_timeout_seconds(settings)  # type: ignore[arg-type]
