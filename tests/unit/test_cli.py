"""Unit tests for CasePilot CLI (mocked HTTP only; no paid calls)."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from casepilot.cli import (
    CLIENT_TIMEOUT_SLACK_SECONDS,
    DEFAULT_OBJECTIVE,
    BackendClient,
    build_parser,
    client_timeout_seconds,
    run,
)
from casepilot.shared import (
    BackendRoutes,
    Diagnosis,
    FollowUpTaskPayload,
    InvestigationResult,
    NextAction,
    Proposal,
    ProposalDecisionResponse,
    ProposalStatus,
    RunRecord,
    RunStatus,
    StartInvestigationResponse,
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


def _result(**overrides: Any) -> InvestigationResult:
    data: dict[str, Any] = {
        "run_id": "run-1",
        "case_id": "CASE-001",
        "status": RunStatus.COMPLETED,
        "diagnosis": Diagnosis.MISSING_DOCUMENT,
        "diagnosis_summary": "Missing ID scan",
        "explanation": "Required document not on file.",
        "evidence": [],
        "uncertainty": None,
        "unresolved_questions": [],
        "next_action": NextAction.CREATE_FOLLOW_UP_TASK,
        "next_action_explanation": "Create a follow-up task",
        "approval_required": True,
        "proposed_task": FollowUpTaskPayload(
            title="Request ID",
            body="Need ID scan",
            case_id="CASE-001",
        ),
    }
    data.update(overrides)
    return InvestigationResult(**data)


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


class TestParser:
    def test_investigate_defaults(self) -> None:
        args = build_parser().parse_args(["investigate", "CASE-001"])
        assert args.command == "investigate"
        assert args.case_id == "CASE-001"
        assert args.objective == DEFAULT_OBJECTIVE
        assert args.json is False

    def test_investigate_objective_and_json(self) -> None:
        args = build_parser().parse_args(
            ["investigate", "CASE-9", "--objective", "Find blocker", "--json"]
        )
        assert args.objective == "Find blocker"
        assert args.json is True

    def test_all_subcommands(self) -> None:
        p = build_parser()
        assert p.parse_args(["run", "run-1"]).run_id == "run-1"
        assert p.parse_args(["proposal", "prop-1"]).proposal_id == "prop-1"
        assert p.parse_args(["approve", "prop-1"]).proposal_id == "prop-1"
        assert p.parse_args(["reject", "prop-1", "--reason", "nope"]).reason == "nope"


class TestCliCommands:
    def test_investigate_success_human_and_json(self) -> None:
        payload = StartInvestigationResponse(
            run_id="run-1",
            result=_result(),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert request.url.path == BackendRoutes.INVESTIGATIONS
            body = json.loads(request.content)
            assert body["case_id"] == "CASE-001"
            assert body["objective"] == DEFAULT_OBJECTIVE
            assert "Authorization" not in request.headers
            return httpx.Response(200, json=payload.model_dump(mode="json"))

        settings = _settings()
        out = io.StringIO()
        code = run(
            ["investigate", "CASE-001"],
            settings=settings,
            http_client=_mock_client(handler),
            stdout=out,
        )
        assert code == 0
        text = out.getvalue()
        assert "run-1" in text
        assert "missing_document" in text
        assert OPERATOR_TOKEN not in text
        assert SERVICE_TOKEN not in text

        out_json = io.StringIO()
        code = run(
            ["investigate", "CASE-001", "--json"],
            settings=settings,
            http_client=_mock_client(handler),
            stdout=out_json,
        )
        assert code == 0
        parsed = json.loads(out_json.getvalue())
        assert parsed["run_id"] == "run-1"
        assert parsed["result"]["diagnosis"] == "missing_document"

    def test_run_and_proposal_get(self) -> None:
        run_rec = RunRecord(
            run_id="run-1",
            case_id="CASE-001",
            objective="obj",
            status=RunStatus.COMPLETED,
            result=_result(),
            error=None,
            created_at=_now(),
            completed_at=_now(),
        )
        prop = _proposal()

        def handler(request: httpx.Request) -> httpx.Response:
            assert "Authorization" not in request.headers
            if request.url.path == BackendRoutes.RUN.format(run_id="run-1"):
                return httpx.Response(200, json=run_rec.model_dump(mode="json"))
            if request.url.path == BackendRoutes.PROPOSAL.format(proposal_id="prop-1"):
                return httpx.Response(200, json=prop.model_dump(mode="json"))
            return httpx.Response(404, json={"detail": "not found"})

        settings = _settings()
        out = io.StringIO()
        assert (
            run(
                ["run", "run-1", "--json"],
                settings=settings,
                http_client=_mock_client(handler),
                stdout=out,
            )
            == 0
        )
        assert json.loads(out.getvalue())["run_id"] == "run-1"

        out2 = io.StringIO()
        assert (
            run(
                ["proposal", "prop-1", "--json"],
                settings=settings,
                http_client=_mock_client(handler),
                stdout=out2,
            )
            == 0
        )
        assert json.loads(out2.getvalue())["proposal_id"] == "prop-1"

    def test_approve_sends_bearer_token_not_printed(self) -> None:
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

    def test_reject_sends_bearer_and_reason(self) -> None:
        decided = _proposal(
            status=ProposalStatus.REJECTED,
            decided_at=_now(),
            operator_identity="local-operator",
        )
        response = ProposalDecisionResponse(proposal=decided)
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == BackendRoutes.REJECT.format(proposal_id="prop-1")
            captured["auth"] = request.headers.get("Authorization")
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=response.model_dump(mode="json"))

        out = io.StringIO()
        code = run(
            ["reject", "prop-1", "--reason", "duplicate", "--json"],
            settings=_settings(),
            http_client=_mock_client(handler),
            stdout=out,
        )
        assert code == 0
        assert captured["auth"] == f"Bearer {OPERATOR_TOKEN}"
        assert captured["body"]["reason"] == "duplicate"
        assert OPERATOR_TOKEN not in out.getvalue()

    def test_api_error_nonzero_exit(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "case not found"})

        err = io.StringIO()
        out = io.StringIO()
        code = run(
            ["investigate", "MISSING"],
            settings=_settings(),
            http_client=_mock_client(handler),
            stdout=out,
            stderr=err,
        )
        assert code == 1
        assert "case not found" in err.getvalue()
        assert OPERATOR_TOKEN not in err.getvalue()


class TestTimeoutAndProposalSurface:
    def test_client_timeout_exceeds_backend_budget(self) -> None:
        settings = _settings(investigation_timeout_seconds=60.0)
        assert client_timeout_seconds(settings) == 60.0 + CLIENT_TIMEOUT_SLACK_SECONDS
        assert client_timeout_seconds(settings) > settings.investigation_timeout_seconds

        with BackendClient(settings) as backend:
            timeout = backend._client.timeout
            assert timeout is not None
            # httpx.Timeout stores read/pool/connect; default constructor sets all.
            assert float(timeout.read) == client_timeout_seconds(settings)  # type: ignore[arg-type]

    def test_investigate_human_surfaces_proposal_id_and_approve_command(self) -> None:
        """CLI investigate entry: proposal ID + approve callout, absent when none."""

        def handler_for(payload: StartInvestigationResponse):
            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json=payload.model_dump(mode="json"))

            return handler

        with_prop = StartInvestigationResponse(
            run_id="run-1",
            result=_result(proposal_id="prop-9"),
        )
        out = io.StringIO()
        code = run(
            ["investigate", "CASE-001"],
            settings=_settings(),
            http_client=_mock_client(handler_for(with_prop)),
            stdout=out,
        )
        assert code == 0
        text = out.getvalue()
        assert text.startswith("proposal_id: prop-9")
        assert "casepilot approve prop-9" in text

        bare_payload = StartInvestigationResponse(
            run_id="run-1",
            result=_result(proposal_id=None),
        )
        bare_out = io.StringIO()
        assert (
            run(
                ["investigate", "CASE-001"],
                settings=_settings(),
                http_client=_mock_client(handler_for(bare_payload)),
                stdout=bare_out,
            )
            == 0
        )
        bare = bare_out.getvalue()
        assert not bare.startswith("proposal_id:")
        assert "casepilot approve" not in bare


class TestBackendClientDirect:
    def test_uses_backend_base_url(self) -> None:
        hit: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            hit.append(str(request.url))
            payload = StartInvestigationResponse(
                run_id="run-1",
                result=_result(),
            )
            return httpx.Response(200, json=payload.model_dump(mode="json"))

        client = httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:8000",
        )
        with BackendClient(_settings(), client=client) as backend:
            backend.investigate("CASE-001", "obj")
        assert hit[0].startswith("http://127.0.0.1:8000/")
        assert BackendRoutes.INVESTIGATIONS in hit[0]

    def test_transport_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        with BackendClient(_settings(), client=_mock_client(handler)) as backend:
            with pytest.raises(Exception) as exc_info:
                backend.get_run("run-x")
        assert "Backend unreachable" in str(exc_info.value)
