"""Unit tests for shared transport contracts and settings."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from casepilot.shared import (
    BackendRoutes,
    CaseRecord,
    CreateFollowUpTaskRequest,
    Diagnosis,
    DocumentRecord,
    Evidence,
    FollowUpTaskPayload,
    InvestigationResult,
    NextAction,
    ProcessingEvent,
    Proposal,
    ProposalStatus,
    RunStatus,
    Settings,
    SimulatorRoutes,
    StartInvestigationRequest,
    get_settings,
)


def test_diagnosis_and_next_action_values() -> None:
    assert set(Diagnosis) == {
        Diagnosis.MISSING_DOCUMENT,
        Diagnosis.DATA_MISMATCH,
        Diagnosis.PROCESSING_FAILURE,
        Diagnosis.INDETERMINATE,
        Diagnosis.NO_ISSUE,
    }
    assert Diagnosis.MISSING_DOCUMENT == "missing_document"
    assert NextAction.CREATE_FOLLOW_UP_TASK == "create_follow_up_task"
    assert ProposalStatus.PENDING == "pending"
    assert RunStatus.COMPLETED == "completed"


def test_case_document_event_round_trip() -> None:
    now = datetime(2024, 1, 15, 12, 0, tzinfo=UTC)
    case = CaseRecord(
        case_id="CASE-001",
        status="stalled",
        symptom="awaiting documents",
        application_fields={"applicant_name": "Ada"},
        required_documents=["id_scan"],
        created_at=now,
    )
    doc = DocumentRecord(
        document_id="DOC-1",
        case_id="CASE-001",
        doc_type="id_scan",
        status="missing",
        received_at=None,
        fields={},
    )
    event = ProcessingEvent(
        event_id="EVT-1",
        case_id="CASE-001",
        event_type="dependency_failure",
        timestamp=now,
        message="OCR service timeout",
        details={"code": "TIMEOUT"},
    )
    assert CaseRecord.model_validate(case.model_dump()).case_id == "CASE-001"
    assert doc.doc_type == "id_scan"
    assert event.details["code"] == "TIMEOUT"


def test_investigation_result_and_proposal() -> None:
    task = FollowUpTaskPayload(
        title="Request ID scan",
        body="Applicant ID scan is missing.",
        case_id="CASE-001",
    )
    result = InvestigationResult(
        run_id="run-1",
        case_id="CASE-001",
        status=RunStatus.COMPLETED,
        diagnosis=Diagnosis.MISSING_DOCUMENT,
        diagnosis_summary="Missing ID scan",
        explanation="Required document not on file.",
        evidence=[
            Evidence(
                source_system="documents",
                record_id="DOC-1",
                field="status",
                fact="missing",
                retrieval_ref="tool:list_documents#1",
            )
        ],
        uncertainty=None,
        unresolved_questions=[],
        next_action=NextAction.CREATE_FOLLOW_UP_TASK,
        next_action_explanation="Create internal follow-up to request document.",
        approval_required=True,
        proposed_task=task,
        proposal_id="prop-1",
    )
    assert result.approval_required is True
    assert result.proposed_task is not None
    assert result.proposal_id == "prop-1"
    # optional: omit proposal_id for backwards compatibility
    bare = result.model_dump()
    del bare["proposal_id"]
    assert InvestigationResult.model_validate(bare).proposal_id is None

    proposal = Proposal(
        proposal_id="prop-1",
        run_id="run-1",
        case_id="CASE-001",
        action=NextAction.CREATE_FOLLOW_UP_TASK,
        task_payload=task,
        status=ProposalStatus.PENDING,
        payload_digest="abc123",
        created_at=datetime(2024, 1, 15, 12, 0, tzinfo=UTC),
    )
    assert proposal.status is ProposalStatus.PENDING


def test_start_investigation_and_follow_up_request() -> None:
    req = StartInvestigationRequest(case_id="CASE-001", objective="Find blocker")
    assert req.case_id == "CASE-001"

    follow = CreateFollowUpTaskRequest(
        proposal_id="prop-1",
        task=FollowUpTaskPayload(title="t", body="b", case_id="CASE-001"),
    )
    assert follow.proposal_id == "prop-1"


def test_route_path_templates() -> None:
    assert SimulatorRoutes.CASE.format(case_id="CASE-001") == "/v1/cases/CASE-001"
    assert SimulatorRoutes.DOCUMENTS.format(case_id="CASE-001") == "/v1/cases/CASE-001/documents"
    assert SimulatorRoutes.EVENTS.format(case_id="CASE-001") == "/v1/cases/CASE-001/events"
    assert (
        SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id="CASE-001")
        == "/v1/cases/CASE-001/follow-up-tasks"
    )
    assert BackendRoutes.INVESTIGATIONS == "/v1/investigations"
    assert BackendRoutes.RUN.format(run_id="run-1") == "/v1/investigations/run-1"
    assert BackendRoutes.PROPOSAL.format(proposal_id="p1") == "/v1/proposals/p1"
    assert BackendRoutes.APPROVE.format(proposal_id="p1") == "/v1/proposals/p1/approve"
    assert BackendRoutes.REJECT.format(proposal_id="p1") == "/v1/proposals/p1/reject"


def test_settings_defaults_and_required_tokens() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]

    settings = get_settings(
        _env_file=None,
        simulator_service_token="test-sim-token",
        operator_token="test-op-token",
    )
    assert settings.openrouter_model == "z-ai/glm-5.3-flash"
    assert settings.backend_db_path == Path(".data/backend.sqlite3")
    assert settings.simulator_db_path == Path(".data/simulator.sqlite3")
    assert settings.trace_dir == Path("traces")
    assert settings.backend_host == "127.0.0.1"
    assert settings.simulator_port == 8001


def test_settings_secrets_not_leaked_in_repr() -> None:
    settings = get_settings(
        _env_file=None,
        simulator_service_token="super-secret-sim",
        operator_token="super-secret-op",
        openrouter_api_key="sk-secret-key",
    )
    text = repr(settings)
    assert "super-secret-sim" not in text
    assert "super-secret-op" not in text
    assert "sk-secret-key" not in text
    assert settings.simulator_service_token.get_secret_value() == "super-secret-sim"
    assert settings.operator_token.get_secret_value() == "super-secret-op"


def test_settings_extra_env_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMULATOR_SERVICE_TOKEN", "env-sim")
    monkeypatch.setenv("OPERATOR_TOKEN", "env-op")
    monkeypatch.setenv("TOTALLY_UNKNOWN_FLAG", "1")
    settings = Settings(_env_file=None)
    assert settings.simulator_service_token.get_secret_value() == "env-sim"
    assert not hasattr(settings, "totally_unknown_flag")
