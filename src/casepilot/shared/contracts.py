"""Stable HTTP/transport contracts shared by backend and simulator.

These are shape-only Pydantic models and enums — no business rules.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Diagnosis(StrEnum):
    MISSING_DOCUMENT = "missing_document"
    DATA_MISMATCH = "data_mismatch"
    PROCESSING_FAILURE = "processing_failure"
    INDETERMINATE = "indeterminate"
    NO_ISSUE = "no_issue"


class NextAction(StrEnum):
    """Recommended next step. Only CREATE_FOLLOW_UP_TASK is executable via approval."""

    CREATE_FOLLOW_UP_TASK = "create_follow_up_task"
    REQUEST_MISSING_DOCUMENT = "request_missing_document"
    CORRECT_DATA_MISMATCH = "correct_data_mismatch"
    RETRY_PROCESSING = "retry_processing"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    NO_ACTION_REQUIRED = "no_action_required"


class RunStatus(StrEnum):
    COMPLETED = "completed"
    INDETERMINATE = "indeterminate"
    FAILED = "failed"


class ProposalStatus(StrEnum):
    PENDING = "pending"
    REJECTED = "rejected"
    APPROVED = "approved"
    EXECUTED = "executed"
    EXECUTION_FAILED = "execution_failed"


class TurnStatus(StrEnum):
    """Conversation turn/attempt lifecycle: pending → succeeded | failed."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EvidenceProvenance(StrEnum):
    """Whether a snapshot was captured during the original investigation or later."""

    ORIGINAL_RUN = "original_run"
    FOLLOW_UP = "follow_up"


class InterpretationKind(StrEnum):
    INFERENCE = "inference"
    UNCERTAINTY = "uncertainty"


# ---------------------------------------------------------------------------
# Simulator domain records (tool/HTTP payloads)
# ---------------------------------------------------------------------------


class CaseRecord(BaseModel):
    """GET /v1/cases/{case_id} (simulator full record)."""

    case_id: str
    status: str
    symptom: str
    application_fields: dict[str, Any] = Field(default_factory=dict)
    required_documents: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime | None = None


class CaseSummary(BaseModel):
    """GET /v1/cases/{case_id} (backend operator summary)."""

    case_id: str
    status: str
    symptom: str


class CaseListItem(BaseModel):
    """Item in GET /v1/cases (backend)."""

    case_id: str
    status: str
    symptom: str
    pending_review: bool
    needs_execution: bool


class RunSummary(BaseModel):
    """Item in GET /v1/cases/{case_id}/runs."""

    run_id: str
    case_id: str
    status: RunStatus
    created_at: datetime
    completed_at: datetime | None = None


class DocumentRecord(BaseModel):
    """Item in GET /v1/cases/{case_id}/documents"""

    document_id: str
    case_id: str
    doc_type: str
    status: str
    received_at: datetime | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    content_summary: str | None = None


class DocumentListResponse(BaseModel):
    case_id: str
    documents: list[DocumentRecord]


class ProcessingEvent(BaseModel):
    """Item in GET /v1/cases/{case_id}/events"""

    event_id: str
    case_id: str
    event_type: str
    timestamp: datetime
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class EventListResponse(BaseModel):
    case_id: str
    events: list[ProcessingEvent]


class FollowUpTaskPayload(BaseModel):
    """Bounded internal task content; immutable once stored on a proposal."""

    title: str = Field(max_length=200)
    body: str = Field(max_length=4000)
    case_id: str


class CreateFollowUpTaskRequest(BaseModel):
    """POST /v1/cases/{case_id}/follow-up-tasks (service-auth only)."""

    proposal_id: str
    task: FollowUpTaskPayload


class CreateFollowUpTaskResponse(BaseModel):
    task_id: str
    proposal_id: str
    case_id: str
    created: bool = True


# ---------------------------------------------------------------------------
# Investigation / backend contracts
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    source_system: str
    record_id: str
    field: str
    fact: str
    retrieval_ref: str


class InvestigationResult(BaseModel):
    run_id: str
    case_id: str
    status: RunStatus
    diagnosis: Diagnosis
    diagnosis_summary: str
    explanation: str
    evidence: list[Evidence] = Field(default_factory=list)
    uncertainty: str | None = None
    unresolved_questions: list[str] = Field(default_factory=list)
    next_action: NextAction
    next_action_explanation: str
    approval_required: bool = False
    proposed_task: FollowUpTaskPayload | None = None
    proposal_id: str | None = None


class StartInvestigationRequest(BaseModel):
    """POST /v1/investigations"""

    case_id: str
    objective: str = Field(max_length=2000)


class StartInvestigationResponse(BaseModel):
    run_id: str
    result: InvestigationResult


class RunRecord(BaseModel):
    """GET /v1/investigations/{run_id}"""

    run_id: str
    case_id: str
    objective: str
    status: RunStatus
    result: InvestigationResult | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class Proposal(BaseModel):
    """GET /v1/proposals/{proposal_id} — immutable task payload after create."""

    proposal_id: str
    run_id: str
    case_id: str
    action: NextAction
    task_payload: FollowUpTaskPayload
    status: ProposalStatus
    payload_digest: str
    created_at: datetime
    decided_at: datetime | None = None
    operator_identity: str | None = None
    execution_error: str | None = None
    task_id: str | None = None  # simulator-confirmed; never fabricate for legacy rows


class ApproveProposalRequest(BaseModel):
    """POST /v1/proposals/{proposal_id}/approve"""

    operator_identity: str = Field(default="local-operator", max_length=200)


class RejectProposalRequest(BaseModel):
    """POST /v1/proposals/{proposal_id}/reject"""

    operator_identity: str = Field(default="local-operator", max_length=200)
    reason: str | None = Field(default=None, max_length=1000)


class ProposalDecisionResponse(BaseModel):
    proposal: Proposal


# ---------------------------------------------------------------------------
# Run-scoped follow-up conversation
# ---------------------------------------------------------------------------


class EvidenceSnapshot(BaseModel):
    """Server-owned retrieved field value used for grounded citations."""

    snapshot_id: str
    run_id: str
    attempt_id: str | None = None
    source_system: str
    record_id: str
    field_path: str
    value: Any
    retrieval_ref: str
    retrieved_at: datetime
    provenance: EvidenceProvenance


class SourceFactClaim(BaseModel):
    """Model-emitted citation of a server snapshot (no trusted value/metadata)."""

    local_id: str = Field(max_length=64)
    snapshot_id: str
    field_path: str = Field(max_length=500)


class RenderedSourceFact(BaseModel):
    """Server-rendered source fact; values/provenance come only from snapshots."""

    local_id: str
    snapshot_id: str
    source_system: str
    record_id: str
    field_path: str
    value: Any
    retrieval_ref: str
    retrieved_at: datetime
    provenance: EvidenceProvenance


class InterpretationBlock(BaseModel):
    """Labeled inference/uncertainty; must cite source-fact local_ids when factual."""

    text: str = Field(max_length=4000)
    cites: list[str] = Field(default_factory=list)
    kind: InterpretationKind


class FollowUpAgentOutput(BaseModel):
    """Structured follow-up model output before server rendering/validation."""

    source_facts: list[SourceFactClaim] = Field(default_factory=list)
    interpretations: list[InterpretationBlock] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class FollowUpAnswer(BaseModel):
    """Persisted/operator-visible follow-up answer after server grounding."""

    source_facts: list[RenderedSourceFact] = Field(default_factory=list)
    interpretations: list[InterpretationBlock] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ConversationAttempt(BaseModel):
    attempt_id: str
    turn_id: str
    request_key: str
    previous_attempt_id: str | None = None
    status: TurnStatus
    answer: FollowUpAnswer | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class ConversationTurn(BaseModel):
    turn_id: str
    run_id: str
    question: str
    status: TurnStatus
    created_at: datetime
    updated_at: datetime
    attempts: list[ConversationAttempt] = Field(default_factory=list)


class ConversationHistoryResponse(BaseModel):
    """GET /v1/investigations/{run_id}/conversation"""

    run_id: str
    turns: list[ConversationTurn] = Field(default_factory=list)


class SendMessageRequest(BaseModel):
    """POST /v1/investigations/{run_id}/conversation/messages"""

    question: str = Field(max_length=4000)
    idempotency_key: str = Field(min_length=1, max_length=200)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must not be empty or whitespace-only")
        return question


class RetryTurnRequest(BaseModel):
    """POST .../conversation/turns/{turn_id}/retry"""

    idempotency_key: str = Field(min_length=1, max_length=200)
    expected_previous_attempt_id: str


class ConversationTurnResponse(BaseModel):
    """Send/retry response body."""

    turn: ConversationTurn
    attempt: ConversationAttempt
    created: bool = True


# ---------------------------------------------------------------------------
# Documented route paths (format with .format(case_id=..., run_id=..., ...))
# ---------------------------------------------------------------------------


class SimulatorRoutes:
    """Enterprise simulator HTTP paths under /v1."""

    HEALTH = "/health"
    CASES = "/v1/cases"
    CASE = "/v1/cases/{case_id}"
    DOCUMENTS = "/v1/cases/{case_id}/documents"
    EVENTS = "/v1/cases/{case_id}/events"
    FOLLOW_UP_TASKS = "/v1/cases/{case_id}/follow-up-tasks"


class BackendRoutes:
    """Investigation backend HTTP paths under /v1."""

    HEALTH = "/health"
    INVESTIGATIONS = "/v1/investigations"
    RUN = "/v1/investigations/{run_id}"
    CASES = "/v1/cases"
    CASE = "/v1/cases/{case_id}"
    CASE_RUNS = "/v1/cases/{case_id}/runs"
    CASE_PROPOSALS = "/v1/cases/{case_id}/proposals"
    PROPOSAL = "/v1/proposals/{proposal_id}"
    APPROVE = "/v1/proposals/{proposal_id}/approve"
    REJECT = "/v1/proposals/{proposal_id}/reject"
    CONVERSATION = "/v1/investigations/{run_id}/conversation"
    CONVERSATION_MESSAGES = "/v1/investigations/{run_id}/conversation/messages"
    CONVERSATION_RETRY = "/v1/investigations/{run_id}/conversation/turns/{turn_id}/retry"
