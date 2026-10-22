"""Minimal shared scripted-model helpers for unit and e2e tests.

E2E import (repo root must be importable; unit bootstraps sys.path):
  from tests.helpers import _finding, _tool_then_finding

Signatures:
  _finding(case_id: str = "CASE-001", **overrides) -> AgentFinding
    default evidence uses DOC-001-A / DOC-002-A for CASE-001 / other case_ids
  _tool_then_finding(finding, *, tools: list[str] | None = None) -> list[AIMessage]
    default tools: ["get_case", "get_documents", "get_events"]
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from casepilot.backend.agent import AgentFinding
from casepilot.shared.contracts import (
    Diagnosis,
    Evidence,
    FollowUpTaskPayload,
    NextAction,
)

_DEFAULT_CASE = "CASE-001"
_DEFAULT_TOOLS = ["get_case", "get_documents", "get_events"]


def _tool_then_finding(
    finding: AgentFinding,
    *,
    tools: list[str] | None = None,
) -> list[AIMessage]:
    names = tools if tools is not None else list(_DEFAULT_TOOLS)
    msgs: list[AIMessage] = [
        AIMessage(
            content="",
            tool_calls=[{"name": name, "args": {}, "id": f"t{i}", "type": "tool_call"}],
        )
        for i, name in enumerate(names)
    ]
    msgs.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "AgentFinding",
                    "args": finding.model_dump(mode="json"),
                    "id": "final",
                    "type": "tool_call",
                }
            ],
        )
    )
    return msgs


def _finding(case_id: str = _DEFAULT_CASE, **overrides: object) -> AgentFinding:
    """Scripted AgentFinding; evidence ids track CASE-001 / CASE-002 fixtures."""
    doc_id = "DOC-001-A" if case_id == _DEFAULT_CASE else "DOC-002-A"
    base: dict[str, object] = dict(
        diagnosis=Diagnosis.MISSING_DOCUMENT,
        diagnosis_summary="Required document missing",
        explanation="Checklist shows a required document is not on file",
        evidence=[
            Evidence(
                source_system="cases",
                record_id=case_id,
                field="status",
                fact="stalled",
                retrieval_ref="get_case",
            ),
            Evidence(
                source_system="documents",
                record_id=doc_id,
                field="status",
                fact="received",
                retrieval_ref="get_documents",
            ),
        ],
        next_action=NextAction.CREATE_FOLLOW_UP_TASK,
        next_action_explanation="Create internal follow-up for missing document",
        proposed_task=FollowUpTaskPayload(
            title=f"Follow up missing docs for {case_id}",
            body=f"Obtain missing required documentation for {case_id}",
            case_id=case_id,
        ),
    )
    base.update(overrides)
    return AgentFinding(**base)  # type: ignore[arg-type]
