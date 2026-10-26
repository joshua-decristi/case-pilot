"""P0: domain reason survives capture/snapshot/trace; provider CoT + secrets redact."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from casepilot.backend.db import BackendStore
from casepilot.backend.evidence import (
    RetrievalContext,
    RetrievedIndex,
    persist_index_snapshots,
)
from casepilot.backend.pilot_observability import (
    capture_leaf_exposure_facts,
    tool_exposure_facts,
)
from casepilot.backend.trace import TraceWriter, _redact
from casepilot.shared.contracts import (
    Diagnosis,
    EvidenceProvenance,
    InvestigationResult,
    NextAction,
    ProcessingEvent,
    RunStatus,
)

DOMAIN_REASON = "required_document_absent"
COT_SECRET = "secret-chain-of-thought"
API_SECRET = "super-secret-key"


def _event_with_domain_reason() -> ProcessingEvent:
    return ProcessingEvent(
        event_id="EVT-001-B",
        case_id="CASE-001",
        event_type="underwriting_gate",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        message="blocked",
        details={
            "reason": DOMAIN_REASON,
            "document_type": "proof_of_income",
            "pause_reason": "awaiting_docs",
            "rejection_reason": "incomplete_packet",
            # Nested CoT / secrets must still redact inside metadata.
            "metadata": {
                "reasoning_content": COT_SECRET,
                "Reasoning_Text": COT_SECRET,
                "api_key": API_SECRET,
                "reasoning_tokens": 42,
            },
        },
    )


def _seed_run(store: BackendStore, run_id: str = "run-redact") -> None:
    store.insert_run(
        run_id=run_id,
        case_id="CASE-001",
        objective="test",
        status=RunStatus.COMPLETED,
        result=InvestigationResult(
            run_id=run_id,
            case_id="CASE-001",
            status=RunStatus.COMPLETED,
            diagnosis=Diagnosis.MISSING_DOCUMENT,
            diagnosis_summary="s",
            explanation="e",
            evidence=[],
            next_action=NextAction.NO_ACTION_REQUIRED,
            next_action_explanation="n",
        ),
    )


def test_redact_preserves_domain_reason_keys_redacts_cot_and_secrets() -> None:
    payload = {
        "reason": DOMAIN_REASON,
        "pause_reason": "awaiting_docs",
        "rejection_reason": "incomplete_packet",
        "reasoning": COT_SECRET,
        "reasoning_content": COT_SECRET,
        "reasoning_details": {"steps": COT_SECRET},
        "reasoning_text": COT_SECRET,
        "REASONING_CONTENT": COT_SECRET,
        "reasoning_tokens": 7,
        "api_key": API_SECRET,
        "nested": {"token": "t-1", "reason": "domain-nested"},
    }
    out = _redact(payload)
    assert out["reason"] == DOMAIN_REASON
    assert out["pause_reason"] == "awaiting_docs"
    assert out["rejection_reason"] == "incomplete_packet"
    assert out["reasoning_tokens"] == 7
    assert out["nested"]["reason"] == "domain-nested"
    assert out["reasoning"] == "[REDACTED_REASONING]"
    assert out["reasoning_content"] == "[REDACTED_REASONING]"
    assert out["reasoning_details"] == "[REDACTED_REASONING]"
    assert out["reasoning_text"] == "[REDACTED_REASONING]"
    assert out["REASONING_CONTENT"] == "[REDACTED_REASONING]"
    assert out["api_key"] == "[REDACTED]"
    assert out["nested"]["token"] == "[REDACTED]"


def test_retrieved_index_snapshots_trace_preserve_domain_reason(tmp_path: Path) -> None:
    """RetrievedIndex → persisted snapshots → TraceWriter; model-visible reason intact."""
    event = _event_with_domain_reason()
    run_id = "run-redact"
    store = BackendStore(tmp_path / "backend.sqlite3")
    _seed_run(store, run_id)

    idx = RetrievedIndex(
        context=RetrievalContext(
            run_id=run_id,
            case_id="CASE-001",
            provenance=EvidenceProvenance.ORIGINAL_RUN,
        )
    )
    idx.note_events([event])

    # Agent-side index keeps the live domain value (get_events path).
    assert idx.events["EVT-001-B"].details["reason"] == DOMAIN_REASON

    cap = idx.captures[0]
    assert cap.payload["details"]["reason"] == DOMAIN_REASON
    assert cap.payload["details"]["pause_reason"] == "awaiting_docs"
    assert cap.payload["details"]["rejection_reason"] == "incomplete_packet"
    assert cap.payload["details"]["metadata"]["reasoning_content"] == "[REDACTED_REASONING]"
    assert cap.payload["details"]["metadata"]["Reasoning_Text"] == "[REDACTED_REASONING]"
    assert cap.payload["details"]["metadata"]["api_key"] == "[REDACTED]"
    assert cap.payload["details"]["metadata"]["reasoning_tokens"] == 42

    snaps = persist_index_snapshots(store, idx)
    by_path = {s.field_path: s for s in snaps if s.record_id == "EVT-001-B"}
    assert by_path["details.reason"].value == DOMAIN_REASON
    assert by_path["details.pause_reason"].value == "awaiting_docs"
    assert by_path["details.rejection_reason"].value == "incomplete_packet"
    assert by_path["details.metadata.reasoning_content"].value == "[REDACTED_REASONING]"
    assert by_path["details.metadata.Reasoning_Text"].value == "[REDACTED_REASONING]"
    assert by_path["details.metadata.api_key"].value == "[REDACTED]"
    assert by_path["details.metadata.reasoning_tokens"].value == 42

    # Model-visible leaf exposure (collector path) keeps domain reason string.
    leaf_facts = capture_leaf_exposure_facts(
        idx.captures, provenance=EvidenceProvenance.ORIGINAL_RUN.value
    )
    leaf_by_path = {f["field_path"]: f["value"] for f in leaf_facts}
    assert leaf_by_path["details.reason"] == DOMAIN_REASON

    # Descriptors as returned to the model after persist (value from snapshot rows).
    descriptors = [
        {
            "snapshot_id": s.snapshot_id,
            "record_id": s.record_id,
            "field_path": s.field_path,
            "provenance": s.provenance.value,
            "attempt_id": s.attempt_id,
            "value": s.value,
        }
        for s in snaps
        if s.field_path == "details.reason"
    ]
    facts = tool_exposure_facts(descriptors)
    assert facts[0]["value"] == DOMAIN_REASON
    assert DOMAIN_REASON not in (COT_SECRET,)  # sanity

    tw = TraceWriter(tmp_path / "traces", run_id)
    tw.emit("tool", facts=facts)
    tw.emit(
        "capture",
        details={
            "reason": DOMAIN_REASON,
            "reasoning_content": COT_SECRET,
            "api_key": API_SECRET,
        },
    )
    line = (tmp_path / "traces" / f"{run_id}.jsonl").read_text(encoding="utf-8")
    assert DOMAIN_REASON in line
    assert COT_SECRET not in line
    assert API_SECRET not in line
    assert "[REDACTED_REASONING]" in line
    assert "[REDACTED]" in line
