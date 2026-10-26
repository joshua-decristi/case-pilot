"""Diagnostics codes for strict ``render_follow_up_answer`` reject branches."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from casepilot.backend.evidence import (
    SAFE_UNCERTAINTY,
    FollowUpRenderRejectCode,
    render_follow_up_answer,
)
from casepilot.shared.contracts import (
    EvidenceProvenance,
    EvidenceSnapshot,
    FollowUpAgentOutput,
    InterpretationBlock,
    InterpretationKind,
    SourceFactClaim,
)


def _snap(**overrides: object) -> EvidenceSnapshot:
    base = dict(
        snapshot_id="snap-1",
        run_id="r",
        source_system="documents",
        record_id="DOC-X",
        field_path="status",
        value="missing",
        retrieval_ref="ref",
        retrieved_at=datetime(2024, 1, 2, tzinfo=UTC),
        provenance=EvidenceProvenance.ORIGINAL_RUN,
    )
    base.update(overrides)
    return EvidenceSnapshot(**base)  # type: ignore[arg-type]


def _assert_safe_reject(result, code: FollowUpRenderRejectCode) -> None:
    assert result.ok is False
    assert result.reject_code is code
    assert result.reject_code == code.value
    assert result.answer.source_facts == []
    assert result.answer.interpretations == []
    assert result.answer.notes == [SAFE_UNCERTAINTY]


def test_reject_codes_are_stable_strings() -> None:
    assert issubclass(FollowUpRenderRejectCode, Enum)
    assert FollowUpRenderRejectCode.NO_CONTENT == "no_content"
    assert FollowUpRenderRejectCode.INVALID_NOTES == "invalid_notes"
    assert FollowUpRenderRejectCode.EMPTY_LOCAL_ID == "empty_local_id"
    assert FollowUpRenderRejectCode.DUPLICATE_LOCAL_ID == "duplicate_local_id"
    assert FollowUpRenderRejectCode.UNKNOWN_SNAPSHOT == "unknown_snapshot"
    assert FollowUpRenderRejectCode.FIELD_PATH_MISMATCH == "field_path_mismatch"
    assert FollowUpRenderRejectCode.UNRENDERABLE_VALUE == "unrenderable_value"
    assert FollowUpRenderRejectCode.INVALID_INTERPRETATION_KIND == "invalid_interpretation_kind"
    assert FollowUpRenderRejectCode.EMPTY_INTERPRETATION_CITES == "empty_interpretation_cites"
    assert FollowUpRenderRejectCode.UNKNOWN_INTERPRETATION_CITE == "unknown_interpretation_cite"


def test_happy_path_has_no_reject_code() -> None:
    snap = _snap()
    out = render_follow_up_answer(
        FollowUpAgentOutput(
            source_facts=[
                SourceFactClaim(local_id="s1", snapshot_id="snap-1", field_path="status")
            ],
            interpretations=[
                InterpretationBlock(
                    text="status missing",
                    cites=["s1"],
                    kind=InterpretationKind.INFERENCE,
                )
            ],
        ),
        [snap],
    )
    assert out.ok is True
    assert out.reject_code is None
    assert out.answer.source_facts[0].value == "missing"
    assert SAFE_UNCERTAINTY not in out.answer.notes


def test_notes_only_success_has_no_reject_code() -> None:
    out = render_follow_up_answer(FollowUpAgentOutput(notes=["read_only"]), [])
    assert out.ok is True
    assert out.reject_code is None


def test_no_content_empty_output() -> None:
    out = render_follow_up_answer(FollowUpAgentOutput(), [])
    _assert_safe_reject(out, FollowUpRenderRejectCode.NO_CONTENT)


def test_invalid_notes_free_text_only() -> None:
    lie = "All required documents have been received."
    out = render_follow_up_answer(FollowUpAgentOutput(notes=[lie]), [])
    _assert_safe_reject(out, FollowUpRenderRejectCode.INVALID_NOTES)
    assert lie not in out.answer.notes


def test_invalid_notes_bogus_server_notes_kwarg() -> None:
    out = render_follow_up_answer(FollowUpAgentOutput(), [], server_notes=["not_a_code"])
    _assert_safe_reject(out, FollowUpRenderRejectCode.INVALID_NOTES)


def test_empty_local_id() -> None:
    out = render_follow_up_answer(
        FollowUpAgentOutput(
            source_facts=[SourceFactClaim(local_id="", snapshot_id="snap-1", field_path="status")]
        ),
        [_snap()],
    )
    _assert_safe_reject(out, FollowUpRenderRejectCode.EMPTY_LOCAL_ID)


def test_duplicate_local_id() -> None:
    out = render_follow_up_answer(
        FollowUpAgentOutput(
            source_facts=[
                SourceFactClaim(local_id="s1", snapshot_id="snap-1", field_path="status"),
                SourceFactClaim(local_id="s1", snapshot_id="snap-1", field_path="status"),
            ]
        ),
        [_snap()],
    )
    _assert_safe_reject(out, FollowUpRenderRejectCode.DUPLICATE_LOCAL_ID)


def test_unknown_snapshot() -> None:
    out = render_follow_up_answer(
        FollowUpAgentOutput(
            source_facts=[SourceFactClaim(local_id="s1", snapshot_id="nope", field_path="status")]
        ),
        [_snap()],
    )
    _assert_safe_reject(out, FollowUpRenderRejectCode.UNKNOWN_SNAPSHOT)


def test_field_path_mismatch() -> None:
    out = render_follow_up_answer(
        FollowUpAgentOutput(
            source_facts=[SourceFactClaim(local_id="s1", snapshot_id="snap-1", field_path="other")]
        ),
        [_snap()],
    )
    _assert_safe_reject(out, FollowUpRenderRejectCode.FIELD_PATH_MISMATCH)


def test_invalid_interpretation_kind() -> None:
    # Bypass pydantic so defensive kind check is reachable.
    bad_block = InterpretationBlock.model_construct(
        text="guess",
        cites=["s1"],
        kind="not_a_kind",  # type: ignore[arg-type]
    )
    out = render_follow_up_answer(
        FollowUpAgentOutput(
            source_facts=[
                SourceFactClaim(local_id="s1", snapshot_id="snap-1", field_path="status")
            ],
            interpretations=[bad_block],
        ),
        [_snap()],
    )
    _assert_safe_reject(out, FollowUpRenderRejectCode.INVALID_INTERPRETATION_KIND)


def test_empty_interpretation_cites() -> None:
    out = render_follow_up_answer(
        FollowUpAgentOutput(
            source_facts=[
                SourceFactClaim(local_id="s1", snapshot_id="snap-1", field_path="status")
            ],
            interpretations=[
                InterpretationBlock(text="guess", cites=[], kind=InterpretationKind.INFERENCE)
            ],
        ),
        [_snap()],
    )
    _assert_safe_reject(out, FollowUpRenderRejectCode.EMPTY_INTERPRETATION_CITES)


def test_unknown_interpretation_cite() -> None:
    out = render_follow_up_answer(
        FollowUpAgentOutput(
            source_facts=[
                SourceFactClaim(local_id="s1", snapshot_id="snap-1", field_path="status")
            ],
            interpretations=[
                InterpretationBlock(
                    text="guess",
                    cites=["missing"],
                    kind=InterpretationKind.INFERENCE,
                )
            ],
        ),
        [_snap()],
    )
    _assert_safe_reject(out, FollowUpRenderRejectCode.UNKNOWN_INTERPRETATION_CITE)


def test_unknown_interpretation_cite_without_facts() -> None:
    out = render_follow_up_answer(
        FollowUpAgentOutput(
            interpretations=[
                InterpretationBlock(
                    text="guess",
                    cites=["s1"],
                    kind=InterpretationKind.UNCERTAINTY,
                )
            ]
        ),
        [_snap()],
    )
    _assert_safe_reject(out, FollowUpRenderRejectCode.UNKNOWN_INTERPRETATION_CITE)


def test_reject_code_never_embeds_model_text() -> None:
    """Diagnostics must be fixed codes — never model prose or claim values."""
    prose = "secret model claim about DOC-SECRET-999"
    cases = [
        FollowUpAgentOutput(notes=[prose]),
        FollowUpAgentOutput(
            source_facts=[SourceFactClaim(local_id="s1", snapshot_id="nope", field_path=prose)]
        ),
        FollowUpAgentOutput(
            source_facts=[
                SourceFactClaim(local_id="s1", snapshot_id="snap-1", field_path="status")
            ],
            interpretations=[
                InterpretationBlock(text=prose, cites=["ghost"], kind=InterpretationKind.INFERENCE)
            ],
        ),
    ]
    for output in cases:
        out = render_follow_up_answer(output, [_snap()])
        assert out.ok is False
        assert out.reject_code is not None
        assert prose not in str(out.reject_code)
        assert prose not in out.answer.notes
        assert "DOC-SECRET-999" not in str(out.reject_code)
