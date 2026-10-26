"""Follow-up retrieval coverage: pure classification and proof checks.

Natural eligibility (spec): the actual pre-turn model context *after*
filtering/truncation — not the investigation answer alone — decides whether a
follow-up is retrieval-optional or retrieval-required. Required new-retrieval
coverage then needs a full proof chain. Unknown/missing instrumentation is
``not_assessed``, never evidence of availability or successful coverage.

This module does not call the backend, mutate state, or invent context. The
pilot runner must capture and pass the artifacts below.

Runner instrumentation (required inputs)
---------------------------------------
Target facts (from frozen scenario references):
  ``target_facts`` — ``record_id`` + ``field_path`` pairs the follow-up needs.

Pre-turn (immediately before the fixed follow-up turn):
  ``pre_turn_model_context`` — facts actually present in the answering model's
  established context after runtime filtering/truncation. ``None`` means the
  runner lacked this capture → classification is not assessed.

Post-turn proof (only material for retrieval-required coverage):
  ``attempt_id`` — scored attempt id.
  ``attempt_follow_up_snapshots`` — snapshots with ``provenance=follow_up`` and
  ``attempt_id`` equal to this attempt (authoritative store rows). ``None`` =
  missing capture.
  ``answering_model_exposed`` — snapshot content that actually reached the
  answering model for this turn (tool results / context after retrieval,
  post-filter). ``None`` = missing capture. Store presence alone is insufficient.
  ``answer_citations`` — grounded answer citations (snapshot_id + record_id +
  field_path from the rendered answer). ``None`` = missing answer/citations
  capture; empty list = known answer with zero citations.

Exported API
------------
  Types: ``TargetFact``, ``ExposedFact``, ``AttemptSnapshot``, ``AnswerCitation``,
  ``RetrievalClass``, ``CoverageStatus``, ``RetrievalClassification``,
  ``CoverageAssessment``, ``RepetitionCoverage``, ``ScenarioCoverageCounts``,
  ``CohortCoverageSummary``.
  Functions: ``fact_key``, ``classify_retrieval``, ``assess_required_coverage``,
  ``assess_followup_coverage``, ``summarize_cohort_coverage``.

  ``summarize_cohort_coverage(..., mandated_scenario_ids=...)`` — P0 gate is true
  only when each mandated scenario_id has ≥1 ``covered`` repetition. Omitting
  mandated IDs never defaults the gate to true (avoids incidental cross-scenario
  coverage). Global totals remain descriptive only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Inputs the runner must supply (plain, serializable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetFact:
    """Scenario reference target: record_id + field_path."""

    record_id: str
    field_path: str


@dataclass(frozen=True)
class ExposedFact:
    """One fact observed in actual model-visible context (pre-turn or answering)."""

    record_id: str
    field_path: str
    snapshot_id: str = ""
    provenance: str = ""
    attempt_id: str | None = None
    value: Any = None


@dataclass(frozen=True)
class AttemptSnapshot:
    """Authoritative attempt-scoped snapshot row (expected provenance=follow_up)."""

    snapshot_id: str
    record_id: str
    field_path: str
    provenance: str
    attempt_id: str | None
    retrieval_ref: str = ""
    value: Any = None


@dataclass(frozen=True)
class AnswerCitation:
    """Grounded citation from the rendered follow-up answer (not free text)."""

    snapshot_id: str
    record_id: str
    field_path: str
    provenance: str = ""


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


class RetrievalClass(StrEnum):
    """Whether this natural setup needed new retrieval for target facts."""

    REQUIRED = "retrieval_required"
    OPTIONAL = "retrieval_optional"
    NOT_ASSESSED = "not_assessed"


class CoverageStatus(StrEnum):
    """Required-new-retrieval coverage outcome for one repetition."""

    COVERED = "covered"
    """Required, and full proof chain succeeded."""

    NOT_COVERED = "not_covered"
    """Required, instrumentation complete, but proof chain failed."""

    NOT_ELIGIBLE = "not_eligible"
    """Optional: valid answer-from-context; cannot count as required coverage."""

    NOT_ASSESSED = "not_assessed"
    """Missing proof/instrumentation; never pass or covered."""


@dataclass(frozen=True)
class RetrievalClassification:
    retrieval_class: RetrievalClass
    present_targets: tuple[tuple[str, str], ...]
    absent_targets: tuple[tuple[str, str], ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageAssessment:
    status: CoverageStatus
    retrieval_class: RetrievalClass
    relevant_new_snapshot_ids: tuple[str, ...] = ()
    exposed_new_snapshot_ids: tuple[str, ...] = ()
    grounded_new_snapshot_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepetitionCoverage:
    """One fixed scenario/repetition slot coverage result."""

    scenario_id: str
    repetition: int
    assessment: CoverageAssessment

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "repetition": self.repetition,
            "assessment": self.assessment.to_dict(),
        }


@dataclass(frozen=True)
class ScenarioCoverageCounts:
    """Per-scenario tallies; gate uses covered on mandated IDs only."""

    scenario_id: str
    n: int
    required: int
    optional: int
    classification_not_assessed: int
    covered: int
    not_covered: int
    not_eligible: int
    coverage_not_assessed: int

    @property
    def required_coverage_demonstrated(self) -> bool:
        """≥1 covered on this scenario (0/n is incomplete)."""
        return self.covered >= 1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["required_coverage_demonstrated"] = self.required_coverage_demonstrated
        return d


@dataclass
class CohortCoverageSummary:
    """Cohort tallies. Gate is mandated-scenario-scoped, never global covered."""

    n: int
    required: int
    optional: int
    classification_not_assessed: int
    covered: int
    not_covered: int
    not_eligible: int
    coverage_not_assessed: int
    required_coverage_demonstrated: bool
    mandated_scenario_ids: tuple[str, ...] = ()
    by_scenario: dict[str, ScenarioCoverageCounts] = field(default_factory=dict)
    repetitions: list[RepetitionCoverage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "required": self.required,
            "optional": self.optional,
            "classification_not_assessed": self.classification_not_assessed,
            "covered": self.covered,
            "not_covered": self.not_covered,
            "not_eligible": self.not_eligible,
            "coverage_not_assessed": self.coverage_not_assessed,
            "required_coverage_demonstrated": self.required_coverage_demonstrated,
            "mandated_scenario_ids": list(self.mandated_scenario_ids),
            "by_scenario": {k: v.to_dict() for k, v in self.by_scenario.items()},
            "notes": list(self.notes),
            "repetitions": [r.to_dict() for r in self.repetitions],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fact_key(record_id: str, field_path: str) -> tuple[str, str]:
    """Canonical target key: stripped record_id + field_path."""
    return ((record_id or "").strip(), (field_path or "").strip())


def _target_keys(targets: Sequence[TargetFact]) -> tuple[tuple[str, str], ...]:
    keys = (fact_key(t.record_id, t.field_path) for t in targets)
    return tuple(k for k in keys if k != ("", ""))


def _keys_from_exposed(facts: Iterable[ExposedFact]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for f in facts:
        k = fact_key(f.record_id, f.field_path)
        if k != ("", ""):
            out.add(k)
    return out


def _is_follow_up_provenance(value: str) -> bool:
    return (value or "").strip().lower() == "follow_up"


# ---------------------------------------------------------------------------
# Classification (pre-turn context only)
# ---------------------------------------------------------------------------


def classify_retrieval(
    target_facts: Sequence[TargetFact],
    pre_turn_model_context: Sequence[ExposedFact] | None,
) -> RetrievalClassification:
    """Classify retrieval-optional vs required from actual pre-turn model context.

    ``pre_turn_model_context is None`` → not assessed (missing instrumentation).
    Empty context with non-empty targets → all absent → required.
    Empty targets → optional (nothing to retrieve for coverage).
    """
    keys = _target_keys(target_facts)
    if pre_turn_model_context is None:
        return RetrievalClassification(
            retrieval_class=RetrievalClass.NOT_ASSESSED,
            present_targets=(),
            absent_targets=(),
            notes=(
                "pre_turn_model_context missing: cannot prove target availability "
                "from actual filtered/truncated model context",
            ),
        )

    available = _keys_from_exposed(pre_turn_model_context)
    present = tuple(k for k in keys if k in available)
    absent = tuple(k for k in keys if k not in available)

    if not keys or not absent:
        return RetrievalClassification(
            retrieval_class=RetrievalClass.OPTIONAL,
            present_targets=present,
            absent_targets=(),
            notes=(
                ("all target facts present in pre-turn model context",)
                if keys
                else ("no target facts declared; retrieval-optional by default",)
            ),
        )

    return RetrievalClassification(
        retrieval_class=RetrievalClass.REQUIRED,
        present_targets=present,
        absent_targets=absent,
        notes=("one or more target facts absent from pre-turn model context",),
    )


# ---------------------------------------------------------------------------
# Required coverage proof chain
# ---------------------------------------------------------------------------


def assess_required_coverage(
    *,
    absent_targets: Sequence[tuple[str, str]],
    attempt_id: str,
    attempt_follow_up_snapshots: Sequence[AttemptSnapshot] | None,
    answering_model_exposed: Sequence[ExposedFact] | None,
    answer_citations: Sequence[AnswerCitation] | None,
) -> CoverageAssessment:
    """Prove required new-retrieval coverage for absent targets.

    Chain (all required; any missing instrumentation → not_assessed):
      1. New attempt-scoped FOLLOW_UP snapshots relevant to absent targets
      2. Those snapshot contents exposed to the answering model
      3. Grounded answer citation of that new information

    Redundant fetches of already-present targets do not count. Complete
    instrumentation with a broken chain → not_covered (finding, not proof gap).
    """
    notes: list[str] = []
    absent = tuple(k for k in absent_targets if k != ("", ""))
    if not absent:
        return CoverageAssessment(
            status=CoverageStatus.NOT_ELIGIBLE,
            retrieval_class=RetrievalClass.OPTIONAL,
            notes=("no absent targets; not eligible for required coverage",),
        )

    aid = (attempt_id or "").strip()
    if not aid:
        return CoverageAssessment(
            status=CoverageStatus.NOT_ASSESSED,
            retrieval_class=RetrievalClass.REQUIRED,
            notes=("attempt_id missing; cannot scope FOLLOW_UP snapshots",),
        )

    if attempt_follow_up_snapshots is None:
        return CoverageAssessment(
            status=CoverageStatus.NOT_ASSESSED,
            retrieval_class=RetrievalClass.REQUIRED,
            notes=("attempt_follow_up_snapshots missing; cannot prove new retrieval",),
        )
    if answering_model_exposed is None:
        return CoverageAssessment(
            status=CoverageStatus.NOT_ASSESSED,
            retrieval_class=RetrievalClass.REQUIRED,
            notes=("answering_model_exposed missing; store rows alone are insufficient",),
        )
    if answer_citations is None:
        return CoverageAssessment(
            status=CoverageStatus.NOT_ASSESSED,
            retrieval_class=RetrievalClass.REQUIRED,
            notes=("answer_citations missing; cannot prove grounded citation",),
        )

    # 1. Relevant new attempt-scoped FOLLOW_UP snapshots for absent targets.
    relevant: list[AttemptSnapshot] = []
    for snap in attempt_follow_up_snapshots:
        if not _is_follow_up_provenance(snap.provenance):
            continue
        if (snap.attempt_id or "").strip() != aid:
            continue
        if not (snap.snapshot_id or "").strip():
            continue
        if not (snap.retrieval_ref or "").strip():
            # Mirror grader honesty: retrieval needs an observable ref.
            continue
        key = fact_key(snap.record_id, snap.field_path)
        if key in absent:
            relevant.append(snap)

    relevant_ids = tuple(dict.fromkeys(s.snapshot_id for s in relevant))
    if not relevant:
        notes.append(
            "no attempt-scoped FOLLOW_UP snapshot matches absent targets "
            "(redundant or irrelevant fetches do not count)"
        )
        return CoverageAssessment(
            status=CoverageStatus.NOT_COVERED,
            retrieval_class=RetrievalClass.REQUIRED,
            relevant_new_snapshot_ids=(),
            notes=tuple(notes),
        )

    relevant_by_id = {s.snapshot_id: s for s in relevant}

    # 2. Exposure to answering model.
    # Prefer exact snapshot_id; fact-key fallback only when exposure omits snapshot_id
    # (so citing a different unexposed snap of the same field cannot piggyback).
    def _is_exposed(snap: AttemptSnapshot) -> bool:
        sid = (snap.snapshot_id or "").strip()
        key = fact_key(snap.record_id, snap.field_path)
        for f in answering_model_exposed:
            f_sid = (f.snapshot_id or "").strip()
            f_key = fact_key(f.record_id, f.field_path)
            if f_sid and f_sid == sid:
                return True
            if not f_sid and f_key == key:
                return True
        return False

    exposed_ids_t = tuple(dict.fromkeys(sid for sid, s in relevant_by_id.items() if _is_exposed(s)))
    if not exposed_ids_t:
        notes.append("relevant new FOLLOW_UP content did not reach answering model context")
        return CoverageAssessment(
            status=CoverageStatus.NOT_COVERED,
            retrieval_class=RetrievalClass.REQUIRED,
            relevant_new_snapshot_ids=relevant_ids,
            exposed_new_snapshot_ids=(),
            notes=tuple(notes),
        )

    # 3. Grounded answer citation of exposed new information.
    grounded: list[str] = []
    for cite in answer_citations:
        sid = (cite.snapshot_id or "").strip()
        if not sid or sid not in exposed_ids_t:
            continue
        matched = relevant_by_id.get(sid)
        if matched is None:
            continue
        # Citation must align with the snapshot's record/field (no id-only handwave).
        if fact_key(cite.record_id, cite.field_path) != fact_key(
            matched.record_id, matched.field_path
        ):
            continue
        grounded.append(sid)
    grounded_t = tuple(dict.fromkeys(grounded))
    if not grounded_t:
        notes.append("answer lacks grounded citation of exposed new FOLLOW_UP content")
        return CoverageAssessment(
            status=CoverageStatus.NOT_COVERED,
            retrieval_class=RetrievalClass.REQUIRED,
            relevant_new_snapshot_ids=relevant_ids,
            exposed_new_snapshot_ids=exposed_ids_t,
            grounded_new_snapshot_ids=(),
            notes=tuple(notes),
        )

    return CoverageAssessment(
        status=CoverageStatus.COVERED,
        retrieval_class=RetrievalClass.REQUIRED,
        relevant_new_snapshot_ids=relevant_ids,
        exposed_new_snapshot_ids=exposed_ids_t,
        grounded_new_snapshot_ids=grounded_t,
        notes=("required new-retrieval coverage proven",),
    )


def assess_followup_coverage(
    *,
    target_facts: Sequence[TargetFact],
    pre_turn_model_context: Sequence[ExposedFact] | None,
    attempt_id: str = "",
    attempt_follow_up_snapshots: Sequence[AttemptSnapshot] | None = None,
    answering_model_exposed: Sequence[ExposedFact] | None = None,
    answer_citations: Sequence[AnswerCitation] | None = None,
) -> CoverageAssessment:
    """Classify from pre-turn context, then assess required coverage if applicable.

    Optional setups are scored as not_eligible for required-coverage counts;
    missing classification proof yields not_assessed end-to-end.
    """
    classification = classify_retrieval(target_facts, pre_turn_model_context)

    if classification.retrieval_class is RetrievalClass.NOT_ASSESSED:
        return CoverageAssessment(
            status=CoverageStatus.NOT_ASSESSED,
            retrieval_class=RetrievalClass.NOT_ASSESSED,
            notes=classification.notes,
        )

    if classification.retrieval_class is RetrievalClass.OPTIONAL:
        return CoverageAssessment(
            status=CoverageStatus.NOT_ELIGIBLE,
            retrieval_class=RetrievalClass.OPTIONAL,
            notes=classification.notes
            + ("answering from existing evidence is valid; not required coverage",),
        )

    return assess_required_coverage(
        absent_targets=classification.absent_targets,
        attempt_id=attempt_id,
        attempt_follow_up_snapshots=attempt_follow_up_snapshots,
        answering_model_exposed=answering_model_exposed,
        answer_citations=answer_citations,
    )


def _tally_reps(reps: Sequence[RepetitionCoverage]) -> tuple[int, int, int, int, int, int, int]:
    """required, optional, class_na, covered, not_covered, not_eligible, cov_na."""
    required = optional = class_na = 0
    covered = not_covered = not_eligible = cov_na = 0
    for rep in reps:
        a = rep.assessment
        if a.retrieval_class is RetrievalClass.REQUIRED:
            required += 1
        elif a.retrieval_class is RetrievalClass.OPTIONAL:
            optional += 1
        else:
            class_na += 1
        if a.status is CoverageStatus.COVERED:
            covered += 1
        elif a.status is CoverageStatus.NOT_COVERED:
            not_covered += 1
        elif a.status is CoverageStatus.NOT_ELIGIBLE:
            not_eligible += 1
        else:
            cov_na += 1
    return required, optional, class_na, covered, not_covered, not_eligible, cov_na


def summarize_cohort_coverage(
    repetitions: Sequence[RepetitionCoverage],
    *,
    mandated_scenario_ids: Sequence[str] | None = None,
) -> CohortCoverageSummary:
    """Aggregate fixed repetitions with per-scenario counts.

    ``required_coverage_demonstrated`` is true only when every id in
    ``mandated_scenario_ids`` has ≥1 ``covered`` repetition (spec: retrieve 0/3
    is incomplete). Global ``covered`` is descriptive only — never the gate.

    When ``mandated_scenario_ids`` is omitted or empty, the gate stays false
    (no silent default that would accept incidental cross-scenario coverage).
    """
    notes: list[str] = []
    required, optional, class_na, covered, not_covered, not_eligible, cov_na = _tally_reps(
        repetitions
    )

    # Stable per-scenario order: first appearance in repetitions, then mandated.
    order: list[str] = []
    seen: set[str] = set()
    for rep in repetitions:
        sid = (rep.scenario_id or "").strip()
        if sid and sid not in seen:
            seen.add(sid)
            order.append(sid)

    mandated = tuple(
        dict.fromkeys(s.strip() for s in (mandated_scenario_ids or ()) if (s or "").strip())
    )
    for sid in mandated:
        if sid not in seen:
            seen.add(sid)
            order.append(sid)

    by_scenario: dict[str, ScenarioCoverageCounts] = {}
    for sid in order:
        subset = [r for r in repetitions if (r.scenario_id or "").strip() == sid]
        r, o, cls_na, cov, nc, ne, cov_na_s = _tally_reps(subset)
        by_scenario[sid] = ScenarioCoverageCounts(
            scenario_id=sid,
            n=len(subset),
            required=r,
            optional=o,
            classification_not_assessed=cls_na,
            covered=cov,
            not_covered=nc,
            not_eligible=ne,
            coverage_not_assessed=cov_na_s,
        )

    if not mandated:
        demonstrated = False
        notes.append(
            "mandated_scenario_ids not provided; required_coverage_demonstrated stays "
            "false (gate requires explicit scenario filter, not global covered count)"
        )
    else:
        missing = [sid for sid in mandated if sid not in by_scenario or by_scenario[sid].n == 0]
        failed = [
            sid
            for sid in mandated
            if sid in by_scenario and not by_scenario[sid].required_coverage_demonstrated
        ]
        demonstrated = not missing and not failed
        if missing:
            notes.append("mandated scenario(s) missing from cohort: " + ", ".join(missing))
        for sid in failed:
            sc = by_scenario[sid]
            notes.append(
                f"mandated scenario {sid!r}: {sc.covered}/{sc.n} covered; "
                "P0 retrieval-coverage incomplete"
            )

    return CohortCoverageSummary(
        n=len(repetitions),
        required=required,
        optional=optional,
        classification_not_assessed=class_na,
        covered=covered,
        not_covered=not_covered,
        not_eligible=not_eligible,
        coverage_not_assessed=cov_na,
        required_coverage_demonstrated=demonstrated,
        mandated_scenario_ids=mandated,
        by_scenario=by_scenario,
        repetitions=list(repetitions),
        notes=notes,
    )
