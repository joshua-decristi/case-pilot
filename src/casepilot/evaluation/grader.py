"""Rule-based grader: fixture action codes -> shared NextAction; ID-level evidence.

Does not claim narrative entailment — only category, ID/ref coverage, uncertainty
presence, mapped action codes, and termination. ``create_follow_up_task`` is never
auto-correct against fixture labels (needs manual task-text review).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from casepilot.shared.contracts import InvestigationResult, NextAction, RunStatus

# Ground-truth fixture codes -> shared contract enums (1:1 advisory mapping).
FIXTURE_ACTION_TO_NEXT: dict[str, NextAction] = {
    "request_document": NextAction.REQUEST_MISSING_DOCUMENT,
    "reconcile_data": NextAction.CORRECT_DATA_MISMATCH,
    "investigate_processing": NextAction.RETRY_PROCESSING,
    "escalate": NextAction.ESCALATE_TO_HUMAN,
    "no_action": NextAction.NO_ACTION_REQUIRED,
}

DEFAULT_LABELS_PATH = Path(__file__).resolve().parents[3] / "evals" / "ground_truth" / "labels.json"


@dataclass(frozen=True)
class CaseLabel:
    case_id: str
    expected_categories: tuple[str, ...]
    acceptable_actions: tuple[str, ...]  # fixture codes
    evidence_alternatives: tuple[tuple[str, ...], ...]
    require_uncertainty: bool
    notes: str = ""
    split: str = ""  # development | evaluation | ""


@dataclass(frozen=True)
class GroundTruth:
    version: str
    fixture_ref: str
    development_ids: tuple[str, ...]
    evaluation_ids: tuple[str, ...]
    cases: dict[str, CaseLabel]
    source_path: str
    raw_hash: str

    def case_ids_for_split(self, split: str) -> tuple[str, ...]:
        if split == "development":
            return self.development_ids
        if split == "evaluation":
            return self.evaluation_ids
        if split == "all":
            return tuple(sorted(self.cases))
        raise ValueError(f"unknown split: {split!r} (use development|evaluation|all)")


@dataclass
class CaseGrade:
    case_id: str
    # Each dimension is bool | None (None = not scorable / needs manual).
    category_ok: bool | None
    evidence_ok: bool | None
    uncertainty_ok: bool | None
    action_ok: bool | None
    terminated_ok: bool
    failed: bool
    action_needs_manual: bool
    notes: list[str] = field(default_factory=list)
    predicted_category: str | None = None
    predicted_action: str | None = None
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Scorecard:
    """Aggregate metrics. Denominator always includes failures/timeouts."""

    n: int
    category_correct: int
    evidence_correct: int
    uncertainty_correct: int
    uncertainty_scored: int
    action_correct: int
    action_auto_scored: int
    action_manual_pending: int
    terminated_ok: int
    failures: int
    grades: list[CaseGrade] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def category_accuracy(self) -> float:
        return self.category_correct / self.n if self.n else 0.0

    @property
    def evidence_accuracy(self) -> float:
        return self.evidence_correct / self.n if self.n else 0.0

    @property
    def action_accuracy_auto(self) -> float:
        """Only over cases with automatic action mapping (excludes manual-pending)."""
        return self.action_correct / self.action_auto_scored if self.action_auto_scored else 0.0

    @property
    def uncertainty_accuracy(self) -> float:
        return (
            self.uncertainty_correct / self.uncertainty_scored if self.uncertainty_scored else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "failures": self.failures,
            "category_correct": self.category_correct,
            "category_accuracy": self.category_accuracy,
            "evidence_correct": self.evidence_correct,
            "evidence_accuracy": self.evidence_accuracy,
            "uncertainty_correct": self.uncertainty_correct,
            "uncertainty_scored": self.uncertainty_scored,
            "uncertainty_accuracy": self.uncertainty_accuracy,
            "action_correct": self.action_correct,
            "action_auto_scored": self.action_auto_scored,
            "action_manual_pending": self.action_manual_pending,
            "action_accuracy_auto": self.action_accuracy_auto,
            "terminated_ok": self.terminated_ok,
            "caveats": list(self.caveats),
            "grades": [g.to_dict() for g in self.grades],
        }


def load_ground_truth(path: Path | None = None) -> GroundTruth:
    p = path or DEFAULT_LABELS_PATH
    text = p.read_text(encoding="utf-8")
    data = json.loads(text)
    import hashlib

    raw_hash = hashlib.sha256(text.encode()).hexdigest()
    split = data.get("split") or {}
    cases: dict[str, CaseLabel] = {}
    dev = tuple(split.get("development") or ())
    eva = tuple(split.get("evaluation") or ())
    split_of = {cid: "development" for cid in dev}
    split_of.update({cid: "evaluation" for cid in eva})

    for cid, raw in (data.get("cases") or {}).items():
        alts = raw.get("required_evidence", {}).get("alternatives") or []
        cases[cid] = CaseLabel(
            case_id=cid,
            expected_categories=tuple(raw.get("expected_categories") or ()),
            acceptable_actions=tuple(raw.get("acceptable_actions") or ()),
            evidence_alternatives=tuple(tuple(a) for a in alts),
            require_uncertainty=bool(raw.get("require_uncertainty")),
            notes=str(raw.get("notes") or ""),
            split=split_of.get(cid, ""),
        )
    return GroundTruth(
        version=str(data.get("version") or ""),
        fixture_ref=str(data.get("fixture_ref") or ""),
        development_ids=dev,
        evaluation_ids=eva,
        cases=cases,
        source_path=str(p),
        raw_hash=raw_hash,
    )


def _retrieved_record_ids(result: InvestigationResult) -> set[str]:
    """record_ids that also carry a non-empty retrieval_ref (observable retrieval).

    Required label IDs must appear as evidence.record_id. retrieval_ref must be
    present as proof of retrieval — but path-tokenizing the ref is NOT a bypass
    for missing record_id citations (no stuffing IDs only into ref strings).
    """
    ids: set[str] = set()
    for ev in result.evidence:
        rid = (ev.record_id or "").strip()
        ref = (ev.retrieval_ref or "").strip()
        if rid and ref:
            ids.add(rid)
    return ids


def _evidence_covers(label: CaseLabel, retrieved_ids: set[str]) -> bool:
    if not label.evidence_alternatives:
        return True
    for alt in label.evidence_alternatives:
        if alt and all(rid in retrieved_ids for rid in alt):
            return True
    return False


def _map_acceptable_actions(label: CaseLabel) -> set[NextAction]:
    out: set[NextAction] = set()
    for code in label.acceptable_actions:
        mapped = FIXTURE_ACTION_TO_NEXT.get(code)
        if mapped is not None:
            out.add(mapped)
    return out


def grade_case(
    label: CaseLabel,
    result: InvestigationResult | None,
    *,
    error: str | None = None,
) -> CaseGrade:
    """Grade one investigation against a label. Failures count as incorrect on auto dims."""
    notes: list[str] = []
    if result is None:
        notes.append(error or "missing result")
        return CaseGrade(
            case_id=label.case_id,
            category_ok=False,
            evidence_ok=False,
            uncertainty_ok=False if label.require_uncertainty else None,
            action_ok=False,
            terminated_ok=False,
            failed=True,
            action_needs_manual=False,
            notes=notes,
        )

    failed = result.status == RunStatus.FAILED
    terminated_ok = result.status != RunStatus.FAILED
    if failed:
        notes.append("run status=failed (counts in denominator as incorrect)")

    cat = result.diagnosis.value
    category_ok = (not failed) and cat in label.expected_categories

    retrieved = _retrieved_record_ids(result)
    evidence_ok = (not failed) and _evidence_covers(label, retrieved)
    if not evidence_ok and not failed:
        notes.append(
            "evidence requires label IDs as record_id with non-empty retrieval_ref "
            "(ID-level only; no entailment; no retrieval_ref token bypass)"
        )

    uncertainty_ok: bool | None
    if label.require_uncertainty:
        has_u = bool((result.uncertainty or "").strip()) or bool(result.unresolved_questions)
        uncertainty_ok = (not failed) and has_u
        if not uncertainty_ok and not failed:
            notes.append("expected non-empty uncertainty/unresolved_questions")
    else:
        uncertainty_ok = None  # not scored

    mapped = _map_acceptable_actions(label)
    predicted = result.next_action
    action_needs_manual = False
    action_ok: bool | None

    if failed:
        action_ok = False
    elif predicted == NextAction.CREATE_FOLLOW_UP_TASK:
        # Never auto-correct: fixture labels don't map to this executable write.
        action_ok = None
        action_needs_manual = True
        notes.append(
            "create_follow_up_task requires manual task-text assessment; "
            "not auto-correct vs fixture action codes"
        )
    elif predicted in mapped:
        action_ok = True
    else:
        action_ok = False
        if not mapped:
            notes.append("no mappable acceptable_actions on label")
        else:
            notes.append(
                f"action {predicted.value} not in mapped acceptable "
                f"{sorted(a.value for a in mapped)}"
            )

    return CaseGrade(
        case_id=label.case_id,
        category_ok=category_ok,
        evidence_ok=evidence_ok,
        uncertainty_ok=uncertainty_ok,
        action_ok=action_ok,
        terminated_ok=terminated_ok,
        failed=failed,
        action_needs_manual=action_needs_manual,
        notes=notes,
        predicted_category=cat,
        predicted_action=predicted.value,
        status=result.status.value,
    )


def grade_results(
    gt: GroundTruth,
    results: dict[str, InvestigationResult | None],
    *,
    errors: dict[str, str] | None = None,
) -> Scorecard:
    """Aggregate grades. Denominator = len(results); failures count as misses."""
    errors = errors or {}
    grades: list[CaseGrade] = []
    caveats = [
        "Evidence metric: required IDs must be evidence.record_id with non-empty "
        "retrieval_ref (no ref-token bypass; no semantic entailment claim).",
        "create_follow_up_task is never auto-scored correct against fixture labels.",
        "Failures/timeouts are included in the denominator as incorrect.",
    ]

    cat_ok = ev_ok = unc_ok = unc_n = act_ok = act_n = act_manual = term_ok = fails = 0
    for cid, result in results.items():
        label = gt.cases.get(cid)
        if label is None:
            grades.append(
                CaseGrade(
                    case_id=cid,
                    category_ok=False,
                    evidence_ok=False,
                    uncertainty_ok=None,
                    action_ok=False,
                    terminated_ok=False,
                    failed=True,
                    action_needs_manual=False,
                    notes=[f"no ground-truth label for {cid}"],
                )
            )
            fails += 1
            continue
        g = grade_case(label, result, error=errors.get(cid))
        grades.append(g)
        if g.failed:
            fails += 1
        if g.terminated_ok:
            term_ok += 1
        if g.category_ok:
            cat_ok += 1
        if g.evidence_ok:
            ev_ok += 1
        if g.uncertainty_ok is not None:
            unc_n += 1
            if g.uncertainty_ok:
                unc_ok += 1
        if g.action_needs_manual or g.action_ok is None:
            act_manual += 1
        else:
            act_n += 1
            if g.action_ok:
                act_ok += 1

    return Scorecard(
        n=len(results),
        category_correct=cat_ok,
        evidence_correct=ev_ok,
        uncertainty_correct=unc_ok,
        uncertainty_scored=unc_n,
        action_correct=act_ok,
        action_auto_scored=act_n,
        action_manual_pending=act_manual,
        terminated_ok=term_ok,
        failures=fails,
        grades=grades,
        caveats=caveats,
    )
