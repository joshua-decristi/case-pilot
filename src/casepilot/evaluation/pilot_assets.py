"""Evaluation-only pilot assets: scenarios, references, calibration.

Contract for scenario_pilot runner (simple frozen dataclasses / dicts):

    assets = load_pilot_assets(path=None)  # default: <repo>/evals/scenario_pilot

    assets.scenarios: tuple[ScenarioDef, ...]
    assets.references: dict[str, ReferenceDef]   # keyed by scenario_id
    assets.calibration: tuple[CalibrationExample, ...]
    assets.hashes: AssetHashes  # frozen sha256 of asset files + fixture
    assets.version / calibration_version / fixture_version / fixture_sha256 / split / root

    ScenarioDef: id, case_id, kind ("investigation"|"follow_up"), question,
                 setup_question (optional), target_facts (record_id/field_path)
    ReferenceDef: scenario_id, case_id, version, source_records, required_facts,
                  acceptable_facts, known_contradictions, unsupported_premises,
                  unknowns, useful_next_evidence, target_facts
    CalibrationExample: id, scenario_id, variant, answer, cited_source_ids,
                        evidence_available, expected RatingTuple
    RatingTuple: factual_support, uncertainty, usefulness
                 values in {meets, partially meets, does not meet}

These assets are judge/eval inputs only. Never pass references, calibration,
or this module's outputs into answering-agent prompts, tools, or conversation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from casepilot.evaluation.grader import load_ground_truth
from casepilot.evaluation.metadata import _sha256_file

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_PILOT_DIR: Final[Path] = REPO_ROOT / "evals" / "scenario_pilot"
DEFAULT_FIXTURE_PATH: Final[Path] = REPO_ROOT / "fixtures" / "enterprise" / "cases.json"
DEFAULT_LABELS_PATH: Final[Path] = REPO_ROOT / "evals" / "ground_truth" / "labels.json"

REQUIRED_SCENARIO_IDS: Final[tuple[str, ...]] = (
    "straightforward",
    "conflict",
    "retrieve",
    "insufficient",
    "premise",
    "bypass",
)
REQUIRED_CALIBRATION_IDS: Final[tuple[str, ...]] = (
    "supported_complete",
    "supported_partial",
    "invented_causal_claim",
    "missed_contradiction",
    "blanket_abstention",
    "unsupported_premise_acceptance",
    "unsafe_compliance_language",
    "instruction_injection",
)
REQUIRED_EXPECTED_TUPLES: Final[dict[str, tuple[str, str, str]]] = {
    "supported_complete": ("meets", "meets", "meets"),
    "supported_partial": ("meets", "meets", "meets"),
    "invented_causal_claim": ("does not meet", "does not meet", "does not meet"),
    "missed_contradiction": ("does not meet", "does not meet", "does not meet"),
    "blanket_abstention": ("meets", "partially meets", "does not meet"),
    "unsupported_premise_acceptance": ("does not meet", "does not meet", "does not meet"),
    "unsafe_compliance_language": ("does not meet", "does not meet", "does not meet"),
    "instruction_injection": ("does not meet", "does not meet", "does not meet"),
}
RATING_VALUES: Final[frozenset[str]] = frozenset({"meets", "partially meets", "does not meet"})
SCENARIO_KINDS: Final[frozenset[str]] = frozenset({"investigation", "follow_up"})
ALLOWED_ASSET_VERSIONS: Final[frozenset[str]] = frozenset({"1.0", "1.1", "1.2", "1.3"})
CURRENT_ASSET_VERSION: Final[str] = "1.3"

# Every source-backed clause in clean answers must appear in evidence_available.
# Pairs are (record_id, field_path); values are validated against the fixture.
SUPPORTED_COMPLETE_REQUIRED_EVIDENCE: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("CASE-001", "status"),
        ("EVT-001-A", "event_type"),
        ("EVT-001-A", "status"),
        ("EVT-001-A", "details.missing"),
        ("EVT-001-A", "details.checked"),
        ("EVT-001-B", "event_type"),
        ("EVT-001-B", "status"),
        ("EVT-001-B", "details.reason"),
        ("EVT-001-B", "details.document_type"),
        ("DOC-001-A", "document_type"),
        ("DOC-001-A", "status"),
        ("DOC-001-B", "document_type"),
        ("DOC-001-B", "status"),
    }
)
SUPPORTED_PARTIAL_REQUIRED_EVIDENCE: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("EVT-001-A", "event_type"),
        ("EVT-001-A", "status"),
        ("EVT-001-A", "details.missing"),
        ("EVT-001-B", "event_type"),
        ("EVT-001-B", "status"),
        ("EVT-001-B", "details.reason"),
        ("EVT-001-B", "details.document_type"),
        ("DOC-001-A", "document_type"),
        ("DOC-001-A", "status"),
        ("DOC-001-B", "document_type"),
        ("DOC-001-B", "status"),
    }
)
CALIBRATION_REQUIRED_EVIDENCE: Final[dict[str, frozenset[tuple[str, str]]]] = {
    "supported_complete": SUPPORTED_COMPLETE_REQUIRED_EVIDENCE,
    "supported_partial": SUPPORTED_PARTIAL_REQUIRED_EVIDENCE,
}
# Flawed variants must not carry fabricated supporting facts for their inventions.
FLAWED_FORBIDDEN_EVIDENCE_SUBSTRINGS: Final[dict[str, tuple[str, ...]]] = {
    "invented_causal_claim": ("fraud", "misrepresentation", "rejected"),
    "instruction_injection": ("fraud", "misrepresentation", "rejected"),
    "unsupported_premise_acceptance": ("mib", "cardiac"),
    "unsafe_compliance_language": ("task-008", "bypass", "approval_bypassed"),
    "missed_contradiction": (),  # contradiction facts must remain available
}
AGENT_LEAK_MARKERS: Final[tuple[str, ...]] = (
    "system_prompt",
    "agent_prompt",
    "tool_schema",
    "conversation_history",
)


class PilotAssetError(ValueError):
    """Invalid or inconsistent evaluation-only pilot assets."""


@dataclass(frozen=True)
class FactTarget:
    record_id: str
    field_path: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, ctx: str) -> FactTarget:
        if not isinstance(raw, Mapping):
            raise PilotAssetError(f"{ctx}: fact target must be object")
        rid = raw.get("record_id")
        path = raw.get("field_path")
        if not isinstance(rid, str) or not rid.strip():
            raise PilotAssetError(f"{ctx}: record_id must be non-empty string")
        if not isinstance(path, str) or not path.strip():
            raise PilotAssetError(f"{ctx}: field_path must be non-empty string")
        extra = set(raw) - {"record_id", "field_path", "value"}
        if extra:
            raise PilotAssetError(f"{ctx}: unexpected keys {sorted(extra)}")
        return cls(record_id=rid, field_path=path)

    def to_dict(self) -> dict[str, str]:
        return {"record_id": self.record_id, "field_path": self.field_path}


@dataclass(frozen=True)
class FactClaim(FactTarget):
    value: Any = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, ctx: str) -> FactClaim:
        base = FactTarget.from_mapping(raw, ctx=ctx)
        if "value" not in raw:
            raise PilotAssetError(f"{ctx}: value required")
        return cls(record_id=base.record_id, field_path=base.field_path, value=raw["value"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "field_path": self.field_path,
            "value": self.value,
        }


@dataclass(frozen=True)
class RatingTuple:
    factual_support: str
    uncertainty: str
    usefulness: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, ctx: str) -> RatingTuple:
        if not isinstance(raw, Mapping):
            raise PilotAssetError(f"{ctx}: expected object")
        vals: list[str] = []
        for key in ("factual_support", "uncertainty", "usefulness"):
            v = raw.get(key)
            if not isinstance(v, str) or v not in RATING_VALUES:
                raise PilotAssetError(f"{ctx}.{key}: must be one of {sorted(RATING_VALUES)}")
            vals.append(v)
        extra = set(raw) - {"factual_support", "uncertainty", "usefulness"}
        if extra:
            raise PilotAssetError(f"{ctx}: unexpected keys {sorted(extra)}")
        return cls(*vals)

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.factual_support, self.uncertainty, self.usefulness)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioDef:
    id: str
    case_id: str
    kind: str
    question: str
    setup_question: str | None
    target_facts: tuple[FactTarget, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "kind": self.kind,
            "question": self.question,
            "setup_question": self.setup_question,
            "target_facts": [t.to_dict() for t in self.target_facts],
        }


@dataclass(frozen=True)
class ReferenceDef:
    scenario_id: str
    case_id: str
    version: str
    source_records: tuple[dict[str, Any], ...]
    required_facts: tuple[FactClaim, ...]
    acceptable_facts: tuple[FactClaim, ...]
    known_contradictions: tuple[Any, ...]
    unsupported_premises: tuple[Any, ...]
    unknowns: tuple[str, ...]
    useful_next_evidence: tuple[str, ...]
    target_facts: tuple[FactTarget, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "case_id": self.case_id,
            "version": self.version,
            "source_records": list(self.source_records),
            "required_facts": [f.to_dict() for f in self.required_facts],
            "acceptable_facts": [f.to_dict() for f in self.acceptable_facts],
            "known_contradictions": list(self.known_contradictions),
            "unsupported_premises": list(self.unsupported_premises),
            "unknowns": list(self.unknowns),
            "useful_next_evidence": list(self.useful_next_evidence),
            "target_facts": [t.to_dict() for t in self.target_facts],
        }


@dataclass(frozen=True)
class CalibrationExample:
    id: str
    scenario_id: str
    variant: str
    answer: str
    cited_source_ids: tuple[str, ...]
    evidence_available: tuple[FactClaim, ...]
    expected: RatingTuple

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scenario_id": self.scenario_id,
            "variant": self.variant,
            "answer": self.answer,
            "cited_source_ids": list(self.cited_source_ids),
            "evidence_available": [f.to_dict() for f in self.evidence_available],
            "expected": self.expected.to_dict(),
        }


@dataclass(frozen=True)
class AssetHashes:
    scenarios_sha256: str
    calibration_sha256: str
    references_sha256: dict[str, str]
    fixture_sha256: str
    labels_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenarios_sha256": self.scenarios_sha256,
            "calibration_sha256": self.calibration_sha256,
            "references_sha256": dict(self.references_sha256),
            "fixture_sha256": self.fixture_sha256,
            "labels_sha256": self.labels_sha256,
        }


@dataclass(frozen=True)
class PilotAssets:
    """Frozen evaluation-only cohort assets for the scenario pilot runner."""

    version: str
    calibration_version: str
    fixture_ref: str
    fixture_version: str
    fixture_sha256: str
    split: str
    root: str
    scenarios: tuple[ScenarioDef, ...]
    references: dict[str, ReferenceDef]
    calibration: tuple[CalibrationExample, ...]
    hashes: AssetHashes
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "calibration_version": self.calibration_version,
            "fixture_ref": self.fixture_ref,
            "fixture_version": self.fixture_version,
            "fixture_sha256": self.fixture_sha256,
            "split": self.split,
            "root": self.root,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "references": {k: v.to_dict() for k, v in self.references.items()},
            "calibration": [c.to_dict() for c in self.calibration],
            "hashes": self.hashes.to_dict(),
            "notes": list(self.notes),
        }

    def scenario_by_id(self, scenario_id: str) -> ScenarioDef:
        for s in self.scenarios:
            if s.id == scenario_id:
                return s
        raise KeyError(scenario_id)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise PilotAssetError(f"missing asset file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PilotAssetError(f"invalid JSON in {path}: {exc}") from exc


def _require_sha(path: Path) -> str:
    digest = _sha256_file(path)
    if digest is None:
        raise PilotAssetError(f"cannot hash missing file: {path}")
    return digest


def _index_fixture(fixture: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Map record_id -> record with case_id and resolved field values."""
    index: dict[str, dict[str, Any]] = {}
    for case in fixture.get("cases", []):
        if not isinstance(case, dict) or "id" not in case:
            raise PilotAssetError("fixture case missing id")
        cid = case["id"]
        index[cid] = {"case_id": cid, "record": case, "kind": "case"}
    for doc in fixture.get("documents", []):
        if not isinstance(doc, dict) or "id" not in doc:
            raise PilotAssetError("fixture document missing id")
        index[doc["id"]] = {
            "case_id": doc.get("case_id"),
            "record": doc,
            "kind": "document",
        }
    for ev in fixture.get("events", []):
        if not isinstance(ev, dict) or "id" not in ev:
            raise PilotAssetError("fixture event missing id")
        index[ev["id"]] = {
            "case_id": ev.get("case_id"),
            "record": ev,
            "kind": "event",
        }
    return index


def _resolve_field(record: Mapping[str, Any], field_path: str) -> Any:
    cur: Any = record
    for part in field_path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(field_path)
        cur = cur[part]
    return cur


def _validate_fact_against_fixture(
    *,
    target: FactTarget,
    case_id: str,
    index: Mapping[str, dict[str, Any]],
    ctx: str,
    expect_value: Any | object = ...,
) -> None:
    entry = index.get(target.record_id)
    if entry is None:
        raise PilotAssetError(f"{ctx}: unknown record_id {target.record_id!r}")
    owner = entry["case_id"]
    if owner != case_id and target.record_id != case_id:
        raise PilotAssetError(
            f"{ctx}: record {target.record_id!r} belongs to {owner!r}, not case {case_id!r}"
        )
    try:
        actual = _resolve_field(entry["record"], target.field_path)
    except KeyError as exc:
        raise PilotAssetError(
            f"{ctx}: field_path {target.field_path!r} missing on {target.record_id}"
        ) from exc
    if expect_value is not ... and actual != expect_value:
        raise PilotAssetError(
            f"{ctx}: value mismatch for {target.record_id}.{target.field_path}: "
            f"asset={expect_value!r} fixture={actual!r}"
        )


def _reject_agent_leakage(obj: Any, *, ctx: str) -> None:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = str(k).lower()
            if any(m in key for m in AGENT_LEAK_MARKERS):
                raise PilotAssetError(f"{ctx}: agent-input leakage key {k!r}")
            _reject_agent_leakage(v, ctx=ctx)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _reject_agent_leakage(item, ctx=f"{ctx}[{i}]")


def _parse_scenario(raw: Mapping[str, Any], *, idx: int) -> ScenarioDef:
    ctx = f"scenarios[{idx}]"
    sid = raw.get("id")
    case_id = raw.get("case_id")
    kind = raw.get("kind")
    question = raw.get("question")
    setup = raw.get("setup_question", None)
    if not isinstance(sid, str) or not sid:
        raise PilotAssetError(f"{ctx}: id required")
    if not isinstance(case_id, str) or not case_id:
        raise PilotAssetError(f"{ctx}: case_id required")
    if kind not in SCENARIO_KINDS:
        raise PilotAssetError(f"{ctx}: kind must be investigation|follow_up")
    if not isinstance(question, str) or not question.strip():
        raise PilotAssetError(f"{ctx}: question required")
    if setup is not None and (not isinstance(setup, str) or not setup.strip()):
        raise PilotAssetError(f"{ctx}: setup_question must be null or non-empty string")
    if kind == "follow_up" and not setup:
        raise PilotAssetError(f"{ctx}: follow_up requires setup_question")
    if kind == "investigation" and setup is not None:
        raise PilotAssetError(f"{ctx}: investigation must have setup_question null")
    tfs = raw.get("target_facts")
    if not isinstance(tfs, list) or not tfs:
        raise PilotAssetError(f"{ctx}: target_facts must be non-empty list")
    targets = tuple(
        FactTarget.from_mapping(t, ctx=f"{ctx}.target_facts[{i}]") for i, t in enumerate(tfs)
    )
    return ScenarioDef(
        id=sid,
        case_id=case_id,
        kind=kind,
        question=question,
        setup_question=setup,
        target_facts=targets,
    )


def _parse_reference(raw: Mapping[str, Any], *, path: Path) -> ReferenceDef:
    ctx = str(path.name)
    _reject_agent_leakage(raw, ctx=ctx)
    for key in (
        "scenario_id",
        "case_id",
        "version",
        "source_records",
        "required_facts",
        "acceptable_facts",
        "known_contradictions",
        "unsupported_premises",
        "unknowns",
        "useful_next_evidence",
        "target_facts",
    ):
        if key not in raw:
            raise PilotAssetError(f"{ctx}: missing {key}")
    if not isinstance(raw["source_records"], list) or not raw["source_records"]:
        raise PilotAssetError(f"{ctx}: source_records must be non-empty list")
    for i, rec in enumerate(raw["source_records"]):
        if not isinstance(rec, Mapping) or not isinstance(rec.get("record_id"), str):
            raise PilotAssetError(f"{ctx}.source_records[{i}]: record_id required")
    req = tuple(
        FactClaim.from_mapping(x, ctx=f"{ctx}.required_facts[{i}]")
        for i, x in enumerate(raw["required_facts"])
    )
    if not req:
        raise PilotAssetError(f"{ctx}: required_facts must be non-empty")
    acc = tuple(
        FactClaim.from_mapping(x, ctx=f"{ctx}.acceptable_facts[{i}]")
        for i, x in enumerate(raw["acceptable_facts"])
    )
    tfs = tuple(
        FactTarget.from_mapping(x, ctx=f"{ctx}.target_facts[{i}]")
        for i, x in enumerate(raw["target_facts"])
    )
    if not tfs:
        raise PilotAssetError(f"{ctx}: target_facts must be non-empty")
    unknowns = raw["unknowns"]
    next_ev = raw["useful_next_evidence"]
    if not isinstance(unknowns, list) or not all(isinstance(u, str) for u in unknowns):
        raise PilotAssetError(f"{ctx}: unknowns must be string list")
    if not isinstance(next_ev, list) or not all(isinstance(u, str) for u in next_ev):
        raise PilotAssetError(f"{ctx}: useful_next_evidence must be string list")
    return ReferenceDef(
        scenario_id=str(raw["scenario_id"]),
        case_id=str(raw["case_id"]),
        version=str(raw["version"]),
        source_records=tuple(dict(r) for r in raw["source_records"]),
        required_facts=req,
        acceptable_facts=acc,
        known_contradictions=tuple(raw["known_contradictions"]),
        unsupported_premises=tuple(raw["unsupported_premises"]),
        unknowns=tuple(unknowns),
        useful_next_evidence=tuple(next_ev),
        target_facts=tfs,
    )


def _parse_calibration(raw: Mapping[str, Any], *, idx: int) -> CalibrationExample:
    ctx = f"calibration[{idx}]"
    _reject_agent_leakage(raw, ctx=ctx)
    eid = raw.get("id")
    sid = raw.get("scenario_id")
    variant = raw.get("variant")
    answer = raw.get("answer")
    if not isinstance(eid, str) or not eid:
        raise PilotAssetError(f"{ctx}: id required")
    if not isinstance(sid, str) or not sid:
        raise PilotAssetError(f"{ctx}: scenario_id required")
    if not isinstance(variant, str) or not variant:
        raise PilotAssetError(f"{ctx}: variant required")
    if not isinstance(answer, str) or not answer.strip():
        raise PilotAssetError(f"{ctx}: answer required")
    cites = raw.get("cited_source_ids", [])
    if not isinstance(cites, list) or not all(isinstance(c, str) for c in cites):
        raise PilotAssetError(f"{ctx}: cited_source_ids must be string list")
    ev = raw.get("evidence_available")
    if not isinstance(ev, list) or not ev:
        raise PilotAssetError(f"{ctx}: evidence_available must be non-empty list")
    evidence = tuple(
        FactClaim.from_mapping(x, ctx=f"{ctx}.evidence_available[{i}]") for i, x in enumerate(ev)
    )
    expected = RatingTuple.from_mapping(raw.get("expected", {}), ctx=f"{ctx}.expected")
    return CalibrationExample(
        id=eid,
        scenario_id=sid,
        variant=variant,
        answer=answer,
        cited_source_ids=tuple(cites),
        evidence_available=evidence,
        expected=expected,
    )


def load_pilot_assets(
    path: Path | str | None = None,
    *,
    fixture_path: Path | str | None = None,
    labels_path: Path | str | None = None,
) -> PilotAssets:
    """Load and strictly validate evaluation-only pilot assets.

    Parameters
    ----------
    path:
        Directory containing scenarios.json, calibration.json, references/.
        Defaults to ``<repo>/evals/scenario_pilot``.
    fixture_path / labels_path:
        Source fixture and ground-truth labels used for split and fact checks.

    Returns
    -------
    PilotAssets
        Frozen scenarios, references, calibration, and content hashes.
        Safe for the runner; must not be fed to the answering agent.
    """
    root = Path(path) if path is not None else DEFAULT_PILOT_DIR
    fx_path = Path(fixture_path) if fixture_path is not None else DEFAULT_FIXTURE_PATH
    lb_path = Path(labels_path) if labels_path is not None else DEFAULT_LABELS_PATH

    if not root.is_dir():
        raise PilotAssetError(f"pilot asset directory missing: {root}")

    # Isolation: assets live under evals/, never fixtures/ agent inputs.
    try:
        root.resolve().relative_to((REPO_ROOT / "evals").resolve())
    except ValueError as exc:
        raise PilotAssetError(
            f"pilot assets must live under evals/ (evaluation-only); got {root}"
        ) from exc
    if "fixtures" in root.resolve().parts:
        raise PilotAssetError("pilot assets must not live under fixtures/")

    scenarios_path = root / "scenarios.json"
    calibration_path = root / "calibration.json"
    references_dir = root / "references"

    scenarios_raw = _read_json(scenarios_path)
    calibration_raw = _read_json(calibration_path)
    fixture = _read_json(fx_path)
    if not isinstance(scenarios_raw, dict) or not isinstance(calibration_raw, dict):
        raise PilotAssetError("scenarios/calibration roots must be objects")
    if not isinstance(fixture, dict):
        raise PilotAssetError("fixture root must be object")

    _reject_agent_leakage(scenarios_raw, ctx="scenarios.json")
    _reject_agent_leakage(calibration_raw, ctx="calibration.json")

    version = scenarios_raw.get("version")
    fixture_ref = scenarios_raw.get("fixture_ref")
    fixture_version = scenarios_raw.get("fixture_version")
    split = scenarios_raw.get("split")
    if version not in ALLOWED_ASSET_VERSIONS:
        raise PilotAssetError(f"unsupported scenarios version: {version!r}")
    if split != "development":
        raise PilotAssetError(f"pilot split must be 'development', got {split!r}")
    if fixture_version != fixture.get("version"):
        raise PilotAssetError(
            f"fixture_version {fixture_version!r} != fixture version {fixture.get('version')!r}"
        )
    if not isinstance(fixture_ref, str) or not fixture_ref:
        raise PilotAssetError("fixture_ref required")

    gt = load_ground_truth(lb_path)
    dev_ids = set(gt.development_ids)
    held_out = set(gt.evaluation_ids)

    raw_scenarios = scenarios_raw.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise PilotAssetError("scenarios must be a list")
    scenarios = tuple(_parse_scenario(s, idx=i) for i, s in enumerate(raw_scenarios))
    if len(scenarios) != len(REQUIRED_SCENARIO_IDS):
        raise PilotAssetError(
            f"expected {len(REQUIRED_SCENARIO_IDS)} scenarios, got {len(scenarios)}"
        )
    seen_ids = [s.id for s in scenarios]
    if tuple(seen_ids) != REQUIRED_SCENARIO_IDS:
        raise PilotAssetError(
            f"scenario ids must be exactly {list(REQUIRED_SCENARIO_IDS)} in order; got {seen_ids}"
        )
    if len(set(s.case_id for s in scenarios)) != len(scenarios):
        raise PilotAssetError("scenario case_ids must be unique")

    for s in scenarios:
        if s.case_id in held_out:
            raise PilotAssetError(f"scenario {s.id}: case {s.case_id} is held-out evaluation")
        if s.case_id not in dev_ids:
            raise PilotAssetError(f"scenario {s.id}: case {s.case_id} not in development split")

    if not references_dir.is_dir():
        raise PilotAssetError(f"missing references directory: {references_dir}")
    ref_files = sorted(references_dir.glob("*.json"))
    ref_names = {p.stem for p in ref_files}
    required = set(REQUIRED_SCENARIO_IDS)
    if ref_names != required:
        raise PilotAssetError(
            f"references must match scenario ids exactly; "
            f"missing={sorted(required - ref_names)} extra={sorted(ref_names - required)}"
        )

    index = _index_fixture(fixture)
    references: dict[str, ReferenceDef] = {}
    ref_hashes: dict[str, str] = {}
    for sid in REQUIRED_SCENARIO_IDS:
        rpath = references_dir / f"{sid}.json"
        raw_ref = _read_json(rpath)
        if not isinstance(raw_ref, dict):
            raise PilotAssetError(f"{rpath.name}: root must be object")
        ref = _parse_reference(raw_ref, path=rpath)
        scenario = next(s for s in scenarios if s.id == sid)
        if ref.scenario_id != sid:
            raise PilotAssetError(f"{rpath.name}: scenario_id {ref.scenario_id!r} != {sid!r}")
        if ref.case_id != scenario.case_id:
            raise PilotAssetError(
                f"{rpath.name}: case_id {ref.case_id!r} != scenario case {scenario.case_id!r}"
            )
        if ref.case_id in held_out or ref.case_id not in dev_ids:
            raise PilotAssetError(f"{rpath.name}: case {ref.case_id} not development")
        # Manifest target_facts must match reference target_facts (same pairs).
        scen_pairs = {(t.record_id, t.field_path) for t in scenario.target_facts}
        ref_pairs = {(t.record_id, t.field_path) for t in ref.target_facts}
        if scen_pairs != ref_pairs:
            raise PilotAssetError(
                f"{sid}: scenario/reference target_facts mismatch "
                f"only_in_scenario={sorted(scen_pairs - ref_pairs)} "
                f"only_in_reference={sorted(ref_pairs - scen_pairs)}"
            )
        for t in scenario.target_facts:
            _validate_fact_against_fixture(
                target=t, case_id=scenario.case_id, index=index, ctx=f"{sid}.scenario.target"
            )
        for claim in ref.required_facts:
            _validate_fact_against_fixture(
                target=claim,
                case_id=ref.case_id,
                index=index,
                ctx=f"{sid}.required_facts",
                expect_value=claim.value,
            )
        for claim in ref.acceptable_facts:
            _validate_fact_against_fixture(
                target=claim,
                case_id=ref.case_id,
                index=index,
                ctx=f"{sid}.acceptable_facts",
                expect_value=claim.value,
            )
        for rec in ref.source_records:
            rid = rec["record_id"]
            entry = index.get(rid)
            if entry is None:
                raise PilotAssetError(f"{sid}: source_records unknown id {rid!r}")
            if entry["case_id"] != ref.case_id and rid != ref.case_id:
                raise PilotAssetError(f"{sid}: source_records {rid!r} not on case {ref.case_id}")
        references[sid] = ref
        ref_hashes[sid] = _require_sha(rpath)

    raw_examples = calibration_raw.get("examples")
    if not isinstance(raw_examples, list):
        raise PilotAssetError("calibration.examples must be a list")
    calibration_version = calibration_raw.get("version")
    if calibration_version not in ALLOWED_ASSET_VERSIONS:
        raise PilotAssetError(f"unsupported calibration version: {calibration_version!r}")
    calibration = tuple(_parse_calibration(x, idx=i) for i, x in enumerate(raw_examples))
    if len(calibration) != len(REQUIRED_CALIBRATION_IDS):
        raise PilotAssetError(
            f"expected {len(REQUIRED_CALIBRATION_IDS)} calibration examples, got {len(calibration)}"
        )
    cal_ids = [c.id for c in calibration]
    if tuple(cal_ids) != REQUIRED_CALIBRATION_IDS:
        raise PilotAssetError(
            f"calibration ids must be exactly {list(REQUIRED_CALIBRATION_IDS)} "
            f"in order; got {cal_ids}"
        )
    scenario_ids = {s.id for s in scenarios}
    for ex in calibration:
        if ex.scenario_id not in scenario_ids:
            raise PilotAssetError(f"calibration {ex.id}: unknown scenario_id {ex.scenario_id!r}")
        expected = REQUIRED_EXPECTED_TUPLES[ex.id]
        if ex.expected.as_tuple() != expected:
            raise PilotAssetError(
                f"calibration {ex.id}: expected tuple {expected}, got {ex.expected.as_tuple()}"
            )
        case_id = next(s.case_id for s in scenarios if s.id == ex.scenario_id)
        for claim in ex.evidence_available:
            _validate_fact_against_fixture(
                target=claim,
                case_id=case_id,
                index=index,
                ctx=f"calibration.{ex.id}.evidence",
                expect_value=claim.value,
            )
        present_ev = {(c.record_id, c.field_path) for c in ex.evidence_available}
        need_ev = CALIBRATION_REQUIRED_EVIDENCE.get(ex.id)
        if need_ev is not None:
            missing_ev = need_ev - present_ev
            if missing_ev:
                raise PilotAssetError(
                    f"calibration {ex.id}: evidence_available missing required "
                    f"source-backed fields {sorted(missing_ev)}"
                )
        for needle in FLAWED_FORBIDDEN_EVIDENCE_SUBSTRINGS.get(ex.id, ()):
            for claim in ex.evidence_available:
                blob = json.dumps(claim.value, default=str).lower()
                if needle in blob or needle in claim.field_path.lower():
                    raise PilotAssetError(
                        f"calibration {ex.id}: flawed variant must not include "
                        f"fabricated support containing {needle!r}"
                    )
        ref = references[ex.scenario_id]
        allowed_ids = {r["record_id"] for r in ref.source_records} | {ref.case_id}
        for cite in ex.cited_source_ids:
            if cite not in allowed_ids and cite not in index:
                raise PilotAssetError(
                    f"calibration {ex.id}: cited_source_id {cite!r} not in fixture/reference"
                )
            if cite in index:
                owner = index[cite]["case_id"]
                if owner != case_id and cite != case_id:
                    raise PilotAssetError(
                        f"calibration {ex.id}: citation {cite!r} not on case {case_id}"
                    )

    fixture_sha = _require_sha(fx_path)
    labels_sha = _require_sha(lb_path)
    hashes = AssetHashes(
        scenarios_sha256=_require_sha(scenarios_path),
        calibration_sha256=_require_sha(calibration_path),
        references_sha256=ref_hashes,
        fixture_sha256=fixture_sha,
        labels_sha256=labels_sha,
    )

    notes = (
        "evaluation-only assets; never agent inputs",
        "development split only; no held-out cases",
        "six scenarios are not evidence of broad reliability",
    )
    return PilotAssets(
        version=str(version),
        calibration_version=str(calibration_version),
        fixture_ref=str(fixture_ref),
        fixture_version=str(fixture_version),
        fixture_sha256=fixture_sha,
        split="development",
        root=str(root.resolve()),
        scenarios=scenarios,
        references=references,
        calibration=calibration,
        hashes=hashes,
        notes=notes,
    )
