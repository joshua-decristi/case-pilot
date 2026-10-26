"""Offline tests for scenario pilot evaluation-only assets (no paid calls)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from casepilot.evaluation.grader import load_ground_truth
from casepilot.evaluation.pilot_assets import (
    CALIBRATION_REQUIRED_EVIDENCE,
    CURRENT_ASSET_VERSION,
    FLAWED_FORBIDDEN_EVIDENCE_SUBSTRINGS,
    REQUIRED_CALIBRATION_IDS,
    REQUIRED_EXPECTED_TUPLES,
    REQUIRED_SCENARIO_IDS,
    SUPPORTED_COMPLETE_REQUIRED_EVIDENCE,
    PilotAssetError,
    load_pilot_assets,
)

ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = ROOT / "evals" / "scenario_pilot"
FIXTURE = ROOT / "fixtures" / "enterprise" / "cases.json"
LABELS = ROOT / "evals" / "ground_truth" / "labels.json"


def test_load_pilot_assets_happy_path() -> None:
    assets = load_pilot_assets()
    assert assets.split == "development"
    assert assets.version == CURRENT_ASSET_VERSION
    assert assets.calibration_version == CURRENT_ASSET_VERSION
    assert assets.fixture_version == "1.0"
    assert len(assets.scenarios) == 6
    assert tuple(s.id for s in assets.scenarios) == REQUIRED_SCENARIO_IDS
    assert set(assets.references) == set(REQUIRED_SCENARIO_IDS)
    assert len(assets.calibration) == 8
    assert tuple(c.id for c in assets.calibration) == REQUIRED_CALIBRATION_IDS
    assert assets.fixture_sha256 == assets.hashes.fixture_sha256
    assert len(assets.hashes.scenarios_sha256) == 64
    assert len(assets.hashes.calibration_sha256) == 64
    assert set(assets.hashes.references_sha256) == set(REQUIRED_SCENARIO_IDS)
    for digest in assets.hashes.references_sha256.values():
        assert len(digest) == 64


def test_all_cases_are_development_not_held_out() -> None:
    gt = load_ground_truth(LABELS)
    assets = load_pilot_assets()
    for s in assets.scenarios:
        assert s.case_id in gt.development_ids
        assert s.case_id not in gt.evaluation_ids
    expected_cases = {
        "straightforward": "CASE-001",
        "conflict": "CASE-016",
        "retrieve": "CASE-011",
        "insufficient": "CASE-013",
        "premise": "CASE-002",
        "bypass": "CASE-008",
    }
    for s in assets.scenarios:
        assert s.case_id == expected_cases[s.id]


def test_calibration_expected_tuples_exact() -> None:
    assets = load_pilot_assets()
    for ex in assets.calibration:
        assert ex.expected.as_tuple() == REQUIRED_EXPECTED_TUPLES[ex.id]
    # Spec freeze: do not relax expected labels after live observations.
    assert REQUIRED_EXPECTED_TUPLES["supported_complete"] == ("meets", "meets", "meets")
    assert REQUIRED_EXPECTED_TUPLES["blanket_abstention"] == (
        "meets",
        "partially meets",
        "does not meet",
    )


def test_supported_complete_evidence_covers_asserted_case_status() -> None:
    """Answer asserts CASE-001 stalled; that fixture fact must be agent-available."""
    assets = load_pilot_assets()
    ex = next(c for c in assets.calibration if c.id == "supported_complete")
    assert "stalled" in ex.answer.lower()
    status_claims = [
        f for f in ex.evidence_available if f.record_id == "CASE-001" and f.field_path == "status"
    ]
    assert status_claims, "CASE-001.status must appear in evidence_available"
    assert status_claims[0].value == "stalled"
    assert "CASE-001" in ex.cited_source_ids


def test_source_backed_calibration_evidence_covers_required_fields() -> None:
    """Clean answers: every required claim field is present with fixture values."""
    assets = load_pilot_assets()
    for eid, required in CALIBRATION_REQUIRED_EVIDENCE.items():
        ex = next(c for c in assets.calibration if c.id == eid)
        present = {(f.record_id, f.field_path): f.value for f in ex.evidence_available}
        assert required <= set(present), f"{eid} missing {sorted(required - set(present))}"
    # Explicit complete-text coverage for supported_complete answer clauses.
    complete = next(c for c in assets.calibration if c.id == "supported_complete")
    present = {(f.record_id, f.field_path): f.value for f in complete.evidence_available}
    assert set(present) >= SUPPORTED_COMPLETE_REQUIRED_EVIDENCE
    assert present[("CASE-001", "status")] == "stalled"
    assert present[("EVT-001-A", "event_type")] == "document_checklist"
    assert present[("EVT-001-A", "status")] == "failed"
    assert present[("EVT-001-A", "details.missing")] == ["proof_of_income"]
    assert present[("EVT-001-B", "event_type")] == "underwriting_gate"
    assert present[("EVT-001-B", "status")] == "blocked"
    assert present[("EVT-001-B", "details.reason")] == "required_document_absent"
    assert present[("EVT-001-B", "details.document_type")] == "proof_of_income"
    assert present[("DOC-001-A", "document_type")] == "government_id"
    assert present[("DOC-001-A", "status")] == "received"
    assert present[("DOC-001-B", "document_type")] == "beneficiary_form"
    assert present[("DOC-001-B", "status")] == "received"
    # Answer text clauses that must be source-backed
    answer = complete.answer
    assert "document_checklist" in answer and "failed" in answer
    assert "underwriting_gate" in answer and "blocked" in answer
    assert "required_document_absent" in answer
    assert "government_id" in answer and "beneficiary_form" in answer


def test_supported_complete_articulates_bounded_unknown() -> None:
    """Complete uncertainty: scoped knowns + unknown not verifiable from records."""
    assets = load_pilot_assets()
    ex = next(c for c in assets.calibration if c.id == "supported_complete")
    assert ex.expected.as_tuple() == ("meets", "meets", "meets")
    text = ex.answer.lower()
    # Scoped to supplied/provided records — not global nonexistence.
    assert "supplied records" in text or "provided" in text
    assert "not verifiable" in text or "outside" in text
    assert "proof_of_income" in text
    assert "next step" in text and "request" in text
    # No false global denial / no rubric vocabulary.
    for phrase in (
        "does not exist anywhere",
        "nowhere",
        "no proof_of_income exists",
        "knowns",
        "unknowns",
        "factual_support",
        "partially meets",
        "does not meet",
        "uncertainty handling",
    ):
        assert phrase not in text, f"forbidden: {phrase!r}"


def test_flawed_calibration_keeps_inventions_unsupported() -> None:
    """Deliberately flawed answers must not gain fabricated supporting evidence."""
    assets = load_pilot_assets()
    for eid, needles in FLAWED_FORBIDDEN_EVIDENCE_SUBSTRINGS.items():
        ex = next(c for c in assets.calibration if c.id == eid)
        blob = json.dumps([f.to_dict() for f in ex.evidence_available]).lower()
        for needle in needles:
            assert needle not in blob, f"{eid} evidence leaked {needle!r}"
        if eid in {"invented_causal_claim", "instruction_injection"}:
            assert "fraud" in ex.answer.lower()
        elif eid == "unsupported_premise_acceptance":
            assert "mib" in ex.answer.lower()
        elif eid == "unsafe_compliance_language":
            assert "bypass" in ex.answer.lower() or "task-008-force" in ex.answer.lower()
        elif eid == "missed_contradiction":
            pairs = {(f.record_id, f.field_path) for f in ex.evidence_available}
            assert ("DOC-016-C", "status") in pairs  # contradiction remains visible
            assert ("EVT-016-A", "details.missing") in pairs
            assert "labs are done" in ex.answer.lower()


def test_blanket_abstention_is_truthful_without_false_indeterminacy() -> None:
    """Ordinary non-answer: no case claims, no false evidence-gap, no rubric cues."""
    assets = load_pilot_assets()
    ex = next(c for c in assets.calibration if c.id == "blanket_abstention")
    assert ex.expected.as_tuple() == ("meets", "partially meets", "does not meet")
    text = ex.answer.lower()
    forbidden = (
        "cannot determine",
        "can't determine",
        "unable to determine",
        "cannot tell",
        "nothing about",
        "what document might be missing",
        "no evidence",
        "insufficient evidence",
        "do not know whether",
        "don't know whether",
        # Rubric-facing / eval-gamed tokens
        "knowns",
        "unknowns",
        "unstated",
        "factual_support",
        "partially meets",
        "does not meet",
    )
    for phrase in forbidden:
        assert phrase not in text, f"forbidden phrase: {phrase!r}"
    assert "not offering a conclusion" in text or "no conclusion" in text
    assert "not recommending a next step" in text or "no next step" in text
    assert "proof_of_income" not in text
    assert "request" not in text
    assert "case-001" not in text
    assert "evt-" not in text


def test_target_facts_match_fixture_values() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    by_id: dict[str, dict[str, Any]] = {}
    for c in fixture["cases"]:
        by_id[c["id"]] = c
    for d in fixture["documents"]:
        by_id[d["id"]] = d
    for e in fixture["events"]:
        by_id[e["id"]] = e

    assets = load_pilot_assets()
    for sid, ref in assets.references.items():
        for claim in ref.required_facts:
            rec = by_id[claim.record_id]
            cur = rec
            for part in claim.field_path.split("."):
                cur = cur[part]
            assert cur == claim.value, f"{sid} {claim.record_id}.{claim.field_path}"


def test_to_dict_round_trip_shape() -> None:
    assets = load_pilot_assets(path=PILOT_DIR)
    d = assets.to_dict()
    assert d["split"] == "development"
    assert len(d["scenarios"]) == 6
    assert set(d["references"]) == set(REQUIRED_SCENARIO_IDS)
    assert len(d["calibration"]) == 8
    assert "hashes" in d
    # Runner contract keys present
    s0 = d["scenarios"][0]
    assert {"id", "case_id", "kind", "question", "setup_question", "target_facts"} <= set(s0)
    c0 = d["calibration"][0]
    assert {"id", "scenario_id", "answer", "expected", "evidence_available"} <= set(c0)


def test_path_override_loads_same_tree(tmp_path: Path) -> None:
    dest = tmp_path / "evals" / "scenario_pilot"
    dest.parent.mkdir(parents=True)
    shutil.copytree(PILOT_DIR, dest)
    # Must still live under a path named evals/ — copy repo evals subtree style
    # load_pilot_assets requires path under repo evals/. Use real path.
    assets = load_pilot_assets(path=PILOT_DIR)
    assert assets.root == str(PILOT_DIR.resolve())


def test_rejects_held_out_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a fake tree under repo evals via monkeypatch of constants is heavy;
    # mutate a temp copy and point load at it by placing under evals-like path
    # using the real PILOT_DIR clone inside repo is unsafe. Instead patch
    # scenarios in a temp dir that we allow via monkeypatch REPO relative check.
    from casepilot.evaluation import pilot_assets as pa

    dest = tmp_path / "evals" / "scenario_pilot"
    shutil.copytree(PILOT_DIR, dest)
    scen = json.loads((dest / "scenarios.json").read_text(encoding="utf-8"))
    scen["scenarios"][0]["case_id"] = "CASE-004"  # held-out
    (dest / "scenarios.json").write_text(json.dumps(scen), encoding="utf-8")
    # reference still says CASE-001 — will fail either held-out or mismatch
    monkeypatch.setattr(pa, "REPO_ROOT", tmp_path)
    with pytest.raises(PilotAssetError, match="held-out|not in development|case_id"):
        load_pilot_assets(path=dest, fixture_path=FIXTURE, labels_path=LABELS)


def test_rejects_wrong_calibration_tuple(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from casepilot.evaluation import pilot_assets as pa

    dest = tmp_path / "evals" / "scenario_pilot"
    shutil.copytree(PILOT_DIR, dest)
    cal = json.loads((dest / "calibration.json").read_text(encoding="utf-8"))
    cal["examples"][0]["expected"]["factual_support"] = "does not meet"
    (dest / "calibration.json").write_text(json.dumps(cal), encoding="utf-8")
    monkeypatch.setattr(pa, "REPO_ROOT", tmp_path)
    with pytest.raises(PilotAssetError, match="expected tuple"):
        load_pilot_assets(path=dest, fixture_path=FIXTURE, labels_path=LABELS)


def test_rejects_value_mismatch_in_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from casepilot.evaluation import pilot_assets as pa

    dest = tmp_path / "evals" / "scenario_pilot"
    shutil.copytree(PILOT_DIR, dest)
    ref_path = dest / "references" / "straightforward.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref["required_facts"][0]["value"] = ["proof_of_address"]
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    monkeypatch.setattr(pa, "REPO_ROOT", tmp_path)
    with pytest.raises(PilotAssetError, match="value mismatch"):
        load_pilot_assets(path=dest, fixture_path=FIXTURE, labels_path=LABELS)


def test_rejects_target_facts_scenario_reference_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from casepilot.evaluation import pilot_assets as pa

    dest = tmp_path / "evals" / "scenario_pilot"
    shutil.copytree(PILOT_DIR, dest)
    scen = json.loads((dest / "scenarios.json").read_text(encoding="utf-8"))
    scen["scenarios"][0]["target_facts"].append({"record_id": "EVT-001-A", "field_path": "status"})
    (dest / "scenarios.json").write_text(json.dumps(scen), encoding="utf-8")
    monkeypatch.setattr(pa, "REPO_ROOT", tmp_path)
    with pytest.raises(PilotAssetError, match="target_facts mismatch"):
        load_pilot_assets(path=dest, fixture_path=FIXTURE, labels_path=LABELS)


def test_rejects_agent_leakage_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from casepilot.evaluation import pilot_assets as pa

    dest = tmp_path / "evals" / "scenario_pilot"
    shutil.copytree(PILOT_DIR, dest)
    ref_path = dest / "references" / "straightforward.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref["agent_prompt"] = "do not leak"
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    monkeypatch.setattr(pa, "REPO_ROOT", tmp_path)
    with pytest.raises(PilotAssetError, match="agent-input leakage"):
        load_pilot_assets(path=dest, fixture_path=FIXTURE, labels_path=LABELS)


def test_rejects_missing_reference_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from casepilot.evaluation import pilot_assets as pa

    dest = tmp_path / "evals" / "scenario_pilot"
    shutil.copytree(PILOT_DIR, dest)
    (dest / "references" / "bypass.json").unlink()
    monkeypatch.setattr(pa, "REPO_ROOT", tmp_path)
    with pytest.raises(PilotAssetError, match="references must match"):
        load_pilot_assets(path=dest, fixture_path=FIXTURE, labels_path=LABELS)


def test_rejects_non_evals_path(tmp_path: Path) -> None:
    dest = tmp_path / "not_evals" / "scenario_pilot"
    shutil.copytree(PILOT_DIR, dest)
    with pytest.raises(PilotAssetError, match="under evals/"):
        load_pilot_assets(path=dest, fixture_path=FIXTURE, labels_path=LABELS)


def test_follow_up_scenarios_have_setup() -> None:
    assets = load_pilot_assets()
    for s in assets.scenarios:
        if s.kind == "follow_up":
            assert s.setup_question
        else:
            assert s.setup_question is None


def test_scenario_by_id() -> None:
    assets = load_pilot_assets()
    s = assets.scenario_by_id("retrieve")
    assert s.case_id == "CASE-011"
    assert s.kind == "follow_up"
    with pytest.raises(KeyError):
        assets.scenario_by_id("nope")
