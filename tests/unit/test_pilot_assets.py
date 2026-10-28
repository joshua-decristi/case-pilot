"""Offline tests for scenario pilot evaluation-only assets (no paid calls)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from casepilot.evaluation.pilot_assets import (
    CURRENT_ASSET_VERSION,
    REQUIRED_CALIBRATION_IDS,
    REQUIRED_EXPECTED_TUPLES,
    REQUIRED_SCENARIO_IDS,
    PilotAssetError,
    load_pilot_assets,
)

ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = ROOT / "evals" / "scenario_pilot"
FIXTURE = ROOT / "fixtures" / "enterprise" / "cases.json"
LABELS = ROOT / "evals" / "ground_truth" / "labels.json"


def test_load_pilot_assets_happy_path() -> None:
    """Frozen development pack loads with required ids/hashes."""
    assets = load_pilot_assets()
    assert assets.split == "development"
    assert assets.version == CURRENT_ASSET_VERSION
    assert assets.calibration_version == CURRENT_ASSET_VERSION
    assert len(assets.scenarios) == 6
    assert tuple(s.id for s in assets.scenarios) == REQUIRED_SCENARIO_IDS
    assert set(assets.references) == set(REQUIRED_SCENARIO_IDS)
    assert len(assets.calibration) == 8
    assert tuple(c.id for c in assets.calibration) == REQUIRED_CALIBRATION_IDS
    assert assets.fixture_sha256 == assets.hashes.fixture_sha256
    assert len(assets.hashes.scenarios_sha256) == 64
    assert len(assets.hashes.calibration_sha256) == 64


def test_calibration_expected_tuples_exact() -> None:
    """Calibration labels stay frozen; drift would mint false quality claims."""
    assets = load_pilot_assets()
    for ex in assets.calibration:
        assert ex.expected.as_tuple() == REQUIRED_EXPECTED_TUPLES[ex.id]
    assert REQUIRED_EXPECTED_TUPLES["supported_complete"] == ("meets", "meets", "meets")
    assert REQUIRED_EXPECTED_TUPLES["blanket_abstention"] == (
        "meets",
        "partially meets",
        "does not meet",
    )


def test_target_facts_match_fixture_values() -> None:
    """Reference required_facts must match enterprise fixture values."""
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


def test_rejects_wrong_calibration_tuple(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loader fail-closed on calibration expected-tuple drift."""
    from casepilot.evaluation import pilot_assets as pa

    dest = tmp_path / "evals" / "scenario_pilot"
    shutil.copytree(PILOT_DIR, dest)
    cal = json.loads((dest / "calibration.json").read_text(encoding="utf-8"))
    cal["examples"][0]["expected"]["factual_support"] = "does not meet"
    (dest / "calibration.json").write_text(json.dumps(cal), encoding="utf-8")
    monkeypatch.setattr(pa, "REPO_ROOT", tmp_path)
    with pytest.raises(PilotAssetError, match="expected tuple"):
        load_pilot_assets(path=dest, fixture_path=FIXTURE, labels_path=LABELS)
