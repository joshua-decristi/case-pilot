"""DEFERRED test — stage alongside src/casepilot/backend (not part of repo yet).

Backend-specific metadata assertion removed from the committed split version of
tests/unit/test_evaluation.py (commit history must stay green without backend).
When src/casepilot/backend lands: restore the full original suite from
/tmp/casepilot-commit-final/test_evaluation.py — it already covers this via the
original test_metadata_hashes_and_honest_git (prompt hash + prompt_source).
"""

from __future__ import annotations

from pathlib import Path

from casepilot.evaluation.cost import ALLOWED_LIVE_MODEL
from casepilot.evaluation.metadata import collect_metadata

ROOT = Path(__file__).resolve().parents[2]  # repo root when placed in tests/unit/


def test_metadata_backend_prompt_hash() -> None:
    meta = collect_metadata(
        labels_path=ROOT / "evals" / "ground_truth" / "labels.json",
        model=ALLOWED_LIVE_MODEL,
        split="evaluation",
    )
    assert meta.prompt_sha256 and len(meta.prompt_sha256) == 64
    assert meta.prompt_source == "casepilot.backend.agent.SYSTEM_PROMPT"
