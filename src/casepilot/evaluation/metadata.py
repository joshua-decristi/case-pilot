"""Run metadata: fixture/prompt hashes, versions; honest when git unavailable."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git_revision(repo: Path | None = None) -> str:
    """Return git HEAD sha or the literal 'unavailable' (never invent)."""
    cwd = str(repo) if repo is not None else None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
    except OSError, subprocess.SubprocessError:
        return "unavailable"
    if out.returncode != 0:
        return "unavailable"
    rev = (out.stdout or "").strip()
    return rev or "unavailable"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


@dataclass
class RunMetadata:
    timestamp: str
    git_revision: str
    python_version: str
    casepilot_version: str
    package_versions: dict[str, str] = field(default_factory=dict)
    fixture_path: str | None = None
    fixture_sha256: str | None = None
    labels_path: str | None = None
    labels_sha256: str | None = None
    prompt_sha256: str | None = None
    prompt_source: str | None = None
    model: str | None = None
    split: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_metadata(
    *,
    fixture_path: Path | None = None,
    labels_path: Path | None = None,
    model: str | None = None,
    split: str | None = None,
    repo_root: Path | None = None,
) -> RunMetadata:
    root = repo_root or Path(__file__).resolve().parents[3]
    fx = fixture_path or (root / "fixtures" / "enterprise" / "cases.json")
    lb = labels_path or (root / "evals" / "ground_truth" / "labels.json")

    prompt_sha: str | None = None
    prompt_source: str | None = None
    notes: list[str] = []
    try:
        # Import only the prompt string; do not invoke the agent.
        from casepilot.backend.agent import SYSTEM_PROMPT

        prompt_sha = _sha256_text(SYSTEM_PROMPT)
        prompt_source = "casepilot.backend.agent.SYSTEM_PROMPT"
    except Exception as exc:  # noqa: BLE001 — honest gap reporting
        notes.append(f"prompt hash unavailable: {exc}")

    pkgs = {
        "langchain": package_version("langchain"),
        "httpx": package_version("httpx"),
        "fastapi": package_version("fastapi"),
        "pydantic": package_version("pydantic"),
    }

    return RunMetadata(
        timestamp=datetime.now(UTC).isoformat(),
        git_revision=git_revision(root),
        python_version=sys.version.split()[0],
        casepilot_version=package_version("casepilot"),
        package_versions=pkgs,
        fixture_path=str(fx),
        fixture_sha256=_sha256_file(fx),
        labels_path=str(lb),
        labels_sha256=_sha256_file(lb),
        prompt_sha256=prompt_sha,
        prompt_source=prompt_source,
        model=model,
        split=split,
        notes=notes,
    )
