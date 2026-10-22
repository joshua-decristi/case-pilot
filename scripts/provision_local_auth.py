"""Provision local service/operator tokens into .env when missing.

Does not print secret values. Does not overwrite existing token keys.
Does not touch OpenRouter keys. Refuses silent bypass: missing tokens are
reported; write only with --write.
"""

from __future__ import annotations

import argparse
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
EXAMPLE_PATH = ROOT / ".env.example"

REQUIRED = ("SIMULATOR_SERVICE_TOKEN", "OPERATOR_TOKEN")
_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _present_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _KEY_RE.match(s)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if value:
            keys.add(key)
    return keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Append missing tokens to .env (never overwrite existing keys).",
    )
    args = parser.parse_args(argv)

    present = _present_keys(ENV_PATH)
    missing = [k for k in REQUIRED if k not in present]

    if not missing:
        print("local auth tokens: present in .env (values not printed)")
        return 0

    print(
        "local auth tokens missing: " + ", ".join(missing),
        file=sys.stderr,
    )
    if not args.write:
        print(
            "Refusing silent bypass. Re-run with --write to append random "
            f"tokens to {ENV_PATH.name} (existing keys left unchanged).",
            file=sys.stderr,
        )
        if not ENV_PATH.exists() and EXAMPLE_PATH.exists():
            print(
                f"Hint: copy {EXAMPLE_PATH.name} → {ENV_PATH.name} first if needed.",
                file=sys.stderr,
            )
        return 2

    lines: list[str] = []
    if ENV_PATH.is_file():
        text = ENV_PATH.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            lines.append("")
    else:
        print(f"creating {ENV_PATH.name} (tokens only; OpenRouter keys not set)")

    lines.append("# Local auth tokens (generated; never commit real .env)")
    for key in missing:
        # Generate only; do not print the value.
        lines.append(f"{key}={secrets.token_urlsafe(32)}")
    with ENV_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(
        f"wrote {len(missing)} missing token key(s) to {ENV_PATH.name} "
        "(values not printed; existing keys untouched)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
