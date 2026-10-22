"""Deterministic fixture seed entrypoint (delegates to simulator seed CLI).

Default is non-destructive (refuses overwrite). Destructive reseed requires
``--reset``.
"""

from __future__ import annotations

import importlib
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        module = importlib.import_module("casepilot.simulator.seed")
    except ModuleNotFoundError as exc:
        print(
            "Simulator seed module not implemented yet (expected casepilot.simulator.seed).",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    cli = getattr(module, "main", None)
    if cli is None or not callable(cli):
        print("casepilot.simulator.seed must expose main().", file=sys.stderr)
        return 1

    result = cli(args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
