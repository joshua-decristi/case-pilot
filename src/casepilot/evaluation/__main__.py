"""CLI entrypoint for evaluation + operator cost recovery.

  uv run python -m casepilot.evaluation --offline path/to/results.json
  uv run python -m casepilot.evaluation --live --max-cases 2 --budget-usd 5
  uv run python -m casepilot.evaluation --recover-stale-reservations \\
      --operator local --reason "crash" --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from casepilot.evaluation.cost import (
    ALLOWED_LIVE_MODEL,
    DEFAULT_BUDGET_USD,
    DEFAULT_LEDGER_PATH,
    CostLedger,
    CostPause,
)
from casepilot.evaluation.runner import LiveEvalConfig, run_live, run_offline_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m casepilot.evaluation",
        description="CasePilot grader, opt-in live eval, operator cost recovery",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--live",
        action="store_true",
        help="Explicit paid opt-in: call backend/OpenRouter (GLM-only)",
    )
    mode.add_argument(
        "--offline",
        metavar="RESULTS_JSON",
        type=Path,
        help="Grade saved results JSON; no model/backend calls",
    )
    mode.add_argument(
        "--recover-stale-reservations",
        action="store_true",
        help=(
            "Operator: settle open ledger reservations at full reserve "
            "(no refund); audited; pauses further live calls"
        ),
    )
    p.add_argument(
        "--split",
        choices=("development", "evaluation", "all"),
        default="evaluation",
        help="Case split (default: evaluation held-out set)",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Cap cases (0 = all in split). Required sanity bound for live.",
    )
    p.add_argument(
        "--budget-usd",
        type=float,
        default=DEFAULT_BUDGET_USD,
        help=f"Cumulative spend ceiling across runs (max initial {DEFAULT_BUDGET_USD})",
    )
    p.add_argument(
        "--model",
        default=ALLOWED_LIVE_MODEL,
        help=f"Live model ID (only {ALLOWED_LIVE_MODEL} allowed)",
    )
    p.add_argument(
        "--backend-url",
        default="http://127.0.0.1:8000",
        help="Investigation backend base URL",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path("evals/reports"),
        help="Directory for scorecard JSON + cost ledger",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Override ground-truth labels path",
    )
    p.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help=f"Cost ledger path (default {DEFAULT_LEDGER_PATH})",
    )
    p.add_argument(
        "--operator",
        default="",
        help="Operator identity for --recover-stale-reservations",
    )
    p.add_argument(
        "--reason",
        default="",
        help="Reason for --recover-stale-reservations (required with recovery)",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Confirm --recover-stale-reservations (required)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print full report/recovery JSON to stdout",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.recover_stale_reservations:
            ledger_path = args.ledger or DEFAULT_LEDGER_PATH
            if not ledger_path.is_file():
                print(f"error: no ledger at {ledger_path}", file=sys.stderr)
                return 1
            # prices unused for recovery settle-at-reserve; pass empty + explicit
            # reserve_usd so init does not require verified file for crash recovery.
            ledger = CostLedger(
                path=ledger_path,
                budget_usd=DEFAULT_BUDGET_USD,
                prices={},
                reserve_usd=0.01,  # unused when only recovering open rows
            )
            recovered = ledger.recover_stale_reservations(
                operator=args.operator,
                reason=args.reason,
                confirm=bool(args.yes),
            )
            payload = {
                "recovered": recovered,
                "n": len(recovered),
                "ledger": str(ledger_path),
                "note": (
                    "Settled full reserve per open row (no refund). "
                    "pricing_reliable=false until operator re-verifies."
                ),
            }
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"recovered {len(recovered)} stale reservation(s) at full reserve")
                for row in recovered:
                    print(
                        f"  {row['reservation_id']} case={row.get('case_id')} "
                        f"spent=${row['spent_usd']:.6f}"
                    )
                print("pricing marked unreliable; live calls paused")
            return 0

        if args.offline is not None:
            report = run_offline_report(
                args.offline,
                labels_path=args.labels,
                report_dir=args.report_dir,
                split=args.split,
            )
        else:
            if args.max_cases < 0:
                print("error: --max-cases must be >= 0", file=sys.stderr)
                return 2
            if args.budget_usd > DEFAULT_BUDGET_USD + 1e-12:
                print(
                    f"error: --budget-usd may not exceed ${DEFAULT_BUDGET_USD} "
                    "without explicit permission",
                    file=sys.stderr,
                )
                return 2
            if args.live and args.max_cases == 0:
                print(
                    "error: live mode requires explicit --max-cases > 0 "
                    "(refuse unbounded paid runs)",
                    file=sys.stderr,
                )
                return 2
            cfg = LiveEvalConfig(
                live=True,
                split=args.split,
                max_cases=args.max_cases,
                budget_usd=args.budget_usd,
                model=args.model,
                backend_base_url=args.backend_url,
                labels_path=args.labels,
                report_dir=args.report_dir,
                ledger_path=args.ledger,
            )
            report = run_live(cfg)
    except CostPause as exc:
        print(f"cost pause: {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        sc = report.scorecard
        print(f"live={report.live} n={sc.n} failures={sc.failures}")
        print(
            f"category={sc.category_correct}/{sc.n} ({sc.category_accuracy:.2%})  "
            f"evidence={sc.evidence_correct}/{sc.n} ({sc.evidence_accuracy:.2%})  "
            f"action_auto={sc.action_correct}/{sc.action_auto_scored}  "
            f"action_manual_pending={sc.action_manual_pending}  "
            f"uncertainty={sc.uncertainty_correct}/{sc.uncertainty_scored}"
        )
        for c in report.gaps[:12]:
            print(f"gap: {c}")
        if len(report.gaps) > 12:
            print(f"gap: ... {len(report.gaps) - 12} more")
        for note in sc.caveats:
            print(f"caveat: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
