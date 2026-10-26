"""CLI entrypoint for evaluation + operator cost recovery.

  uv run python -m casepilot.evaluation --offline path/to/results.json
  uv run python -m casepilot.evaluation --live --max-cases 2 --budget-usd 5
  uv run python -m casepilot.evaluation --scenario-pilot --pilot-live
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
from casepilot.evaluation.scenario_pilot import PilotConfig, run_scenario_pilot_sync
from casepilot.shared.config import get_settings


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
        "--scenario-pilot",
        action="store_true",
        help=(
            "P0 scenario-pilot cohort (6×3 slots); requires --pilot-live. "
            "After instrumentation fixes use a new --cohort-id (fresh 18 slots); "
            "never selectively replace a prior baseline's slots"
        ),
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
    p.add_argument(
        "--pilot-live",
        action="store_true",
        help="Explicit paid opt-in for --scenario-pilot (shared ledger/judge/backend)",
    )
    p.add_argument(
        "--cohort-id",
        default=None,
        help=(
            "Cohort id for scenario pilot reports. Omit to mint a fresh id. "
            "For a new baseline after fixes, pass a new id (do not reuse a "
            "failed/complete prior cohort to replace its 18 slots)"
        ),
    )
    p.add_argument(
        "--pilot-assets",
        type=Path,
        default=None,
        help="Override evals/scenario_pilot assets directory",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Resume same cohort from report (unstarted slots only; frozen versions). "
            "Not a baseline replacement path — completed/failed slots stay immutable"
        ),
    )
    return p


def _run_scenario_pilot_cli(args: argparse.Namespace) -> int:
    if args.budget_usd > DEFAULT_BUDGET_USD + 1e-12:
        print(
            f"error: --budget-usd may not exceed ${DEFAULT_BUDGET_USD} without explicit permission",
            file=sys.stderr,
        )
        return 2
    if not args.pilot_live:
        print(
            "error: --scenario-pilot requires explicit --pilot-live (refuse silent paid runs)",
            file=sys.stderr,
        )
        return 2
    sett = get_settings()
    pilot_cfg = PilotConfig(
        live=True,
        backend_base_url=args.backend_url,
        report_dir=args.report_dir,
        ledger_path=args.ledger,
        assets_path=args.pilot_assets,
        cohort_id=args.cohort_id,
        model=args.model,
        budget_usd=args.budget_usd,
        resume_path=args.resume,
        backend_db_path=sett.backend_db_path,
        simulator_db_path=sett.simulator_db_path,
        trace_dir=sett.trace_dir,
    )
    if args.resume is None:
        print(
            "scenario_pilot: fresh cohort run "
            f"(cohort_id={pilot_cfg.cohort_id or '(minted)'}); "
            "prior baselines are not replaced — completed/failed slots stay history",
            file=sys.stderr,
        )
    else:
        print(
            "scenario_pilot: resume unstarted slots only "
            f"(report={args.resume}); no selective replacement of prior slots",
            file=sys.stderr,
        )
    pilot_report = run_scenario_pilot_sync(pilot_cfg)
    if args.json:
        print(json.dumps(pilot_report.to_dict(), indent=2, default=str))
    else:
        print(
            f"scenario_pilot cohort={pilot_report.cohort_id} "
            f"status={pilot_report.status} live={pilot_report.live} "
            f"p0={pilot_report.completion.get('p0_measurement_complete')}"
        )
        print(
            f"slots_complete={pilot_report.completion.get('slots_complete')}/"
            f"{pilot_report.completion.get('slots_total')} "
            f"calibration_ok={pilot_report.calibration.get('ok')} "
            f"total_usd={pilot_report.costs.total_usd}"
        )
        for b in pilot_report.blockers[:12]:
            print(f"blocker: {b}")
        for g in pilot_report.gaps[:12]:
            print(f"gap: {g}")
        if len(pilot_report.gaps) > 12:
            print(f"gap: ... {len(pilot_report.gaps) - 12} more")
    return 0 if pilot_report.status == "complete" else 1


def _print_eval_report(eval_report: object, *, as_json: bool) -> None:
    # Local import typing avoidance: EvalReport has scorecard; keep separate from pilot.
    from casepilot.evaluation.runner import EvalReport

    assert isinstance(eval_report, EvalReport)
    if as_json:
        print(json.dumps(eval_report.to_dict(), indent=2))
        return
    sc = eval_report.scorecard
    print(f"live={eval_report.live} n={sc.n} failures={sc.failures}")
    print(
        f"category={sc.category_correct}/{sc.n} ({sc.category_accuracy:.2%})  "
        f"evidence={sc.evidence_correct}/{sc.n} ({sc.evidence_accuracy:.2%})  "
        f"action_auto={sc.action_correct}/{sc.action_auto_scored}  "
        f"action_manual_pending={sc.action_manual_pending}  "
        f"uncertainty={sc.uncertainty_correct}/{sc.uncertainty_scored}"
    )
    for c in eval_report.gaps[:12]:
        print(f"gap: {c}")
    if len(eval_report.gaps) > 12:
        print(f"gap: ... {len(eval_report.gaps) - 12} more")
    for note in sc.caveats:
        print(f"caveat: {note}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.recover_stale_reservations:
            ledger_path = args.ledger or DEFAULT_LEDGER_PATH
            if not ledger_path.is_file():
                print(f"error: no ledger at {ledger_path}", file=sys.stderr)
                return 1
            ledger = CostLedger(
                path=ledger_path,
                budget_usd=DEFAULT_BUDGET_USD,
                prices={},
                reserve_usd=0.01,
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

        if args.scenario_pilot:
            return _run_scenario_pilot_cli(args)

        if args.offline is not None:
            eval_report = run_offline_report(
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
            eval_report = run_live(cfg)
    except CostPause as exc:
        print(f"cost pause: {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_eval_report(eval_report, as_json=bool(args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
