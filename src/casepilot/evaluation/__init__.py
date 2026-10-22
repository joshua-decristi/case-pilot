"""Deterministic evaluation harness (rule-based grader + opt-in live runner).

Backend owns live cost accounting via ``casepilot.evaluation.cost.CostLedger``.
Unit prices come only from verified config — no guessed ceilings in code.
"""

from casepilot.evaluation.cost import (
    ALLOWED_LIVE_MODEL,
    CONSERVATIVE_RESERVE_USD,
    DEFAULT_BUDGET_USD,
    DEFAULT_LEDGER_PATH,
    DEFAULT_VERIFIED_PRICES_PATH,
    LEDGER_DEFAULT_PATH,
    VERIFIED_PRICES_ENV,
    CostLedger,
    CostPause,
    ModelPrice,
    Reservation,
    RunTokenBudget,
    compute_max_reserve_usd,
    cost_from_usage,
    default_run_token_budget,
    get_conservative_reserve_usd,
    get_model_price,
    load_verified_prices,
    validate_live_model,
    verified_prices_path,
)
from casepilot.evaluation.grader import (
    FIXTURE_ACTION_TO_NEXT,
    CaseGrade,
    CaseLabel,
    GroundTruth,
    Scorecard,
    grade_case,
    grade_results,
    load_ground_truth,
)
from casepilot.evaluation.metadata import RunMetadata, collect_metadata
from casepilot.evaluation.runner import EvalCaseResult, LiveEvalConfig, run_live, run_offline_report

__all__ = [
    "ALLOWED_LIVE_MODEL",
    "CONSERVATIVE_RESERVE_USD",
    "DEFAULT_BUDGET_USD",
    "DEFAULT_LEDGER_PATH",
    "DEFAULT_VERIFIED_PRICES_PATH",
    "FIXTURE_ACTION_TO_NEXT",
    "LEDGER_DEFAULT_PATH",
    "VERIFIED_PRICES_ENV",
    "CaseGrade",
    "CaseLabel",
    "CostLedger",
    "CostPause",
    "EvalCaseResult",
    "GroundTruth",
    "LiveEvalConfig",
    "ModelPrice",
    "Reservation",
    "RunMetadata",
    "RunTokenBudget",
    "Scorecard",
    "collect_metadata",
    "compute_max_reserve_usd",
    "cost_from_usage",
    "default_run_token_budget",
    "get_conservative_reserve_usd",
    "get_model_price",
    "grade_case",
    "grade_results",
    "load_ground_truth",
    "load_verified_prices",
    "run_live",
    "run_offline_report",
    "validate_live_model",
    "verified_prices_path",
]
