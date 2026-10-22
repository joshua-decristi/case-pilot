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

__all__ = (
    "ALLOWED_LIVE_MODEL",
    "CONSERVATIVE_RESERVE_USD",
    "CostLedger",
    "CostPause",
    "DEFAULT_BUDGET_USD",
    "DEFAULT_LEDGER_PATH",
    "DEFAULT_VERIFIED_PRICES_PATH",
    "LEDGER_DEFAULT_PATH",
    "ModelPrice",
    "Reservation",
    "RunTokenBudget",
    "VERIFIED_PRICES_ENV",
    "compute_max_reserve_usd",
    "cost_from_usage",
    "default_run_token_budget",
    "get_conservative_reserve_usd",
    "get_model_price",
    "load_verified_prices",
    "validate_live_model",
    "verified_prices_path",
)
