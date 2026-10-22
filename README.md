# CasePilot

Local portfolio demo package (Python 3.14+). CLI/API only — no UI, no Docker.

## Development bootstrap

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Tool commands

```bash
# Install / refresh lock-consistent env
uv sync --extra dev

# Lint
uv run ruff check src
uv run mypy

# Tests
uv run pytest
```

Package import path: `src/casepilot` (set via `pythonpath` in pyproject / `PYTHONPATH=src`).

## Shared contracts & settings

- `casepilot.shared.contracts` — shape-only Pydantic models/enums for simulator + backend HTTP payloads and route templates.
- `casepilot.shared.config.Settings` / `get_settings()` — env-backed settings (`pydantic-settings`); service/operator tokens required, OpenRouter key optional until live runs. See `.env.example` (placeholders only).
- Local auth: `python scripts/provision_local_auth.py` reports missing tokens; `--write` appends random tokens to `.env` without printing or overwriting existing keys.

## Enterprise simulator (standalone)

Seed SQLite from `fixtures/enterprise/cases.json` (non-destructive by default; `--reset` for destructive reseed). App startup never auto-seeds.

```bash
# Tokens required in env (use provision_local_auth --write, or export temps for a one-off)
export SIMULATOR_SERVICE_TOKEN=dev-sim-token
export OPERATOR_TOKEN=dev-op-token

# Seed (module or scripts/ entrypoint)
uv run python -m casepilot.simulator.seed --reset
# equivalent: uv run python scripts/seed.py --reset

# Serve on loopback :8001
uv run uvicorn casepilot.simulator.app:app --host 127.0.0.1 --port 8001
```

### API surface

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/health` | none | liveness |
| GET | `/v1/cases/{case_id}` | Bearer service token | case record |
| GET | `/v1/cases/{case_id}/documents` | Bearer service token | case-scoped docs |
| GET | `/v1/cases/{case_id}/events` | Bearer service token | case-scoped events |
| POST | `/v1/cases/{case_id}/follow-up-tasks` | Bearer service token | create follow-up task |

Follow-up task writes are **idempotent on `proposal_id`**: identical retry returns the original (`created=false`); same `proposal_id` with a different case or payload → **409 Conflict**.

Auth header: `Authorization: Bearer <SIMULATOR_SERVICE_TOKEN>`.

## Cost ledger (live only)

- **Envelope:** $5 USD total, GLM-only (`z-ai/glm-5.3-flash`) — enforced via `casepilot.evaluation.cost.CostLedger`; the backend is the sole ledger mutator.
- **Unit prices** come only from verified config: `evals/pricing/verified_prices.json` (override via `CASEPILOT_VERIFIED_PRICES_PATH`); see `verified_prices.example.json` for the schema. No guessed ceilings in code.
- Default ledger file: `evals/reports/cost_ledger.json` (gitignored). Missing usage on settle fails closed (full reserve charged + pricing marked unreliable); operator-only stale-reservation recovery settles at full reserve (`--recover-stale-reservations`).

## Ground truth labels

`evals/ground_truth/labels.json` holds evaluator-only labels: expected categories, acceptable fixture action codes, evidence predicates, uncertainty requirements, and a development/evaluation split. Grading (`casepilot.evaluation.grader`) is rule-based — category match, record-ID + non-empty `retrieval_ref` evidence coverage, uncertainty presence, mapped action codes, termination — never an LLM judge; `create_follow_up_task` is not auto-corrected against labels. Labels are for scoring only: do not tune prompts against held-out labels and then claim unseen performance.
