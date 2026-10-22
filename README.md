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

## Investigation backend (direct uvicorn)

Investigation orchestration, run storage (`BACKEND_DB_PATH`, default `.data/backend.sqlite3`), JSONL traces (`TRACE_DIR`, default `traces/`), and the single approved-write path to the simulator. Auth: approve/reject require `Authorization: Bearer <OPERATOR_TOKEN>`; all other routes are local-unauthenticated. The backend calls the simulator over HTTP only — it never reads `SIMULATOR_DB_PATH`.

```bash
# Tokens required in env (same .env as the simulator; backend reuses both)
# Live investigations additionally need OPENROUTER_API_KEY; without it the
# run fails visibly with a clear missing-key error — no silent fallback.

# With the simulator already serving on :8001, serve on loopback :8000
uv run uvicorn casepilot.backend.app:app --host 127.0.0.1 --port 8000
```

### API surface

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/health` | none | liveness |
| POST | `/v1/investigations` | none | body `{"case_id": "...", "objective": "..."}`; synchronous agent run |
| GET | `/v1/investigations/{run_id}` | none | stored run record |
| GET | `/v1/proposals/{proposal_id}` | none | proposal with `payload_digest` |
| GET | `/v1/cases/{case_id}/proposals` | none | list proposals for a case |
| POST | `/v1/proposals/{proposal_id}/approve` | Bearer operator token | body `{"operator_identity": "..."}`; executes the proposed write |
| POST | `/v1/proposals/{proposal_id}/reject` | Bearer operator token | body `{"operator_identity": "...", "reason": "..."}`; terminal |

Evidence cited by the run is validated against records actually retrieved this run; a proposal is only created when the agent proposes `create_follow_up_task` and every evidence row is grounded.

## Operator CLI

`casepilot` is a thin HTTPX client over the backend HTTP API (installed via `[project.scripts]`; also `uv run casepilot ...`).

```bash
# After backend is serving on :8000 (settings from env / .env)
uv run casepilot investigate CASE-001              # start a run (default objective)
uv run casepilot run <run_id>                      # fetch a stored run
uv run casepilot proposal <proposal_id>            # fetch a proposal
uv run casepilot approve <proposal_id>             # operator-token write to simulator
uv run casepilot reject <proposal_id> --reason R   # terminal decision

# Machine-readable output on any subcommand
uv run casepilot --json investigate CASE-001
```

Human output surfaces `proposal_id` (and the approve command) first. Errors print to stderr with exit code 1 (API/backend failure) or 2 (configuration error); success exits 0.

### End-to-end without the CLI

```bash
# 1. Investigate a seeded case (returns run result + proposal_id when a write is proposed)
curl -s -X POST http://127.0.0.1:8000/v1/investigations \
  -H 'Content-Type: application/json' \
  -d '{"case_id": "CASE-001", "objective": "Why is this case stalled?"}'

# 2. Read the proposal (idempotency-ready payload digest is stored server-side)
curl -s http://127.0.0.1:8000/v1/proposals/<proposal_id>

# 3. Approve (the ONLY path that writes to the simulator) or reject (terminal)
curl -s -X POST http://127.0.0.1:8000/v1/proposals/<proposal_id>/approve \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -H 'Content-Type: application/json' -d '{}'
```

## Cost ledger (live only)

- **Envelope:** $5 USD total, GLM-only (`z-ai/glm-5.3-flash`) — enforced via `casepilot.evaluation.cost.CostLedger`; the backend is the sole ledger mutator.
- **Unit prices** come only from verified config: `evals/pricing/verified_prices.json` (override via `CASEPILOT_VERIFIED_PRICES_PATH`); see `verified_prices.example.json` for the schema. No guessed ceilings in code.
- Default ledger file: `evals/reports/cost_ledger.json` (gitignored). Missing usage on settle fails closed (full reserve charged + pricing marked unreliable); operator-only stale-reservation recovery settles at full reserve (`--recover-stale-reservations`).

## Ground truth labels

`evals/ground_truth/labels.json` holds evaluator-only labels: expected categories, acceptable fixture action codes, evidence predicates, uncertainty requirements, and a development/evaluation split. Grading (`casepilot.evaluation.grader`) is rule-based — category match, record-ID + non-empty `retrieval_ref` evidence coverage, uncertainty presence, mapped action codes, termination — never an LLM judge; `create_follow_up_task` is not auto-corrected against labels. Labels are for scoring only: do not tune prompts against held-out labels and then claim unseen performance.

## Evaluation

Entrypoint: `uv run python -m casepilot.evaluation` (mutually exclusive `--live` / `--offline`).

```bash
# Offline grade only — no OpenRouter
uv run python -m casepilot.evaluation --offline path/to/results.json
# Live — explicit max cases; budget ≤ $5; GLM-only; default split=evaluation
uv run python -m casepilot.evaluation --live --max-cases 2 --budget-usd 5 --split evaluation
```

Useful flags: `--split {development,evaluation,all}`, `--max-cases` (live requires `> 0`), `--budget-usd` (capped at 5), `--model` (only `z-ai/glm-5.3-flash`), `--backend-url`, `--report-dir` (default `evals/reports`), `--labels`, `--json`. The live runner never sends ground-truth labels and never mutates the ledger — it observes remaining budget and enforces per-session thresholds. Reports land in `evals/reports/` (gitignored).
