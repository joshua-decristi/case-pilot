# Run CasePilot locally

[Overview](../README.md) · [Architecture](architecture.md) · [Evaluation](evaluation.md) · [Operator UI](operator-ui.md)

Run commands from the repository root. Live investigations are paid; offline checks need no OpenRouter key.

## Requirements

- **Python 3.14** (pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** for install and runs
- **Node.js + npm** only if you run the operator UI (`frontend/`)
- OpenRouter API key **only** for live investigations / live eval (not required for lint, typecheck, offline eval, default pytest, or UI chrome without Investigate)

## Setup

```bash
cp .env.example .env
make provision-local-auth   # appends random SIMULATOR_SERVICE_TOKEN + OPERATOR_TOKEN if missing
# Edit .env: set OPENROUTER_API_KEY only when you intend live model calls
make sync                   # uv sync --frozen --all-extras
make seed                   # non-destructive fixture load (refuses non-empty simulator DB)
make dev                    # start simulator + backend; wait on /health

# Operator UI (separate process; needs make dev already up)
npm install --prefix frontend
npm run dev --prefix frontend   # http://127.0.0.1:5173
# Isolated manual env (CSRF allowed origins follow the port):
# FRONTEND_PORT=5174 npm run dev --prefix frontend
# DEV_PORT is also supported; prefer FRONTEND_PORT for explicit UI configuration.
```

### Local auth

`make provision-local-auth` runs `scripts/provision_local_auth.py --write`:

- Fills **missing** `SIMULATOR_SERVICE_TOKEN` and `OPERATOR_TOKEN` with `secrets.token_urlsafe(32)` values
- **Never** prints token values, **never** overwrites existing keys, **never** touches OpenRouter keys
- Without `--write`, reports missing keys and exits 2 (no silent bypass)

### Seed / reset safety

| Command           | Behavior                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `make seed`       | Non-destructive. Loads `fixtures/enterprise/cases.json` only if the simulator DB is empty; otherwise refuses overwrite (exit 1) |
| `make reset-demo` | **Destructive** reseed: `uv run python scripts/seed.py --reset` drops/recreates tables, then loads fixtures                     |
| App startup       | Never auto-seeds                                                                                                                |

Optional seed flags (via `uv run python scripts/seed.py`): `--db-path`, `--fixtures`, `--reset`.

### `.env.example`

| Variable                                  | Role                                                                       |
| ----------------------------------------- | -------------------------------------------------------------------------- |
| `OPENROUTER_API_KEY`                      | Live model calls only                                                      |
| `OPENROUTER_MODEL`                        | Default `z-ai/glm-5.3-flash` (live allowlist is GLM-only; see cost ledger) |
| `SIMULATOR_SERVICE_TOKEN`                 | Backend→simulator auth; prefer `make provision-local-auth`                 |
| `OPERATOR_TOKEN`                          | Approve/reject auth; same                                                  |
| `BACKEND_HOST` / `BACKEND_PORT`           | Backend bind (default `127.0.0.1:8000`)                                    |
| `SIMULATOR_HOST` / `SIMULATOR_PORT`       | Simulator bind (default `127.0.0.1:8001`)                                  |
| `BACKEND_BASE_URL` / `SIMULATOR_BASE_URL` | Client base URLs                                                           |
| `SIMULATOR_DB_PATH`                       | Default `.data/simulator.sqlite3`                                          |
| `BACKEND_DB_PATH`                         | Runs, proposals, approval state — default `.data/backend.sqlite3`          |
| `TRACE_DIR`                               | JSONL traces — default `traces`                                            |

Optional: `CASEPILOT_COST_LEDGER_PATH` overrides the live cost ledger path (default `evals/reports/cost_ledger.json`).

Do not commit `.env`, databases, traces, or generated eval reports (sanitized sample under `docs/` is intentional).

## Usage

With both services running (`make dev`):

```bash
# Live investigation (needs OPENROUTER_API_KEY; charged against global ledger)
uv run casepilot investigate CASE-001
uv run casepilot investigate CASE-001 --objective "Why is this case stalled?"
uv run casepilot investigate CASE-001 --json

# Inspect stored artifacts
uv run casepilot run RUN_ID
uv run casepilot proposal PROPOSAL_ID

# Operator decisions (OPERATOR_TOKEN from env; values never printed)
uv run casepilot approve PROPOSAL_ID
uv run casepilot reject PROPOSAL_ID --reason "Not warranted"
```

When a proposal exists, human-readable CLI output surfaces it first:

```text
proposal_id: <id>
Approve with: casepilot approve <id>
```

Fixtures: 20 cases (`CASE-001` … `CASE-020`), 57 documents, 46 events in `fixtures/enterprise/cases.json`.

### Makefile targets

| Target                                           | What it runs                                                                          |
| ------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `make sync`                                      | `uv sync --frozen --all-extras`                                                       |
| `make provision-local-auth`                      | `uv run python scripts/provision_local_auth.py --write`                               |
| `make dev`                                       | Two-process launcher + `/health` readiness (backend + simulator; UI is separate npm)  |
| `make seed`                                      | Non-destructive seed                                                                  |
| `make reset-demo`                                | Destructive reseed (`--reset`)                                                        |
| `make test`                                      | `uv run pytest` (deterministic; no paid model calls)                                  |
| `make lint`                                      | `uv run ruff check src tests scripts` and `ruff format --check` on the same           |
| `make typecheck`                                 | `uv run mypy`                                                                         |
| `make eval-offline RESULTS=path/to/results.json` | Offline grade only (no paid calls); optional `EVAL_ARGS=`                             |
| `make eval MAX_CASES=N`                          | Live paid opt-in; `BUDGET_USD` defaults to `5` and must be ≤ 5; optional `EVAL_ARGS=` |

## Quality

No hosted CI in this repo; quality gates run locally with `uv` + Python **3.14**:

```bash
make sync       # uv sync --frozen --all-extras (lock-consistent env)
make lint       # ruff check + ruff format --check on src tests scripts
make typecheck  # mypy (strict)
make test       # pytest (deterministic; no paid model calls)
npm run build --prefix frontend   # operator UI typecheck/bundle (when changing frontend)
```

These offline gates never touch OpenRouter. Live `make eval` / UI Investigate are separate explicit opt-ins.

## Troubleshooting

- A non-empty simulator database makes seed refuse to run. Reset only if you intend to destroy its records.
- For UI connection errors, wait for `make dev` health checks and verify `BACKEND_BASE_URL`.
- After changing tokens or configuration, restart the services and Vite. Never paste secrets into frontend code.
- For paused live calls, inspect ledger headroom and pricing status. See [cost controls](architecture.md#cost-controls); do not bypass the guard.
