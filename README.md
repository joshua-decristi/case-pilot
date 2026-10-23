# CasePilot

Local portfolio demo: an agent investigates **synthetic** stalled insurance cases, cites retrieved evidence, and recommends a next action. One simulated write (create follow-up task) runs only after explicit operator approval.

This is an **independent synthetic demonstration**. It is not a product of any insurer, does not use real customer data, and does not model any insurer’s internal systems, workflows, or requirements.

**Stage:** CLI + HTTP APIs only (case list/summary + per-case run history). **No UI. No Docker.** Repository git metadata may be absent (`git_revision: unavailable` in reports until `git init`).

Architecture sketch: [`docs/architecture.md`](docs/architecture.md). Sanitized sample live report: [`docs/sample-evaluation.json`](docs/sample-evaluation.json).

## Results snapshot (honest)

| Gate                | Result                                                                                                                                                                                                                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Offline quality     | **103** pytest cases collected (functionality-focused consolidation from **117** — three negative parameterization cases plus one independent metadata-gap/backend-regression e2e case — no paid reruns); local gates green: Ruff check + format, mypy, pytest (**no paid calls**)                         |
| Live smoke (latest) | **CASE-001** run `82e71af3-2bfa-4145-9b51-8afa58644fc4`: **status `completed`**, diagnosis **`missing_document`**, **5** grounded evidence rows, next_action **`request_missing_document`**, ledger settle **`$0.00169768`** (GLM-only, cost-guarded) |
| Live smoke history  | Attempt 1 failed (async `wrap_model_call`); attempt 2 `indeterminate` / `escalate_to_human` (evidence validation) — retained as history only, not the headline                                                                                        |
| Held-out live eval  | **N=8** (`evaluation` split): category **7/8**, evidence IDs **7/8**, action **6/8**, uncertainty **1/1**, hard fails **0/8** (unchanged)                                                                                                             |
| Ledger snapshot     | Cumulative spent **~$0.01718** USD (`evals/reports/cost_ledger.json` at report time) — **snapshot, not a promised unit cost**                                                                                                                         |

**Do not over-read these numbers:**

- Small held-out **N=8**; not production accuracy or generalization.
- Evidence scores check required **record IDs + non-empty `retrieval_ref`** only — **not** semantic entailment proof that the narrative is correct.
- Action auto-score is rule-mapped codes vs labels; misses are real (e.g. escalate vs correct_data_mismatch on CASE-009/010).
- Labels grade only; **no prompt changes based on labels** for the reported run.
- Live model output is non-deterministic; re-runs may differ.
- Full narratives/tool payloads are omitted from the portable sample; reconciliation sources for smoke are noted in `evals/reports/live-smoke.json`.

Primary artifacts: `evals/reports/live-smoke.json` (CASE-001 attempt **3** headline), `evals/reports/eval_latest.json`, `evals/reports/cost_ledger.json`, sanitized `docs/sample-evaluation.json`.

## What it does

```text
CLI (casepilot)
    | HTTP
Investigation backend  -- LangChain agent -- OpenRouter (live runs only)
    | cost ledger (backend sole mutator) + authenticated HTTP
Enterprise simulator
    | owns
SQLite enterprise records + follow-up tasks
```

- **Simulator** (`casepilot.simulator.app`, default `127.0.0.1:8001`): seeded synthetic cases (**list + get**), documents, processing events; service-token auth; owns `SIMULATOR_DB_PATH`.
- **Backend** (`casepilot.backend.app`, default `127.0.0.1:8000`): investigation orchestration, proposals/approvals, run storage, **case list/summary with action flags**, **per-case run history**, JSONL traces, **global live-cost ledger**; operator-token auth for approve/reject; owns `BACKEND_DB_PATH` and `TRACE_DIR`.

Services bind loopback only. The backend never reads the simulator DB directly (and vice versa).

### Case browsing APIs (backend + simulator)

HTTP-only operator discoverability — **no UI in this change**.

| Method | Path | Auth | Returns |
| --- | --- | --- | --- |
| `GET` | `/v1/cases` (simulator) | service token | full `CaseRecord` list |
| `GET` | `/v1/cases` (backend) | none (loopback) | `CaseListItem[]` — `case_id`, `status`, `symptom`, plus independent flags `pending_review` / `needs_execution` from proposal state |
| `GET` | `/v1/cases/{case_id}` (backend) | none (loopback) | `CaseSummary` — brief `case_id` / `status` / `symptom` (not the full simulator record) |
| `GET` | `/v1/cases/{case_id}/runs` (backend) | none (loopback) | `RunSummary[]` — persisted investigation history for the case, newest first (`run_id`, `status`, timestamps). 404 if case missing in simulator |

Backend list aggregates action flags from persisted proposals in one pass (`pending` → pending review; `approved` or `execution_failed` → needs execution). Run history reads backend SQLite only after a simulator case existence check. Existing investigation / proposal / approve-reject routes are unchanged. Shared `BackendRoutes.CASE_PROPOSALS` / `CASE_RUNS` document case-scoped paths.


## Requirements

- **Python 3.14** (pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** for install and runs
- OpenRouter API key **only** for live investigations / live eval (not required for lint, typecheck, offline eval, or default pytest)

## Setup

```bash
cp .env.example .env
make provision-local-auth   # appends random SIMULATOR_SERVICE_TOKEN + OPERATOR_TOKEN if missing
# Edit .env: set OPENROUTER_API_KEY only when you intend live model calls
make sync                   # uv sync --frozen --all-extras
make seed                   # non-destructive fixture load (refuses non-empty simulator DB)
make dev                    # start simulator + backend; wait on /health
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
| `make reset-demo` | **Destructive** reseed: `python scripts/seed.py --reset` drops/recreates tables, then loads fixtures                            |
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
| `make dev`                                       | Two-process launcher + `/health` readiness                                            |
| `make seed`                                      | Non-destructive seed                                                                  |
| `make reset-demo`                                | Destructive reseed (`--reset`)                                                        |
| `make test`                                      | `uv run pytest` (deterministic; no paid model calls)                                  |
| `make lint`                                      | `uv run ruff check src tests scripts` and `ruff format --check` on the same           |
| `make typecheck`                                 | `uv run mypy`                                                                         |
| `make eval-offline RESULTS=path/to/results.json` | Offline grade only (no paid calls); optional `EVAL_ARGS=`                             |
| `make eval MAX_CASES=N`                          | Live paid opt-in; `BUDGET_USD` defaults to `5` and must be ≤ 5; optional `EVAL_ARGS=` |

## Cost ledger (live only)

Application policy — **not** a provider hard cap:

- **Envelope:** **$5 USD total**, **GLM-only** (`z-ai/glm-5.3-flash`)
- **Backend is the sole ledger mutator** (`casepilot.backend.cost_guard`); investigations reserve/settle/release around billable work
- **Ledger file:** `evals/reports/cost_ledger.json` (override with `CASEPILOT_COST_LEDGER_PATH`)
- Non-allowlisted model, exhausted budget, or `pricing_reliable: false` → live calls pause (`CostPause`)
- Scripted / `CASEPILOT_TEST_MODE` paths do **not** touch the ledger
- Missing usage on settle fails closed (full reserve charged + pricing marked unreliable)

Conservative per-run reserves use local price tables × token ceilings × `max_model_turns` — bounds for guarding, not a quote.

Reported cumulative spend **~$0.01718** is one ledger snapshot after smoke + eval activity — not a guaranteed cost per case.

## Evaluation

Entrypoint: `uv run python -m casepilot.evaluation` (mutually exclusive `--live` / `--offline`).

```bash
# Offline grade only — no OpenRouter
make eval-offline RESULTS=path/to/results.json
# or: uv run python -m casepilot.evaluation --offline path/to/results.json

# Live — explicit max cases; budget ≤ $5; default split=evaluation
make eval MAX_CASES=2
# or: uv run python -m casepilot.evaluation --live --max-cases 2 --budget-usd 5 --split evaluation
```

Useful flags: `--split {development,evaluation,all}`, `--max-cases` (live requires `> 0`), `--budget-usd` (capped at 5), `--model` (only `z-ai/glm-5.3-flash`), `--backend-url`, `--report-dir` (default `evals/reports`), `--labels`, `--json`.

Grading is rule-based against `evals/ground_truth/labels.json` (not an LLM judge). Labels are for scoring — **do not tune prompts against held-out labels** and then claim unseen performance.

## Grounding and approval controls

- **Untrusted evidence:** retrieved records are data, not instructions.
- **Case-bound tools only:** case / documents / events reads; no generic HTTP, SQL, or filesystem tool.
- **Evidence validation:** cited `record_id`s must match records retrieved in that run.
- **Execution budgets** (settings defaults): investigation timeout 60s, tool HTTP timeout 10s, max 12 tool calls, max 8 model turns.
- **No auto-writes:** investigation may store an **immutable proposal**; the model cannot set approval state.
- **Operator gate:** approve/reject require `OPERATOR_TOKEN`. Backend sends the stored payload to the simulator with the service token only after approval.
- **Idempotent task create:** simulator dedupes follow-up tasks by proposal id.
- **Tests:** scripted model hooks (`CASEPILOT_TEST_MODE`) — no paid calls.

Agent stack: **LangChain** (`create_agent`, tools, structured output, OpenRouter chat model). No custom graph authoring in this repo.

## Data layout and independence limits

| Path                             | Visibility                         | Purpose                                                                     |
| -------------------------------- | ---------------------------------- | --------------------------------------------------------------------------- |
| `fixtures/enterprise/cases.json` | Model-visible via tools after seed | Synthetic enterprise source records                                         |
| `evals/ground_truth/labels.json` | Evaluator-only                     | Diagnoses, evidence predicates, actions, **development / evaluation** split |
| `evals/reports/`                 | Generated (gitignored except keep) | Live reports + `cost_ledger.json`                                           |
| `docs/sample-evaluation.json`    | Curated sanitized sample           | Compact honest grades/metadata for portfolio readers                        |
| `SIMULATOR_DB_PATH`              | Simulator                          | Seeded records + executed tasks                                             |
| `BACKEND_DB_PATH`                | Backend                            | Runs, proposals, approval/audit state                                       |
| `TRACE_DIR`                      | Backend                            | JSONL traces (no secrets/tokens)                                            |

**Limitations:**

- Authored synthetic data only — not production insurance performance.
- Labels stay under `evals/ground_truth/`; fixtures must not embed answer keys. Only tool-returned records enter model context.
- Do not tune on **evaluation** split IDs and claim held-out generalization.
- Live model output is non-deterministic; small samples vary. Do not invent accuracy scores beyond the reported rule-based counts.
- Demo tokens + loopback binding ≠ enterprise identity, TLS, or multi-user RBAC.
- One simulated write path (`create_follow_up_task`). No payments, coverage changes, or external messaging.
- The $5 GLM ledger is app policy; use a provider account limit if you need a hard stop.

## Quality

No hosted CI in this repo; quality gates run locally with `uv` + Python **3.14**:

```bash
make sync       # uv sync --frozen --all-extras (lock-consistent env)
make lint       # ruff check + ruff format --check on src tests scripts
make typecheck  # mypy (strict)
make test       # pytest (deterministic; no paid model calls)
```

These offline gates never touch OpenRouter. Live `make eval` is a separate explicit opt-in.

## Stack (actual)

- Python ≥ 3.14, `uv` + committed `uv.lock`, hatchling package `casepilot` (`src/casepilot`)
- FastAPI + Uvicorn + Pydantic Settings + HTTPX
- LangChain + `langchain-openrouter`
- stdlib `sqlite3`
- CLI: `casepilot` → `casepilot.cli:main`
- Eval: `python -m casepilot.evaluation`
- Dev extras: pytest, pytest-asyncio, ruff, mypy

## Out of scope (deferred)

UI, Docker, LangSmith, cloud deploy, multi-agent design, real insurer integrations, autonomous underwriting/payment decisions.
