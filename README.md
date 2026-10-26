# CasePilot

Local portfolio demo: an agent investigates **synthetic** stalled insurance cases, cites retrieved evidence, and recommends a next action. One simulated write (create follow-up task) runs only after explicit operator approval.

This is an **independent synthetic demonstration**. It is not a product of any insurer, does not use real customer data, and does not model any insurer’s internal systems, workflows, or requirements.

**Stage:** CLI + HTTP APIs, plus a **local-only operator UI** (React/Vite) over the same workflow. **No Docker. No production hosting.** Repository git metadata may be absent (`git_revision: unavailable` in reports until `git init`).

Architecture sketch: [`docs/architecture.md`](docs/architecture.md). Sanitized sample live report: [`docs/sample-evaluation.json`](docs/sample-evaluation.json). Operator UI scope: [`.agent/operator-ui/spec.md`](.agent/operator-ui/spec.md) (local agent notes; may be absent from some checkouts).

## Results snapshot (honest)

| Gate                | Result                                                                                                                                                                                                                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Offline quality     | **162** pytest cases collected (includes offline fail-closed, outage harness, task_id approval, grounded chat — no paid reruns); local gates green: Ruff check + format, mypy, pytest (**no paid calls**); operator-UI **frontend build** + reviews also passed                         |
| Live smoke (latest) | **CASE-001** run `82e71af3-2bfa-4145-9b51-8afa58644fc4`: **status `completed`**, diagnosis **`missing_document`**, **5** grounded evidence rows, next_action **`request_missing_document`**, ledger settle **`$0.00169768`** (GLM-only, cost-guarded) |
| Live smoke history  | Attempt 1 failed (async `wrap_model_call`); attempt 2 `indeterminate` / `escalate_to_human` (evidence validation) — retained as history only, not the headline                                                                                        |
| Held-out live eval  | **N=8** (`evaluation` split): category **7/8**, evidence IDs **7/8**, action **6/8**, uncertainty **1/1**, hard fails **0/8** (unchanged)                                                                                                             |
| Operator UI manual  | Checklist **verified** except execution recovery/retry (**unavailable** — no `approved`-unfinished / `execution_failed` state). Incl. **5** authorized live Investigates + offline loading/duplicate-guard/network-error checks (**no further paid calls**). Spec **not** fully validated while recovery remains unobserved. |
| Ledger snapshot     | Cumulative **`$0.02582926`** USD (`evals/reports/cost_ledger.json`); prior baseline **`$0.01718485`**; operator-UI manual work **`$0.00864441`** (**5** live investigates) — **snapshot, not a promised unit cost**                                                                                                                         |

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
CLI (casepilot)  or  Operator UI (Vite :5173 → /api proxy)
    | HTTP (loopback)
Investigation backend  -- LangChain agent -- OpenRouter (live runs only)
    | cost ledger (backend sole mutator) + authenticated HTTP
Enterprise simulator
    | owns
SQLite enterprise records + follow-up tasks
```

- **Simulator** (`casepilot.simulator.app`, default `127.0.0.1:8001`): seeded synthetic cases, documents, processing events; service-token auth; owns `SIMULATOR_DB_PATH`.
- **Backend** (`casepilot.backend.app`, default `127.0.0.1:8000`): investigation orchestration, proposals/approvals, run storage, JSONL traces, **global live-cost ledger**; operator-token auth for approve/reject; owns `BACKEND_DB_PATH` and `TRACE_DIR`.
- **Operator UI** (`frontend/`, Vite default `http://127.0.0.1:5173`): single-operator local shell for cases, findings, evidence, proposals, approve/reject/retry, and grounded run chat. Same-origin `/api` via the Vite dev proxy — **not** a production host. Override listen port with `FRONTEND_PORT` or `DEV_PORT` (CSRF allowlist tracks the chosen port; `strictPort` is on).

Services bind loopback only. The backend never reads the simulator DB directly (and vice versa). The browser never talks to the simulator and never receives `SIMULATOR_SERVICE_TOKEN`.

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

### Operator UI (local)

Local-first visual shell over the existing investigation + human-approval workflow. **Not** production-hosted; packaging/deploy are out of scope.

**Startup** (backend + simulator already running via `make dev`):

```bash
npm install --prefix frontend
npm run dev --prefix frontend
```

Open **http://127.0.0.1:5173** (or `http://127.0.0.1:$FRONTEND_PORT` when set). The Vite dev server proxies same-origin `/api` to the backend (`BACKEND_BASE_URL`, default `127.0.0.1:8000`).

**Credentials (local wiring only):**

| Secret | Browser? | How |
| --- | --- | --- |
| `OPERATOR_TOKEN` (root `.env`) | **No** | Vite proxy attaches it **server-side** to approve/reject requests using **non-`VITE_`** config. Never embed in frontend assets or commit it. |
| `SIMULATOR_SERVICE_TOKEN` | **No** | Backend→simulator only. UI must not call the simulator or receive this credential. |
| `OPENROUTER_API_KEY` | **No** | Backend live runs only. |

This preserves the existing operator-token gate. It is **not** a login system, multi-user auth, or a claim of production browser security.

**Paid model calls:** Investigate hits the real backend investigation path. Do **not** click Investigate / run live evals without **explicit** permission and a configured key + ledger headroom. UI chrome (list/detail/history of already-persisted data) does not need OpenRouter.

#### Screens and workflow (exact scope)

1. **Case list** — seeded cases + status; flags **pending review** vs **approved-but-unfinished / execution failed** separately. No charts, metrics, or global recovery dashboard.
2. **Case detail** — brief summary; **Investigate** with the **default objective only** (no editable objective); latest persisted run; small **history** of earlier runs; unfinished proposals remain visible across runs.
3. **Findings (one page)** — diagnosis/explanation, cited evidence (facts + source/record/field refs), uncertainty, recommended next action, and exact proposed task when present. A result **without** a proposal stays useful (findings only; no approval controls). Evidence lives with findings — **no** document/event/task record browser.
4. **Proposal review** — show the **exact persisted** task payload (title, body, case) + status.
   - **Pending:** **Approve & execute** (immediate, no confirm dialog) or **Reject** (optional inline reason). Disable conflicting controls while a request runs.
   - **Approved unfinished / execution_failed:** show error details; **Retry execution** via the existing approval endpoint + stored payload. **No reject after approval.**
   - **Executed / rejected:** terminal outcome only; show rejection reason or execution error only as applicable — do not mislabel rejection as execution failure.
5. **Loading / errors / history** — synchronous REST; show **Investigating…** until complete (no streaming, tool feed, or fake progress). Block duplicate Investigate while in flight **in the mounted UI only** (not cross-tab/job infra). Inline request failures; keep earlier results if a new run fails. History reopens **persisted** runs/proposals; page reload refetches persisted state. Distinguish network/request errors from confirmed execution failure; refetch before claiming failure. **No** automatic execution or automatic retries.

#### Manual verification checklist

Manual only — **no new automated UI tests** in this workstream. Against a running backend + simulator. Status after authorized manual pass (**5** live Investigate calls; ledger delta **`$0.00864441`**):

| Item | Status |
| --- | --- |
| Open case list and select a case without the CLI | **Verified** |
| See summary; start default investigation | **Verified** (5 live calls) |
| Evidence, uncertainty, recommended action, exact proposed task on one page | **Verified** |
| Approve → persisted **executed** outcome | **Verified** |
| Reject pending proposal **with** a reason | **Verified** |
| Reject pending proposal **without** a reason | **Verified** |
| Reopen earlier investigations; earlier **pending** proposal preserved across a later run | **Verified** |
| Reload pages; recover persisted history and **current** proposal statuses | **Verified** |
| Local startup; CSRF / no service or operator credentials in frontend assets | **Verified** |
| Honest **Investigating…** (controls disabled); in-mounted-UI duplicate Investigate guard (two clicks → one request via pending `fetch` intercept) | **Verified** (ephemeral browser; **no paid calls**) |
| Manual network failure on Investigate → inline error; prior findings preserved; reload clean | **Verified** (forced failure; **no paid calls**) |
| Failed/interrupted execution → explicit **Retry** via existing backend behavior | **Unavailable / not verified** — no `approved`-unfinished or `execution_failed` state observed; do **not** claim coverage |

**Automated gates (separate from manual UI checklist):** **162** pytest + Ruff lint/format + mypy + frontend build; reviews passed.

**Overall:** operator-UI spec is **not fully validated** solely because execution recovery/retry was **unavailable** to observe. Ledger unchanged this pass. Further paid Investigate/eval calls still require **explicit** authorization.

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

## Cost ledger (live only)

Application policy — **not** a provider hard cap:

- **Envelope:** **$5 USD total**, **GLM-only** (`z-ai/glm-5.3-flash`)
- **Backend is the sole ledger mutator** (`casepilot.backend.cost_guard`); investigations reserve/settle/release around billable work
- **Ledger file:** `evals/reports/cost_ledger.json` (override with `CASEPILOT_COST_LEDGER_PATH`)
- Non-allowlisted model, exhausted budget, or `pricing_reliable: false` → live calls pause (`CostPause`)
- Scripted / `CASEPILOT_TEST_MODE` paths do **not** touch the ledger
- Missing usage on settle fails closed (full reserve charged + pricing marked unreliable)

Conservative per-run reserves use local price tables × token ceilings × `max_model_turns` — bounds for guarding, not a quote.

Reported cumulative spend **`$0.02582926`** (baseline **`$0.01718485`** before operator-UI manual investigates; this-work delta **`$0.00864441`**) is a ledger snapshot — not a guaranteed cost per case.

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

**Scenario pilot** (paid; separate mode):

```bash
uv run python -m casepilot.evaluation --scenario-pilot --pilot-live --budget-usd 5
```

Six development scenarios × 3 reps + calibration gate. Requires explicit `--pilot-live`. Shared canonical ledger (`CASEPILOT_COST_LEDGER_PATH` / default `evals/reports/cost_ledger.json`) must match backend and CLI; local DB traces must match. Fresh `--cohort-id` after instrumentation fixes — never selectively replace started slots (resume unstarted only). Generated reports are ignored. Judge is same-model (GLM) and fallible — not ground truth.

## Grounding and approval controls

- **Untrusted evidence:** retrieved records are data, not instructions.
- **Case-bound tools only:** case / documents / events reads; no generic HTTP, SQL, or filesystem tool.
- **Evidence validation:** cited `record_id`s must match records retrieved in that run.
- **Execution budgets** (settings defaults): investigation timeout 60s, tool HTTP timeout 10s, max 12 tool calls, max 8 model turns.
- **No auto-writes:** investigation may store an **immutable proposal**; the model cannot set approval state.
- **Operator gate:** approve/reject require `OPERATOR_TOKEN`. Backend sends the stored payload to the simulator with the service token only after approval.
- **Idempotent task create:** simulator dedupes follow-up tasks by proposal id.
- **Confirmed task identity:** on successful approve/execute, backend persists the simulator `task_id` on the proposal (nullable for legacy rows). Operator UI shows task id/title/body when present; never fabricates ids for historical executions.
- **Run-scoped grounded chat:** completed/indeterminate runs support follow-up Q&A under `/v1/investigations/{run_id}/conversation`. Answers cite server-owned field snapshots only (model cannot invent values/times). Send/retry are idempotent; interrupted pending attempts recover on startup. Operator UI `ChatPanel` for eligible runs.
- **Tests:** scripted model hooks (`CASEPILOT_TEST_MODE`) — no paid calls.
- **Fail-closed offline tests:** with `CASEPILOT_TEST_MODE=1` and no installed scripted/hook model, `default_model_factory` refuses live OpenRouter fallback (investigation fails offline instead of billing).
- **Simulator outage harness (test-only):** when `CASEPILOT_TEST_MODE=1`, simulator exposes `PUT /v1/_test/outage` and `GET /v1/_test/mode` to inject reversible 503s on domain routes (health stays up). Routes are **not** registered without test mode.

Agent stack: **LangChain** (`create_agent`, tools, structured output, OpenRouter chat model). No custom graph authoring in this repo.

## Data layout and independence limits

| Path                             | Visibility                         | Purpose                                                                     |
| -------------------------------- | ---------------------------------- | --------------------------------------------------------------------------- |
| `fixtures/enterprise/cases.json` | Model-visible via tools after seed | Synthetic enterprise source records                                         |
| `evals/ground_truth/labels.json` | Evaluator-only                     | Diagnoses, evidence predicates, actions, **development / evaluation** split |
| `evals/reports/`                 | Generated (gitignored except keep) | Live reports + `cost_ledger.json`                                           |
| `docs/sample-evaluation.json`    | Curated sanitized sample           | Compact honest grades/metadata for portfolio readers                        |
| `SIMULATOR_DB_PATH`              | Simulator                          | Seeded records + executed tasks                                             |
| `BACKEND_DB_PATH`                | Backend                            | Runs, proposals, approval/audit state, conversation turns, evidence snapshots |
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
npm run build --prefix frontend   # operator UI typecheck/bundle (when changing frontend)
```

These offline gates never touch OpenRouter. Live `make eval` / UI Investigate are separate explicit opt-ins.

## Stack (actual)

- Python ≥ 3.14, `uv` + committed `uv.lock`, hatchling package `casepilot` (`src/casepilot`)
- FastAPI + Uvicorn + Pydantic Settings + HTTPX
- LangChain + `langchain-openrouter`
- stdlib `sqlite3`
- CLI: `casepilot` → `casepilot.cli:main`
- Eval: `python -m casepilot.evaluation`
- Operator UI: React + TypeScript + Vite + shadcn/ui under `frontend/` (local dev only; `/api` proxy)
- Dev extras: pytest, pytest-asyncio, ruff, mypy

## Out of scope (deferred)

Docker, production frontend hosting/packaging, cloud deploy, CI, LangSmith, login/RBAC/multi-user auth, multi-agent design, streaming tool traces, full insurance-record browser, bulk/auto retries, custom investigation objectives, real insurer integrations, autonomous underwriting/payment decisions.
