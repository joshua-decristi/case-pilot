# Evaluation and evidence

[Overview](../README.md) · [Architecture](architecture.md) · [Quickstart](quickstart.md)

## Recorded results

Small synthetic demonstrations, not production reliability estimates. Generated reports under `evals/reports/` are ignored and may be absent from a checkout; the [sanitized sample](sample-evaluation.json) is portable.

| Check                          | Recorded result                                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------------- |
| Held-out evaluation            | N=8; category 7/8, evidence IDs 7/8, action 6/8, uncertainty 1/1, hard failures 0/8                     |
| CASE-001 live smoke, attempt 3 | Completed; missing document; five accepted evidence rows; request missing document; settled $0.00169768 |

Sources: local `evals/reports/eval_latest.json` and `evals/reports/live-smoke.json`. The prior README recorded smoke attempt 1 failing at async model-call middleware and attempt 2 returning indeterminate/escalate after evidence validation. Those historical observations are not independently traceable in the current attempt-3 artifact. The headline smoke produced no proposal, so does not demonstrate approval/execution.

**Do not over-read these numbers:**

- Small held-out **N=8**; not production accuracy or generalization.
- Evidence scores check required **record IDs + non-empty `retrieval_ref`** only — **not** semantic entailment proof that the narrative is correct.
- Action auto-score is rule-mapped codes vs labels; misses are real (e.g. escalate vs correct_data_mismatch on CASE-009/010).
- Labels grade only; **no prompt changes based on labels** for the reported run.
- Live model output is non-deterministic; re-runs may differ.
- Full narratives/tool payloads are omitted from the portable sample; reconciliation sources for smoke are noted in `evals/reports/live-smoke.json`.

Primary artifacts: `evals/reports/live-smoke.json` (CASE-001 attempt **3** headline), `evals/reports/eval_latest.json`, `evals/reports/cost_ledger.json`, sanitized `docs/sample-evaluation.json`.

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

### Trace lookup for eval results

Tracing and grading are **separate**. When a live or scripted eval case invokes the backend agent, the same execution can emit a sanitized OTLP trace correlated by the **application run ID already stored on the eval result**. OTLP export does not score cases, alter labels, or touch the cost ledger.

```bash
# 1. Optional: start any loopback OTLP HTTP collector on the configured endpoint.

# 2. Run eval that actually executes the agent (live paid path shown).
#    Export uses TRACING_ENABLED + TRACING_OTLP_ENDPOINT
#    (default http://127.0.0.1:6006/v1/traces).
make eval MAX_CASES=1

# 3. Open the result JSON under evals/reports/ (or your --report-dir).
#    Copy the existing application run_id from the case result.

# 4. In the collector/viewer, search/filter by that run ID
#    (same ID as JSONL traces/<run_id>.jsonl).
```

| Path | Agent execution? | OTLP execution trace? | Grades? |
| --- | --- | --- | --- |
| `make eval` / `--live` | Yes (paid) | Yes, if `TRACING_ENABLED` and a loopback collector is up | Yes |
| Scenario pilot live slots | Yes (paid) | Same as other agent runs | Pilot/judge path (not held-out labels) |
| `make eval-offline` / `--offline` / `run_offline_report` | **No** | **No** — report-only grading never reruns the agent for telemetry | Yes, from saved results |

Do **not** transmit eval labels or expected answers to the backend for tracing. Do **not** rerun an agent solely to populate a viewer. Scores and cost behavior must match with tracing enabled or disabled; the grader and cost guard remain authoritative. Export settings: [quickstart](quickstart.md#backend-otlp-export). Architecture: [observability](architecture.md#local-observability-sanitized-otlp).

**Scenario pilot** (paid; separate mode):

```bash
uv run python -m casepilot.evaluation --scenario-pilot --pilot-live --budget-usd 5
```

Six development scenarios × 3 reps + calibration gate. Requires explicit `--pilot-live`. Shared canonical ledger (`CASEPILOT_COST_LEDGER_PATH` / default `evals/reports/cost_ledger.json`) must match backend and CLI; local DB traces must match. Fresh `--cohort-id` after instrumentation fixes — never selectively replace started slots (resume unstarted only). Generated reports are ignored. Judge is same-model (GLM) and fallible — not ground truth.

## Data layout and independence limits

| Path                             | Visibility                         | Purpose                                                                       |
| -------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------- |
| `fixtures/enterprise/cases.json` | Model-visible via tools after seed | Synthetic enterprise source records                                           |
| `evals/ground_truth/labels.json` | Evaluator-only                     | Diagnoses, evidence predicates, actions, **development / evaluation** split   |
| `evals/reports/`                 | Generated (gitignored except keep) | Live reports + `cost_ledger.json`                                             |
| `docs/sample-evaluation.json`    | Curated sanitized sample           | Compact honest grades/metadata for portfolio readers                          |
| `SIMULATOR_DB_PATH`              | Simulator                          | Seeded records + executed tasks                                               |
| `BACKEND_DB_PATH`                | Backend                            | Runs, proposals, approval/audit state, conversation turns, evidence snapshots |
| `TRACE_DIR`                      | Backend                            | JSONL traces (no secrets/tokens); default `traces/`                           |

**Limitations:**

- Authored synthetic data only — not production insurance performance.
- Labels stay under `evals/ground_truth/`; fixtures must not embed answer keys. Only tool-returned records enter model context.
- Demo tokens + loopback binding ≠ enterprise identity, TLS, or multi-user RBAC.
- One simulated write path (`create_follow_up_task`). No payments, coverage changes, or external messaging.
- The $5 GLM ledger is app policy; use a provider account limit if you need a hard stop.

## Cost interpretation

The ledger is mutable operational state, not a benchmark. The inspected local ledger contained $0.63585653 cumulative spend; the former README total of $0.02582926 was an earlier snapshot. Neither is a promised per-case cost. Read your configured ledger for current headroom.

The held-out report recorded $0.01381785 session spend and $0.01718485 cumulative spend at that run. Subsequent scenario-pilot cohorts are separate development experiments: later passing cohorts do not erase earlier safety failures or establish held-out reliability.

Runtime validation checks retrieved records and field paths; the reported evidence grade checks required record IDs and retrieval references. Neither proves semantic entailment.
