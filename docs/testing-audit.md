# Testing suite audit report

**Governing redesign:** [testing-risk-redesign.md](testing-risk-redesign.md) — **approved / implemented / final-validated**.  
**First-pass** ([testing-audit-spec.md](testing-audit-spec.md)): historical **403 → 382**; assertion-for-assertion preservation **superseded**.

**Current: 183 passed** (includes **+3** review restores).  
Trail: **403 → 382 → 183** (−**220** from 403 = **54.6%**; −**199** from 382 = **52.1%**).  
No equivalence to the old suite. No paid calls; prod/eval assets unchanged; no loop packing.

---

## Counts

| Checkpoint | Collected | Notes |
| --- | ---: | --- |
| Original baseline | **403** | |
| First-pass consolidation | **382** | −21 via replacement maps |
| Risk redesign + review restores | **183** | Delete-without-replacement + **3** critical restores |

### Restores (+3) — safety gating corrections

| Area | Change | Why kept |
| --- | --- | --- |
| Eval scoring | `test_evaluation` **25** (+2) | Empty-ref scoring truth; **offline report** public pipeline |
| Follow-up | `test_followup` **8** (+1) | **Timeout → failed** attempt path |

These are representative critical paths, not a return to exhaustive branch coverage.

### Current buckets (file sums → 183)

| Bucket | N | Files |
| --- | ---: | --- |
| Orchestration | **28** | scenario_pilot 24, cli 2, metadata_backend_prompt 1, contracts 1 |
| Assessment | **40** | pilot_safety 13, pilot_coverage 11, pilot_observability 11, eval_coverage_contract 5 |
| Eval core | **37** | evaluation **25**, evaluation_judge 12 |
| Backend domain | **44** | backend 18, evidence_snapshots 9, followup **8**, inv_exec_limits 3, fu_exec_limits 3, trace_redaction 2, approval_attempt_tracing 1 |
| Lifecycle / process | **29** | simulator 7, conversation_store 6, fixtures 5, conversation_routes 4, pilot_assets 4, approval_task_id 3 |
| Boundary | **4** | test_backend_workflow 4 |
| E2E | **1** | two_process_smoke 1 |
| **Total** | **183** | |

---

## Per-file inventory (183)

| File | N |
| --- | ---: |
| `tests/unit/test_evaluation.py` | 25 |
| `tests/unit/test_scenario_pilot.py` | 24 |
| `tests/unit/test_backend.py` | 18 |
| `tests/unit/test_pilot_safety.py` | 13 |
| `tests/unit/test_evaluation_judge.py` | 12 |
| `tests/unit/test_pilot_observability.py` | 11 |
| `tests/unit/test_pilot_coverage.py` | 11 |
| `tests/unit/test_evidence_snapshots.py` | 9 |
| `tests/unit/test_followup.py` | 8 |
| `tests/integration/test_simulator.py` | 7 |
| `tests/unit/test_conversation_store.py` | 6 |
| `tests/unit/test_fixtures.py` | 5 |
| `tests/unit/test_eval_coverage_contract.py` | 5 |
| `tests/unit/test_pilot_assets.py` | 4 |
| `tests/unit/test_conversation_routes.py` | 4 |
| `tests/boundary/test_backend_workflow.py` | 4 |
| `tests/unit/test_investigation_execution_limits.py` | 3 |
| `tests/unit/test_followup_execution_limits.py` | 3 |
| `tests/unit/test_approval_task_id.py` | 3 |
| `tests/unit/test_trace_evidence_redaction.py` | 2 |
| `tests/unit/test_cli.py` | 2 |
| `tests/unit/test_metadata_backend_prompt.py` | 1 |
| `tests/unit/test_contracts.py` | 1 |
| `tests/unit/test_approval_attempt_tracing.py` | 1 |
| `tests/e2e/test_two_process_smoke.py` | 1 |
| **Sum** | **183** |

**Deleted files (no replacement):** `test_followup_render_diagnostics.py`, `test_followup_output_diagnostics.py`.

---

## Retained guarantees (risk — not equivalence)

| Domain | How |
| --- | --- |
| **Authorization / execution** | Boundary approve idempotent / reject terminal / outage retry / invalid evidence; unauthorized **denial** at boundary (`test_approval_attempt_tracing` + workflow); task-id + crash recovery; E2E smoke restart reapproval |
| **Domain / evidence / follow-up** | Backend evidence paths; snapshots; follow-up cite / bad cite / tools / **timeout→failed** / fail-closed test mode; conversation send/retry/startup recovery |
| **Budget / charging** | Ledger reserve/settle/release, global cap, lock; live opt-in/budget/model gates; cost pause paths; focused limits; scripted ledger bypass **documented only** |
| **Privacy** | Trace redaction secrets + reasoning / domain reasons preserved |
| **Scoring truth** | Grader rules incl. **empty retrieval ref**; scorecard failures-in-denominator; judge fail-closed set |
| **Report / resume / calibration** | **Offline report** pipeline; coverage gate contract (5); pilot cal/resume/ledger preflight |
| **Fixtures / GT** | `test_fixtures` still exercises **GT split** composition — see losses note below |
| **Deployment** | Real-process E2E only; **no browser suite** (manual Playwright/CLI only) |

---

## Accepted losses (explicit, non-equivalent)

| Lost | Clarification |
| --- | --- |
| **Auth-denial trace emission schema** | Unauthorized **denial behavior remains** tested; exact forbidden-attempt **trace/event schema** not retained as its own surface |
| **Exhaustive asset-loader mutation** | Broad mutation matrix gone; **not** “no composition anywhere” — `test_fixtures` still retains **GT split** composition |
| **Individual composition micro-asserts** | Many one-off composition checks dropped; representative composition may still appear inside kept fixtures/workflows |
| **Exact render / parser / diagnostics** | Entire follow-up render+output diagnostic files removed (codes, emit plumbing, prose) |
| Schema/enum/route-template-only, private helpers, repetitive orchestration plumbing | Per risk redesign intentional reduction |

**Not accepted as “just formatting”:** authz denial, spend/charging, scoring-truth, P0 gate, resume/cal false quality or unintended spend.

---

## Final validation (confirmed)

| Gate | Result |
| --- | --- |
| `pytest` | **183 passed**, **4.85s**, one baseline langchain Pydantic warning |
| Duration vs prior eras | Single-run sample only — **no stronger speed claim** |
| `make lint` | **PASS** (**68** formatted) |
| `make typecheck` / mypy | **PASS** (**35** source files) |
| Spec review (small) | **PASS** (after +3 restores) |
| Quality/simplicity (medium) | **PASS** (after +3 restores) |
| Manual Playwright/CLI (both sessions) | **PASS** (backend + eval; eval **viewer-only** caveat) |
| Browser test suite | **none** (by design) |

```bash
make test          # 183 passed
make lint          # PASS, 68 formatted
make typecheck     # PASS, 35 source files
```

---

## Historical first-pass (403 → 382)

| Wave | Net |
| --- | ---: |
| Backend 6→4, eval gate 9→5, judge 12→3, scorecard 3→1, follow-up thin −3, recovery −1 | **−21 → 382** |
| Grader 11→3 pack | **rejected** (count packing) |

---

**Done.** Risk-era suite **183** final-validated; accepted losses intentional and non-equivalent; +3 restores kept for safety gating.
