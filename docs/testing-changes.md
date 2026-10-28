# Testing redesign — chronological slice log

Per-commit record of retained risk guarantees and accepted losses.
Final inventory and suite totals: see [testing-audit.md](testing-audit.md) (final commit only).
No cross-slice equivalence claims.

## 1. Backend workflows (boundary + unit trim)

**Files:** `tests/boundary/*`, `tests/unit/test_backend.py`

**Change:** Six mock-flow backend unit nodes replaced by four real-app-boundary workflow tests (`test_backend_workflow.py`). Remaining `test_backend.py` keeps spend guard, redaction, evidence validation, and fail-closed units. Module headers marked historically applied (no pending-approval language).

**Retained:** investigate → approve/reject → execute at real backend+simulator+SQLite boundary; reject terminal; outage retry; invalid evidence does not write proposal; idempotent approve side effects; unit coverage for spend/redaction/evidence/fail-closed.

**Accepted loss:** Mock route/schema helper units and exact mock-flow assertion shapes not preserved; denial-trace schema not owned here.

## 2. Eval coverage / orchestration gate contract

**Files:** `tests/unit/test_eval_coverage_contract.py`, `test_pilot_coverage.py`, `test_scenario_pilot.py`

**Change:** Nine original gate nodes folded into fixture-driven `test_eval_coverage_contract.py`. `test_pilot_coverage` keeps classify/assess honesty; `test_scenario_pilot` keeps focused orchestration (large low-value plumbing deleted). Header on contract module marked historically applied.

**Retained:** Required-retrieval coverage gate; cohort reevaluate/report gate integrity; assessment status honesty paths still in pilot_coverage.

**Accepted loss:** Exhaustive scenario-pilot orchestration micro-branches and repetitive gate plumbing; no claim of equivalent node-for-node coverage vs pre-redesign scenario_pilot.

## 3. Eval scoring harness

**Files:** `tests/unit/test_evaluation.py`

**Change:** Risk-based trim of deterministic evaluation harness units; keep scoring-truth paths (failures in denominator, empty retrieval-ref rules, offline report/resume integrity including restored empty-ref and offline report cases).

**Retained:** Deterministic scoring truth; cost/budget gates touched by harness; report pipeline honesty needed to avoid false quality claims.

**Accepted loss:** Low-impact harness helper/edge assertion volume; not every historical input vector.

## 4. Evaluation judge

**Files:** `tests/unit/test_evaluation_judge.py`

**Change:** Collapse judge suite to focused fail-closed contract tests (scripted; no paid calls).

**Retained:** Judge fail-closed on invalid/malformed outcomes; dimension assessment contracts that gate false quality claims.

**Accepted loss:** Broad judge helper/branch matrix and repetitive serialization/plumbing asserts.

## 5a. Pilot safety

**Files:** `tests/unit/test_pilot_safety.py`

**Change:** Adversarial pilot safety suite trimmed to high-impact phase/scenario safety checks.

**Retained:** Deterministic safety assessment for forbidden attempts, task/proposal state facts, and phase safety evidence needed to catch unsafe pilot progression.

**Accepted loss:** Exhaustive safety micro-fact combinations and low-signal repetition.

## 5b. Pilot observability

**Files:** `tests/unit/test_pilot_observability.py`

**Change:** Observability collector suite trimmed to fail-closed pre-turn/event integrity and critical collector contracts.

**Retained:** Pre-turn fail-closed / not-assessed honesty and collector behaviors that would otherwise hide invalid pilot runs.

**Accepted loss:** Redundant event-schema and tracing-plumbing asserts; exact emission shape volume.

## 6. Follow-up + evidence snapshots (shared guarantees)

**Files:** `tests/unit/test_followup.py`, `test_evidence_snapshots.py`; delete `test_followup_output_diagnostics.py`, `test_followup_render_diagnostics.py`

**Change:** Focused follow-up agent units (incl. timeout→failed restore) and evidence snapshot security/grounding; remove render/output diagnostics files entirely (no replacement).

**Retained:** Representative follow-up including timeout→failed; evidence snapshot isolation/grounding and sensitive handling shared with follow-up path.

**Accepted loss:** Exact render/parser/follow-up diagnostics files and their assertion volume — deleted without replacement.

## 7. Conversation recovery

**Files:** `tests/unit/test_conversation_store.py`, `test_conversation_routes.py`

**Change:** Trim store/route conversation tests to atomicity, retry/recovery, and essential route behavior.

**Retained:** Conversation persistence atomicity and retry recovery; route-level create/read behaviors needed for restart-safe follow-up continuity.

**Accepted loss:** Redundant route-template and store helper asserts.

## 8a. Approval identity and denial boundary

**Files:** `tests/unit/test_approval_task_id.py`, `test_approval_attempt_tracing.py`

**Change:** Keep non-derived task_id guarantees; reduce attempt tracing to unauthorized-approve denied at boundary. Drop proposal task_id column migration unit.

**Retained:** Approval task_id correctness; unauthorized callers cannot approve (denial at boundary).

**Accepted loss:** Auth-denial **trace emission schema** (behavior retained, schema not); migration micro-test.

## 8b. Investigation and follow-up execution limits

**Files:** `tests/unit/test_followup_execution_limits.py`, `test_investigation_execution_limits.py`

**Change:** Small risk-based trims; keep charging-boundary-relevant budget/input/tool limit enforcement units.

**Retained:** Investigation and follow-up execution limit enforcement (budget/input/tool) that gates unintended spend.

**Accepted loss:** Minor redundant limit-edge volume only.

## 9. Simulator integration

**Files:** `tests/integration/test_simulator.py`

**Change:** Trim simulator integration to representative concurrency/conflict/seed/outage contracts.

**Retained:** Simulator behaviors required for real investigation/execution integration confidence (conflict, seed, outage-route).

**Accepted loss:** Exhaustive simulator route micro-matrix beyond representative contracts.

## 10a. Contracts surface

**Files:** `tests/unit/test_contracts.py`

**Change:** Massive schema/enum/serialization trim to secret-not-leaked contract.

**Retained:** Secrets must not leak via contract/serialization surfaces.

**Accepted loss:** Schema/enum/serialization/route-template-only coverage deleted without replacement.

## 10b. CLI surface

**Files:** `tests/unit/test_cli.py`

**Change:** Trim CLI units to bearer-token-not-printed and client timeout safety.

**Retained:** CLI must not print bearer tokens; client timeout behavior that avoids hung paid/live calls.

**Accepted loss:** Broader CLI help/plumbing/argparse coverage.
