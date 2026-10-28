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
