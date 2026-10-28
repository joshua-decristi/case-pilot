# Testing redesign — chronological slice log

Per-commit record of retained risk guarantees and accepted losses.
Final inventory and suite totals: see [testing-audit.md](testing-audit.md) (final commit only).
No cross-slice equivalence claims.

## 1. Backend workflows (boundary + unit trim)

**Files:** `tests/boundary/*`, `tests/unit/test_backend.py`

**Change:** Six mock-flow backend unit nodes replaced by four real-app-boundary workflow tests (`test_backend_workflow.py`). Remaining `test_backend.py` keeps spend guard, redaction, evidence validation, and fail-closed units. Module headers marked historically applied (no pending-approval language).

**Retained:** investigate → approve/reject → execute at real backend+simulator+SQLite boundary; reject terminal; outage retry; invalid evidence does not write proposal; idempotent approve side effects; unit coverage for spend/redaction/evidence/fail-closed.

**Accepted loss:** Mock route/schema helper units and exact mock-flow assertion shapes not preserved; denial-trace schema not owned here.
