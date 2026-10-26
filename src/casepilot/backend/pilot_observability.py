"""Backend-owned read-only pilot observability collector.

Opt-in only. The scenario pilot runner calls this after HTTP phase calls to
assemble authoritative before/after write-audit rows plus follow-up coverage,
model-visible evidence, usage/cost, and forbidden-attempt provenance from:

  - BackendStore (proposals, evidence snapshots, run case_id)
  - simulator SQLite (follow_up_tasks + record→case mapping)
  - existing JSONL traces (pre-turn context, tool exposure, end meta, forbidden)

No public HTTP routes. No eval-reference injection. Absent instrumentation
yields ``None`` on proof fields so safety/coverage stay ``not_assessed``.

---------------------------------------------------------------------------
Runner contract (scenario_pilot; not wired by this module)
---------------------------------------------------------------------------
1. Construct only with ``enabled=True`` under explicit pilot live/offline opt-in.
   ``enabled=False`` (default) → every method returns ``None``.

2. Write audit — before and after each phase HTTP call::

     before = collector.capture_write_audit(case_id)
     after  = collector.capture_write_audit(case_id)
     proposals_before = proposal_facts_from_audit(before)  # None if unusable
     tasks_before     = task_facts_from_audit(before)
     write_audit_captured = bool(
         before and after and before.captured and after.captured
     )

   Empty tuples from a captured audit are real zeros, not missing evidence.

3. After follow-up HTTP (need attempt_id)::

     obs = collector.observe_followup(case_id=..., run_id=..., attempt_id=...)

   Map onto PhaseCapture / judge inputs:
     - pre_turn_model_context
     - answering_model_exposed   # model-visible tool facts (not all store rows)
     - attempt_follow_up_snapshots
     - persisted_snapshots       # SnapshotProvenanceFact with ACTUAL case_id
     - forbidden_attempts
     - usage / cost_usd / provider_model  # from follow_up_end; None = gap
   ``None`` proof fields mean missing instrumentation — do not invent empty proof.
   ``obs.gaps`` lists missing end-meta / unresolved snapshot case ids, etc.

4. After investigation::

     obs = collector.observe_investigation(case_id=..., run_id=...)

   Same mapping; ``answering_model_exposed`` comes from investigation tool
   ``facts`` (model-visible), gated on ``exposure_instrumented`` / facts key.
   ``pre_turn_model_context`` stays None (N/A).

5. Snapshot provenance — **do not** re-tag with the intended phase case_id.
   ``obs.persisted_snapshots`` already carries the authoritative source case_id
   derived from simulator record mapping (and cases source_system). If any row
   cannot be resolved → ``persisted_snapshots is None`` + gap
   ``snapshot_case_unresolved`` → safety not_assessed (never pretend intended).

6. Never POST approve/reject or create tasks from the runner. Record local
   ForbiddenAttemptFact if the runner itself would make a forbidden call.

Trace events understood:
  - ``pre_turn_context`` — post-filter established context (follow-up)
  - ``tool`` with ``facts`` — content exposed to the answering model
  - ``follow_up_end`` / ``run_end`` — usage, cost_usd, provider_model,
    exposure_instrumented
  - ``forbidden_attempt`` / ``unknown_tool_attempt``
  - ``tool_attempt_batch`` — attempted tool names on runtime error (no args);
    execution_status is ``not_executed`` only with limiter-abort proof, else
    ``unknown`` (never invent per-call executed/blocked labels)
  - ``model_turn_batches`` — all AIMessage tool-name batches after success/error
    (names+count only; includes structured-output tool attempts; end aggregate)
  - ``follow_up_output`` — untrusted schema refs (claim ids / kind+cites / note
    codes) + render reject_code; never free interpretation text/values/CoT
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from casepilot.backend.db import BackendStore
from casepilot.evaluation.pilot_coverage import AttemptSnapshot, ExposedFact
from casepilot.evaluation.pilot_safety import (
    ForbiddenAttemptFact,
    ProposalStateFact,
    SnapshotProvenanceFact,
    TaskStateFact,
    collect_proposal_states,
    forbidden_attempt_fact,
    task_state_fact,
)

# Trace event names (stable contract with followup/agent/app instrumentation).
EVENT_PRE_TURN = "pre_turn_context"
EVENT_TOOL = "tool"
EVENT_FORBIDDEN = "forbidden_attempt"
EVENT_UNKNOWN_TOOL = "unknown_tool_attempt"
EVENT_TOOL_ATTEMPT_BATCH = "tool_attempt_batch"
EVENT_MODEL_TURN_BATCHES = "model_turn_batches"
EVENT_FOLLOW_UP_OUTPUT = "follow_up_output"
EVENT_FOLLOW_UP_END = "follow_up_end"
EVENT_RUN_END = "run_end"

# Batch execution proof levels for EVENT_TOOL_ATTEMPT_BATCH (aggregate only).
EXEC_STATUS_NOT_EXECUTED = "not_executed"  # entire batch aborted before run
EXEC_STATUS_UNKNOWN = "unknown"  # no proof tools ran or were blocked

# Diagnostics label — never feed these refs back into the model.
UNTRUSTED_MODEL_OUTPUT_LABEL = "untrusted"

_FOLLOW_UP_PROVENANCE = "follow_up"
_ORIGINAL_PROVENANCE = "original_run"

# Simulator source_system values used by evidence captures.
_SRC_CASES = "cases"
_SRC_DOCUMENTS = "documents"
_SRC_EVENTS = "events"


@dataclass(frozen=True)
class WriteAuditSnapshot:
    """Authoritative proposal + task rows for one case at one instant."""

    proposals: tuple[Any, ...]
    tasks: tuple[TaskStateFact, ...]
    """True only when both backend proposals and simulator tasks were readable."""
    captured: bool
    gaps: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseObservation:
    """Read-only observation for one completed phase.

    ``None`` on coverage/proof fields = missing or incomplete instrumentation
    (fail-closed) → not_assessed. Empty tuples mean *known* zero rows/facts only
    when the instrumented event reported an empty set — never a partial parse.
    Usage/cost/provider ``None`` means missing from end trace (see gaps) —
    never coerce missing to zero.
    """

    persisted_snapshots: tuple[SnapshotProvenanceFact, ...] | None = None
    pre_turn_model_context: tuple[ExposedFact, ...] | None = None
    answering_model_exposed: tuple[ExposedFact, ...] | None = None
    attempt_follow_up_snapshots: tuple[AttemptSnapshot, ...] | None = None
    forbidden_attempts: tuple[ForbiddenAttemptFact, ...] = ()
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    provider_model: str | None = None
    gaps: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()


@dataclass
class PilotObservability:
    """Opt-in collector. Disabled instance is a safe no-op."""

    enabled: bool = False
    backend_db_path: Path | None = None
    simulator_db_path: Path | None = None
    trace_dir: Path | None = None
    _store: BackendStore | None = field(default=None, repr=False, compare=False)
    _sim_case_cache: dict[tuple[str, str], str | None] = field(
        default_factory=dict, repr=False, compare=False
    )

    # ----- construction -----------------------------------------------------

    @classmethod
    def open(
        cls,
        *,
        enabled: bool,
        backend_db_path: Path,
        simulator_db_path: Path,
        trace_dir: Path,
        store: BackendStore | None = None,
    ) -> PilotObservability:
        """Factory. When disabled, paths are retained but never opened."""
        return cls(
            enabled=bool(enabled),
            backend_db_path=Path(backend_db_path),
            simulator_db_path=Path(simulator_db_path),
            trace_dir=Path(trace_dir),
            _store=store,
        )

    def close(self) -> None:
        return None

    def _get_store(self) -> BackendStore | None:
        if not self.enabled or self.backend_db_path is None:
            return None
        if self._store is not None:
            return self._store
        if not self.backend_db_path.exists():
            return None
        self._store = BackendStore(self.backend_db_path)
        return self._store

    # ----- write audit (proposals + tasks) ----------------------------------

    def capture_write_audit(self, case_id: str) -> WriteAuditSnapshot | None:
        """Read proposals + simulator tasks for ``case_id``.

        Returns ``None`` when collector disabled. ``captured=False`` when either
        source is unreadable (caller must not set write_audit_captured).
        """
        if not self.enabled:
            return None
        gaps: list[str] = []
        proposals: tuple[Any, ...] | None = None
        tasks: tuple[TaskStateFact, ...] | None = None

        store = self._get_store()
        if store is None:
            gaps.append("backend_store_unreadable")
        else:
            try:
                rows = store.list_proposals_for_case(case_id)
                proposals = tuple(rows)
            except Exception as exc:  # noqa: BLE001
                gaps.append(f"proposals_read_error:{type(exc).__name__}")

        task_rows = self._list_simulator_tasks(case_id)
        if task_rows is None:
            gaps.append("simulator_tasks_unreadable")
        else:
            tasks = tuple(task_state_fact(r) for r in task_rows)

        captured = proposals is not None and tasks is not None
        return WriteAuditSnapshot(
            proposals=proposals if proposals is not None else (),
            tasks=tasks if tasks is not None else (),
            captured=captured,
            gaps=tuple(gaps),
        )

    def _list_simulator_tasks(self, case_id: str) -> list[dict[str, Any]] | None:
        path = self.simulator_db_path
        if path is None or not path.exists():
            return None
        try:
            conn = sqlite3.connect(str(path), timeout=5.0)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT task_id, proposal_id, case_id FROM follow_up_tasks"
                    " WHERE case_id = ? ORDER BY created_at, task_id",
                    (case_id,),
                ).fetchall()
                return [
                    {
                        "task_id": str(r["task_id"]),
                        "proposal_id": str(r["proposal_id"]),
                        "case_id": str(r["case_id"]),
                    }
                    for r in rows
                ]
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return None

    # ----- phase observation ------------------------------------------------

    def observe_followup(
        self,
        *,
        case_id: str,
        run_id: str,
        attempt_id: str,
    ) -> PhaseObservation | None:
        """Collect follow-up coverage + safety + cost provenance after HTTP."""
        if not self.enabled:
            return None
        gaps: list[str] = []
        signals: list[str] = []

        store = self._get_store()
        store_rows: list[Any] | None = None
        if store is None:
            gaps.append("backend_store_unreadable")
        else:
            try:
                store_rows = list(store.list_evidence_snapshots(run_id))
                signals.append("store_snapshots")
            except Exception as exc:  # noqa: BLE001
                gaps.append(f"snapshots_read_error:{type(exc).__name__}")

        events = self._read_attempt_events(run_id, attempt_id)
        run_events = self._read_run_events(run_id)
        if events:
            signals.append("attempt_trace")
        else:
            gaps.append("attempt_trace_missing")
        if run_events:
            signals.append("run_trace")

        pre_turn, pre_gaps = self._parse_pre_turn(events, store_rows or [])
        gaps.extend(pre_gaps)
        if pre_turn is not None:
            signals.append(EVENT_PRE_TURN)

        exposed: tuple[ExposedFact, ...] | None = None
        # Follow-up: pre_turn_context marks the instrumented runtime revision
        # that also emits tool facts for model exposure. Exposure is only
        # "complete" when parse returns a tuple (incl. empty); None = fail-closed.
        if pre_turn is not None:
            exposed, exp_gaps = self._parse_tool_exposure(events, store_rows or [])
            gaps.extend(exp_gaps)
            if exposed is not None:
                signals.append("answering_model_exposed")

        attempt_snaps: tuple[AttemptSnapshot, ...] | None = None
        if store_rows is not None:
            attempt_snaps = self._attempt_follow_up_snapshots(store_rows, attempt_id)
            signals.append("attempt_follow_up_snapshots")

        snap_facts, snap_gaps = self._snapshot_facts_actual_case(store_rows, run_id=run_id)
        gaps.extend(snap_gaps)
        if snap_facts is not None:
            signals.append("persisted_snapshots")

        forbidden = self._dedupe_forbidden(
            self._parse_forbidden(events) + self._parse_forbidden(run_events)
        )
        if forbidden:
            signals.append(EVENT_FORBIDDEN)

        usage, cost, provider, meta_gaps = self._parse_end_meta(
            events, end_events=(EVENT_FOLLOW_UP_END,)
        )
        gaps.extend(meta_gaps)
        if not meta_gaps or "phase_end_trace_missing" not in meta_gaps:
            signals.append("end_meta")

        return PhaseObservation(
            persisted_snapshots=snap_facts,
            pre_turn_model_context=pre_turn,
            answering_model_exposed=exposed,
            attempt_follow_up_snapshots=attempt_snaps,
            forbidden_attempts=tuple(forbidden),
            usage=usage,
            cost_usd=cost,
            provider_model=provider,
            gaps=tuple(gaps),
            signals=tuple(signals),
        )

    def observe_investigation(
        self,
        *,
        case_id: str,
        run_id: str,
    ) -> PhaseObservation | None:
        """Collect investigation snapshots, model exposure, cost, forbidden."""
        if not self.enabled:
            return None
        gaps: list[str] = []
        signals: list[str] = []
        store = self._get_store()
        store_rows: list[Any] | None = None
        if store is None:
            gaps.append("backend_store_unreadable")
        else:
            try:
                store_rows = list(store.list_evidence_snapshots(run_id))
                signals.append("store_snapshots")
            except Exception as exc:  # noqa: BLE001
                gaps.append(f"snapshots_read_error:{type(exc).__name__}")

        events = self._read_run_events(run_id)
        if events:
            signals.append("run_trace")
        else:
            gaps.append("run_trace_missing")

        exposed: tuple[ExposedFact, ...] | None = None
        if _exposure_instrumented(events):
            exposed, exp_gaps = self._parse_tool_exposure(events, store_rows or [])
            gaps.extend(exp_gaps)
            if exposed is not None:
                signals.append("answering_model_exposed")
        else:
            gaps.append("investigation_exposure_not_instrumented")

        snap_facts, snap_gaps = self._snapshot_facts_actual_case(store_rows, run_id=run_id)
        gaps.extend(snap_gaps)
        if snap_facts is not None:
            signals.append("persisted_snapshots")

        forbidden = self._dedupe_forbidden(self._parse_forbidden(events))
        if forbidden:
            signals.append(EVENT_FORBIDDEN)

        usage, cost, provider, meta_gaps = self._parse_end_meta(events, end_events=(EVENT_RUN_END,))
        gaps.extend(meta_gaps)
        if not meta_gaps or "phase_end_trace_missing" not in meta_gaps:
            signals.append("end_meta")

        return PhaseObservation(
            persisted_snapshots=snap_facts,
            pre_turn_model_context=None,
            answering_model_exposed=exposed,
            attempt_follow_up_snapshots=None,
            forbidden_attempts=tuple(forbidden),
            usage=usage,
            cost_usd=cost,
            provider_model=provider,
            gaps=tuple(gaps),
            signals=tuple(signals),
        )

    # ----- actual case_id provenance ----------------------------------------

    def _snapshot_facts_actual_case(
        self,
        store_rows: list[Any] | None,
        *,
        run_id: str,
    ) -> tuple[tuple[SnapshotProvenanceFact, ...] | None, list[str]]:
        """Build provenance facts with *actual* source case_id (never intended).

        Any unresolved record→case mapping → whole set ``None`` + gap so safety
        stays not_assessed rather than silently tagging the phase case.
        """
        gaps: list[str] = []
        if store_rows is None:
            return None, gaps
        out: list[SnapshotProvenanceFact] = []
        unresolved = 0
        for row in store_rows:
            source = str(_get(row, "source_system") or "")
            record_id = str(_get(row, "record_id") or "")
            actual = self.resolve_record_case_id(source, record_id)
            if actual is None or not actual:
                unresolved += 1
                continue
            out.append(
                SnapshotProvenanceFact(
                    snapshot_id=_snap_id(row),
                    run_id=str(_get(row, "run_id") or run_id),
                    case_id=actual,
                    record_id=record_id,
                    field_path=str(_get(row, "field_path") or ""),
                    provenance=_prov(row) or None,
                    attempt_id=_get(row, "attempt_id"),
                )
            )
        if unresolved:
            gaps.append(f"snapshot_case_unresolved:{unresolved}")
            return None, gaps
        return tuple(out), gaps

    def resolve_record_case_id(self, source_system: str, record_id: str) -> str | None:
        """Map a retrieved record to its authoritative simulator/source case_id."""
        src = (source_system or "").strip()
        rid = (record_id or "").strip()
        if not rid:
            return None
        key = (src, rid)
        if key in self._sim_case_cache:
            return self._sim_case_cache[key]

        actual: str | None
        if src in (_SRC_CASES, "case"):
            # Case record id is the case_id; confirm in simulator when possible.
            actual = self._sim_lookup_case_exists(rid) or rid
        elif src == _SRC_DOCUMENTS:
            actual = self._sim_lookup_row_case(
                "SELECT case_id FROM documents WHERE document_id = ?", rid
            )
        elif src == _SRC_EVENTS:
            actual = self._sim_lookup_row_case("SELECT case_id FROM events WHERE event_id = ?", rid)
        else:
            # Unknown source_system — do not guess from intended/run case.
            actual = None

        self._sim_case_cache[key] = actual
        return actual

    def _sim_lookup_case_exists(self, case_id: str) -> str | None:
        path = self.simulator_db_path
        if path is None or not path.exists():
            return None
        try:
            conn = sqlite3.connect(str(path), timeout=5.0)
            try:
                row = conn.execute(
                    "SELECT case_id FROM cases WHERE case_id = ?", (case_id,)
                ).fetchone()
                return str(row[0]) if row else None
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return None

    def _sim_lookup_row_case(self, sql: str, record_id: str) -> str | None:
        path = self.simulator_db_path
        if path is None or not path.exists():
            return None
        try:
            conn = sqlite3.connect(str(path), timeout=5.0)
            try:
                row = conn.execute(sql, (record_id,)).fetchone()
                if row is None:
                    return None
                return str(row[0]) if row[0] is not None else None
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return None

    # ----- trace I/O --------------------------------------------------------

    def _attempt_trace_path(self, run_id: str, attempt_id: str) -> Path | None:
        if self.trace_dir is None:
            return None
        return self.trace_dir / f"{run_id}:{attempt_id}.jsonl"

    def _run_trace_path(self, run_id: str) -> Path | None:
        if self.trace_dir is None:
            return None
        return self.trace_dir / f"{run_id}.jsonl"

    def _read_attempt_events(self, run_id: str, attempt_id: str) -> list[dict[str, Any]]:
        path = self._attempt_trace_path(run_id, attempt_id)
        return _read_jsonl(path) if path is not None else []

    def _read_run_events(self, run_id: str) -> list[dict[str, Any]]:
        path = self._run_trace_path(run_id)
        return _read_jsonl(path) if path is not None else []

    # ----- parsers ----------------------------------------------------------

    @staticmethod
    def _parse_pre_turn(
        events: Sequence[dict[str, Any]],
        store_rows: Sequence[Any],
    ) -> tuple[tuple[ExposedFact, ...] | None, list[str]]:
        """Parse pre_turn_context. ``()`` = known empty; ``None`` = missing/invalid."""
        matched = [e for e in events if e.get("event") == EVENT_PRE_TURN]
        if not matched:
            return None, []
        facts_raw = matched[-1].get("facts")
        if facts_raw is None:
            return (), []
        if not isinstance(facts_raw, list):
            return None, ["pre_turn_facts_malformed"]
        out, gaps = _facts_with_authoritative_values(facts_raw, store_rows)
        # Whole-set fail-closed: any unresolved item → None (not partial tuple).
        if out is None:
            return None, gaps
        return tuple(out), gaps

    @staticmethod
    def _parse_tool_exposure(
        events: Sequence[dict[str, Any]],
        store_rows: Sequence[Any],
    ) -> tuple[tuple[ExposedFact, ...] | None, list[str]]:
        """Parse tool exposure. ``()`` = known zero tools/facts; ``None`` = incomplete."""
        by_id = {_snap_id(s): s for s in store_rows if _snap_id(s)}
        out: list[ExposedFact] = []
        seen: set[str] = set()
        unresolved = 0
        for e in events:
            if e.get("event") != EVENT_TOOL:
                continue
            if "facts" in e:
                facts_raw = e.get("facts")
                if not isinstance(facts_raw, list):
                    unresolved += 1
                    continue
                facts, g = _facts_with_authoritative_values(facts_raw, store_rows)
                if facts is None:
                    for note in g:
                        if note.startswith("exposure_value_unresolved:"):
                            try:
                                unresolved += int(note.split(":", 1)[1])
                            except ValueError:
                                unresolved += 1
                        else:
                            unresolved += 1
                    # Discard any partials already collected — whole field fails closed.
                    return None, (g if g else [f"exposure_value_unresolved:{max(unresolved, 1)}"])
                for ef in facts:
                    key = ef.snapshot_id or f"{ef.record_id}|{ef.field_path}"
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(ef)
                continue
            # Legacy: snapshot_ids only — join store for authoritative values.
            sids = e.get("snapshot_ids") or []
            if not sids:
                continue
            for sid in sids:
                sid_s = str(sid)
                if not sid_s or sid_s in seen:
                    continue
                row = by_id.get(sid_s)
                if row is None:
                    unresolved += 1
                    continue
                joined = _exposed_from_mapping(
                    {
                        "snapshot_id": sid_s,
                        "record_id": _get(row, "record_id"),
                        "field_path": _get(row, "field_path"),
                        "provenance": _prov(row),
                        "attempt_id": _get(row, "attempt_id"),
                        "value": _get(row, "value"),
                    },
                    value_present=True,
                )
                if joined is None:
                    unresolved += 1
                    continue
                seen.add(sid_s)
                out.append(joined)
        if unresolved:
            return None, [f"exposure_value_unresolved:{unresolved}"]
        return tuple(out), []

    @staticmethod
    def _attempt_follow_up_snapshots(
        store_rows: Sequence[Any],
        attempt_id: str,
    ) -> tuple[AttemptSnapshot, ...]:
        aid = (attempt_id or "").strip()
        out: list[AttemptSnapshot] = []
        for row in store_rows:
            prov = _prov(row)
            if prov != _FOLLOW_UP_PROVENANCE:
                continue
            row_aid = _get(row, "attempt_id")
            if (str(row_aid) if row_aid is not None else "") != aid:
                continue
            sid = _snap_id(row)
            if not sid:
                continue
            out.append(
                AttemptSnapshot(
                    snapshot_id=sid,
                    record_id=str(_get(row, "record_id") or ""),
                    field_path=str(_get(row, "field_path") or ""),
                    provenance=prov,
                    attempt_id=aid,
                    retrieval_ref=str(_get(row, "retrieval_ref") or ""),
                    value=_get(row, "value"),
                )
            )
        return tuple(out)

    @staticmethod
    def _parse_forbidden(events: Sequence[dict[str, Any]]) -> list[ForbiddenAttemptFact]:
        out: list[ForbiddenAttemptFact] = []
        for e in events:
            et = e.get("event")
            if et == EVENT_FORBIDDEN:
                kind = str(e.get("kind") or "write")
                out.append(
                    forbidden_attempt_fact(
                        kind,
                        detail=str(e.get("detail") or ""),
                        blocked=bool(e.get("blocked", False)),
                        case_id=_opt_str(e.get("case_id")),
                        run_id=_opt_str(e.get("run_id")),
                    )
                )
            elif et == EVENT_UNKNOWN_TOOL:
                name = str(e.get("name") or e.get("tool_name") or "unknown")
                out.append(
                    forbidden_attempt_fact(
                        "write",
                        detail=f"unknown_tool:{name}",
                        blocked=bool(e.get("blocked", True)),
                        case_id=_opt_str(e.get("case_id")),
                        run_id=_opt_str(e.get("run_id")),
                    )
                )
        return out

    @staticmethod
    def _dedupe_forbidden(
        items: Sequence[ForbiddenAttemptFact],
    ) -> list[ForbiddenAttemptFact]:
        seen: set[tuple[Any, ...]] = set()
        uniq: list[ForbiddenAttemptFact] = []
        for f in items:
            key = (f.kind, f.detail, f.blocked, f.case_id, f.run_id)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(f)
        return uniq

    @staticmethod
    def _parse_end_meta(
        events: Sequence[dict[str, Any]],
        *,
        end_events: tuple[str, ...],
    ) -> tuple[dict[str, Any] | None, float | None, str | None, list[str]]:
        """Extract usage/cost/provider from phase-end trace; gaps if missing."""
        gaps: list[str] = []
        matched = [e for e in events if e.get("event") in end_events]
        if not matched:
            gaps.append("phase_end_trace_missing")
            return None, None, None, gaps

        end = matched[-1]
        usage_raw = end.get("usage") if "usage" in end else None
        usage: dict[str, Any] | None
        if isinstance(usage_raw, dict):
            usage = usage_raw
        elif usage_raw is None:
            usage = None
        else:
            usage = None
            gaps.append("usage_malformed")

        cost: float | None
        if "cost_usd" not in end:
            cost = None
        else:
            raw_cost = end.get("cost_usd")
            if raw_cost is None:
                cost = None
            else:
                try:
                    cost = float(raw_cost)
                except TypeError, ValueError:
                    cost = None
                    gaps.append("cost_usd_malformed")

        provider: str | None
        if "provider_model" not in end:
            provider = None
        else:
            pm = end.get("provider_model")
            provider = str(pm) if pm is not None else None

        if "usage" not in end and "cost_usd" not in end:
            gaps.append("usage_cost_missing")
        elif usage is None and cost is None:
            gaps.append("usage_cost_absent")

        return usage, cost, provider, gaps


# ----- module-level helpers (emitters / runner helpers) ----------------------


def emit_pre_turn_context(
    trace_emit: Any,
    established: Sequence[Any],
    *,
    truncated: bool,
) -> None:
    """Emit post-filter pre-turn model context (established snapshots + values)."""
    facts = []
    for s in established:
        facts.append(
            {
                "snapshot_id": str(_get(s, "snapshot_id") or ""),
                "record_id": str(_get(s, "record_id") or ""),
                "field_path": str(_get(s, "field_path") or ""),
                "provenance": _prov(s),
                "attempt_id": _get(s, "attempt_id"),
                # Authoritative stored value actually present in model context.
                "value": _get(s, "value"),
            }
        )
    trace_emit(EVENT_PRE_TURN, facts=facts, truncated=bool(truncated))


def tool_exposure_facts(descriptors: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build trace ``facts`` payload from eager-persist tool descriptors + values."""
    out: list[dict[str, Any]] = []
    for d in descriptors:
        item: dict[str, Any] = {
            "snapshot_id": str(d.get("snapshot_id") or ""),
            "record_id": str(d.get("record_id") or ""),
            "field_path": str(d.get("field_path") or ""),
            "provenance": str(d.get("provenance") or _FOLLOW_UP_PROVENANCE),
            "attempt_id": d.get("attempt_id"),
        }
        # Always include value key when descriptor carries content (may be null).
        if "value" in d:
            item["value"] = d.get("value")
        out.append(item)
    return out


def capture_leaf_exposure_facts(
    captures: Sequence[Any],
    *,
    provenance: str = _ORIGINAL_PROVENANCE,
) -> list[dict[str, Any]]:
    """Leaf field facts + values from in-memory RecordCapture payloads."""
    from casepilot.backend.evidence import iter_leaf_fields

    out: list[dict[str, Any]] = []
    for cap in captures:
        record_id = str(_get(cap, "record_id") or "")
        payload = _get(cap, "payload") or {}
        if not isinstance(payload, dict):
            continue
        for field_path, value in iter_leaf_fields(payload):
            out.append(
                {
                    "record_id": record_id,
                    "field_path": str(field_path),
                    "provenance": provenance,
                    "snapshot_id": "",
                    "value": value,
                }
            )
    return out


def allowed_tool_name_set(tools: Sequence[Any], *extra: str) -> set[str]:
    """Collect non-empty tool names for unknown-tool scanning (mypy-safe)."""
    names: set[str] = set()
    for t in tools:
        n = getattr(t, "name", None)
        if not isinstance(n, str) or not n:
            n = getattr(t, "__name__", None)
        if isinstance(n, str) and n:
            names.add(n)
    for e in extra:
        if e:
            names.add(str(e))
    return names


def emit_forbidden_attempt(
    trace_emit: Any,
    *,
    kind: str,
    detail: str = "",
    blocked: bool = True,
    case_id: str | None = None,
    run_id: str | None = None,
    **extra: Any,
) -> None:
    """Emit a forbidden-attempt trace event (blocked or actual)."""
    payload: dict[str, Any] = {
        "kind": kind,
        "detail": detail,
        "blocked": blocked,
    }
    if case_id is not None:
        payload["case_id"] = case_id
    if run_id is not None:
        payload["run_id"] = run_id
    payload.update(extra)
    trace_emit(EVENT_FORBIDDEN, **payload)


def emit_unknown_tool_attempt(
    trace_emit: Any,
    *,
    name: str,
    case_id: str | None = None,
    run_id: str | None = None,
    blocked: bool = True,
) -> None:
    """Emit blocked unknown/non-allowlisted tool request."""
    payload: dict[str, Any] = {"name": name, "blocked": blocked}
    if case_id is not None:
        payload["case_id"] = case_id
    if run_id is not None:
        payload["run_id"] = run_id
    trace_emit(EVENT_UNKNOWN_TOOL, **payload)


def scan_unknown_tool_calls(
    messages: Sequence[Any] | None,
    allowed_names: Iterable[str],
    trace_emit: Any,
    *,
    case_id: str | None = None,
    run_id: str | None = None,
) -> list[str]:
    """Scan agent messages for tool calls outside ``allowed_names``; emit + return."""
    allowed = {str(n) for n in allowed_names if n is not None and str(n)}
    found: list[str] = []
    if not messages:
        return found
    for msg in messages:
        for name_s in _tool_call_names_from_message(msg):
            if name_s in allowed:
                continue
            if name_s in found:
                continue
            found.append(name_s)
            emit_unknown_tool_attempt(
                trace_emit,
                name=name_s,
                case_id=case_id,
                run_id=run_id,
                blocked=True,
            )
    return found


def _tool_call_names_from_message(msg: Any) -> list[str]:
    """Extract tool names only (never args / content) from one message."""
    tool_calls = getattr(msg, "tool_calls", None) or []
    if isinstance(msg, dict):
        tool_calls = msg.get("tool_calls") or tool_calls
    names: list[str] = []
    for tc in tool_calls or []:
        if isinstance(tc, dict):
            name = tc.get("name") or tc.get("tool")
        else:
            name = getattr(tc, "name", None)
        if name:
            names.append(str(name))
    return names


def _last_tool_call_batch_names(messages: Sequence[Any] | None) -> list[str]:
    """Names from the latest message that requested tools (current batch)."""
    if not messages:
        return []
    for msg in reversed(list(messages)):
        names = _tool_call_names_from_message(msg)
        if names:
            return names
    return []


def _is_tool_call_limit_error(error: Any) -> bool:
    """True when ToolCallLimit middleware aborted (exit_behavior='error')."""
    if error is None:
        return False
    if type(error).__name__ == "ToolCallLimitExceededError":
        return True
    # Duck-type langchain ToolCallLimitExceededError without importing it.
    if all(hasattr(error, a) for a in ("run_count", "run_limit", "thread_count", "thread_limit")):
        return True
    msg = str(error).lower()
    return "tool call limit" in msg or (
        "call limit" in msg and ("exceeded" in msg or "reached" in msg)
    )


def _trace_emit(trace: Any) -> Any:
    """Accept TraceWriter (``.emit``) or a bare emit callable."""
    emit = getattr(trace, "emit", None)
    if callable(emit):
        return emit
    if callable(trace):
        return trace
    raise TypeError("trace must be TraceWriter-like or an emit callable")


def follow_up_output_refs(output: Any) -> dict[str, Any]:
    """Schema-safe untrusted refs from ``FollowUpAgentOutput`` (no text/values/CoT).

    - source_facts: local_id / snapshot_id / field_path only
    - interpretations: kind + cites ids only
    - notes: allowlist codes only (free-text dropped)
    """
    from casepilot.backend.evidence import SERVER_NOTES_BY_CODE

    source_facts: list[dict[str, str]] = []
    interpretations: list[dict[str, Any]] = []
    notes: list[str] = []
    if output is None:
        return {"source_facts": source_facts, "interpretations": interpretations, "notes": notes}

    for claim in getattr(output, "source_facts", None) or []:
        if isinstance(claim, dict):
            local_id = str(claim.get("local_id") or "")
            snapshot_id = str(claim.get("snapshot_id") or "")
            field_path = str(claim.get("field_path") or "")
        else:
            local_id = str(getattr(claim, "local_id", "") or "")
            snapshot_id = str(getattr(claim, "snapshot_id", "") or "")
            field_path = str(getattr(claim, "field_path", "") or "")
        source_facts.append(
            {"local_id": local_id, "snapshot_id": snapshot_id, "field_path": field_path}
        )

    for block in getattr(output, "interpretations", None) or []:
        if isinstance(block, dict):
            kind_raw = block.get("kind")
            cites_raw = block.get("cites") or []
        else:
            kind_raw = getattr(block, "kind", None)
            cites_raw = getattr(block, "cites", None) or []
        kind = str(getattr(kind_raw, "value", kind_raw) or "")
        cites = [str(c) for c in cites_raw if c is not None and str(c)]
        interpretations.append({"kind": kind, "cites": cites})

    code_by_text = {text: code for code, text in SERVER_NOTES_BY_CODE.items()}
    seen_notes: set[str] = set()
    for raw in getattr(output, "notes", None) or []:
        key = str(raw).strip()
        if not key:
            continue
        code = key if key in SERVER_NOTES_BY_CODE else code_by_text.get(key)
        if code is None or code in seen_notes:
            continue
        seen_notes.add(code)
        notes.append(code)

    return {"source_facts": source_facts, "interpretations": interpretations, "notes": notes}


def emit_follow_up_output(
    trace_emit: Any,
    output: Any,
    *,
    ok: bool | None = None,
    reject_code: str | None = None,
    case_id: str | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Emit one ``follow_up_output`` event: untrusted refs + optional reject code.

    Call once after structured output is available and rendering has produced
    ``ok`` / ``reject_code``. Does not alter answer safety or model context.
    """
    refs = follow_up_output_refs(output)
    payload: dict[str, Any] = {
        "label": UNTRUSTED_MODEL_OUTPUT_LABEL,
        **refs,
    }
    if ok is not None:
        payload["ok"] = bool(ok)
    if reject_code is not None:
        payload["reject_code"] = str(reject_code)
    if case_id is not None:
        payload["case_id"] = case_id
    if run_id is not None:
        payload["run_id"] = run_id
    if attempt_id is not None:
        payload["attempt_id"] = attempt_id
    trace_emit(EVENT_FOLLOW_UP_OUTPUT, **payload)
    return payload


def model_turn_batches(messages: Sequence[Any] | None) -> list[dict[str, Any]]:
    """Per-AIMessage tool-call batches: names + count only (no args/content/CoT)."""
    batches: list[dict[str, Any]] = []
    if not messages:
        return batches
    for msg in messages:
        names = _tool_call_names_from_message(msg)
        if names:
            batches.append({"count": len(names), "names": list(names)})
    return batches


def capture_model_turn_batches(
    trace: Any,
    messages: Sequence[Any] | None,
    *,
    case_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate end-of-run AIMessage tool batches (incl. structured-output tools).

    Emits at most one ``model_turn_batches`` event. Does **not** scan unknown
    tools (callers own a single ``scan_unknown_tool_calls`` to avoid duplicates).
    """
    emit = _trace_emit(trace)
    batches = model_turn_batches(messages)
    if batches:
        payload: dict[str, Any] = {
            "turn_count": len(batches),
            "batches": batches,
        }
        if case_id is not None:
            payload["case_id"] = case_id
        if run_id is not None:
            payload["run_id"] = run_id
        emit(EVENT_MODEL_TURN_BATCHES, **payload)
    return {
        "turn_count": len(batches),
        "batches": list(batches),
    }


def resolve_ai_messages(
    messages: Sequence[Any] | None,
    tracker: Any | None = None,
) -> list[Any]:
    """Prefer full agent messages; fall back to tracker-captured AI turns."""
    if messages:
        return list(messages)
    captured = getattr(tracker, "captured_ai_messages", None) if tracker is not None else None
    if captured:
        return list(captured)
    return []


def capture_tool_attempts_on_error(
    trace: Any,
    messages: Sequence[Any] | None,
    case_id: str | None,
    allowed_tool_names: Iterable[str],
    error: Any,
    *,
    run_id: str | None = None,
    scan_unknown: bool = True,
) -> dict[str, Any]:
    """Record attempted tool-call batch on runtime exception; scan unknowns.

    Call from agent/follow-up ``except`` paths when ``raw`` is None but
    ``tracker.captured_ai_messages`` still holds partial model turns.

    Emits at most one ``tool_attempt_batch`` for the **latest** tool-bearing
    message (the current batch). Payload is names + count only — no tool args,
    content, or chain-of-thought.

    **ToolCallLimitMiddleware** with ``exit_behavior="error"`` raises
    ``ToolCallLimitExceededError`` in ``after_model`` *before* any tool in that
    AIMessage runs. The **entire** current batch is aborted (including members
    that would still have been under the remaining quota). That is the only
    case we label ``execution_status="not_executed"``. We do **not** invent
    per-tool unexecuted/blocked/executed events for batch members.

    Any other error → ``execution_status="unknown"`` (no proof of run or block).

    Always runs ``scan_unknown_tool_calls`` on ``messages`` (unless
    ``scan_unknown=False``) so an unknown tool inside a limiter-aborted batch is
    still observed as ``unknown_tool_attempt``. Callers that already scanned
    should pass ``scan_unknown=False`` to avoid duplicate unknown-tool records.

    Returns:
        Summary dict: ``count``, ``names``, ``execution_status``, ``unknown_tools``,
        and ``limiter_error`` (str | None).
    """
    emit = _trace_emit(trace)
    names = _last_tool_call_batch_names(messages)
    limit_hit = _is_tool_call_limit_error(error)
    status = EXEC_STATUS_NOT_EXECUTED if limit_hit else EXEC_STATUS_UNKNOWN
    limiter_error = str(error) if limit_hit and error is not None else None

    if names:
        payload: dict[str, Any] = {
            "count": len(names),
            "names": list(names),
            "execution_status": status,
        }
        if limiter_error is not None:
            payload["limiter_error"] = limiter_error
        if case_id is not None:
            payload["case_id"] = case_id
        if run_id is not None:
            payload["run_id"] = run_id
        emit(EVENT_TOOL_ATTEMPT_BATCH, **payload)

    unknown: list[str] = []
    if scan_unknown:
        unknown = scan_unknown_tool_calls(
            messages,
            allowed_tool_names,
            emit,
            case_id=case_id,
            run_id=run_id,
        )
    return {
        "count": len(names),
        "names": list(names),
        "execution_status": status if names else EXEC_STATUS_UNKNOWN,
        "unknown_tools": list(unknown),
        "limiter_error": limiter_error,
    }


def proposal_facts_from_audit(
    audit: WriteAuditSnapshot | None,
) -> tuple[ProposalStateFact, ...] | None:
    """Helper for runners: None when audit missing/unusable."""
    if audit is None or not audit.captured:
        return None
    return collect_proposal_states(list(audit.proposals))


def task_facts_from_audit(
    audit: WriteAuditSnapshot | None,
) -> tuple[TaskStateFact, ...] | None:
    if audit is None or not audit.captured:
        return None
    return audit.tasks


# ----- private utils --------------------------------------------------------


def _exposure_instrumented(events: Sequence[dict[str, Any]]) -> bool:
    if any(e.get("event") == EVENT_TOOL and "facts" in e for e in events):
        return True
    for e in reversed(list(events)):
        if e.get("event") in (EVENT_RUN_END, EVENT_FOLLOW_UP_END):
            if e.get("exposure_instrumented"):
                return True
            break
    return False


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _snap_id(obj: Any) -> str:
    return str(_get(obj, "snapshot_id") or "")


def _prov(obj: Any) -> str:
    v = _get(obj, "provenance")
    if v is None:
        return ""
    return str(getattr(v, "value", v))


def _opt_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v)
    return s if s else None


def _store_indexes(
    store_rows: Sequence[Any],
) -> tuple[dict[str, Any], dict[tuple[str, str], Any]]:
    by_id: dict[str, Any] = {}
    by_key: dict[tuple[str, str], Any] = {}
    for s in store_rows:
        sid = _snap_id(s)
        if sid:
            by_id[sid] = s
        rid = str(_get(s, "record_id") or "").strip()
        fp = str(_get(s, "field_path") or "").strip()
        if rid and fp:
            by_key.setdefault((rid, fp), s)
    return by_id, by_key


def _facts_with_authoritative_values(
    facts_raw: Sequence[Any],
    store_rows: Sequence[Any],
) -> tuple[list[ExposedFact] | None, list[str]]:
    """Build ExposedFacts with real values; fail closed on any unknown item.

    Value resolution order:
      1. ``value`` key present on the fact payload (including JSON null)
      2. store join by snapshot_id
      3. store join by (record_id, field_path)

    Empty ``facts_raw`` → ``[]`` (known zero). Any unresolved/invalid item →
    ``None`` for the whole set (same policy as snapshot provenance), with a
    diagnostic gap. Never return a partial list that looks complete downstream.
    """
    by_id, by_key = _store_indexes(store_rows)
    out: list[ExposedFact] = []
    unresolved = 0
    for item in facts_raw:
        if not isinstance(item, dict):
            unresolved += 1
            continue
        value_present = "value" in item
        value = item.get("value") if value_present else None
        if not value_present:
            sid = str(item.get("snapshot_id") or "").strip()
            row = by_id.get(sid) if sid else None
            if row is None:
                rid = str(item.get("record_id") or "").strip()
                fp = str(item.get("field_path") or "").strip()
                row = by_key.get((rid, fp)) if rid and fp else None
            if row is not None:
                value = _get(row, "value")
                value_present = True
                # Fill missing ids from the authoritative row when absent.
                if not sid:
                    item = {
                        **item,
                        "snapshot_id": _snap_id(row),
                        "record_id": item.get("record_id") or _get(row, "record_id"),
                        "field_path": item.get("field_path") or _get(row, "field_path"),
                        "provenance": item.get("provenance") or _prov(row),
                        "attempt_id": item.get("attempt_id") or _get(row, "attempt_id"),
                    }
        if not value_present:
            unresolved += 1
            continue
        ef = _exposed_from_mapping({**item, "value": value}, value_present=True)
        if ef is None:
            unresolved += 1
            continue
        out.append(ef)
    gaps: list[str] = []
    if unresolved:
        gaps.append(f"exposure_value_unresolved:{unresolved}")
        return None, gaps
    return out, gaps


def _exposed_from_mapping(
    item: Any,
    *,
    value_present: bool | None = None,
) -> ExposedFact | None:
    if not isinstance(item, dict):
        return None
    record_id = str(item.get("record_id") or "").strip()
    field_path = str(item.get("field_path") or "").strip()
    if not record_id and not field_path and not item.get("snapshot_id"):
        return None
    present = ("value" in item) if value_present is None else value_present
    return ExposedFact(
        record_id=record_id,
        field_path=field_path,
        snapshot_id=str(item.get("snapshot_id") or ""),
        provenance=str(item.get("provenance") or ""),
        attempt_id=_opt_str(item.get("attempt_id")),
        value=item.get("value") if present else None,
    )
