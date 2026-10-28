"""Backend approval workflow at the real-app boundary (applied redesign).

Real backend + SimulatorClient + real simulator ASGI + isolated SQLite.
Scripted model only (external model boundary). No paid calls; no .env credentials.

Historically applied map (6 mock-flow unit nodes → these 4 boundary tests; net −2):
  - tests/unit/test_backend.py::test_api_investigate_get_approve_reject_flow
  - tests/unit/test_backend.py::test_reject_is_terminal
  - tests/unit/test_backend.py::test_execution_failed_retry
  - tests/unit/test_backend.py::test_proposal_get
  - tests/unit/test_backend.py::test_case_proposal_discoverability_api
  - tests/unit/test_backend.py::test_invalid_evidence_no_write_proposal

Sibling unit files still own crash recovery, task_id, attempt denial, budget,
redaction, simulator contracts, two-process smoke, and follow-up beyond create-on-approve.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from casepilot.backend.app import create_app as create_backend_app
from casepilot.backend.client import SimulatorClient
from casepilot.backend.db import BackendStore
from casepilot.backend.hooks import TEST_MODE_ENV, clear_test_hooks
from casepilot.backend.scripted import ScriptedChatModel
from casepilot.shared.config import get_settings
from casepilot.shared.contracts import (
    BackendRoutes,
    Diagnosis,
    Evidence,
    NextAction,
    ProposalStatus,
    RunStatus,
)
from casepilot.simulator.app import clear_outage, set_outage
from casepilot.simulator.app import create_app as create_simulator_app
from casepilot.simulator.seed import seed

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from tests.helpers import _finding, _tool_then_finding  # noqa: E402

CASE_ID = "CASE-001"
TOKEN_SIM = "boundary-sim-token"
TOKEN_OP = "boundary-op-token"


def _settings(tmp_path: Path, **extra: object):
    base: dict[str, object] = {
        "_env_file": None,
        "simulator_service_token": TOKEN_SIM,
        "operator_token": TOKEN_OP,
        "backend_db_path": tmp_path / "backend.sqlite3",
        "simulator_db_path": tmp_path / "simulator.sqlite3",
        "trace_dir": tmp_path / "traces",
        "simulator_base_url": "http://simulator.test",
        "openrouter_api_key": None,
        "investigation_timeout_seconds": 30.0,
        "max_tool_calls": 12,
        "max_model_turns": 8,
        "tracing_enabled": False,
    }
    base.update(extra)
    return get_settings(**base)


def _sim_task_rows(db_path: Path, proposal_id: str | None = None) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if proposal_id is None:
            return list(conn.execute("SELECT * FROM follow_up_tasks").fetchall())
        return list(
            conn.execute(
                "SELECT * FROM follow_up_tasks WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchall()
        )
    finally:
        conn.close()


@dataclass
class WorkflowStack:
    """Independent in-process backend↔simulator stack for one scenario."""

    backend_app: object
    store: BackendStore
    sim_http: httpx.AsyncClient
    sim_db: Path
    finding: object


@asynccontextmanager
async def _workflow_stack(
    tmp_path: Path,
    *,
    finding=None,
) -> AsyncIterator[WorkflowStack]:
    """Seeded real simulator ASGI + backend with scripted model; isolated DBs."""
    sim_db = tmp_path / "simulator.sqlite3"
    assert seed(db_path=sim_db, reset=True) == 0

    settings = _settings(tmp_path)
    sim_app = create_simulator_app(settings=settings, db_path=sim_db)
    # Eager state: ASGITransport has no lifespan in httpx 0.28.
    sim_app.state.service_token = TOKEN_SIM
    sim_app.state.db_path = sim_db

    sim_http = httpx.AsyncClient(
        transport=ASGITransport(app=sim_app),
        base_url="http://simulator.test",
        headers={"Authorization": f"Bearer {TOKEN_SIM}"},
    )
    client = SimulatorClient(settings, client=sim_http)
    store = BackendStore(settings.backend_db_path)
    resolved = finding if finding is not None else _finding(CASE_ID)

    def factory(_s):
        return ScriptedChatModel(responses=_tool_then_finding(resolved))

    backend_app = create_backend_app(settings, model_factory=factory, store=store, client=client)
    try:
        yield WorkflowStack(
            backend_app=backend_app,
            store=store,
            sim_http=sim_http,
            sim_db=sim_db,
            finding=resolved,
        )
    finally:
        store.close()
        await sim_http.aclose()


@pytest.fixture(autouse=True)
def _clean_hooks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(TEST_MODE_ENV, "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    clear_test_hooks()
    clear_outage()
    yield
    clear_test_hooks()
    clear_outage()


@pytest.mark.asyncio
async def test_investigate_approve_persists_follow_up_idempotently(tmp_path: Path):
    """Investigate → unauthorized blocked → approve executes once → sim row durable."""
    async with _workflow_stack(tmp_path) as stack:
        finding = stack.finding
        assert finding.proposed_task is not None
        async with AsyncClient(
            transport=ASGITransport(app=stack.backend_app),
            base_url="http://backend.test",
        ) as ac:
            health = await ac.get(BackendRoutes.HEALTH)
            assert health.status_code == 200

            inv = await ac.post(
                BackendRoutes.INVESTIGATIONS,
                json={"case_id": CASE_ID, "objective": "Why stalled?"},
            )
            assert inv.status_code == 200, inv.text
            body = inv.json()
            run_id = body["run_id"]
            result = body["result"]
            assert result["status"] == RunStatus.COMPLETED.value
            assert result["approval_required"] is True
            pid = result["proposal_id"]
            assert pid
            # Retained finding/evidence (scripted AgentFinding survives validation)
            assert result["diagnosis"] == Diagnosis.MISSING_DOCUMENT.value
            assert result["diagnosis_summary"] == finding.diagnosis_summary
            assert result["next_action"] == NextAction.CREATE_FOLLOW_UP_TASK.value
            assert result["proposed_task"] == finding.proposed_task.model_dump()
            evidence = result["evidence"]
            assert len(evidence) == len(finding.evidence) == 2
            assert evidence == [e.model_dump() for e in finding.evidence]
            assert {e["record_id"] for e in evidence} == {CASE_ID, "DOC-001-A"}
            assert {e["retrieval_ref"] for e in evidence} == {"get_case", "get_documents"}

            run = await ac.get(BackendRoutes.RUN.format(run_id=run_id))
            assert run.status_code == 200
            assert run.json()["status"] == RunStatus.COMPLETED.value
            assert run.json()["result"]["proposal_id"] == pid
            assert run.json()["result"]["evidence"] == evidence

            prop = await ac.get(BackendRoutes.PROPOSAL.format(proposal_id=pid))
            assert prop.status_code == 200
            assert prop.json()["status"] == ProposalStatus.PENDING.value
            assert prop.json()["task_payload"] == finding.proposed_task.model_dump()
            missing = await ac.get(BackendRoutes.PROPOSAL.format(proposal_id="nope"))
            assert missing.status_code == 404

            listed = await ac.get(BackendRoutes.CASE_PROPOSALS.format(case_id=CASE_ID))
            assert listed.status_code == 200
            rows = listed.json()
            assert len(rows) == 1
            assert rows[0]["proposal_id"] == pid
            assert rows[0]["case_id"] == CASE_ID
            empty = await ac.get(BackendRoutes.CASE_PROPOSALS.format(case_id="CASE-NONE"))
            assert empty.status_code == 200
            assert empty.json() == []

            # No execution before authorization (approve + reject gates)
            assert _sim_task_rows(stack.sim_db, pid) == []
            assert (
                await ac.post(
                    BackendRoutes.APPROVE.format(proposal_id=pid),
                    json={"operator_identity": "alice"},
                )
            ).status_code == 401
            assert (
                await ac.post(
                    BackendRoutes.APPROVE.format(proposal_id=pid),
                    json={"operator_identity": "alice"},
                    headers={"Authorization": "Bearer wrong"},
                )
            ).status_code == 403
            assert (
                await ac.post(
                    BackendRoutes.REJECT.format(proposal_id=pid),
                    json={"operator_identity": "x"},
                )
            ).status_code == 401
            assert (
                await ac.post(
                    BackendRoutes.REJECT.format(proposal_id=pid),
                    json={"operator_identity": "x"},
                    headers={"Authorization": "Bearer wrong"},
                )
            ).status_code == 403
            assert _sim_task_rows(stack.sim_db, pid) == []
            still = await ac.get(BackendRoutes.PROPOSAL.format(proposal_id=pid))
            assert still.json()["status"] == ProposalStatus.PENDING.value

            ok = await ac.post(
                BackendRoutes.APPROVE.format(proposal_id=pid),
                json={"operator_identity": "alice"},
                headers={"Authorization": f"Bearer {TOKEN_OP}"},
            )
            assert ok.status_code == 200, ok.text
            executed = ok.json()["proposal"]
            assert executed["status"] == ProposalStatus.EXECUTED.value
            assert executed["operator_identity"] == "alice"
            assert executed["task_payload"] == finding.proposed_task.model_dump()
            task_id = executed["task_id"]
            assert task_id  # simulator-assigned identity, persisted on proposal

            sim_rows = _sim_task_rows(stack.sim_db, pid)
            assert len(sim_rows) == 1
            assert sim_rows[0]["task_id"] == task_id
            assert sim_rows[0]["case_id"] == CASE_ID
            assert sim_rows[0]["title"] == finding.proposed_task.title
            assert sim_rows[0]["body"] == finding.proposed_task.body

            # Repeated approval cannot duplicate side effects
            again = await ac.post(
                BackendRoutes.APPROVE.format(proposal_id=pid),
                json={"operator_identity": "alice"},
                headers={"Authorization": f"Bearer {TOKEN_OP}"},
            )
            assert again.status_code == 200
            assert again.json()["proposal"]["status"] == ProposalStatus.EXECUTED.value
            assert again.json()["proposal"]["task_id"] == task_id
            assert len(_sim_task_rows(stack.sim_db, pid)) == 1
            assert len(_sim_task_rows(stack.sim_db)) == 1


@pytest.mark.asyncio
async def test_reject_is_terminal_without_simulator_side_effects(tmp_path: Path):
    """Rejection ends the proposal; later approve is 409 and sim stays empty."""
    async with _workflow_stack(tmp_path) as stack:
        async with AsyncClient(
            transport=ASGITransport(app=stack.backend_app),
            base_url="http://backend.test",
        ) as ac:
            inv = await ac.post(
                BackendRoutes.INVESTIGATIONS,
                json={"case_id": CASE_ID, "objective": "x"},
            )
            assert inv.status_code == 200, inv.text
            pid = inv.json()["result"]["proposal_id"]
            assert pid

            assert (
                await ac.post(
                    BackendRoutes.REJECT.format(proposal_id=pid),
                    json={"operator_identity": "bob", "reason": "nope"},
                )
            ).status_code == 401

            rej = await ac.post(
                BackendRoutes.REJECT.format(proposal_id=pid),
                json={"operator_identity": "bob", "reason": "nope"},
                headers={"Authorization": f"Bearer {TOKEN_OP}"},
            )
            assert rej.status_code == 200, rej.text
            assert rej.json()["proposal"]["status"] == ProposalStatus.REJECTED.value

            appr = await ac.post(
                BackendRoutes.APPROVE.format(proposal_id=pid),
                json={"operator_identity": "bob"},
                headers={"Authorization": f"Bearer {TOKEN_OP}"},
            )
            assert appr.status_code == 409
            assert _sim_task_rows(stack.sim_db, pid) == []
            assert _sim_task_rows(stack.sim_db) == []


@pytest.mark.asyncio
async def test_simulator_outage_fails_execution_then_retry_persists(tmp_path: Path):
    """Real simulator outage → execution_failed; recovery executes intended task once."""
    async with _workflow_stack(tmp_path) as stack:
        finding = stack.finding
        assert finding.proposed_task is not None
        async with AsyncClient(
            transport=ASGITransport(app=stack.backend_app),
            base_url="http://backend.test",
        ) as ac:
            inv = await ac.post(
                BackendRoutes.INVESTIGATIONS,
                json={"case_id": CASE_ID, "objective": "x"},
            )
            assert inv.status_code == 200, inv.text
            pid = inv.json()["result"]["proposal_id"]
            assert pid

            set_outage(True)
            try:
                fail = await ac.post(
                    BackendRoutes.APPROVE.format(proposal_id=pid),
                    json={"operator_identity": "ops"},
                    headers={"Authorization": f"Bearer {TOKEN_OP}"},
                )
                assert fail.status_code == 200, fail.text
                failed = fail.json()["proposal"]
                assert failed["status"] == ProposalStatus.EXECUTION_FAILED.value
                assert failed["execution_error"]
                assert failed.get("task_id") in (None, "")
                assert _sim_task_rows(stack.sim_db, pid) == []
            finally:
                set_outage(False)

            retry = await ac.post(
                BackendRoutes.APPROVE.format(proposal_id=pid),
                json={"operator_identity": "ops"},
                headers={"Authorization": f"Bearer {TOKEN_OP}"},
            )
            assert retry.status_code == 200, retry.text
            executed = retry.json()["proposal"]
            assert executed["status"] == ProposalStatus.EXECUTED.value
            assert executed["task_payload"] == finding.proposed_task.model_dump()
            task_id = executed["task_id"]
            assert task_id

            rows = _sim_task_rows(stack.sim_db, pid)
            assert len(rows) == 1
            assert rows[0]["task_id"] == task_id
            assert rows[0]["title"] == finding.proposed_task.title

            again = await ac.post(
                BackendRoutes.APPROVE.format(proposal_id=pid),
                json={"operator_identity": "ops"},
                headers={"Authorization": f"Bearer {TOKEN_OP}"},
            )
            assert again.status_code == 200
            assert again.json()["proposal"]["task_id"] == task_id
            assert len(_sim_task_rows(stack.sim_db, pid)) == 1


@pytest.mark.asyncio
async def test_invalid_evidence_writes_neither_proposal_nor_simulator_task(
    tmp_path: Path,
):
    """Fabricated evidence cannot create a proposal or simulator follow-up."""
    forged = _finding(
        CASE_ID,
        evidence=[
            Evidence(
                source_system="documents",
                record_id="DOC-FORGED",
                field="status",
                fact="missing",
                retrieval_ref="lie",
            )
        ],
    )
    async with _workflow_stack(tmp_path, finding=forged) as stack:
        async with AsyncClient(
            transport=ASGITransport(app=stack.backend_app),
            base_url="http://backend.test",
        ) as ac:
            inv = await ac.post(
                BackendRoutes.INVESTIGATIONS,
                json={"case_id": CASE_ID, "objective": "x"},
            )
            assert inv.status_code == 200, inv.text
            body = inv.json()
            result = body["result"]
            assert result["status"] == "indeterminate"
            assert result["evidence"] == []
            assert result["approval_required"] is False
            assert result["proposed_task"] is None
            assert result["proposal_id"] is None
            assert result["next_action"] == NextAction.ESCALATE_TO_HUMAN.value

            listed = await ac.get(BackendRoutes.CASE_PROPOSALS.format(case_id=CASE_ID))
            assert listed.status_code == 200
            assert listed.json() == []
            assert stack.store.get_proposal_by_run(body["run_id"]) is None
            assert _sim_task_rows(stack.sim_db) == []
