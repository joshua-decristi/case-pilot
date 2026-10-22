"""Unit tests for investigation backend (scripted model, no paid calls)."""

from __future__ import annotations

import json
import sys
import tempfile
from functools import cache
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from casepilot.backend.agent import payload_digest, run_investigation
from casepilot.backend.app import create_app
from casepilot.backend.client import SimulatorClient
from casepilot.backend.db import BackendStore
from casepilot.backend.evidence import RetrievedIndex, validate_evidence
from casepilot.backend.hooks import (
    TEST_MODE_ENV,
    clear_test_hooks,
    install_scripted_model,
    is_test_mode,
)
from casepilot.backend.scripted import ScriptedChatModel
from casepilot.backend.trace import TraceWriter
from casepilot.shared.config import get_settings
from casepilot.shared.contracts import (
    BackendRoutes,
    CaseRecord,
    CreateFollowUpTaskRequest,
    CreateFollowUpTaskResponse,
    Diagnosis,
    DocumentListResponse,
    EventListResponse,
    Evidence,
    FollowUpTaskPayload,
    NextAction,
    ProposalStatus,
    RunStatus,
    SimulatorRoutes,
    StartInvestigationRequest,
)
from casepilot.simulator.db import connect
from casepilot.simulator.seed import seed
from casepilot.simulator.store import SimulatorStore

# Repo root not on tool.pytest pythonpath (src only); enable tests.helpers for unit+e2e.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from tests.helpers import _finding, _tool_then_finding  # noqa: E402

CASE_ID = "CASE-001"
TOKEN_SIM = "test-sim-token"
TOKEN_OP = "test-op-token"


def _settings(tmp_path: Path, **extra: object):
    base: dict[str, object] = {
        "_env_file": None,
        "simulator_service_token": TOKEN_SIM,
        "operator_token": TOKEN_OP,
        "backend_db_path": tmp_path / "backend.sqlite3",
        "trace_dir": tmp_path / "traces",
        "simulator_base_url": "http://simulator.test",
        "openrouter_api_key": None,
        "investigation_timeout_seconds": 30.0,
        "max_tool_calls": 12,
        "max_model_turns": 8,
    }
    base.update(extra)
    return get_settings(**base)


@cache
def _case001_bundle() -> tuple[CaseRecord, DocumentListResponse, EventListResponse]:
    """CASE-001 contracts via public seed + SimulatorStore (no fixture field mapping)."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "sim.sqlite3"
        assert seed(db_path=db_path, reset=True) == 0
        conn = connect(db_path)
        try:
            store = SimulatorStore(conn)
            case = store.get_case(CASE_ID)
            assert case is not None
            docs = store.list_documents(CASE_ID)
            events = store.list_events(CASE_ID)
            return (
                case,
                DocumentListResponse(case_id=CASE_ID, documents=docs),
                EventListResponse(case_id=CASE_ID, events=events),
            )
        finally:
            conn.close()


def _case() -> CaseRecord:
    return _case001_bundle()[0]


def _docs() -> DocumentListResponse:
    return _case001_bundle()[1]


def _events() -> EventListResponse:
    return _case001_bundle()[2]


def _case001_index() -> RetrievedIndex:
    case, docs, events = _case001_bundle()
    idx = RetrievedIndex()
    idx.note_case(case)
    idx.note_documents(docs.documents)
    idx.note_events(events.events)
    return idx


def _mock_simulator(*, fail_write: bool = False, conflict: bool = False) -> httpx.AsyncClient:
    case, docs, events = _case001_bundle()
    tasks: dict[str, CreateFollowUpTaskRequest] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {TOKEN_SIM}":
            return httpx.Response(401, json={"detail": "unauthorized"})
        path = request.url.path
        if path == SimulatorRoutes.CASE.format(case_id=CASE_ID):
            return httpx.Response(200, json=case.model_dump(mode="json"))
        if path == SimulatorRoutes.DOCUMENTS.format(case_id=CASE_ID):
            return httpx.Response(200, json=docs.model_dump(mode="json"))
        if path == SimulatorRoutes.EVENTS.format(case_id=CASE_ID):
            return httpx.Response(200, json=events.model_dump(mode="json"))
        if path == SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id=CASE_ID):
            body = CreateFollowUpTaskRequest.model_validate_json(request.content)
            if fail_write:
                return httpx.Response(503, json={"detail": "down"})
            if body.proposal_id in tasks:
                prev = tasks[body.proposal_id]
                if prev.task != body.task or prev.proposal_id != body.proposal_id:
                    return httpx.Response(409, json={"detail": "conflict"})
                if conflict:
                    return httpx.Response(409, json={"detail": "conflict"})
                return httpx.Response(
                    200,
                    json=CreateFollowUpTaskResponse(
                        task_id="TASK-EXISTING",
                        proposal_id=body.proposal_id,
                        case_id=CASE_ID,
                        created=False,
                    ).model_dump(mode="json"),
                )
            tasks[body.proposal_id] = body
            return httpx.Response(
                200,
                json=CreateFollowUpTaskResponse(
                    task_id="TASK-1",
                    proposal_id=body.proposal_id,
                    case_id=CASE_ID,
                    created=True,
                ).model_dump(mode="json"),
            )
        if path.startswith("/v1/cases/"):
            return httpx.Response(404, json={"detail": "missing"})
        return httpx.Response(404, json={"detail": "no route"})

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(
        transport=transport,
        base_url="http://simulator.test",
        headers={"Authorization": f"Bearer {TOKEN_SIM}"},
    )


@pytest.fixture(autouse=True)
def _clean_hooks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(TEST_MODE_ENV, "1")
    clear_test_hooks()
    yield
    clear_test_hooks()


# ---------------------------------------------------------------------------
# hooks / evidence / digest
# ---------------------------------------------------------------------------


def test_scripted_install_requires_test_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(TEST_MODE_ENV, raising=False)
    assert not is_test_mode()
    with pytest.raises(RuntimeError, match=TEST_MODE_ENV):
        install_scripted_model([AIMessage(content="x")])


def test_evidence_rejects_unknown_record():
    idx = RetrievedIndex()
    idx.note_documents(_docs().documents)
    ok, errs = validate_evidence(
        [
            Evidence(
                source_system="documents",
                record_id="DOC-FAKE",
                field="status",
                fact="x",
                retrieval_ref="get_documents",
            )
        ],
        idx,
    )
    assert ok == []
    assert errs and "DOC-FAKE" in errs[0]


def test_evidence_accepts_retrieved_doc():
    idx = RetrievedIndex()
    idx.note_documents(_docs().documents)
    ok, errs = validate_evidence(
        [
            Evidence(
                source_system="documents",
                record_id="DOC-001-A",
                field="status",
                fact="missing",
                retrieval_ref="get_documents",
            )
        ],
        idx,
    )
    assert len(ok) == 1 and not errs


def test_evidence_case001_fixture_good_paths():
    """Regression: CASE-001 fixture nested/top-level paths accept via validate_evidence."""
    idx = _case001_index()
    good = [
        Evidence(
            source_system="cases",
            record_id=CASE_ID,
            field="required_documents",
            fact="includes proof_of_income",
            retrieval_ref="get_case",
        ),
        Evidence(
            source_system="documents",
            record_id="DOC-001-A",
            field="status",
            fact="received",
            retrieval_ref="get_documents",
        ),
        Evidence(
            source_system="documents",
            record_id="DOC-001-A",
            field="content_summary",
            fact="license on file",
            retrieval_ref="get_documents",
        ),
        Evidence(
            source_system="documents",
            record_id="DOC-001-B",
            field="status",
            fact="received",
            retrieval_ref="get_documents",
        ),
        Evidence(
            source_system="documents",
            record_id="DOC-001-B",
            field="doc_type",
            fact="beneficiary_form",
            retrieval_ref="get_documents",
        ),
        Evidence(
            source_system="events",
            record_id="EVT-001-A",
            field="details.missing",
            fact="proof_of_income missing from checklist",
            retrieval_ref="get_events",
        ),
        Evidence(
            source_system="events",
            record_id="EVT-001-B",
            field="details.reason",
            fact="required_document_absent",
            retrieval_ref="get_events",
        ),
        Evidence(
            source_system="events",
            record_id="EVT-001-B",
            field="details.document_type",
            fact="proof_of_income",
            retrieval_ref="get_events",
        ),
        Evidence(
            source_system="documents",
            record_id="DOC-001-A",
            field="fields.name",
            fact="Maria Chen",
            retrieval_ref="get_documents",
        ),
        Evidence(
            source_system="cases",
            record_id=CASE_ID,
            field="application_fields.applicant_name",
            fact="Maria Chen",
            retrieval_ref="get_case",
        ),
    ]
    ok, errs = validate_evidence(good, idx)
    assert not errs, errs
    assert len(ok) == len(good)


@pytest.mark.parametrize(
    ("source_system", "record_id", "field", "fact", "retrieval_ref", "err_substr"),
    [
        (
            "documents",
            "DOC-001-A",
            "status/content_summary",
            "received",
            "get_documents",
            "status/content_summary",
        ),
        (
            "events",
            "EVT-001-A",
            "details.not_a_real_key",
            "x",
            "get_events",
            None,
        ),
        (
            "events",
            "EVT-001-B",
            "reason",
            "required_document_absent",
            "get_events",
            None,
        ),
    ],
    ids=["slash-multi-field", "fake-nested-key", "bare-nested-key"],
)
def test_evidence_case001_rejects_bad_field_path(
    source_system: str,
    record_id: str,
    field: str,
    fact: str,
    retrieval_ref: str,
    err_substr: str | None,
):
    idx = _case001_index()
    bad = Evidence(
        source_system=source_system,  # type: ignore[arg-type]
        record_id=record_id,
        field=field,
        fact=fact,
        retrieval_ref=retrieval_ref,
    )
    ok, errs = validate_evidence([bad], idx)
    assert ok == [] and errs
    if err_substr is not None:
        assert err_substr in errs[0]


def test_payload_digest_stable():
    t = FollowUpTaskPayload(title="a", body="b", case_id="C")
    assert payload_digest(t) == payload_digest(t)
    assert payload_digest(t) != payload_digest(
        FollowUpTaskPayload(title="a", body="B", case_id="C")
    )


def test_trace_redacts_secrets(tmp_path: Path):
    tw = TraceWriter(tmp_path, "run-1")
    tw.emit("x", authorization="Bearer super-secret", nested={"api_key": "abc"})
    line = (tmp_path / "run-1.jsonl").read_text(encoding="utf-8")
    assert "super-secret" not in line
    assert "abc" not in line
    assert "[REDACTED]" in line


# ---------------------------------------------------------------------------
# investigation + API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_case001_scripted_investigation_accepts_correct_evidence_paths(tmp_path: Path):
    """CASE-001 fixture tools + nested evidence → grounded proposal, digest, no cost/secrets."""
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    finding = _finding(
        diagnosis=Diagnosis.MISSING_DOCUMENT,
        diagnosis_summary="proof_of_income missing",
        explanation="Required proof_of_income never received; ID and beneficiary on file.",
        evidence=[
            Evidence(
                source_system="cases",
                record_id=CASE_ID,
                field="required_documents",
                fact="proof_of_income is required",
                retrieval_ref="get_case",
            ),
            Evidence(
                source_system="documents",
                record_id="DOC-001-A",
                field="status",
                fact="government_id received",
                retrieval_ref="get_documents",
            ),
            Evidence(
                source_system="documents",
                record_id="DOC-001-A",
                field="content_summary",
                fact="ID on file",
                retrieval_ref="get_documents",
            ),
            Evidence(
                source_system="events",
                record_id="EVT-001-A",
                field="details.missing",
                fact="checklist missing proof_of_income",
                retrieval_ref="get_events",
            ),
            Evidence(
                source_system="events",
                record_id="EVT-001-B",
                field="details.reason",
                fact="required_document_absent",
                retrieval_ref="get_events",
            ),
        ],
        next_action=NextAction.CREATE_FOLLOW_UP_TASK,
        next_action_explanation="Request proof_of_income",
        proposed_task=FollowUpTaskPayload(
            title="Request proof of income",
            body="Obtain proof_of_income for CASE-001",
            case_id=CASE_ID,
        ),
    )

    def factory(_s):
        return ScriptedChatModel(
            responses=_tool_then_finding(finding, tools=["get_case", "get_documents", "get_events"])
        )

    outcome = await run_investigation(
        settings=settings,
        store=store,
        client=client,
        case_id=CASE_ID,
        objective="Why stalled?",
        model_factory=factory,
    )
    result = outcome.result
    assert result.status == RunStatus.COMPLETED
    assert result.diagnosis == Diagnosis.MISSING_DOCUMENT
    assert result.approval_required is True
    assert result.proposal_id is not None
    assert len(result.evidence) == 5
    fields = {e.field for e in result.evidence}
    assert "details.missing" in fields and "details.reason" in fields
    assert "content_summary" in fields

    prop = store.get_proposal(result.proposal_id)
    assert prop is not None
    assert prop.status == ProposalStatus.PENDING
    assert prop.task_payload == finding.proposed_task
    assert prop.payload_digest == payload_digest(finding.proposed_task)
    stored = store.get_run(result.run_id)
    assert stored is not None and stored.result is not None
    assert stored.result.proposal_id == result.proposal_id
    assert outcome.cost_usd is None
    trace = (settings.trace_dir / f"{result.run_id}.jsonl").read_text(encoding="utf-8")
    assert TOKEN_SIM not in trace
    assert "run_start" in trace
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_api_investigate_get_approve_reject_flow(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    finding = _finding()

    def factory(_s):
        return ScriptedChatModel(responses=_tool_then_finding(finding))

    app = create_app(settings, model_factory=factory, store=store, client=client)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        h = await ac.get(BackendRoutes.HEALTH)
        assert h.status_code == 200

        resp = await ac.post(
            BackendRoutes.INVESTIGATIONS,
            json=StartInvestigationRequest(case_id=CASE_ID, objective="Why stalled?").model_dump(),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        run_id = body["run_id"]
        assert body["result"]["approval_required"] is True
        pid = body["result"]["proposal_id"]
        assert pid

        run = await ac.get(BackendRoutes.RUN.format(run_id=run_id))
        assert run.status_code == 200
        assert run.json()["status"] == "completed"
        assert run.json()["result"]["proposal_id"] == pid

        # reject without auth
        bad = await ac.post(
            BackendRoutes.REJECT.format(proposal_id=pid),
            json={"operator_identity": "x"},
        )
        assert bad.status_code == 401

        # wrong token
        bad2 = await ac.post(
            BackendRoutes.REJECT.format(proposal_id=pid),
            json={"operator_identity": "x"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert bad2.status_code == 403

        # approve
        ok = await ac.post(
            BackendRoutes.APPROVE.format(proposal_id=pid),
            json={"operator_identity": "alice"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert ok.status_code == 200, ok.text
        assert ok.json()["proposal"]["status"] == "executed"
        assert ok.json()["proposal"]["operator_identity"] == "alice"
        # payload immutable
        assert ok.json()["proposal"]["task_payload"] == finding.proposed_task.model_dump()

        # second approve is idempotent success
        again = await ac.post(
            BackendRoutes.APPROVE.format(proposal_id=pid),
            json={"operator_identity": "alice"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert again.status_code == 200
        assert again.json()["proposal"]["status"] == "executed"

    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_reject_is_terminal(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    finding = _finding()

    def factory(_s):
        return ScriptedChatModel(responses=_tool_then_finding(finding))

    app = create_app(settings, model_factory=factory, store=store, client=client)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.post(
            BackendRoutes.INVESTIGATIONS,
            json={"case_id": CASE_ID, "objective": "x"},
        )
        pid = resp.json()["result"]["proposal_id"]
        rej = await ac.post(
            BackendRoutes.REJECT.format(proposal_id=pid),
            json={"operator_identity": "bob", "reason": "nope"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert rej.json()["proposal"]["status"] == "rejected"
        appr = await ac.post(
            BackendRoutes.APPROVE.format(proposal_id=pid),
            json={"operator_identity": "bob"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert appr.status_code == 409
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_execution_failed_retry(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    # First client fails writes
    http_fail = _mock_simulator(fail_write=True)
    client_fail = SimulatorClient(settings, client=http_fail)
    finding = _finding()

    def factory(_s):
        return ScriptedChatModel(responses=_tool_then_finding(finding))

    app = create_app(settings, model_factory=factory, store=store, client=client_fail)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.post(
            BackendRoutes.INVESTIGATIONS,
            json={"case_id": CASE_ID, "objective": "x"},
        )
        pid = resp.json()["result"]["proposal_id"]
        fail = await ac.post(
            BackendRoutes.APPROVE.format(proposal_id=pid),
            json={"operator_identity": "ops"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert fail.json()["proposal"]["status"] == "execution_failed"
        assert fail.json()["proposal"]["execution_error"]

    await http_fail.aclose()

    # Swap client to healthy simulator and retry same proposal
    http_ok = _mock_simulator()
    client_ok = SimulatorClient(settings, client=http_ok)
    app.state.backend.client = client_ok  # type: ignore[attr-defined]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        retry = await ac.post(
            BackendRoutes.APPROVE.format(proposal_id=pid),
            json={"operator_identity": "ops"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert retry.status_code == 200, retry.text
        assert retry.json()["proposal"]["status"] == "executed"
        # payload unchanged
        assert retry.json()["proposal"]["task_payload"]["title"] == (
            f"Follow up missing docs for {CASE_ID}"
        )

    store.close()
    await http_ok.aclose()


@pytest.mark.asyncio
async def test_missing_api_key_fails_visibly_without_hook(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(TEST_MODE_ENV, raising=False)
    clear_test_hooks()
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    outcome = await run_investigation(
        settings=settings,
        store=store,
        client=client,
        case_id=CASE_ID,
        objective="x",
        # default factory → no key, no hook
    )
    result = outcome.result
    assert result.status == RunStatus.FAILED
    assert "OPENROUTER_API_KEY" in (result.explanation or "")
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_test_mode_http_hook_installs_script(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(TEST_MODE_ENV, "1")
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    # model_factory uses default which consults hooks
    app = create_app(settings, store=store, client=client)
    finding = _finding(
        next_action=NextAction.NO_ACTION_REQUIRED,
        next_action_explanation="ok",
        proposed_task=None,
        evidence=[
            Evidence(
                source_system="cases",
                record_id=CASE_ID,
                field="status",
                fact="stalled",
                retrieval_ref="get_case",
            )
        ],
    )
    responses = _tool_then_finding(finding, tools=["get_case"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        mode = await ac.get("/v1/_test/mode")
        assert mode.status_code == 200
        installed = await ac.put(
            "/v1/_test/scripted-model",
            json={"responses": [m.model_dump(mode="json") for m in responses]},
        )
        assert installed.status_code == 200
        resp = await ac.post(
            BackendRoutes.INVESTIGATIONS,
            json={"case_id": CASE_ID, "objective": "x"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["result"]["status"] == "completed"
        assert resp.json()["result"]["approval_required"] is False
    store.close()
    await http.aclose()


def test_test_routes_absent_without_test_mode(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(TEST_MODE_ENV, raising=False)
    settings = _settings(tmp_path)
    app = create_app(settings, store=BackendStore(settings.backend_db_path))
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/v1/_test/scripted-model" not in paths
    assert "/v1/_test/mode" not in paths


@pytest.mark.asyncio
async def test_proposal_get(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    finding = _finding()

    def factory(_s):
        return ScriptedChatModel(responses=_tool_then_finding(finding))

    app = create_app(settings, model_factory=factory, store=store, client=client)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.post(
            BackendRoutes.INVESTIGATIONS,
            json={"case_id": CASE_ID, "objective": "x"},
        )
        pid = resp.json()["result"]["proposal_id"]
        got = await ac.get(BackendRoutes.PROPOSAL.format(proposal_id=pid))
        assert got.status_code == 200
        assert got.json()["status"] == "pending"
        missing = await ac.get(BackendRoutes.PROPOSAL.format(proposal_id="nope"))
        assert missing.status_code == 404
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_case_proposal_discoverability_api(tmp_path: Path):
    """proposal_id on result + GET /v1/cases/{case_id}/proposals."""
    from casepilot.backend.app import CASE_PROPOSALS

    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    finding = _finding()

    def factory(_s):
        return ScriptedChatModel(responses=_tool_then_finding(finding))

    app = create_app(settings, model_factory=factory, store=store, client=client)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.post(
            BackendRoutes.INVESTIGATIONS,
            json={"case_id": CASE_ID, "objective": "x"},
        )
        assert resp.status_code == 200
        pid = resp.json()["result"]["proposal_id"]
        assert pid
        listed = await ac.get(CASE_PROPOSALS.format(case_id=CASE_ID))
        assert listed.status_code == 200
        rows = listed.json()
        assert len(rows) == 1
        assert rows[0]["proposal_id"] == pid
        assert rows[0]["case_id"] == CASE_ID
        empty = await ac.get(CASE_PROPOSALS.format(case_id="CASE-NONE"))
        assert empty.status_code == 200
        assert empty.json() == []
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_invalid_evidence_no_write_proposal(tmp_path: Path):
    """Fabricated evidence → indeterminate/escalate, empty evidence, no proposal (API)."""
    from casepilot.backend.app import CASE_PROPOSALS

    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    finding = _finding(
        evidence=[
            Evidence(
                source_system="documents",
                record_id="DOC-FORGED",
                field="status",
                fact="missing",
                retrieval_ref="lie",
            )
        ]
    )

    def factory(_s):
        return ScriptedChatModel(responses=_tool_then_finding(finding))

    app = create_app(settings, model_factory=factory, store=store, client=client)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.post(
            BackendRoutes.INVESTIGATIONS,
            json={"case_id": CASE_ID, "objective": "x"},
        )
        body = resp.json()["result"]
        assert body["status"] == "indeterminate"
        assert body["evidence"] == []
        assert body["approval_required"] is False
        assert body["proposed_task"] is None
        assert body["proposal_id"] is None
        assert body["next_action"] == "escalate_to_human"
        listed = await ac.get(CASE_PROPOSALS.format(case_id=CASE_ID))
        assert listed.json() == []
        assert store.get_proposal_by_run(resp.json()["run_id"]) is None
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_crash_recovery_approved_then_retry(tmp_path: Path):
    """APPROVED left after crash (sim commit unknown) retries identical payload."""
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    finding = _finding()

    def factory(_s):
        return ScriptedChatModel(responses=_tool_then_finding(finding))

    app = create_app(settings, model_factory=factory, store=store, client=client)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        resp = await ac.post(
            BackendRoutes.INVESTIGATIONS,
            json={"case_id": CASE_ID, "objective": "x"},
        )
        pid = resp.json()["result"]["proposal_id"]
        # Simulate process crash after marking approved, before/without confirm:
        crashed = store.transition_proposal(
            pid,
            from_statuses={ProposalStatus.PENDING},
            to_status=ProposalStatus.APPROVED,
            operator_identity="ops",
        )
        assert crashed is not None
        assert crashed.status == ProposalStatus.APPROVED

        retry = await ac.post(
            BackendRoutes.APPROVE.format(proposal_id=pid),
            json={"operator_identity": "ops"},
            headers={"Authorization": f"Bearer {TOKEN_OP}"},
        )
        assert retry.status_code == 200, retry.text
        assert retry.json()["proposal"]["status"] == "executed"
        assert retry.json()["proposal"]["task_payload"] == finding.proposed_task.model_dump()
    store.close()
    await http.aclose()


@pytest.mark.asyncio
async def test_scripted_path_does_not_touch_cost_ledger(tmp_path: Path, monkeypatch):
    ledger_file = tmp_path / "ledger.json"
    monkeypatch.setenv("CASEPILOT_COST_LEDGER_PATH", str(ledger_file))
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    finding = _finding(
        next_action=NextAction.NO_ACTION_REQUIRED,
        next_action_explanation="none",
        proposed_task=None,
        evidence=[
            Evidence(
                source_system="cases",
                record_id=CASE_ID,
                field="status",
                fact="stalled",
                retrieval_ref="get_case",
            )
        ],
    )

    def factory(_s):
        return ScriptedChatModel(responses=_tool_then_finding(finding, tools=["get_case"]))

    await run_investigation(
        settings=settings,
        store=store,
        client=client,
        case_id=CASE_ID,
        objective="x",
        model_factory=factory,
    )
    assert not ledger_file.exists()
    store.close()
    await http.aclose()


def test_generic_500_does_not_leak_exception(tmp_path: Path):
    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)

    def boom(_s):
        raise RuntimeError("secret-db-password-should-not-leak")

    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    app = create_app(settings, model_factory=boom, store=store, client=client)

    # Force unhandled path via a custom route
    @app.get("/boom")
    async def _boom() -> None:
        raise RuntimeError("secret-db-password-should-not-leak")

    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=False) as tc:
        r = tc.get("/boom")
        assert r.status_code == 500
        assert r.json() == {"detail": "internal error"}
        assert "secret" not in r.text
    store.close()


def test_guard_live_call_temp_ledger_no_provider(tmp_path: Path, monkeypatch):
    """Real temp ledger + create_reservation path without any provider HTTP."""
    from langchain_openrouter import ChatOpenRouter

    from casepilot.backend.cost_guard import (
        INPUT_TOKENS_PER_TURN,
        MAX_OUTPUT_TOKENS,
        estimate_request_tokens,
        get_ledger,
        guard_live_call,
        release_live,
        settle_live,
    )
    from casepilot.evaluation.cost import ALLOWED_LIVE_MODEL, CostPause, ModelPrice

    ledger_file = tmp_path / "cost_ledger.json"
    monkeypatch.setenv("CASEPILOT_COST_LEDGER_PATH", str(ledger_file))
    settings = _settings(tmp_path, max_model_turns=2)

    # Inject verified test prices (no paid provider; fail-closed production needs file).
    prices = {
        ALLOWED_LIVE_MODEL: ModelPrice(
            model=ALLOWED_LIVE_MODEL,
            input_usd_per_mtok=0.10,
            output_usd_per_mtok=0.40,
            source="unit-test",
            verified_at="1970-01-01T00:00:00Z",
            verified=True,
        )
    }
    lg = get_ledger(prices=prices)

    # Fake live model instance (no network) — isinstance check only.
    live = ChatOpenRouter(
        model=ALLOWED_LIVE_MODEL, api_key="test-not-used", max_tokens=MAX_OUTPUT_TOKENS
    )
    ledger, reservation = guard_live_call(
        model=live,
        model_name=ALLOWED_LIVE_MODEL,
        run_id="run-test-1",
        case_id=CASE_ID,
        settings=settings,
        ledger=lg,
    )
    assert ledger is not None and reservation is not None
    assert reservation.amount_usd > 0
    assert ledger_file.exists()
    snap = ledger.snapshot()
    assert float(snap["reserved_usd"]) == pytest.approx(reservation.amount_usd)

    # Release without provider start (safe path).
    release_live(ledger, reservation, detail="test no provider")
    snap2 = ledger.snapshot()
    assert float(snap2["reserved_usd"]) == pytest.approx(0.0)
    assert float(snap2["spent_usd"]) == pytest.approx(0.0)

    # Second reserve then settle full (unknown actual) + pause.
    ledger2, res2 = guard_live_call(
        model=live,
        model_name=ALLOWED_LIVE_MODEL,
        run_id="run-test-2",
        case_id=CASE_ID,
        settings=settings,
        ledger=get_ledger(prices=prices),
    )
    assert res2 is not None and ledger2 is not None
    usage, spent = settle_live(ledger2, res2, messages=None, force_full_reserve=True)
    assert spent == pytest.approx(res2.amount_usd)
    assert usage is not None and usage.get("settled_at_full_reserve") is True
    snap3 = ledger2.snapshot()
    assert snap3.get("pricing_reliable") is False

    # Further reserve must fail closed.
    with pytest.raises(CostPause):
        guard_live_call(
            model=live,
            model_name=ALLOWED_LIVE_MODEL,
            run_id="run-test-3",
            case_id=CASE_ID,
            settings=settings,
            ledger=get_ledger(prices=prices),
        )

    # Input estimator enforces per-turn assumption basis.
    huge = "x" * (INPUT_TOKENS_PER_TURN * 4)
    assert (
        estimate_request_tokens(messages=[type("M", (), {"content": huge})()])
        > INPUT_TOKENS_PER_TURN
    )


def test_default_model_factory_sets_max_tokens_and_provider_caps():
    """Production factory → serialized offline payload carries GLM caps/routing."""
    from casepilot.backend.agent import default_model_factory
    from casepilot.backend.cost_guard import (
        MAX_OUTPUT_TOKENS,
        MAX_PRICE_COMPLETION_USD_PER_MTOK,
        MAX_PRICE_PROMPT_USD_PER_MTOK,
        live_model_price,
        live_request_payload_preview,
    )
    from casepilot.evaluation.cost import ALLOWED_LIVE_MODEL

    price = live_model_price()
    assert price.verified is True
    assert price.max_price_prompt == MAX_PRICE_PROMPT_USD_PER_MTOK
    assert price.max_price_completion == MAX_PRICE_COMPLETION_USD_PER_MTOK

    settings = get_settings(
        _env_file=None,
        simulator_service_token=TOKEN_SIM,
        operator_token=TOKEN_OP,
        openrouter_api_key="sk-test",
        openrouter_model=ALLOWED_LIVE_MODEL,
    )
    model = default_model_factory(settings)
    assert getattr(model, "max_tokens", None) == MAX_OUTPUT_TOKENS
    payload = live_request_payload_preview(model)
    assert payload["model"] == ALLOWED_LIVE_MODEL
    assert payload["max_tokens"] == MAX_OUTPUT_TOKENS
    assert payload["provider"] == {
        "require_parameters": True,
        "max_price": {
            "prompt": MAX_PRICE_PROMPT_USD_PER_MTOK,
            "completion": MAX_PRICE_COMPLETION_USD_PER_MTOK,
        },
    }
    assert payload["reasoning"] == {"effort": "low", "exclude": True}


def test_realistic_complete_prompt_fits_input_cap():
    """Full tools + schema + multi-turn tool results under per-turn byte-token cap."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    from casepilot.backend.agent import SYSTEM_PROMPT, AgentFinding
    from casepilot.backend.cost_guard import (
        INPUT_TOKENS_PER_TURN,
        estimate_request_tokens,
        estimate_text_tokens,
    )

    # 1 byte == 1 token conservatism
    assert estimate_text_tokens("abc") == 3
    assert estimate_text_tokens("€") == len("€".encode())

    schema = AgentFinding.model_json_schema()
    tools = [
        {
            "name": "get_case",
            "description": "Fetch the investigation case record",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_documents",
            "description": "Fetch documents",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_events",
            "description": "Fetch events",
            "parameters": {"type": "object", "properties": {}},
        },
        {"name": "AgentFinding", "description": "final", "parameters": schema},
    ]
    case_json = json.dumps(
        {
            "case_id": CASE_ID,
            "status": "stalled",
            "symptom": "x" * 400,
            "application_fields": {f"k{i}": "v" * 40 for i in range(30)},
            "required_documents": ["aps", "lab", "exam"],
        }
    )
    docs_json = json.dumps(
        {
            "case_id": CASE_ID,
            "documents": [
                {
                    "document_id": f"DOC-{i}",
                    "fields": {f"f{j}": "d" * 40 for j in range(12)},
                    "content_summary": "c" * 300,
                }
                for i in range(10)
            ],
        }
    )
    events_json = json.dumps(
        {
            "case_id": CASE_ID,
            "events": [
                {
                    "event_id": f"EVT-{i}",
                    "message": "m" * 120,
                    "details": {"note": "n" * 80},
                }
                for i in range(12)
            ],
        }
    )
    messages = [
        HumanMessage(content="Investigate case. Objective: " + ("why stalled? " * 30)),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_case", "args": {}, "id": "1", "type": "tool_call"}],
        ),
        ToolMessage(content=case_json, tool_call_id="1", name="get_case"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_documents", "args": {}, "id": "2", "type": "tool_call"}],
        ),
        ToolMessage(content=docs_json, tool_call_id="2", name="get_documents"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_events", "args": {}, "id": "3", "type": "tool_call"}],
        ),
        ToolMessage(content=events_json, tool_call_id="3", name="get_events"),
    ]
    est = estimate_request_tokens(
        messages=messages,
        system_message=SystemMessage(content=SYSTEM_PROMPT),
        tools=tools,
    )
    assert est < INPUT_TOKENS_PER_TURN, f"realistic prompt est {est} >= cap {INPUT_TOKENS_PER_TURN}"


def test_trace_redacts_reasoning(tmp_path: Path):
    tw = TraceWriter(tmp_path, "run-r")
    tw.emit("model", reasoning_content="secret-chain-of-thought", ok=True)
    line = (tmp_path / "run-r.jsonl").read_text(encoding="utf-8")
    assert "secret-chain-of-thought" not in line
    assert "[REDACTED_REASONING]" in line


@pytest.mark.asyncio
async def test_run_investigation_ainvoke_installs_async_live_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """run_investigation async path installs live guard; tracker captures usage/input cap."""
    from langchain.agents.middleware import AgentMiddleware

    import casepilot.backend.agent as agent_mod
    from casepilot.backend.cost_guard import LiveCallTracker

    settings = _settings(tmp_path)
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    finding = _finding(
        next_action=NextAction.NO_ACTION_REQUIRED,
        next_action_explanation="none",
        proposed_task=None,
        evidence=[
            Evidence(
                source_system="cases",
                record_id=CASE_ID,
                field="status",
                fact="stalled",
                retrieval_ref="get_case",
            )
        ],
    )
    responses = _tool_then_finding(finding, tools=["get_case"])

    # Force live-guard install without OpenRouter / paid calls.
    monkeypatch.setattr(agent_mod, "is_live_model", lambda _m: True)
    # Skip ledger reservation (still offline).
    monkeypatch.setattr(
        agent_mod,
        "guard_live_call",
        lambda **_kw: (None, None),
    )
    seen: dict[str, object] = {}

    real_make = agent_mod.make_live_guard_middleware

    def _wrap(tracker: LiveCallTracker):
        seen["tracker"] = tracker
        mw = real_make(tracker)
        seen["mw"] = mw
        return mw

    monkeypatch.setattr(agent_mod, "make_live_guard_middleware", _wrap)

    def factory(_s):
        return ScriptedChatModel(responses=list(responses))

    outcome = await run_investigation(
        settings=settings,
        store=store,
        client=client,
        case_id=CASE_ID,
        objective="offline async middleware",
        model_factory=factory,
    )
    assert outcome.result.status == RunStatus.COMPLETED
    tracker = seen.get("tracker")
    assert isinstance(tracker, LiveCallTracker)
    assert tracker.provider_started is True
    assert len(tracker.captured_ai_messages) >= 1
    assert tracker.last_input_estimate > 0
    assert tracker.last_input_estimate <= tracker.input_cap
    mw = seen.get("mw")
    assert mw is not None
    assert type(mw).awrap_model_call is not AgentMiddleware.awrap_model_call
    store.close()
    await http.aclose()
