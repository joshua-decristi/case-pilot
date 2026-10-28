"""Backend unit tests: spend guard, redaction, evidence, fail-closed (no paid calls).

Route/schema mock-flow and helper units deleted; boundary covers workflow CRUD.
"""

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


def _docs() -> DocumentListResponse:
    return _case001_bundle()[1]


def _case001_index() -> RetrievedIndex:
    case, docs, events = _case001_bundle()
    idx = RetrievedIndex()
    idx.note_case(case)
    idx.note_documents(docs.documents)
    idx.note_events(events.events)
    return idx


def _mock_simulator() -> httpx.AsyncClient:
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
            if body.proposal_id in tasks:
                prev = tasks[body.proposal_id]
                if prev.task != body.task or prev.proposal_id != body.proposal_id:
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


def test_scripted_install_requires_test_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(TEST_MODE_ENV, raising=False)
    assert not is_test_mode()
    with pytest.raises(RuntimeError, match=TEST_MODE_ENV):
        install_scripted_model([AIMessage(content="x")])


def test_default_model_factory_test_mode_no_script_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """CASEPILOT_TEST_MODE=1 + no hook must never fall through to paid provider."""
    from casepilot.backend.agent import default_model_factory
    from casepilot.evaluation.cost import ALLOWED_LIVE_MODEL

    monkeypatch.setenv(TEST_MODE_ENV, "1")
    clear_test_hooks()
    settings = get_settings(
        _env_file=None,
        simulator_service_token=TOKEN_SIM,
        operator_token=TOKEN_OP,
        openrouter_api_key="sk-would-be-paid",
        openrouter_model=ALLOWED_LIVE_MODEL,
        backend_db_path=tmp_path / "b.sqlite3",
        trace_dir=tmp_path / "traces",
    )
    with pytest.raises(RuntimeError, match="refusing live paid provider fallback"):
        default_model_factory(settings)


@pytest.mark.asyncio
async def test_investigation_test_mode_no_script_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Investigation with default factory in test mode + no script stays offline."""
    monkeypatch.setenv(TEST_MODE_ENV, "1")
    clear_test_hooks()
    settings = _settings(tmp_path, openrouter_api_key="sk-would-be-paid")
    store = BackendStore(settings.backend_db_path)
    http = _mock_simulator()
    client = SimulatorClient(settings, client=http)
    outcome = await run_investigation(
        settings=settings,
        store=store,
        client=client,
        case_id=CASE_ID,
        objective="x",
    )
    result = outcome.result
    assert result.status == RunStatus.FAILED
    assert "refusing live paid provider fallback" in (result.explanation or "")
    store.close()
    await http.aclose()


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


def test_evidence_case001_rejects_bad_field_path():
    """Representative bad path: slash multi-field must not validate."""
    idx = _case001_index()
    bad = Evidence(
        source_system="documents",
        record_id="DOC-001-A",
        field="status/content_summary",
        fact="received",
        retrieval_ref="get_documents",
    )
    ok, errs = validate_evidence([bad], idx)
    assert ok == [] and errs
    assert "status/content_summary" in errs[0]


def test_trace_redacts_secrets(tmp_path: Path):
    tw = TraceWriter(tmp_path, "run-1")
    tw.emit("x", authorization="Bearer super-secret", nested={"api_key": "abc"})
    line = (tmp_path / "run-1.jsonl").read_text(encoding="utf-8")
    assert "super-secret" not in line
    assert "abc" not in line
    assert "[REDACTED]" in line


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
    )
    result = outcome.result
    assert result.status == RunStatus.FAILED
    assert "OPENROUTER_API_KEY" in (result.explanation or "")
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

    release_live(ledger, reservation, detail="test no provider")
    snap2 = ledger.snapshot()
    assert float(snap2["reserved_usd"]) == pytest.approx(0.0)
    assert float(snap2["spent_usd"]) == pytest.approx(0.0)

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

    with pytest.raises(CostPause):
        guard_live_call(
            model=live,
            model_name=ALLOWED_LIVE_MODEL,
            run_id="run-test-3",
            case_id=CASE_ID,
            settings=settings,
            ledger=get_ledger(prices=prices),
        )

    huge = "x" * (INPUT_TOKENS_PER_TURN * 4)
    assert (
        estimate_request_tokens(messages=[type("M", (), {"content": huge})()])
        > INPUT_TOKENS_PER_TURN
    )


def test_default_model_factory_sets_max_tokens_and_provider_caps(
    monkeypatch: pytest.MonkeyPatch,
):
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

    monkeypatch.delenv(TEST_MODE_ENV, raising=False)
    clear_test_hooks()

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

    monkeypatch.setattr(agent_mod, "is_live_model", lambda _m: True)
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
