"""FastAPI investigation backend."""

from __future__ import annotations

import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from casepilot.backend.agent import (
    ModelFactory,
    default_model_factory,
    payload_digest,
    run_investigation,
)
from casepilot.backend.client import SimulatorClient, SimulatorError
from casepilot.backend.db import BackendStore
from casepilot.backend.hooks import (
    TEST_MODE_ENV,
    has_test_hook,
    install_scripted_model,
    is_test_mode,
)
from casepilot.backend.trace import TraceWriter
from casepilot.shared.config import Settings, get_settings
from casepilot.shared.contracts import (
    ApproveProposalRequest,
    BackendRoutes,
    CaseListItem,
    CaseSummary,
    CreateFollowUpTaskRequest,
    NextAction,
    Proposal,
    ProposalDecisionResponse,
    ProposalStatus,
    RejectProposalRequest,
    RunRecord,
    RunSummary,
    StartInvestigationRequest,
    StartInvestigationResponse,
)

# Backward-compatible alias for callers/tests that imported the local constant.
CASE_PROPOSALS = BackendRoutes.CASE_PROPOSALS


@dataclass
class AppState:
    settings: Settings
    store: BackendStore
    client: SimulatorClient
    model_factory: ModelFactory


def _operator_auth(
    settings: Settings,
    authorization: str | None,
) -> None:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="operator Authorization Bearer required")
    token = authorization.split(" ", 1)[1].strip()
    expected = settings.operator_token.get_secret_value()
    # Timing-safe compare; unequal lengths are a mismatch (not a raise).
    if len(token) != len(expected) or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="invalid operator token")


def _assert_payload_integrity(prop: Proposal) -> None:
    """Ensure stored task payload still matches the immutable digest."""
    actual = payload_digest(prop.task_payload)
    if len(actual) != len(prop.payload_digest) or not secrets.compare_digest(
        actual, prop.payload_digest
    ):
        raise HTTPException(status_code=409, detail="stored payload digest mismatch")


def create_app(
    settings: Settings | None = None,
    *,
    model_factory: ModelFactory | None = None,
    store: BackendStore | None = None,
    client: SimulatorClient | None = None,
) -> FastAPI:
    """Build the backend app. Optional injectables support deterministic tests.

    When settings/store/client are omitted (uvicorn entry), resources are created
    during lifespan so importing the module does not require env yet.
    """

    eager = settings is not None or store is not None or client is not None
    backend_state: AppState | None = None
    if eager:
        cfg = settings or get_settings()
        st = store or BackendStore(cfg.backend_db_path)
        cl = client or SimulatorClient(cfg)
        backend_state = AppState(
            settings=cfg,
            store=st,
            client=cl,
            model_factory=model_factory or default_model_factory,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal backend_state
        owns_store = False
        owns_client = False
        if backend_state is None:
            cfg = get_settings()
            st = BackendStore(cfg.backend_db_path)
            cl = SimulatorClient(cfg)
            owns_store = True
            owns_client = True
            backend_state = AppState(
                settings=cfg,
                store=st,
                client=cl,
                model_factory=model_factory or default_model_factory,
            )
        app.state.backend = backend_state
        try:
            yield
        finally:
            if owns_client:
                await backend_state.client.aclose()
            if owns_store:
                backend_state.store.close()

    app = FastAPI(title="CasePilot Backend", lifespan=lifespan)
    if backend_state is not None:
        # Eager state so ASGI test clients without lifespan still work
        app.state.backend = backend_state

    def _backend(request: Request) -> AppState:
        return request.app.state.backend  # type: ignore[no-any-return]

    @app.get(BackendRoutes.HEALTH)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(BackendRoutes.INVESTIGATIONS)
    async def start_investigation(
        body: StartInvestigationRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Start investigation; includes usage/cost_usd/meta for eval (extra keys)."""
        backend = _backend(request)
        if not body.case_id.strip():
            raise HTTPException(status_code=422, detail="case_id required")
        outcome = await run_investigation(
            settings=backend.settings,
            store=backend.store,
            client=backend.client,
            case_id=body.case_id.strip(),
            objective=body.objective,
            model_factory=backend.model_factory,
        )
        result = outcome.result
        payload: dict[str, Any] = StartInvestigationResponse(
            run_id=result.run_id, result=result
        ).model_dump(mode="json")
        # Eval-facing cost metadata (not on shared InvestigationResult contract).
        if outcome.usage is not None:
            payload["usage"] = outcome.usage
            payload["token_usage"] = outcome.usage
            payload["model_usage"] = outcome.usage
        if outcome.cost_usd is not None:
            payload["cost_usd"] = outcome.cost_usd
            payload["total_cost_usd"] = outcome.cost_usd
        meta: dict[str, Any] = {}
        if outcome.usage is not None:
            meta["usage"] = outcome.usage
        if outcome.cost_usd is not None:
            meta["cost_usd"] = outcome.cost_usd
        if outcome.provider_model is not None:
            meta["provider_model"] = outcome.provider_model
            meta["model"] = outcome.provider_model
            payload["provider_model"] = outcome.provider_model
        if meta:
            payload["meta"] = meta
            payload["trace_summary"] = dict(meta)
        return payload

    @app.get(BackendRoutes.RUN, response_model=RunRecord)
    async def get_run(run_id: str, request: Request) -> RunRecord:
        record = _backend(request).store.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return record

    @app.get(BackendRoutes.PROPOSAL, response_model=Proposal)
    async def get_proposal(proposal_id: str, request: Request) -> Proposal:
        prop = _backend(request).store.get_proposal(proposal_id)
        if prop is None:
            raise HTTPException(status_code=404, detail="proposal not found")
        return prop

    @app.get(BackendRoutes.CASES, response_model=list[CaseListItem])
    async def list_cases(request: Request) -> list[CaseListItem]:
        backend = _backend(request)
        try:
            cases = await backend.client.list_cases()
        except SimulatorError as exc:
            raise HTTPException(
                status_code=exc.status_code or 502,
                detail=str(exc),
            ) from exc
        flags = backend.store.case_action_flags()
        return [
            CaseListItem(
                case_id=case.case_id,
                status=case.status,
                symptom=case.symptom,
                pending_review=flags.get(case.case_id, (False, False))[0],
                needs_execution=flags.get(case.case_id, (False, False))[1],
            )
            for case in cases
        ]

    @app.get(BackendRoutes.CASE, response_model=CaseSummary)
    async def get_case(case_id: str, request: Request) -> CaseSummary:
        try:
            case = await _backend(request).client.get_case(case_id)
        except SimulatorError as exc:
            if exc.status_code == 404:
                raise HTTPException(status_code=404, detail=f"case not found: {case_id}") from exc
            raise HTTPException(
                status_code=exc.status_code or 502,
                detail=str(exc),
            ) from exc
        return CaseSummary(case_id=case.case_id, status=case.status, symptom=case.symptom)

    @app.get(BackendRoutes.CASE_RUNS, response_model=list[RunSummary])
    async def list_case_runs(case_id: str, request: Request) -> list[RunSummary]:
        backend = _backend(request)
        try:
            await backend.client.get_case(case_id)
        except SimulatorError as exc:
            if exc.status_code == 404:
                raise HTTPException(status_code=404, detail=f"case not found: {case_id}") from exc
            raise HTTPException(
                status_code=exc.status_code or 502,
                detail=str(exc),
            ) from exc
        return backend.store.list_runs_for_case(case_id)

    @app.get(CASE_PROPOSALS, response_model=list[Proposal])
    async def list_case_proposals(case_id: str, request: Request) -> list[Proposal]:
        """Discover proposals for a case."""
        return _backend(request).store.list_proposals_for_case(case_id)

    @app.post(BackendRoutes.APPROVE, response_model=ProposalDecisionResponse)
    async def approve_proposal(
        proposal_id: str,
        body: ApproveProposalRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> ProposalDecisionResponse:
        backend = _backend(request)
        _operator_auth(backend.settings, authorization)
        prop = backend.store.get_proposal(proposal_id)
        if prop is None:
            raise HTTPException(status_code=404, detail="proposal not found")
        _assert_payload_integrity(prop)
        if prop.action != NextAction.CREATE_FOLLOW_UP_TASK:
            raise HTTPException(status_code=400, detail="action is not executable")
        if prop.status == ProposalStatus.REJECTED:
            raise HTTPException(status_code=409, detail="proposal rejected")
        if prop.status == ProposalStatus.EXECUTED:
            return ProposalDecisionResponse(proposal=prop)
        if prop.status not in {
            ProposalStatus.PENDING,
            ProposalStatus.APPROVED,
            ProposalStatus.EXECUTION_FAILED,
        }:
            raise HTTPException(
                status_code=409,
                detail=f"cannot approve from status {prop.status.value}",
            )

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        # pending -> approved; approved/execution_failed stay approved for retry
        if prop.status == ProposalStatus.PENDING:
            updated = backend.store.transition_proposal(
                proposal_id,
                from_statuses={ProposalStatus.PENDING},
                to_status=ProposalStatus.APPROVED,
                operator_identity=body.operator_identity,
                request_id=request_id,
                clear_execution_error=True,
            )
            if updated is None:
                raise HTTPException(status_code=409, detail="approval race; retry")
            prop = updated
        else:
            # Crash-recovery / retry: re-enter APPROVED with stored payload only
            updated = backend.store.transition_proposal(
                proposal_id,
                from_statuses={ProposalStatus.APPROVED, ProposalStatus.EXECUTION_FAILED},
                to_status=ProposalStatus.APPROVED,
                operator_identity=body.operator_identity,
                request_id=request_id,
                clear_execution_error=True,
            )
            if updated is None:
                prop = backend.store.get_proposal(proposal_id)
                if prop is None:
                    raise HTTPException(status_code=404, detail="proposal not found")
                if prop.status == ProposalStatus.EXECUTED:
                    return ProposalDecisionResponse(proposal=prop)
                raise HTTPException(status_code=409, detail="approval race; retry")
            prop = updated

        TraceWriter(backend.settings.trace_dir, prop.run_id).emit(
            "proposal_approved",
            proposal_id=proposal_id,
            operator_identity=body.operator_identity,
            request_id=request_id,
        )

        # Execute with immutable stored payload only
        try:
            await backend.client.create_follow_up_task(
                prop.case_id,
                CreateFollowUpTaskRequest(
                    proposal_id=prop.proposal_id,
                    task=prop.task_payload,
                ),
            )
        except SimulatorError as exc:
            failed = backend.store.transition_proposal(
                proposal_id,
                from_statuses={ProposalStatus.APPROVED},
                to_status=ProposalStatus.EXECUTION_FAILED,
                execution_error=str(exc),
            )
            TraceWriter(backend.settings.trace_dir, prop.run_id).emit(
                "proposal_execution_failed",
                proposal_id=proposal_id,
                error=str(exc),
            )
            if failed is None:
                prop = backend.store.get_proposal(proposal_id)
                if prop is None:
                    raise HTTPException(status_code=404, detail="proposal not found") from None
                return ProposalDecisionResponse(proposal=prop)
            return ProposalDecisionResponse(proposal=failed)

        executed = backend.store.transition_proposal(
            proposal_id,
            from_statuses={ProposalStatus.APPROVED},
            to_status=ProposalStatus.EXECUTED,
            clear_execution_error=True,
        )
        TraceWriter(backend.settings.trace_dir, prop.run_id).emit(
            "proposal_executed",
            proposal_id=proposal_id,
        )
        if executed is None:
            prop = backend.store.get_proposal(proposal_id)
            if prop is None:
                raise HTTPException(status_code=404, detail="proposal not found")
            return ProposalDecisionResponse(proposal=prop)
        return ProposalDecisionResponse(proposal=executed)

    @app.post(BackendRoutes.REJECT, response_model=ProposalDecisionResponse)
    async def reject_proposal(
        proposal_id: str,
        body: RejectProposalRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> ProposalDecisionResponse:
        backend = _backend(request)
        _operator_auth(backend.settings, authorization)
        prop = backend.store.get_proposal(proposal_id)
        if prop is None:
            raise HTTPException(status_code=404, detail="proposal not found")
        _assert_payload_integrity(prop)
        if prop.status != ProposalStatus.PENDING:
            raise HTTPException(
                status_code=409,
                detail=f"cannot reject from status {prop.status.value}",
            )
        updated = backend.store.transition_proposal(
            proposal_id,
            from_statuses={ProposalStatus.PENDING},
            to_status=ProposalStatus.REJECTED,
            operator_identity=body.operator_identity,
            execution_error=body.reason,
        )
        if updated is None:
            raise HTTPException(status_code=409, detail="reject race; retry")
        TraceWriter(backend.settings.trace_dir, updated.run_id).emit(
            "proposal_rejected",
            proposal_id=proposal_id,
            operator_identity=body.operator_identity,
        )
        return ProposalDecisionResponse(proposal=updated)

    # ----- explicit test-only hook (never registered live) -----
    if is_test_mode():

        @app.put("/v1/_test/scripted-model")
        async def put_scripted_model(payload: dict[str, Any]) -> dict[str, str]:
            """Install scripted AIMessage turns. Requires CASEPILOT_TEST_MODE=1."""
            responses = payload.get("responses")
            if not isinstance(responses, list) or not responses:
                raise HTTPException(status_code=422, detail="responses list required")
            try:
                install_scripted_model(responses)
            except RuntimeError as exc:
                raise HTTPException(status_code=403, detail="forbidden") from exc
            return {"status": "installed", "turns": str(len(responses))}

        @app.get("/v1/_test/mode")
        async def test_mode_flag() -> dict[str, str]:
            return {
                "test_mode": "1",
                "env": TEST_MODE_ENV,
                "scripted_hook": "1" if has_test_hook() else "0",
            }

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, _exc: Exception) -> JSONResponse:
        # Generic 500 — never leak exception text/secrets to clients.
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    return app


# Uvicorn entry: casepilot.backend.app:app
app = create_app()
