"""FastAPI enterprise simulator application."""

import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status

from casepilot.shared import (
    CaseRecord,
    CreateFollowUpTaskRequest,
    CreateFollowUpTaskResponse,
    DocumentListResponse,
    EventListResponse,
    Settings,
    SimulatorRoutes,
    get_settings,
)
from casepilot.simulator.auth import require_service_token
from casepilot.simulator.db import apply_schema, connect
from casepilot.simulator.store import ConflictError, SimulatorStore

TEST_MODE_ENV = "CASEPILOT_TEST_MODE"
# Process-local reversible outage flag (manual/browser tests only).
_outage_enabled = False


def is_test_mode() -> bool:
    return os.environ.get(TEST_MODE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def set_outage(enabled: bool) -> None:
    """Toggle isolated simulator outage. Requires CASEPILOT_TEST_MODE."""
    if not is_test_mode():
        raise RuntimeError(
            f"set_outage requires {TEST_MODE_ENV}=1 "
            "(refusing outage injection outside explicit test mode)"
        )
    global _outage_enabled
    _outage_enabled = bool(enabled)


def clear_outage() -> None:
    global _outage_enabled
    _outage_enabled = False


def outage_enabled() -> bool:
    return _outage_enabled


def create_app(
    settings: Settings | None = None,
    db_path: Path | None = None,
) -> FastAPI:
    """Build the simulator app.

    Does **not** seed or reset the database on startup — seed is explicit via
    ``casepilot.simulator.seed``. Ensures schema exists so an empty file works
    for health checks; domain data must be loaded separately.

    Each request opens its own SQLite connection (simple concurrency).
    """
    resolved_settings = settings or get_settings()
    resolved_db = Path(db_path) if db_path is not None else resolved_settings.simulator_db_path
    service_token = resolved_settings.simulator_service_token.get_secret_value()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Schema only — never load/overwrite fixtures here.
        bootstrap = connect(resolved_db)
        try:
            apply_schema(bootstrap, reset=False)
        finally:
            bootstrap.close()
        app.state.service_token = service_token
        app.state.db_path = resolved_db
        yield

    application = FastAPI(
        title="CasePilot Enterprise Simulator",
        lifespan=lifespan,
    )

    def get_store(request: Request) -> Iterator[SimulatorStore]:
        path: Path = request.app.state.db_path
        conn = connect(path)
        try:
            yield SimulatorStore(conn)
        finally:
            conn.close()

    StoreDep = Annotated[SimulatorStore, Depends(get_store)]
    AuthDep = Annotated[None, Depends(require_service_token)]

    def _reject_if_outage() -> None:
        if _outage_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="simulator outage (test injection)",
            )

    @application.get(SimulatorRoutes.HEALTH)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get(SimulatorRoutes.CASES, response_model=list[CaseRecord])
    def list_cases(store: StoreDep, _auth: AuthDep) -> list[CaseRecord]:
        _reject_if_outage()
        return store.list_cases()

    @application.get(SimulatorRoutes.CASE, response_model=CaseRecord)
    def get_case(case_id: str, store: StoreDep, _auth: AuthDep) -> CaseRecord:
        _reject_if_outage()
        record = store.get_case(case_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case not found: {case_id}",
            )
        return record

    @application.get(SimulatorRoutes.DOCUMENTS, response_model=DocumentListResponse)
    def get_documents(case_id: str, store: StoreDep, _auth: AuthDep) -> DocumentListResponse:
        _reject_if_outage()
        if not store.case_exists(case_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case not found: {case_id}",
            )
        return DocumentListResponse(
            case_id=case_id,
            documents=store.list_documents(case_id),
        )

    @application.get(SimulatorRoutes.EVENTS, response_model=EventListResponse)
    def get_events(case_id: str, store: StoreDep, _auth: AuthDep) -> EventListResponse:
        _reject_if_outage()
        if not store.case_exists(case_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case not found: {case_id}",
            )
        return EventListResponse(
            case_id=case_id,
            events=store.list_events(case_id),
        )

    @application.post(
        SimulatorRoutes.FOLLOW_UP_TASKS,
        response_model=CreateFollowUpTaskResponse,
    )
    def create_follow_up_task(
        case_id: str,
        body: CreateFollowUpTaskRequest,
        store: StoreDep,
        _auth: AuthDep,
    ) -> CreateFollowUpTaskResponse:
        _reject_if_outage()
        if body.task.case_id != case_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="task.case_id must match path case_id",
            )
        if not store.case_exists(case_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case not found: {case_id}",
            )
        try:
            return store.create_follow_up_task(
                case_id=case_id,
                proposal_id=body.proposal_id,
                task=body.task,
            )
        except ConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

    # ----- explicit test-only outage injection (never registered live) -----
    if is_test_mode():

        @application.get("/v1/_test/mode")
        def test_mode_flag() -> dict[str, str]:
            return {
                "test_mode": "1",
                "env": TEST_MODE_ENV,
                "outage": "1" if _outage_enabled else "0",
            }

        @application.put("/v1/_test/outage")
        def put_outage(payload: dict[str, Any]) -> dict[str, str]:
            enabled = payload.get("enabled")
            if not isinstance(enabled, bool):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="enabled bool required",
                )
            try:
                set_outage(enabled)
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="forbidden",
                ) from exc
            return {"status": "ok", "outage": "1" if enabled else "0"}

    return application


class _LazyApp:
    """Resolve ``create_app()`` on first ASGI use so import works without env."""

    def __init__(self) -> None:
        self._app: FastAPI | None = None

    def _get(self) -> FastAPI:
        if self._app is None:
            self._app = create_app()
        return self._app

    def __getattr__(self, name: str) -> object:
        return getattr(self._get(), name)

    async def __call__(self, scope: object, receive: object, send: object) -> None:
        app = self._get()
        await app(scope, receive, send)  # type: ignore[arg-type]


app = _LazyApp()
