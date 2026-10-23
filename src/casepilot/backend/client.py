"""Authenticated HTTP client for the enterprise simulator."""

from __future__ import annotations

from typing import Any

import httpx

from casepilot.shared.config import Settings
from casepilot.shared.contracts import (
    CaseRecord,
    CreateFollowUpTaskRequest,
    CreateFollowUpTaskResponse,
    DocumentListResponse,
    EventListResponse,
    SimulatorRoutes,
)


class SimulatorError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SimulatorClient:
    """Case-scoped reads + service-auth follow-up writes."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._owns_client = client is None
        token = settings.simulator_service_token.get_secret_value()
        self._client = client or httpx.AsyncClient(
            base_url=settings.simulator_base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.tool_http_timeout_seconds),
            headers={"Authorization": f"Bearer {token}"},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, path: str) -> Any:
        last_exc: Exception | None = None
        for attempt in range(2):  # one retry on transient transport failure
            try:
                resp = await self._client.get(path)
                if resp.status_code == 404:
                    raise SimulatorError(f"not found: {path}", status_code=404)
                if resp.status_code in {401, 403}:
                    raise SimulatorError(
                        "simulator authentication failed",
                        status_code=resp.status_code,
                    )
                if resp.status_code >= 500:
                    raise SimulatorError(
                        f"simulator unavailable ({resp.status_code})",
                        status_code=resp.status_code,
                    )
                if resp.status_code >= 400:
                    raise SimulatorError(
                        f"simulator error {resp.status_code}: {resp.text[:200]}",
                        status_code=resp.status_code,
                    )
                return resp.json()
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt == 0:
                    continue
                raise SimulatorError(f"simulator transport error: {exc}") from exc
        raise SimulatorError(f"simulator transport error: {last_exc}")

    async def list_cases(self) -> list[CaseRecord]:
        data = await self._get(SimulatorRoutes.CASES)
        if not isinstance(data, list):
            raise SimulatorError("simulator list_cases: expected array response")
        return [CaseRecord.model_validate(item) for item in data]

    async def get_case(self, case_id: str) -> CaseRecord:
        data = await self._get(SimulatorRoutes.CASE.format(case_id=case_id))
        return CaseRecord.model_validate(data)

    async def get_documents(self, case_id: str) -> DocumentListResponse:
        data = await self._get(SimulatorRoutes.DOCUMENTS.format(case_id=case_id))
        return DocumentListResponse.model_validate(data)

    async def get_events(self, case_id: str) -> EventListResponse:
        data = await self._get(SimulatorRoutes.EVENTS.format(case_id=case_id))
        return EventListResponse.model_validate(data)

    async def create_follow_up_task(
        self, case_id: str, body: CreateFollowUpTaskRequest
    ) -> CreateFollowUpTaskResponse:
        path = SimulatorRoutes.FOLLOW_UP_TASKS.format(case_id=case_id)
        try:
            resp = await self._client.post(path, json=body.model_dump(mode="json"))
        except httpx.TransportError as exc:
            raise SimulatorError(f"simulator transport error: {exc}") from exc
        if resp.status_code in {401, 403}:
            raise SimulatorError("simulator authentication failed", status_code=resp.status_code)
        if resp.status_code == 409:
            raise SimulatorError(
                f"simulator idempotency conflict: {resp.text[:200]}",
                status_code=409,
            )
        if resp.status_code >= 500:
            raise SimulatorError(
                f"simulator unavailable ({resp.status_code})",
                status_code=resp.status_code,
            )
        if resp.status_code >= 400:
            raise SimulatorError(
                f"simulator error {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
            )
        return CreateFollowUpTaskResponse.model_validate(resp.json())
