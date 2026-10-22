"""CasePilot operator CLI — thin HTTPX client over the investigation backend."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, TextIO

import httpx
from pydantic import BaseModel, ValidationError

from casepilot.shared import (
    ApproveProposalRequest,
    BackendRoutes,
    Proposal,
    ProposalDecisionResponse,
    RejectProposalRequest,
    RunRecord,
    StartInvestigationRequest,
    StartInvestigationResponse,
    get_settings,
)
from casepilot.shared.config import Settings

DEFAULT_OBJECTIVE = "Investigate why this case is stalled and recommend the next action."

# CLI must outlive the backend's own investigation deadline so the server can
# return a structured timeout/failure instead of the client hanging up first.
CLIENT_TIMEOUT_SLACK_SECONDS = 15.0


def client_timeout_seconds(settings: Settings) -> float:
    return settings.investigation_timeout_seconds + CLIENT_TIMEOUT_SLACK_SECONDS


class ApiError(Exception):
    """Explicit backend/transport failure for nonzero CLI exit."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        text = response.text.strip()
        return text or f"HTTP {response.status_code}"
    if isinstance(data, dict) and "detail" in data:
        detail = data["detail"]
        if isinstance(detail, str):
            return detail
        return json.dumps(detail)
    return json.dumps(data)


def _human_lines(value: Any, *, indent: int = 0) -> list[str]:
    prefix = "  " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_human_lines(item, indent=indent + 1))
            else:
                lines.append(f"{prefix}{key}: {item}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_human_lines(item, indent=indent + 1))
            else:
                lines.append(f"{prefix}- {item}")
        return lines
    return [f"{prefix}{value}"]


def _find_proposal_id(payload: Any) -> str | None:
    """Pull optional proposal_id from investigation/proposal-shaped payloads."""
    if not isinstance(payload, dict):
        return None
    top = payload.get("proposal_id")
    if isinstance(top, str) and top:
        return top
    result = payload.get("result")
    if isinstance(result, dict):
        nested = result.get("proposal_id")
        if isinstance(nested, str) and nested:
            return nested
    proposal = payload.get("proposal")
    if isinstance(proposal, dict):
        nested = proposal.get("proposal_id")
        if isinstance(nested, str) and nested:
            return nested
    return None


def format_output(model: BaseModel, *, as_json: bool) -> str:
    payload = model.model_dump(mode="json")
    if as_json:
        return json.dumps(payload, indent=2)
    lines = _human_lines(payload)
    # Surface proposal_id first so operators can approve without digging JSON.
    proposal_id = _find_proposal_id(payload)
    if proposal_id is not None:
        lines = [
            f"proposal_id: {proposal_id}",
            f"Approve with: casepilot approve {proposal_id}",
            "",
        ] + lines
    return "\n".join(lines)


class BackendClient:
    """HTTP client for backend routes; operator token only on decision endpoints."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self._settings = settings
        base = settings.backend_base_url.rstrip("/") or (
            f"http://{settings.backend_host}:{settings.backend_port}"
        )
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base,
            timeout=httpx.Timeout(client_timeout_seconds(settings)),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> BackendClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _headers(self, *, operator: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if operator:
            token = self._settings.operator_token.get_secret_value()
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        operator: bool = False,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = self._client.request(
                method,
                path,
                headers=self._headers(operator=operator),
                json=json_body,
            )
        except httpx.RequestError as exc:
            raise ApiError(f"Backend unreachable: {exc}") from exc

        if response.status_code >= 400:
            raise ApiError(
                _error_detail(response),
                status_code=response.status_code,
            )
        if response.status_code == 204 or not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError("Backend returned non-JSON response") from exc

    def investigate(self, case_id: str, objective: str) -> StartInvestigationResponse:
        body = StartInvestigationRequest(case_id=case_id, objective=objective)
        data = self._request(
            "POST",
            BackendRoutes.INVESTIGATIONS,
            json_body=body.model_dump(mode="json"),
        )
        try:
            return StartInvestigationResponse.model_validate(data)
        except ValidationError as exc:
            raise ApiError(f"Invalid investigation response: {exc}") from exc

    def get_run(self, run_id: str) -> RunRecord:
        path = BackendRoutes.RUN.format(run_id=run_id)
        data = self._request("GET", path)
        try:
            return RunRecord.model_validate(data)
        except ValidationError as exc:
            raise ApiError(f"Invalid run response: {exc}") from exc

    def get_proposal(self, proposal_id: str) -> Proposal:
        path = BackendRoutes.PROPOSAL.format(proposal_id=proposal_id)
        data = self._request("GET", path)
        try:
            return Proposal.model_validate(data)
        except ValidationError as exc:
            raise ApiError(f"Invalid proposal response: {exc}") from exc

    def approve(self, proposal_id: str) -> ProposalDecisionResponse:
        path = BackendRoutes.APPROVE.format(proposal_id=proposal_id)
        body = ApproveProposalRequest()
        data = self._request(
            "POST",
            path,
            operator=True,
            json_body=body.model_dump(mode="json"),
        )
        try:
            return ProposalDecisionResponse.model_validate(data)
        except ValidationError as exc:
            raise ApiError(f"Invalid approve response: {exc}") from exc

    def reject(self, proposal_id: str, reason: str | None = None) -> ProposalDecisionResponse:
        path = BackendRoutes.REJECT.format(proposal_id=proposal_id)
        body = RejectProposalRequest(reason=reason)
        data = self._request(
            "POST",
            path,
            operator=True,
            json_body=body.model_dump(mode="json", exclude_none=True),
        )
        try:
            return ProposalDecisionResponse.model_validate(data)
        except ValidationError as exc:
            raise ApiError(f"Invalid reject response: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text",
    )

    parser = argparse.ArgumentParser(
        prog="casepilot",
        description="CasePilot local operator CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inv = sub.add_parser(
        "investigate",
        parents=[shared],
        help="Start an investigation for a case",
    )
    p_inv.add_argument("case_id", help="Case identifier")
    p_inv.add_argument(
        "--objective",
        default=DEFAULT_OBJECTIVE,
        help="Bounded investigation objective",
    )

    p_run = sub.add_parser(
        "run",
        parents=[shared],
        help="Fetch a stored investigation run",
    )
    p_run.add_argument("run_id", help="Investigation run identifier")

    p_prop = sub.add_parser(
        "proposal",
        parents=[shared],
        help="Fetch an action proposal",
    )
    p_prop.add_argument("proposal_id", help="Proposal identifier")

    p_app = sub.add_parser(
        "approve",
        parents=[shared],
        help="Approve a pending proposal (operator token)",
    )
    p_app.add_argument("proposal_id", help="Proposal identifier")

    p_rej = sub.add_parser(
        "reject",
        parents=[shared],
        help="Reject a pending proposal (operator token)",
    )
    p_rej.add_argument("proposal_id", help="Proposal identifier")
    p_rej.add_argument("--reason", default=None, help="Optional rejection reason")

    return parser


def run(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    http_client: httpx.Client | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Parse argv, call backend, print result. Returns process exit code."""
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = settings if settings is not None else get_settings()
    except ValidationError as exc:
        print(f"Configuration error: {exc}", file=err)
        return 2

    as_json = bool(getattr(args, "json", False))

    try:
        with BackendClient(cfg, client=http_client) as client:
            if args.command == "investigate":
                result: BaseModel = client.investigate(args.case_id, args.objective)
            elif args.command == "run":
                result = client.get_run(args.run_id)
            elif args.command == "proposal":
                result = client.get_proposal(args.proposal_id)
            elif args.command == "approve":
                result = client.approve(args.proposal_id)
            elif args.command == "reject":
                result = client.reject(args.proposal_id, reason=args.reason)
            else:  # pragma: no cover - argparse enforces choices
                print(f"Unknown command: {args.command}", file=err)
                return 2
            print(format_output(result, as_json=as_json), file=out)
    except ApiError as exc:
        status = f" (HTTP {exc.status_code})" if exc.status_code is not None else ""
        print(f"Error{status}: {exc.message}", file=err)
        return 1

    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(argv))


if __name__ == "__main__":
    main()
