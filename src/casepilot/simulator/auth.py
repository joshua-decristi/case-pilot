"""Service-token authentication for simulator domain endpoints."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, Request, status


def require_service_token(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Require ``Authorization: Bearer <simulator_service_token>``."""
    expected: str = request.app.state.service_token
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service token",
            headers={"WWW-Authenticate": "Bearer"},
        )
