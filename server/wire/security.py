"""Bearer token verification for protected endpoints."""

import secrets

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings

_bearer_scheme = HTTPBearer(auto_error=False)


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> None:
    """FastAPI dependency: raise 401 unless a valid Bearer token is provided.

    Uses a constant-time comparison to avoid leaking token length/content
    through timing side channels.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header.",
        )

    if not secrets.compare_digest(credentials.credentials, settings.API_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
        )


ADMIN_HEADER = "X-Admin-Token"


def verify_admin(x_admin_token: str = Header(default="", alias=ADMIN_HEADER)) -> None:
    """FastAPI dependency: raise unless the operator's admin token is sent.

    Always used BESIDE `verify_token`, never instead of it. A separate header
    rather than a second Bearer, so the dashboard keeps reading with the API
    token and only an action carries the admin one.

    403 when no admin token is configured at all: the controls are off on this
    deployment, which is a different answer from "wrong token" (401).
    """
    if not settings.ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator controls are disabled on this server: ADMIN_TOKEN is not set.",
        )
    if not x_admin_token or not secrets.compare_digest(x_admin_token, settings.ADMIN_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token.",
        )
