from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from app.auth.cookies import read_access_cookie
from app.auth.jwt_handler import verify_token
from app.auth.models import UserPublic, UserRole
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# auto_error=False: a missing Authorization header is not fatal here — the
# browser client authenticates via the httpOnly magik_access cookie instead
# (see _resolve_token below). Bearer stays supported for API/CLI/test clients.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login/form", auto_error=False)


def _resolve_token(request: Request, header_token: str | None) -> str | None:
    """Cookie first (the browser SPA), Authorization header as fallback (API
    clients, curl, CI, the test suite). Never trust both silently swapped —
    whichever is present wins; if both are present the cookie wins since it's
    what the browser actually sent, and a stale/forged header should not be
    able to override a legitimate cookie session."""
    return read_access_cookie(request) or header_token


def _build_user_from_payload(payload: dict) -> UserPublic:
    from datetime import datetime, timezone

    try:
        role = UserRole(payload.get("role", "user"))
    except ValueError:
        # Unrecognized role (e.g. a still-valid "guest" JWT issued before guest
        # mode was removed) — treat as unauthenticated rather than a 500.
        raise ValueError("Unrecognized role in token") from None

    return UserPublic(
        user_id=payload["sub"],
        email=payload["email"],
        role=role,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )


async def get_current_user(
    request: Request, header_token: str | None = Depends(oauth2_scheme)
) -> UserPublic:
    """
    FastAPI dependency — extracts and verifies the JWT (from the httpOnly
    access cookie, or an Authorization: Bearer header for non-browser clients).
    Returns the authenticated UserPublic. Raises HTTP 401 on failure.

    Used on every protected route to ensure user_id is always JWT-sourced,
    never from a form field or header that a caller could forge.
    """
    token = _resolve_token(request, header_token)
    if not settings.AUTH_ENABLED:
        # Dev bypass: return the default dev user when auth is disabled
        from datetime import datetime, timezone

        return UserPublic(
            user_id=settings.DEFAULT_DEV_USER_ID,
            email="dev@local",
            role=UserRole.USER,
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )

    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = verify_token(token, expected_type="access")
        return _build_user_from_payload(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_admin_user(
    current_user: UserPublic = Depends(get_current_user),
) -> UserPublic:
    """Dependency that additionally requires the ADMIN role."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


async def optional_current_user(
    request: Request, header_token: str | None = Depends(oauth2_scheme),
) -> UserPublic | None:
    """
    Like get_current_user but returns None instead of raising 401.
    Use on public endpoints that behave differently when authenticated.
    """
    token = _resolve_token(request, header_token)
    if not settings.AUTH_ENABLED or token is None:
        return None
    try:
        payload = verify_token(token, expected_type="access")
        return _build_user_from_payload(payload)
    except ValueError:
        return None
