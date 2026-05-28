from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.auth.jwt_handler import verify_token
from app.auth.models import UserPublic, UserRole
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login/form", auto_error=False)


def _build_user_from_payload(payload: dict) -> UserPublic:
    from datetime import datetime, timezone
    return UserPublic(
        user_id=payload["sub"],
        email=payload["email"],
        role=UserRole(payload.get("role", "user")),
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> UserPublic:
    """
    FastAPI dependency — extracts and verifies the JWT bearer token.
    Returns the authenticated UserPublic. Raises HTTP 401 on failure.

    Used on every protected route to ensure user_id is always JWT-sourced,
    never from a form field or header that a caller could forge.
    """
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
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return _build_user_from_payload(payload)


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
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[UserPublic]:
    """
    Like get_current_user but returns None instead of raising 401.
    Use on public endpoints that behave differently when authenticated.
    """
    if not settings.AUTH_ENABLED or token is None:
        return None
    try:
        payload = verify_token(token, expected_type="access")
        return _build_user_from_payload(payload)
    except ValueError:
        return None
