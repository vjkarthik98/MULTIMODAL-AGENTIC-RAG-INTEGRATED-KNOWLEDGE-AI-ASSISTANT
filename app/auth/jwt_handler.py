from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from jose import JWTError, jwt

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ALGORITHM = settings.JWT_ALGORITHM
_SECRET = settings.JWT_SECRET_KEY


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def issue_tokens(user_id: str, email: str, role: str) -> Dict[str, Any]:
    """Issue an access + refresh token pair for the given user."""
    now = _utcnow()
    access_exp = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_exp = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    access_payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "type": "access",
        "jti": str(uuid.uuid4()),
        "iat": int(now.timestamp()),
        "exp": int(access_exp.timestamp()),
    }

    refresh_payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "type": "refresh",
        "jti": str(uuid.uuid4()),
        "iat": int(now.timestamp()),
        "exp": int(refresh_exp.timestamp()),
    }

    access_token = jwt.encode(access_payload, _SECRET, algorithm=_ALGORITHM)
    refresh_token = jwt.encode(refresh_payload, _SECRET, algorithm=_ALGORITHM)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


def verify_token(token: str, expected_type: str = "access") -> Dict[str, Any]:
    """
    Decode and verify a JWT. Returns the payload dict.
    Raises ValueError with a safe message on any failure.
    """
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
    except JWTError as exc:
        logger.warning(event="jwt_verify_failed", error=str(exc))
        raise ValueError("Invalid or expired token") from exc

    if payload.get("type") != expected_type:
        raise ValueError(f"Expected {expected_type} token, got {payload.get('type')}")

    if not payload.get("sub"):
        raise ValueError("Token missing subject")

    return payload


def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    """Exchange a valid refresh token for a new access token."""
    payload = verify_token(refresh_token, expected_type="refresh")
    return issue_tokens(
        user_id=payload["sub"],
        email=payload["email"],
        role=payload["role"],
    )
