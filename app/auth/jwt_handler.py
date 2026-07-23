from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ALGORITHM = settings.JWT_ALGORITHM
_SECRET = settings.JWT_SECRET_KEY


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def issue_tokens(user_id: str, email: str, role: str) -> dict[str, Any]:
    """Issue an access + refresh token pair for the given user."""
    from app.auth.token_blacklist import get_user_token_generation

    now = _utcnow()
    access_exp = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_exp = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    # gen = token generation counter; incremented on password change / logout-all.
    # Tokens with gen < current generation are rejected even if not individually revoked.
    gen = get_user_token_generation(user_id)

    access_payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "type": "access",
        "jti": str(uuid.uuid4()),
        "gen": gen,
        "iat": int(now.timestamp()),
        "exp": int(access_exp.timestamp()),
    }

    refresh_payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "type": "refresh",
        "jti": str(uuid.uuid4()),
        "gen": gen,
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


def verify_token(token: str, expected_type: str = "access") -> dict[str, Any]:
    """
    Decode and verify a JWT.
    Also checks the Redis blacklist (individual revocation) and token
    generation (bulk revocation on password change / logout-all).
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

    # ── Blacklist check (individual revocation via logout) ────────────────────
    jti = payload.get("jti", "")
    if jti:
        from app.auth.token_blacklist import is_revoked

        if is_revoked(jti):
            logger.warning(event="jwt_revoked_token_rejected", jti=jti[:8])
            raise ValueError("Token has been revoked")

    # ── Generation check (bulk revocation via password change / logout-all) ──
    user_id = payload["sub"]
    token_gen = payload.get("gen", 0)
    from app.auth.token_blacklist import get_user_token_generation

    current_gen = get_user_token_generation(user_id)
    if current_gen > 0 and token_gen < current_gen:
        logger.warning(
            event="jwt_stale_generation_rejected",
            user_id=user_id,
            token_gen=token_gen,
            current_gen=current_gen,
        )
        raise ValueError("Token has been invalidated — please log in again")

    return payload


def issue_guest_token(guest_user_id: str) -> dict[str, Any]:
    """Issue a single short-lived access token for an ephemeral guest session.

    No refresh token — guest sessions are not renewable; they convert or expire.
    verify_token() requires no changes: it validates type=access, JTI blacklist,
    and expiry exactly as it does for real user tokens.
    """
    now = _utcnow()
    ttl_sec = settings.GUEST_SESSION_TTL_HOURS * 3600
    exp = now + timedelta(seconds=ttl_sec)
    payload = {
        "sub": guest_user_id,
        "email": "",
        "role": "guest",
        "type": "access",
        "jti": str(uuid.uuid4()),
        "gen": 0,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    token = jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": ttl_sec,
    }


def refresh_access_token(refresh_token: str) -> dict[str, Any]:
    """Exchange a valid refresh token for a new access token.

    Implements refresh token rotation: the consumed refresh token's JTI is
    revoked immediately after verification so each refresh token can only be
    used once.
    """
    payload = verify_token(refresh_token, expected_type="refresh")

    # Revoke the consumed refresh token (rotation — one-use enforcement)
    from app.auth.token_blacklist import revoke_token

    jti = payload.get("jti", "")
    exp = payload.get("exp", 0)
    if jti and exp:
        revoke_token(jti, exp)

    return issue_tokens(
        user_id=payload["sub"],
        email=payload["email"],
        role=payload["role"],
    )
