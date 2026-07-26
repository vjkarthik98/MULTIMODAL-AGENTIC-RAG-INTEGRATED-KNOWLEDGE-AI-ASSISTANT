"""Guest mode endpoints.

POST /auth/guest          — create ephemeral guest session (public)
GET  /auth/guest/limits   — get remaining allowances (guest JWT required)
POST /auth/guest/convert  — register with email/password and migrate guest data
POST /auth/guest/migrate  — migrate guest data after Google OAuth (real JWT required)

Architecture note: Google OAuth conversion uses two calls:
  1. Normal /auth/google flow → frontend receives real JWT
  2. POST /auth/guest/migrate with real JWT (Authorization header) + guest_token in body
This keeps the existing OAuth callback unchanged and avoids complexity of
passing state through the OAuth redirect round-trip.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user, require_real_user
from app.auth.guest_service import (
    check_guest_create_rate,
    create_guest_session,
    get_guest_limits,
    migrate_guest_to_user,
)
from app.auth.jwt_handler import issue_guest_token, issue_tokens, verify_token
from app.auth.models import GuestConvertRequest, GuestSessionResponse, UserPublic, UserRole
from app.utils.logger import get_logger
from app.utils.net import resolve_client_ip as _client_ip

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["guest"])


# ── POST /auth/guest ──────────────────────────────────────────────────────────


@router.post("/guest", status_code=201, response_model=GuestSessionResponse)
async def create_guest(request: Request) -> GuestSessionResponse:
    """Create an ephemeral guest session.

    Returns a guest JWT valid for GUEST_SESSION_TTL_HOURS hours.
    No login required. IP-rate-limited to prevent abuse (5 per IP per 10 min).
    """
    client_ip = _client_ip(request)
    if not await asyncio.to_thread(check_guest_create_rate, client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many guest sessions created from this IP. Please try again later.",
        )

    guest_id = await asyncio.to_thread(create_guest_session)
    token_data = issue_guest_token(guest_id)
    limits = await asyncio.to_thread(get_guest_limits, guest_id)

    return GuestSessionResponse(
        access_token=token_data["access_token"],
        token_type="bearer",
        expires_in=token_data["expires_in"],
        guest_user_id=guest_id,
        queries_left=limits["queries_left"],
        uploads_left=limits["uploads_left"],
    )


# ── GET /auth/guest/limits ────────────────────────────────────────────────────


@router.get("/guest/limits")
async def guest_limits(
    current_user: UserPublic = Depends(get_current_user),
) -> dict:
    """Return current guest usage and remaining allowances.

    Requires a valid guest JWT. Real user tokens return zeroed limits to avoid
    frontend display confusion.
    """
    if current_user.role != UserRole.GUEST:
        return {"queries_left": 0, "uploads_left": 0, "queries_used": 0, "uploads_used": 0}
    limits = await asyncio.to_thread(get_guest_limits, current_user.user_id)
    return limits


# ── POST /auth/guest/convert (email/password path) ───────────────────────────


@router.post("/guest/convert")
async def convert_guest(req: GuestConvertRequest) -> dict:
    """Convert a guest session to a full account using email + password.

    Validates the guest JWT, registers and immediately activates a new account
    (skipping OTP — the guest already demonstrated intent through product use),
    migrates all guest data (Qdrant, BM25, filesystem) to the new user_id, then
    returns a full JWT pair.

    For Google OAuth conversion use POST /auth/guest/migrate instead.
    """
    if not req.email or not req.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide email and password. For Google sign-in use POST /auth/guest/migrate.",
        )

    # Validate the guest token
    guest_id = _validate_guest_token(req.guest_token)

    # Register + immediately activate (skip OTP for guest conversion)
    from app.auth.models import RegisterRequest
    from app.auth.service import AuthService

    svc = AuthService()
    try:
        real_user = await asyncio.to_thread(
            svc.register_and_activate,
            RegisterRequest(email=req.email, password=req.password),
        )
    except ValueError as exc:
        msg = str(exc)
        if "already exists" in msg.lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=msg) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg) from exc

    # Migrate guest data to real user
    stats = await asyncio.to_thread(migrate_guest_to_user, guest_id, real_user.user_id)

    # Revoke guest JWT
    _revoke_guest_jwt(req.guest_token)

    # Issue real JWT pair
    tokens = issue_tokens(real_user.user_id, real_user.email, real_user.role.value)

    logger.info(
        event="guest_converted_email",
        guest_id=guest_id,
        real_user_id=real_user.user_id,
        migration_stats=stats,
    )

    return {
        **tokens,
        "email": real_user.email,
        "migration_stats": stats,
    }


# ── POST /auth/guest/migrate (Google OAuth path) ─────────────────────────────


class GuestMigrateRequest(BaseModel):
    guest_token: str = Field(..., min_length=10)


@router.post("/guest/migrate")
async def migrate_guest(
    req: GuestMigrateRequest,
    current_user: UserPublic = Depends(require_real_user),
) -> dict:
    """Migrate guest data to an already-authenticated real user account.

    Called by the frontend after a successful Google OAuth login, when the user
    had an active guest session. Requires a real user JWT in the Authorization
    header plus the guest_token in the request body.
    """
    guest_id = _validate_guest_token(req.guest_token)

    stats = await asyncio.to_thread(migrate_guest_to_user, guest_id, current_user.user_id)

    _revoke_guest_jwt(req.guest_token)

    logger.info(
        event="guest_migrated_google",
        guest_id=guest_id,
        real_user_id=current_user.user_id,
        migration_stats=stats,
    )

    return {"migrated": True, "migration_stats": stats}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_guest_token(token: str) -> str:
    """Decode and validate a guest JWT. Returns guest_user_id."""
    try:
        payload = verify_token(token, expected_type="access")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    if payload.get("role") != "guest":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provided token is not a guest token.",
        )
    return payload["sub"]


def _revoke_guest_jwt(token: str) -> None:
    """Add the guest token's JTI to the blacklist immediately."""
    try:
        from jose import jwt as _jwt

        from app.core.config import settings

        payload = _jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        jti = payload.get("jti", "")
        exp = payload.get("exp", 0)
        if jti and exp:
            from app.auth.token_blacklist import revoke_token

            revoke_token(jti, exp)
    except Exception:
        pass
