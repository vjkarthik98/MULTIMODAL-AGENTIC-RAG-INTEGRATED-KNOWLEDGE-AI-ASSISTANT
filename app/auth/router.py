from __future__ import annotations

import asyncio
import os
import urllib.parse
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm

from app.auth.dependencies import get_current_user
from app.auth.jwt_handler import issue_tokens, refresh_access_token, verify_token
from app.auth.models import LoginRequest, LogoutRequest, RefreshRequest, RegisterRequest, TokenPair, UserPublic
from app.auth.mfa import MFAService
from app.auth.oauth import build_google_auth_url, exchange_google_code, get_or_create_oauth_user, google_oauth_enabled
from app.auth.service import AuthService
from app.auth.token_blacklist import revoke_all_user_tokens, revoke_token
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])
_svc  = AuthService()
_mfa  = MFAService()


# ── Register ──────────────────────────────────────────────────────────────────

@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest) -> UserPublic:
    """Create a new account using email + password."""
    try:
        user = await asyncio.to_thread(_svc.register, req)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return user


# ── Login (JSON body) ─────────────────────────────────────────────────────────

@router.post("/login")
async def login(req: LoginRequest):
    """
    Authenticate with email + password.
    If MFA is enabled, returns {"mfa_required": true, "mfa_token": "..."}.
    Otherwise returns a full TokenPair.
    """
    try:
        user = await asyncio.to_thread(_svc.authenticate, req.email, req.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if MFA is enabled for this user
    mfa_enabled = await asyncio.to_thread(_mfa.is_enabled, user.user_id)
    if mfa_enabled:
        mfa_token = await asyncio.to_thread(_mfa.begin_login, user.user_id)
        return {"mfa_required": True, "mfa_token": mfa_token}

    tokens = issue_tokens(user.user_id, user.email, user.role.value)
    return TokenPair(**tokens)


# ── Login (OAuth2 form — enables /docs "Authorize" button) ───────────────────

@router.post("/login/form", response_model=TokenPair, include_in_schema=False)
async def login_form(form: OAuth2PasswordRequestForm = Depends()) -> TokenPair:
    """OAuth2 password flow — used by the interactive API docs."""
    try:
        user = await asyncio.to_thread(_svc.authenticate, form.username, form.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )
    tokens = issue_tokens(user.user_id, user.email, user.role.value)
    return TokenPair(**tokens)


# ── Refresh ───────────────────────────────────────────────────────────────────

@router.post("/refresh", response_model=TokenPair)
async def refresh(req: RefreshRequest) -> TokenPair:
    """Exchange a valid refresh token for a new access + refresh token pair."""
    try:
        tokens = await asyncio.to_thread(refresh_access_token, req.refresh_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenPair(**tokens)


# ── Current user profile ──────────────────────────────────────────────────────

@router.get("/me", response_model=UserPublic)
async def me(current_user: UserPublic = Depends(get_current_user)) -> UserPublic:
    """Return the currently authenticated user's profile."""
    return current_user


# ── Google OAuth2 ─────────────────────────────────────────────────────────────

@router.get("/google", summary="Sign in with Google")
async def google_login() -> RedirectResponse:
    """
    Step 1 — Redirect the user to Google's consent screen.
    The browser follows this redirect automatically.
    """
    if not google_oauth_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google login is not configured on this server",
        )
    try:
        url, _ = build_google_auth_url()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return RedirectResponse(url=url)


@router.get("/callback/google", summary="Google OAuth2 callback")
async def google_callback(
    code: str = "",
    state: str = "",
    error: str = "",
) -> RedirectResponse:
    """
    Step 2 — Google redirects back here with ?code=...&state=...
    Exchanges the code for user info, get-or-creates the account, issues a JWT,
    then redirects the browser to the Gradio UI with the token in URL params.
    """
    _gradio_url = os.getenv("FRONTEND_URL", os.getenv("GRADIO_URL", "http://localhost:5173"))

    if error:
        return RedirectResponse(
            url=f"{_gradio_url}?oauth_error={urllib.parse.quote(error, safe='')}",
            status_code=302,
        )

    if not code or not state:
        return RedirectResponse(
            url=f"{_gradio_url}?oauth_error=missing_params",
            status_code=302,
        )

    try:
        userinfo = await exchange_google_code(code, state)
    except ValueError as exc:
        return RedirectResponse(
            url=f"{_gradio_url}?oauth_error={urllib.parse.quote(str(exc), safe='')}",
            status_code=302,
        )

    email = userinfo["email"]

    try:
        user = await asyncio.to_thread(get_or_create_oauth_user, email, "google")
    except Exception as exc:
        logger.error(event="google_oauth_user_create_failed", error=str(exc))
        return RedirectResponse(
            url=f"{_gradio_url}?oauth_error=account_creation_failed",
            status_code=302,
        )

    tokens = issue_tokens(user.user_id, user.email, user.role.value)
    logger.info(event="google_oauth_login_success", email=email, user_id=user.user_id)

    redirect_url = (
        f"{_gradio_url}"
        f"?magik_token={urllib.parse.quote(tokens['access_token'], safe='')}"
        f"&magik_refresh={urllib.parse.quote(tokens['refresh_token'], safe='')}"
        f"&magik_email={urllib.parse.quote(email, safe='')}"
    )
    return RedirectResponse(url=redirect_url, status_code=302)


# ── Logout (revoke current token) ────────────────────────────────────────────

@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(
    request: Request,
    body: Optional[LogoutRequest] = None,
    current_user: UserPublic = Depends(get_current_user),
) -> dict:
    """Revoke the current access token — and the refresh token, if surrendered —
    immediately via the Redis blacklist, so logout fully ends the session rather
    than leaving a long-lived refresh token usable until it naturally expires."""
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.removeprefix("Bearer ").strip()
    if token:
        try:
            payload = verify_token(token, expected_type="access")
            jti = payload.get("jti", "")
            exp = payload.get("exp", 0)
            if jti:
                revoke_token(jti, exp)
        except ValueError:
            pass   # already invalid — no-op

    if body and body.refresh_token:
        try:
            r_payload = verify_token(body.refresh_token, expected_type="refresh")
            r_jti = r_payload.get("jti", "")
            r_exp = r_payload.get("exp", 0)
            if r_jti:
                revoke_token(r_jti, r_exp)
        except ValueError:
            pass   # already invalid — no-op

    logger.info(event="user_logged_out", user_id=current_user.user_id)
    return {"status": "ok", "message": "Logged out successfully"}


# ── Logout-all (revoke ALL tokens via generation bump) ────────────────────────

@router.post("/logout-all", status_code=status.HTTP_200_OK)
async def logout_all(
    current_user: UserPublic = Depends(get_current_user),
) -> dict:
    """Invalidate ALL active tokens for this user (useful if account compromised)."""
    revoke_all_user_tokens(current_user.user_id)
    logger.info(event="user_all_tokens_revoked", user_id=current_user.user_id)
    return {"status": "ok", "message": "All sessions have been terminated"}


# ── Password change ───────────────────────────────────────────────────────────

from pydantic import BaseModel as _BM, Field as _F


class PasswordChangeRequest(_BM):
    current_password: str = _F(..., min_length=1, max_length=128)
    new_password:     str = _F(..., min_length=8, max_length=128)


@router.post("/password", status_code=status.HTTP_200_OK)
async def change_password(
    req: PasswordChangeRequest,
    current_user: UserPublic = Depends(get_current_user),
) -> dict:
    """
    Change password after verifying the current one.
    Revokes ALL existing tokens (forces re-login on all devices).
    """
    try:
        await asyncio.to_thread(
            _svc.change_password,
            current_user.user_id,
            req.current_password,
            req.new_password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # Invalidate all tokens — user must log in again
    revoke_all_user_tokens(current_user.user_id)
    logger.info(event="password_changed", user_id=current_user.user_id)
    return {"status": "ok", "message": "Password changed. All sessions have been terminated."}


# ── MFA — Enrol ───────────────────────────────────────────────────────────────

class MFACodeRequest(_BM):
    code: str = _F(..., min_length=6, max_length=10)


class MFAVerifyLoginRequest(_BM):
    mfa_token: str = _F(..., min_length=10)
    code:      str = _F(..., min_length=6, max_length=10)


@router.post("/mfa/enroll", status_code=status.HTTP_200_OK)
async def mfa_enroll(
    current_user: UserPublic = Depends(get_current_user),
) -> dict:
    """
    Step 1 of MFA enrolment.
    Returns TOTP secret + QR code URI. Show the QR in the UI.
    User must scan with an authenticator app then confirm via /mfa/verify-enroll.
    """
    try:
        data = await asyncio.to_thread(_mfa.enroll_start, current_user.user_id, current_user.email)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return data


@router.post("/mfa/verify-enroll", status_code=status.HTTP_200_OK)
async def mfa_verify_enroll(
    req: MFACodeRequest,
    current_user: UserPublic = Depends(get_current_user),
) -> dict:
    """
    Step 2 of MFA enrolment — confirm with first TOTP code.
    Returns single-use backup codes (show once, store safely).
    """
    try:
        backup_codes = await asyncio.to_thread(_mfa.verify_enroll, current_user.user_id, req.code)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return {
        "status": "ok",
        "message": "MFA enabled successfully",
        "backup_codes": backup_codes,
    }


@router.post("/mfa/verify", response_model=TokenPair)
async def mfa_verify_login(req: MFAVerifyLoginRequest) -> TokenPair:
    """
    After password login returns mfa_required=true, verify the TOTP code here.
    Returns the full JWT access + refresh token pair.
    """
    try:
        user_id = await asyncio.to_thread(_mfa.verify_login, req.mfa_token, req.code)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = _svc.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    tokens = issue_tokens(user.user_id, user.email, user.role.value)
    return TokenPair(**tokens)


@router.post("/mfa/disable", status_code=status.HTTP_200_OK)
async def mfa_disable(
    req: MFACodeRequest,
    current_user: UserPublic = Depends(get_current_user),
) -> dict:
    """Disable MFA after confirming with a valid TOTP code."""
    try:
        await asyncio.to_thread(_mfa.disable, current_user.user_id, req.code)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return {"status": "ok", "message": "MFA disabled"}


@router.get("/mfa/status", status_code=status.HTTP_200_OK)
async def mfa_status(
    current_user: UserPublic = Depends(get_current_user),
) -> dict:
    """Check whether MFA is enabled for the current user."""
    enabled = await asyncio.to_thread(_mfa.is_enabled, current_user.user_id)
    return {"mfa_enabled": enabled}


# ── GDPR self-delete ──────────────────────────────────────────────────────────

@router.delete("/me", status_code=status.HTTP_200_OK)
async def delete_me(
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> dict:
    """
    GDPR right-to-erasure: purge all data for the authenticated user
    from Qdrant, Redis, and MongoDB, then deactivate the account.
    """
    user_id = current_user.user_id

    try:
        from app.memory.memory_manager import MemoryManager
        manager = MemoryManager()
        await asyncio.to_thread(manager.gdpr_purge, user_id)
    except Exception as exc:
        logger.warning(event="gdpr_memory_purge_failed", user_id=user_id, error=str(exc))

    try:
        from app.core.infra_registry import infra
        bm25 = infra.get_bm25()
        if bm25 and hasattr(bm25, "purge_by_session"):
            await asyncio.to_thread(bm25.purge_by_session, user_id)
    except Exception as exc:
        logger.warning(event="gdpr_bm25_purge_failed", user_id=user_id, error=str(exc))

    try:
        await asyncio.to_thread(_svc.deactivate, user_id)
    except Exception as exc:
        logger.warning(event="gdpr_deactivate_failed", user_id=user_id, error=str(exc))

    # Revoke all tokens — account is gone
    revoke_all_user_tokens(user_id)

    logger.info(event="gdpr_self_purge_completed", user_id=user_id)

    return {
        "status": "ok",
        "message": "All your data has been purged and your account deactivated",
        "user_id": user_id,
    }
