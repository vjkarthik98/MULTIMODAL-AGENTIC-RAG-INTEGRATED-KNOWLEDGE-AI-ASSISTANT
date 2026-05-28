from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm

from app.auth.dependencies import get_current_user
from app.auth.jwt_handler import issue_tokens, refresh_access_token
from app.auth.models import LoginRequest, RefreshRequest, RegisterRequest, TokenPair, UserPublic
from app.auth.oauth import build_google_auth_url, exchange_google_code, get_or_create_oauth_user, google_oauth_enabled
from app.auth.service import AuthService
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])
_svc = AuthService()


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

@router.post("/login", response_model=TokenPair)
async def login(req: LoginRequest) -> TokenPair:
    """Authenticate with email + password. Returns JWT access + refresh tokens."""
    try:
        user = await asyncio.to_thread(_svc.authenticate, req.email, req.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )
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


@router.get("/callback/google", response_model=TokenPair, summary="Google OAuth2 callback")
async def google_callback(
    code: str = "",
    state: str = "",
    error: str = "",
) -> TokenPair:
    """
    Step 2 — Google redirects back here with ?code=...&state=...
    We exchange the code for user info, get-or-create the account,
    and return a JWT TokenPair exactly like a normal login.
    """
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Google login was cancelled or failed: {error}",
        )

    if not code or not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing code or state parameter from Google",
        )

    try:
        userinfo = await exchange_google_code(code, state)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )

    email = userinfo["email"]

    try:
        user = await asyncio.to_thread(get_or_create_oauth_user, email, "google")
    except Exception as exc:
        logger.error(event="google_oauth_user_create_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to create user account")

    tokens = issue_tokens(user.user_id, user.email, user.role.value)
    logger.info(event="google_oauth_login_success", email=email, user_id=user.user_id)
    return TokenPair(**tokens)


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

    logger.info(event="gdpr_self_purge_completed", user_id=user_id)

    return {
        "status": "ok",
        "message": "All your data has been purged and your account deactivated",
        "user_id": user_id,
    }
