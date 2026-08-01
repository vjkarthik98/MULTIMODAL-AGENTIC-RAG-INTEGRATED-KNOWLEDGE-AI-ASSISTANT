from __future__ import annotations

import asyncio
import os
import urllib.parse

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user
from app.auth.jwt_handler import issue_tokens, refresh_access_token, verify_token
from app.auth.mfa import MFAService, _issue_mfa_token
from app.auth.models import (
    ForgotPasswordRequest,
    LoginRequest,
    LogoutRequest,
    OTPResendRequest,
    OTPVerifyRequest,
    RefreshRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenPair,
    UserPublic,
)
from app.auth.oauth import (
    build_google_auth_url,
    exchange_google_code,
    get_or_create_oauth_user,
    google_oauth_enabled,
)
from app.auth.service import AuthService
from app.auth.token_blacklist import revoke_all_user_tokens, revoke_token
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])
_svc = AuthService()
_mfa = MFAService()


# ── Register ──────────────────────────────────────────────────────────────────


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    """
    Create an inactive account and send an OTP to verify email ownership.
    Returns {"otp_required": true, "otp_token": "..."} — the client must
    call /auth/verify-otp to activate the account and receive full tokens.
    If OTP sending fails, the account is deleted so the user can retry cleanly.
    """
    from app.auth.email_service import send_otp_email
    from app.auth.otp_store import generate_otp, store_otp

    try:
        user = await asyncio.to_thread(_svc.register, req)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    otp = generate_otp()
    try:
        await asyncio.to_thread(store_otp, user.user_id, otp)
        await asyncio.to_thread(send_otp_email, user.email, otp)
    except Exception as exc:
        logger.error(event="register_otp_send_failed", error=str(exc), user_id=user.user_id)
        try:
            await asyncio.to_thread(_svc.delete_user, user.user_id)
        except Exception:
            pass  # best-effort rollback
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not send verification email. Please try again.",
        ) from exc

    otp_token = await asyncio.to_thread(_issue_mfa_token, user.user_id)
    logger.info(event="register_otp_sent", user_id=user.user_id)
    return {"otp_required": True, "otp_token": otp_token}


# ── Login (JSON body) ─────────────────────────────────────────────────────────


@router.post("/login")
async def login(req: LoginRequest):
    """
    Step 1 of login: verify email + password.
    Always returns {"otp_required": true, "otp_token": "..."} — the client
    must present the 6-digit code sent to the user's email at /auth/verify-otp.
    """
    try:
        user = await asyncio.to_thread(_svc.authenticate, req.email, req.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    # Fixed demo account (app/bin/seed_demo_account.py) — always skips OTP.
    # It's a single publicly-shared login with a known password, so an email
    # round-trip would only add friction for whoever's holding the
    # credentials, not add any real verification value.
    if user.is_demo:
        tokens = issue_tokens(user.user_id, user.email, user.role.value)
        logger.info(event="login_demo_account", user_id=user.user_id)
        return TokenPair(**tokens).model_dump()

    # Check for a trusted device token — skip OTP if valid
    from app.auth.email_service import send_otp_email
    from app.auth.otp_store import generate_otp, store_otp, verify_device_token

    if req.device_token:
        try:
            trusted = await asyncio.to_thread(verify_device_token, req.device_token, user.user_id)
            if trusted:
                tokens = issue_tokens(user.user_id, user.email, user.role.value)
                logger.info(event="login_trusted_device", user_id=user.user_id)
                return {**TokenPair(**tokens).model_dump(), "device_token": req.device_token}
        except Exception:
            pass  # Redis unavailable — fall through to OTP

    # Generate and email the OTP
    otp = generate_otp()
    try:
        await asyncio.to_thread(store_otp, user.user_id, otp)
        await asyncio.to_thread(send_otp_email, user.email, otp)
    except Exception as exc:
        logger.error(event="otp_send_failed", error=str(exc), user_id=user.user_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not send verification email. Please try again.",
        ) from exc

    otp_token = await asyncio.to_thread(_issue_mfa_token, user.user_id)
    logger.info(event="otp_challenge_issued", user_id=user.user_id)
    return {"otp_required": True, "otp_token": otp_token}


# ── Verify email OTP ──────────────────────────────────────────────────────────


@router.post("/verify-otp")
async def verify_otp(req: OTPVerifyRequest):
    """
    Step 2 of login: submit the 6-digit OTP received by email.
    Returns a full JWT access + refresh token pair on success.
    """
    from jose import JWTError
    from jose import jwt as jose_jwt

    from app.auth.otp_store import verify_otp as _verify_otp

    # Decode the otp_token to get user_id (reuses the mfa_challenge token)
    try:
        payload = jose_jwt.decode(
            req.otp_token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        if payload.get("type") != "mfa_challenge":
            raise ValueError("Invalid OTP token type")
        user_id = payload["sub"]
    except (JWTError, KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OTP session expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    try:
        ok = await asyncio.to_thread(_verify_otp, user_id, req.code)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect code. Please try again.",
        )

    user = _svc.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    # Activate account if this was a registration OTP (account starts inactive)
    if not user.is_active:
        await asyncio.to_thread(_svc.activate_user, user_id)

    # Issue a trusted-device token so this browser skips OTP on future logins
    from app.auth.otp_store import generate_reset_token, store_device_token

    device_token = generate_reset_token()  # reuse same 32-byte random generator
    try:
        await asyncio.to_thread(store_device_token, device_token, user_id)
    except Exception:
        device_token = None  # Redis unavailable — don't break login, just skip

    tokens = issue_tokens(user.user_id, user.email, user.role.value)
    logger.info(event="otp_verified_complete", user_id=user_id)
    return {**TokenPair(**tokens).model_dump(), "device_token": device_token}


# ── Resend OTP ─────────────────────────────────────────────────────────────────


@router.post("/resend-otp")
async def resend_otp(req: OTPResendRequest):
    """
    Resend a fresh 6-digit code for a pending login/registration OTP
    challenge. Rate-limited: at least OTP_RESEND_COOLDOWN_SECONDS between
    sends, and at most OTP_RESEND_MAX_PER_WINDOW within OTP_RESEND_WINDOW_SECONDS
    — resending is not itself a way to keep the OTP session alive indefinitely
    or to hammer the mail provider.
    """
    from jose import JWTError
    from jose import jwt as jose_jwt

    from app.auth.email_service import send_otp_email
    from app.auth.otp_store import check_and_record_resend, generate_otp, store_otp

    try:
        payload = jose_jwt.decode(
            req.otp_token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        if payload.get("type") != "mfa_challenge":
            raise ValueError("Invalid OTP token type")
        user_id = payload["sub"]
    except (JWTError, KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OTP session expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    user = _svc.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    try:
        allowed, retry_after = await asyncio.to_thread(check_and_record_resend, user_id)
    except Exception as exc:
        logger.error(event="otp_resend_rate_check_failed", error=str(exc), user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not resend verification email. Please try again.",
        ) from exc

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Please wait {retry_after}s before requesting another code.",
            headers={"Retry-After": str(max(1, retry_after))},
        )

    otp = generate_otp()
    try:
        await asyncio.to_thread(store_otp, user_id, otp)
        await asyncio.to_thread(send_otp_email, user.email, otp)
    except Exception as exc:
        logger.error(event="otp_resend_send_failed", error=str(exc), user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not resend verification email. Please try again.",
        ) from exc

    logger.info(event="otp_resent", user_id=user_id)
    return {"sent": True, "cooldown_seconds": settings.OTP_RESEND_COOLDOWN_SECONDS}


# ── Login (OAuth2 form — enables /docs "Authorize" button) ───────────────────


@router.post("/login/form", response_model=TokenPair, include_in_schema=False)
async def login_form(form: OAuth2PasswordRequestForm = Depends()) -> TokenPair:
    """OAuth2 password flow — used by the interactive API docs (DEBUG mode only)."""
    if not settings.DEBUG:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    try:
        user = await asyncio.to_thread(_svc.authenticate, form.username, form.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
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
        ) from exc
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
        raise HTTPException(status_code=503, detail=str(exc)) from exc

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


# ── Forgot password ───────────────────────────────────────────────────────────


@router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(req: ForgotPasswordRequest) -> dict:
    """
    Send a password-reset link to the given email address.
    Always returns 200 regardless of whether the email exists — prevents
    user enumeration attacks.
    """
    from app.auth.email_service import send_password_reset_email
    from app.auth.otp_store import generate_reset_token, store_reset_token

    user = _svc.get_by_email(req.email)
    if user:
        token = generate_reset_token()
        try:
            await asyncio.to_thread(store_reset_token, token, user.user_id)
            await asyncio.to_thread(send_password_reset_email, req.email, token)
            logger.info(event="password_reset_requested", user_id=user.user_id)
        except Exception as exc:
            logger.error(event="password_reset_email_failed", error=str(exc))
    else:
        logger.info(event="password_reset_unknown_email", email=req.email)

    # Always return the same response to prevent user enumeration
    return {"status": "ok", "message": "If that email is registered, a reset link has been sent."}


# ── Reset password (consume the emailed token) ────────────────────────────────


@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(req: ResetPasswordRequest) -> dict:
    """
    Validate the one-time reset token and set the new password.
    The token is consumed immediately (single-use).
    All existing tokens for the user are revoked, forcing a fresh login.
    """
    from app.auth.otp_store import consume_reset_token

    try:
        user_id = await asyncio.to_thread(consume_reset_token, req.token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        await asyncio.to_thread(_svc.reset_password, user_id, req.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    revoke_all_user_tokens(user_id)
    # Revoke trusted devices so all remembered browsers must re-verify after a reset
    try:
        from app.auth.otp_store import revoke_device_tokens

        await asyncio.to_thread(revoke_device_tokens, user_id)
    except Exception:
        pass
    logger.info(event="password_reset_complete", user_id=user_id)
    return {"status": "ok", "message": "Password updated. Please sign in with your new password."}


# ── Logout (revoke current token) ────────────────────────────────────────────


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(
    request: Request,
    body: LogoutRequest | None = None,
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
            pass  # already invalid — no-op

    if body and body.refresh_token:
        try:
            r_payload = verify_token(body.refresh_token, expected_type="refresh")
            r_jti = r_payload.get("jti", "")
            r_exp = r_payload.get("exp", 0)
            if r_jti:
                revoke_token(r_jti, r_exp)
        except ValueError:
            pass  # already invalid — no-op

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


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)


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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # Invalidate all tokens — user must log in again
    revoke_all_user_tokens(current_user.user_id)
    logger.info(event="password_changed", user_id=current_user.user_id)
    return {"status": "ok", "message": "Password changed. All sessions have been terminated."}


# ── MFA — Enrol ───────────────────────────────────────────────────────────────


class MFACodeRequest(BaseModel):
    code: str = Field(..., min_length=6, max_length=10)


class MFAVerifyLoginRequest(BaseModel):
    mfa_token: str = Field(..., min_length=10)
    code: str = Field(..., min_length=6, max_length=10)


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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
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
        ) from exc

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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
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
    Full account deletion — wipes every trace of the user across all stores:
    Qdrant (vectors), BM25 index, Redis (session cache, OTP keys, device tokens,
    JWT revocation list), MongoDB (messages, summaries, sessions, user document),
    and the user's directory on disk (uploaded files, processed documents, images).
    """
    import shutil

    from app.utils.paths import DATA_ROOT

    user_id = current_user.user_id
    from app.core.infra_registry import infra

    # 1. Qdrant — delete all vectors for this user across both collections
    try:
        vs = infra.get_vector_store()
        if vs:
            await asyncio.to_thread(vs.gdpr_purge, user_id)
    except Exception as exc:
        logger.warning(event="gdpr_qdrant_purge_failed", user_id=user_id, error=str(exc))

    # 2. MongoDB — messages, summaries, chat_sessions (direct, no MemoryManager wrapper)
    try:
        mongo = infra.get_mongo()
        if mongo:
            await asyncio.to_thread(mongo.purge_user, user_id)
    except Exception as exc:
        logger.warning(event="gdpr_mongo_purge_failed", user_id=user_id, error=str(exc))

    # 3. Redis — session cache, OTP keys, device tokens
    try:
        redis_mem = infra.get_memory()
        if redis_mem and hasattr(redis_mem, "purge_user"):
            await asyncio.to_thread(redis_mem.purge_user, user_id)
    except Exception as exc:
        logger.warning(event="gdpr_redis_session_purge_failed", user_id=user_id, error=str(exc))
    try:
        from app.auth.otp_store import delete_otp_keys, revoke_device_tokens

        await asyncio.to_thread(delete_otp_keys, user_id)
        await asyncio.to_thread(revoke_device_tokens, user_id)
    except Exception as exc:
        logger.warning(event="gdpr_redis_auth_purge_failed", user_id=user_id, error=str(exc))

    # 4. BM25 index entries + disk (KB files, pickle, staging, images, etc.)
    try:
        bm25 = infra.get_bm25()
        if bm25 and hasattr(bm25, "delete_by_source"):
            kb_dir = DATA_ROOT / user_id / "knowledge_base"
            if kb_dir.exists():
                for f in kb_dir.iterdir():
                    if f.is_file():
                        await asyncio.to_thread(bm25.delete_by_source, f.name, user_id)
    except Exception as exc:
        logger.warning(event="gdpr_bm25_purge_failed", user_id=user_id, error=str(exc))
    try:
        user_dir = DATA_ROOT / user_id
        if user_dir.exists():
            await asyncio.to_thread(shutil.rmtree, str(user_dir), True)
    except Exception as exc:
        logger.warning(event="gdpr_disk_purge_failed", user_id=user_id, error=str(exc))

    # 5. JWT tokens — blacklist all active tokens immediately
    revoke_all_user_tokens(user_id)

    # 6. MongoDB user document — hard delete last so auth works for all steps above
    try:
        await asyncio.to_thread(_svc.delete_user, user_id)
    except Exception as exc:
        logger.warning(event="gdpr_user_doc_delete_failed", user_id=user_id, error=str(exc))

    logger.info(event="gdpr_self_purge_completed", user_id=user_id)

    return {
        "status": "ok",
        "message": "Your account and all associated data have been permanently deleted.",
        "user_id": user_id,
    }
