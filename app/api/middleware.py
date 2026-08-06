"""
Auth middleware — attaches the verified UserPublic to request.state.user
on every request that carries a Bearer token.

Public routes (/auth/*, /health, /docs, /openapi.json, /redoc) are
passed through without requiring a token.  All other routes still need
to call Depends(get_current_user) in their signature to enforce auth —
this middleware is complementary, not a replacement.

It also:
- Stamps request.state.request_id for correlation
- Logs the authenticated user_id on every request for auditing
"""

from __future__ import annotations

import hmac
import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Routes that do NOT require authentication
_PUBLIC_PREFIXES = (
    "/auth/",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
)

# Liveness/probe endpoints hit by orchestrators, load balancers, and IDE port
# forwarders (e.g. VS Code Remote-SSH probes every locally-listening port on a
# short interval). They carry no diagnostic value and flood the access log, so
# they are logged at DEBUG instead of INFO — the request is still handled and
# still traced, just not surfaced at the default log level.
_QUIET_PATHS = frozenset(
    (
        "/health",
        "/status",  # renamed from "/metrics" in Phase 31 — see app/main.py
    )
)


class AuthMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.user = None

        start = time.perf_counter()

        # Attempt token extraction for ALL routes — so request.state.user
        # is always populated when a token is present, even on public routes.
        token = _extract_bearer(request)
        if token:
            try:
                from datetime import datetime, timezone

                from app.auth.jwt_handler import verify_token
                from app.auth.models import UserPublic, UserRole

                payload = verify_token(token, expected_type="access")
                request.state.user = UserPublic(
                    user_id=payload["sub"],
                    email=payload["email"],
                    role=UserRole(payload.get("role", "user")),
                    is_active=True,
                    created_at=datetime.now(timezone.utc),
                )
            except Exception:
                # Invalid token — don't attach user; route-level dependency
                # will raise 401 if it's a protected route.
                pass

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        user_id = request.state.user.user_id if request.state.user else "anonymous"

        _log = logger.debug if request.url.path in _QUIET_PATHS else logger.info
        _log(
            event="http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
            request_id=request_id,
            user_id=user_id,
        )

        response.headers["X-Request-ID"] = request_id
        return response


def _extract_bearer(request: Request) -> str | None:
    from app.auth.cookies import read_access_cookie

    cookie_token = read_access_cookie(request)
    if cookie_token:
        return cookie_token
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        return token if token else None
    return None


class CSRFMiddleware(BaseHTTPMiddleware):
    """Double-submit CSRF check for cookie-authenticated state-changing requests.

    Scope: only fires when (a) the request is a mutating method and (b) a
    magik_access or magik_refresh cookie is actually present — i.e. only once
    a real cookie session exists to forge. Pre-auth endpoints (login,
    register, verify-otp, forgot/reset-password, the OAuth handshake) carry no
    session cookie yet, so they're naturally exempt without a path allowlist.

    Requests authenticated via `Authorization: Bearer` instead of a cookie are
    also exempt: a cross-site page cannot attach a custom header to a forged
    request, so bearer-token API/CLI/test clients are not a CSRF target by
    construction — only the cookie-based browser session needs this.
    """

    _SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

    async def dispatch(self, request: Request, call_next) -> Response:
        from app.auth.cookies import CSRF_COOKIE, read_access_cookie, read_refresh_cookie

        if request.method not in self._SAFE_METHODS and not request.headers.get("Authorization"):
            if read_access_cookie(request) or read_refresh_cookie(request):
                cookie_csrf = request.cookies.get(CSRF_COOKIE, "")
                header_csrf = request.headers.get("X-CSRF-Token", "")
                if not cookie_csrf or not header_csrf or not hmac.compare_digest(cookie_csrf, header_csrf):
                    from fastapi.responses import JSONResponse

                    return JSONResponse(
                        status_code=403,
                        content={"detail": "CSRF validation failed"},
                    )

        return await call_next(request)
