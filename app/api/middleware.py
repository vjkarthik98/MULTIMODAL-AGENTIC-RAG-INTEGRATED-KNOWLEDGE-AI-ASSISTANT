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
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        return token if token else None
    return None
