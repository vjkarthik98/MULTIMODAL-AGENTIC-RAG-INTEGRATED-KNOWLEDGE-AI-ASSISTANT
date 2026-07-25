"""Shared client-IP resolution.

Single source of truth for turning a FastAPI Request into a client IP used for
rate limiting and abuse guards. X-Forwarded-For is only trusted when the
direct TCP peer is a configured trusted proxy (settings.TRUSTED_PROXY_IPS) —
otherwise it's attacker-controlled and any IP-keyed limit built on it (rate
limits, guest quotas) is trivially bypassed by spoofing the header.
"""

from __future__ import annotations

from fastapi import Request

from app.core.config import settings


def resolve_client_ip(request: Request) -> str:
    direct_ip = request.client.host if request.client else "unknown"
    if direct_ip in settings.TRUSTED_PROXY_IPS:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return direct_ip
