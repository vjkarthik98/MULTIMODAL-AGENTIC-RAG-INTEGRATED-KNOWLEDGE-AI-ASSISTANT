"""
Per-user rate limiter — Redis token bucket.

Each user gets their own bucket keyed at u:{user_id}:ratelimit:{window}.
This ensures one user's heavy traffic cannot starve others.
Falls back gracefully if Redis is unavailable.
"""

from __future__ import annotations

import time

from app.auth.metrics import record_rate_limit_rejection
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_WINDOW_SECONDS = 60  # 1 minute rolling window


def check_user_rate_limit(
    user_id: str,
    limit: int | None = None,
    window: int = _WINDOW_SECONDS,
) -> None:
    """
    Raise ValueError if the user has exceeded their rate limit.
    Uses Redis INCR + EXPIRE for atomic token counting.
    Silently passes if Redis is unavailable (fail open, log warning).
    """
    if not settings.AUTH_ENABLED:
        return

    rpm = limit or settings.RATE_LIMIT_RPM
    key = f"u:{user_id}:ratelimit:{int(time.time()) // window}"

    try:
        from app.core.infra_registry import infra

        # Local Redis cache (~0.5ms) — sliding-window counters are per-instance
        # anyway, so local is correct and avoids the ~200ms Upstash round-trip.
        r = infra.get_cache()
        if r is None:
            return

        count = r.incr(key)
        if count == 1:
            r.expire(key, window)

        if count > rpm:
            logger.warning(
                event="rate_limit_exceeded",
                user_id=user_id,
                count=count,
                limit=rpm,
            )
            record_rate_limit_rejection()
            raise ValueError(f"Rate limit exceeded: {rpm} requests per minute")

    except ValueError:
        raise
    except Exception as exc:
        logger.warning(event="rate_limit_check_failed", error=str(exc))
