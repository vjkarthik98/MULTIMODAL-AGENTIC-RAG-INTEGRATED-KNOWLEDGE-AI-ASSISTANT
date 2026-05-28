"""
Per-user rate limiter — Redis token bucket.

Each user gets their own bucket keyed at u:{user_id}:ratelimit:{window}.
This ensures one user's heavy traffic cannot starve others.
Falls back gracefully if Redis is unavailable.
"""
from __future__ import annotations

import time
from typing import Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_WINDOW_SECONDS = 60  # 1 minute rolling window


def check_user_rate_limit(
    user_id: str,
    limit: Optional[int] = None,
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
        redis = infra.get_memory()
        if redis is None or not hasattr(redis, "_redis"):
            return

        r = redis._redis
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
            raise ValueError(f"Rate limit exceeded: {rpm} requests per minute")

    except ValueError:
        raise
    except Exception as exc:
        logger.warning(event="rate_limit_check_failed", error=str(exc))
