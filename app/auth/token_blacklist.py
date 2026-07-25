"""JWT token revocation via local Redis blacklist + in-process TTL cache.

On logout or password change, the token's jti is added to Redis with a TTL
matching the token's remaining lifetime. Every authenticated request checks
the blacklist before accepting the token.

Redis key schema:  REVOKED_TOKEN:{jti}  →  "1"  (TTL = remaining seconds)

PERFORMANCE: blacklist + generation reads are wrapped in a module-level
in-process TTL cache (settings.AUTH_REVOCATION_CACHE_TTL, default 30s) so the
common case (a valid, non-revoked token) does ZERO network on the request path.
On a cache miss we read LOCAL Redis (~0.5ms via infra.get_cache()), never the
Upstash cloud client (~200ms). Revocations made on this instance invalidate the
in-process entry immediately; cross-instance revocations propagate within the TTL.

Falls back gracefully if Redis is unavailable (logs warning, allows token).
This is intentional: a Redis outage should not lock all users out.
"""

from __future__ import annotations

import time

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_REVOKED_PREFIX = "REVOKED_TOKEN:"

# IN-PROCESS TTL CACHES — {key: (value, expires_at_monotonic)}
_revoked_cache: dict[str, tuple[bool, float]] = {}
_gen_cache: dict[str, tuple[int, float]] = {}


def _ttl() -> int:
    return getattr(settings, "AUTH_REVOCATION_CACHE_TTL", 30)


def _redis():
    """Local Redis cache handle (NOT Upstash) — hot-path, ~0.5ms/call."""
    try:
        from app.core.infra_registry import infra

        return infra.get_cache()
    except Exception:
        return None


def revoke_token(jti: str, exp: int) -> None:
    """
    Add a token JTI to the blacklist.
    TTL is set to the token's remaining lifetime so Redis auto-expires it.

    Args:
        jti: The JWT ID claim from the token payload.
        exp: The JWT expiry timestamp (Unix seconds).
    """
    r = _redis()
    if r is None:
        logger.warning(event="token_blacklist_redis_unavailable", jti=jti[:8])
        return

    ttl = max(1, int(exp - time.time()))
    key = f"{_REVOKED_PREFIX}{jti}"
    try:
        r.set(key, "1", ex=ttl)
        # Invalidate any cached "not revoked" answer so this takes effect immediately.
        _revoked_cache[jti] = (True, time.monotonic() + _ttl())
        logger.info(event="token_revoked", jti=jti[:8], ttl_seconds=ttl)
    except Exception as exc:
        logger.warning(event="token_revoke_failed", jti=jti[:8], error=str(exc))


def is_revoked(jti: str) -> bool:
    """
    Return True if the token has been revoked.

    Reads the in-process TTL cache first; on miss, reads local Redis and caches.
    Falls back to False (allow) if Redis is unavailable, to prevent a Redis
    outage from locking all authenticated users out.
    """
    now = time.monotonic()
    cached = _revoked_cache.get(jti)
    if cached is not None and cached[1] > now:
        return cached[0]

    r = _redis()
    if r is None:
        logger.warning(event="token_blacklist_check_redis_unavailable", jti=jti[:8])
        return False  # fail-open: Redis outage ≠ lock-out

    key = f"{_REVOKED_PREFIX}{jti}"
    try:
        revoked = r.exists(key) == 1
        if _ttl() > 0:
            _revoked_cache[jti] = (revoked, now + _ttl())
        return revoked
    except Exception as exc:
        logger.warning(event="token_blacklist_check_failed", jti=jti[:8], error=str(exc))
        return False  # fail-open


def revoke_all_user_tokens(user_id: str) -> int:
    """
    Scan and revoke all known tokens for a user (used on password change / GDPR purge).
    Returns number of tokens revoked.

    Note: This requires storing a user→jti mapping, which we do NOT do to avoid
    the fan-out cost. Instead, we bump a per-user 'generation' counter in Redis.
    Tokens issued before the current generation are rejected.
    """
    r = _redis()
    if r is None:
        return 0

    gen_key = f"TOKEN_GEN:{user_id}"
    try:
        new_gen = r.incr(gen_key)
        r.expire(gen_key, 60 * 60 * 24 * 8)  # 8 days — covers max refresh TTL
        # Invalidate cached generation so the new value takes effect immediately.
        _gen_cache[user_id] = (int(new_gen), time.monotonic() + _ttl())
        logger.info(event="user_tokens_revoked_via_generation", user_id=user_id, generation=new_gen)
        return 1  # generation bumped
    except Exception as exc:
        logger.warning(event="user_token_gen_bump_failed", user_id=user_id, error=str(exc))
        return 0


def get_user_token_generation(user_id: str) -> int:
    """Return the current valid token generation for a user (0 = any generation valid)."""
    now = time.monotonic()
    cached = _gen_cache.get(user_id)
    if cached is not None and cached[1] > now:
        return cached[0]

    r = _redis()
    if r is None:
        return 0
    gen_key = f"TOKEN_GEN:{user_id}"
    try:
        val = r.get(gen_key)
        gen = int(val) if val else 0
        if _ttl() > 0:
            _gen_cache[user_id] = (gen, now + _ttl())
        return gen
    except Exception:
        return 0
