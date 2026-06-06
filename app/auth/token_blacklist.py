"""JWT token revocation via Redis blacklist.

On logout or password change, the token's jti is added to Redis with a TTL
matching the token's remaining lifetime. Every authenticated request checks
the blacklist before accepting the token.

Redis key schema:  REVOKED_TOKEN:{jti}  →  "1"  (TTL = remaining seconds)

Falls back gracefully if Redis is unavailable (logs warning, allows token).
This is intentional: a Redis outage should not lock all users out.
"""
from __future__ import annotations

import time
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

_REVOKED_PREFIX = "REVOKED_TOKEN:"


def _redis():
    try:
        from app.core.infra_registry import infra
        mem = infra.get_memory()
        if mem is not None and mem.client is not None:
            return mem.client
    except Exception:
        pass
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
        r.setex(key, ttl, "1")
        logger.info(event="token_revoked", jti=jti[:8], ttl_seconds=ttl)
    except Exception as exc:
        logger.warning(event="token_revoke_failed", jti=jti[:8], error=str(exc))


def is_revoked(jti: str) -> bool:
    """
    Return True if the token has been revoked.

    Falls back to False (allow) if Redis is unavailable, to prevent
    a Redis outage from locking all authenticated users out.
    """
    r = _redis()
    if r is None:
        logger.warning(event="token_blacklist_check_redis_unavailable", jti=jti[:8])
        return False   # fail-open: Redis outage ≠ lock-out

    key = f"{_REVOKED_PREFIX}{jti}"
    try:
        return r.exists(key) == 1
    except Exception as exc:
        logger.warning(event="token_blacklist_check_failed", jti=jti[:8], error=str(exc))
        return False   # fail-open


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
        r.expire(gen_key, 60 * 60 * 24 * 8)   # 8 days — covers max refresh TTL
        logger.info(event="user_tokens_revoked_via_generation", user_id=user_id, generation=new_gen)
        return 1  # generation bumped
    except Exception as exc:
        logger.warning(event="user_token_gen_bump_failed", user_id=user_id, error=str(exc))
        return 0


def get_user_token_generation(user_id: str) -> int:
    """Return the current valid token generation for a user (0 = any generation valid)."""
    r = _redis()
    if r is None:
        return 0
    gen_key = f"TOKEN_GEN:{user_id}"
    try:
        val = r.get(gen_key)
        return int(val) if val else 0
    except Exception:
        return 0
