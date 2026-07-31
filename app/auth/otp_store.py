"""Redis-backed storage for email OTPs and password-reset tokens.

Key schema
----------
OTP value:       otp:{user_id}          →  "<6-digit code>"   TTL = OTP_TTL_SECONDS
OTP attempts:    otp_attempts:{user_id} →  "<int count>"      TTL = OTP_LOCKOUT_SECONDS
OTP lock:        otp_locked:{user_id}   →  "1"                TTL = OTP_LOCKOUT_SECONDS
Reset token:     reset:{token}          →  "<user_id>"        TTL = RESET_TOKEN_TTL_SECONDS

All keys are single-use where applicable (deleted after successful consume).
Falls back gracefully when Redis is unavailable — callers receive a RuntimeError
so the endpoint returns a 503 rather than silently doing nothing.
"""

from __future__ import annotations

import random
import secrets
import string

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_OTP_PREFIX = "otp:"
_ATTEMPTS_PREFIX = "otp_attempts:"
_LOCKED_PREFIX = "otp_locked:"
_RESET_PREFIX = "reset:"


def _redis():
    from app.core.infra_registry import infra

    mem = infra.get_memory()
    if mem is None or mem.client is None:
        raise RuntimeError("Redis unavailable — cannot store OTP")
    return mem.client


# ── OTP ───────────────────────────────────────────────────────────────────────


def generate_otp() -> str:
    """Return a cryptographically random 6-digit string."""
    return "".join(random.SystemRandom().choices(string.digits, k=6))


def store_otp(user_id: str, otp: str) -> None:
    """Persist OTP in Redis with TTL. Overwrites any existing pending OTP."""
    r = _redis()
    key = f"{_OTP_PREFIX}{user_id}"
    r.setex(key, settings.OTP_TTL_SECONDS, otp)
    # Reset attempt counter whenever a fresh OTP is issued
    r.delete(f"{_ATTEMPTS_PREFIX}{user_id}")
    logger.info(event="otp_stored", user_id=user_id, ttl=settings.OTP_TTL_SECONDS)


def verify_otp(user_id: str, submitted: str) -> bool:
    """
    Verify OTP. Deletes it on success (single-use).
    Raises ValueError on lockout.
    Returns False on wrong code (increments attempt counter).
    """
    r = _redis()

    # Check lockout
    if r.exists(f"{_LOCKED_PREFIX}{user_id}"):
        raise ValueError("Too many incorrect attempts. Please request a new code.")

    stored = r.get(f"{_OTP_PREFIX}{user_id}")
    if stored is None:
        raise ValueError("OTP has expired or was never issued. Please request a new code.")

    # Decode bytes if needed
    if isinstance(stored, bytes):
        stored = stored.decode()

    if stored.strip() != submitted.strip():
        # Increment attempt counter
        attempts_key = f"{_ATTEMPTS_PREFIX}{user_id}"
        attempts = r.incr(attempts_key)
        r.expire(attempts_key, settings.OTP_LOCKOUT_SECONDS)

        remaining = settings.OTP_MAX_ATTEMPTS - int(attempts)
        if remaining <= 0:
            # Lock the account for OTP retries
            r.setex(f"{_LOCKED_PREFIX}{user_id}", settings.OTP_LOCKOUT_SECONDS, "1")
            r.delete(f"{_OTP_PREFIX}{user_id}")
            logger.warning(event="otp_locked_out", user_id=user_id)
            raise ValueError(
                f"Too many incorrect attempts. Try again in "
                f"{settings.OTP_LOCKOUT_SECONDS // 60} minutes."
            )

        logger.warning(event="otp_wrong_code", user_id=user_id, attempts=int(attempts))
        return False

    # Correct — consume and clear counters
    r.delete(f"{_OTP_PREFIX}{user_id}")
    r.delete(f"{_ATTEMPTS_PREFIX}{user_id}")
    r.delete(f"{_LOCKED_PREFIX}{user_id}")
    logger.info(event="otp_verified", user_id=user_id)
    return True


_RESEND_COOLDOWN_PREFIX = "otp_resend_cooldown:"
_RESEND_COUNT_PREFIX = "otp_resend_count:"


def check_and_record_resend(user_id: str) -> tuple[bool, int]:
    """Atomically check the resend cooldown/cap and record this attempt.

    Two independent limits: a short cooldown between individual resends
    (stops a single click-happy user or a naive retry loop from hammering
    the mail provider), and a cap on total resends per rolling window (stops
    someone using resend as a way to keep re-triggering emails indefinitely).

    Returns (allowed, retry_after_seconds). When not allowed, retry_after_seconds
    is how long the caller must wait before trying again.
    """
    r = _redis()
    cooldown_key = f"{_RESEND_COOLDOWN_PREFIX}{user_id}"
    count_key = f"{_RESEND_COUNT_PREFIX}{user_id}"

    cooldown_ttl = r.ttl(cooldown_key)
    if cooldown_ttl and cooldown_ttl > 0:
        return False, int(cooldown_ttl)

    count = r.incr(count_key)
    if count == 1:
        r.expire(count_key, settings.OTP_RESEND_WINDOW_SECONDS)
    if count > settings.OTP_RESEND_MAX_PER_WINDOW:
        window_ttl = r.ttl(count_key)
        return False, (
            int(window_ttl) if window_ttl and window_ttl > 0 else settings.OTP_RESEND_WINDOW_SECONDS
        )

    r.setex(cooldown_key, settings.OTP_RESEND_COOLDOWN_SECONDS, "1")
    return True, 0


def delete_otp(user_id: str) -> None:
    """Remove all OTP keys for a user (used on resend to reset state)."""
    r = _redis()
    r.delete(
        f"{_OTP_PREFIX}{user_id}",
        f"{_ATTEMPTS_PREFIX}{user_id}",
        f"{_LOCKED_PREFIX}{user_id}",
    )


# ── Trusted device tokens ────────────────────────────────────────────────────

_DEVICE_PREFIX = "device:"
_DEVICE_TTL = 60 * 60 * 24 * 30  # 30 days


def store_device_token(token: str, user_id: str) -> None:
    """Persist a trusted-device token for 30 days."""
    r = _redis()
    r.setex(f"{_DEVICE_PREFIX}{token}", _DEVICE_TTL, user_id)
    logger.info(event="device_token_stored", user_id=user_id)


def verify_device_token(token: str, user_id: str) -> bool:
    """Return True if the token exists and belongs to this user."""
    if not token:
        return False
    r = _redis()
    stored = r.get(f"{_DEVICE_PREFIX}{token}")
    if stored is None:
        return False
    if isinstance(stored, bytes):
        stored = stored.decode()
    return stored == user_id


def delete_otp_keys(user_id: str) -> None:
    """Remove all OTP-related Redis keys for a user (called on account deletion)."""
    try:
        r = _redis()
        r.delete(
            f"{_OTP_PREFIX}{user_id}",
            f"{_ATTEMPTS_PREFIX}{user_id}",
            f"{_LOCKED_PREFIX}{user_id}",
            f"{_RESEND_COOLDOWN_PREFIX}{user_id}",
            f"{_RESEND_COUNT_PREFIX}{user_id}",
        )
        logger.info(event="otp_keys_deleted", user_id=user_id)
    except Exception as exc:
        logger.warning(event="otp_keys_delete_failed", user_id=user_id, error=str(exc))


def revoke_device_tokens(user_id: str) -> None:
    """Remove all device tokens for a user (called on password change/reset)."""
    # Redis doesn't support wildcard delete efficiently — we scan with a pattern
    r = _redis()
    pattern = f"{_DEVICE_PREFIX}*"
    cursor = 0
    deleted = 0
    while True:
        cursor, keys = r.scan(cursor, match=pattern, count=100)
        for key in keys:
            val = r.get(key)
            if val and (val.decode() if isinstance(val, bytes) else val) == user_id:
                r.delete(key)
                deleted += 1
        if cursor == 0:
            break
    if deleted:
        logger.info(event="device_tokens_revoked", user_id=user_id, count=deleted)


# ── Password-reset tokens ─────────────────────────────────────────────────────


def generate_reset_token() -> str:
    """Return a cryptographically random URL-safe 32-byte token (256 bits)."""
    return secrets.token_urlsafe(32)


def store_reset_token(token: str, user_id: str) -> None:
    """Persist reset token → user_id mapping with TTL."""
    r = _redis()
    key = f"{_RESET_PREFIX}{token}"
    r.setex(key, settings.RESET_TOKEN_TTL_SECONDS, user_id)
    logger.info(event="reset_token_stored", user_id=user_id, ttl=settings.RESET_TOKEN_TTL_SECONDS)


def consume_reset_token(token: str) -> str:
    """
    Look up and delete a reset token atomically.
    Returns the associated user_id.
    Raises ValueError if token is invalid or expired.
    """
    r = _redis()
    key = f"{_RESET_PREFIX}{token}"
    user_id = r.getdel(key)  # atomic get-and-delete (Redis >= 6.2)
    if user_id is None:
        raise ValueError("Reset link is invalid or has expired.")
    if isinstance(user_id, bytes):
        user_id = user_id.decode()
    logger.info(event="reset_token_consumed", user_id=user_id)
    return user_id
