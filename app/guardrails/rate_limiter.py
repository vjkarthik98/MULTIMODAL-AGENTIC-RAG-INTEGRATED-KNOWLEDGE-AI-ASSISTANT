"""Guardrail block rate limiter.

Tracks consecutive GuardrailBlocked events per session_id and per IP.
After N consecutive blocks within a time window, the session/IP is
temporarily banned and all further requests are rejected without processing.

Backend: Redis TTL keys (shared across all uvicorn workers).
Falls back silently to in-process deque if Redis is unavailable.

Policy (from policies.yaml):
  rate_limit.block_threshold       — blocks before temp-ban (default 5)
  rate_limit.window_sec            — rolling window in seconds (default 60)
  rate_limit.ban_duration_sec      — how long the ban lasts (default 300)
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import structlog

from app.guardrails.audit import audit_decision
from app.guardrails.exceptions import GuardrailBlocked
from app.guardrails.metrics import record_block

logger = structlog.get_logger(__name__)

# ── Policy ────────────────────────────────────────────────────────────────────
_block_threshold: int   = 5
_window_sec: float      = 60.0
_ban_duration_sec: float = 300.0
_policy_loaded           = False
_lock                    = threading.Lock()


def _load_policy() -> None:
    global _block_threshold, _window_sec, _ban_duration_sec, _policy_loaded
    if _policy_loaded:
        return
    try:
        from app.guardrails._policy_loader import get_policy
        rl = get_policy().get("rate_limit", {})
        _block_threshold  = int(rl.get("block_threshold", 5))
        _window_sec       = float(rl.get("window_sec", 60.0))
        _ban_duration_sec = float(rl.get("ban_duration_sec", 300.0))
    except Exception as e:
        logger.warning("rate_limiter_policy_load_failed", error=str(e))
    _policy_loaded = True


# ── Redis helpers ─────────────────────────────────────────────────────────────

def _get_redis():
    """Return the raw Redis client or None if unavailable."""
    try:
        from app.core.infra_registry import infra
        mem = infra.get_memory()
        if mem and hasattr(mem, "_redis"):
            return mem._redis
    except Exception:
        pass
    return None


def _redis_is_banned(r, key_prefix: str, identifier: str, now: float) -> tuple[bool, float]:
    """Check Redis ban key. Returns (is_banned, remaining_seconds)."""
    try:
        ban_key = f"RATE_LIMIT:ban:{key_prefix}:{identifier}"
        ttl = r.ttl(ban_key)
        if ttl > 0:
            return True, float(ttl)
    except Exception:
        pass
    return False, 0.0


def _redis_record_block(r, key_prefix: str, identifier: str, now: float) -> bool:
    """
    Increment the block counter in Redis using a sorted set (score=timestamp).
    Prune old entries, check threshold. If threshold exceeded, set ban key.
    Returns True if ban was just triggered.
    """
    try:
        count_key = f"RATE_LIMIT:blocks:{key_prefix}:{identifier}"
        ban_key   = f"RATE_LIMIT:ban:{key_prefix}:{identifier}"
        pipe = r.pipeline()
        # Add current timestamp as member
        pipe.zadd(count_key, {str(now): now})
        # Remove entries outside window
        pipe.zremrangebyscore(count_key, "-inf", now - _window_sec)
        # Count remaining
        pipe.zcard(count_key)
        # Expire the sorted set after window
        pipe.expire(count_key, int(_window_sec) + 10)
        results = pipe.execute()
        count = results[2]
        if count >= _block_threshold:
            r.setex(ban_key, int(_ban_duration_sec), "1")
            return True
    except Exception as e:
        logger.warning("rate_limiter_redis_record_failed", error=str(e))
    return False


def _redis_reset(r, key_prefix: str, identifier: str) -> None:
    try:
        r.delete(
            f"RATE_LIMIT:blocks:{key_prefix}:{identifier}",
            f"RATE_LIMIT:ban:{key_prefix}:{identifier}",
        )
    except Exception:
        pass


# ── In-process fallback (single-worker dev mode) ──────────────────────────────

@dataclass
class _State:
    block_times: Deque[float] = field(default_factory=deque)
    banned_until: float = 0.0


_session_state: Dict[str, _State] = {}
_ip_state: Dict[str, _State] = {}


def _get_state(store: Dict[str, _State], key: str) -> _State:
    if key not in store:
        store[key] = _State()
    return store[key]


def _local_is_banned(state: _State, now: float) -> bool:
    return state.banned_until > now


def _local_record_block(state: _State, now: float) -> bool:
    while state.block_times and (now - state.block_times[0]) > _window_sec:
        state.block_times.popleft()
    state.block_times.append(now)
    if len(state.block_times) >= _block_threshold:
        state.banned_until = now + _ban_duration_sec
        return True
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def check_and_record(
    session_id: str,
    ip: Optional[str] = None,
    surface: str = "api",
    correlation_id: str = "",
) -> None:
    """Record a block event. Bans the session/IP if threshold is exceeded."""
    _load_policy()
    now = time.time()
    r   = _get_redis()

    with _lock:
        if session_id:
            banned = False
            if r is not None:
                banned = _redis_record_block(r, "session", session_id, now)
            else:
                s_state = _get_state(_session_state, session_id)
                banned  = _local_record_block(s_state, now)

            if banned:
                logger.warning(
                    "rate_limiter_session_banned",
                    session_id=session_id,
                    ban_duration_sec=_ban_duration_sec,
                    surface=surface,
                    backend="redis" if r else "local",
                )
                record_block("rate_limit_session", surface)
                audit_decision(
                    surface=surface,
                    guard_type="rate_limit",
                    action="ban",
                    reason="session_block_threshold_exceeded",
                    session_id=session_id,
                    correlation_id=correlation_id,
                    latency_ms=0.0,
                )

        if ip:
            banned = False
            if r is not None:
                banned = _redis_record_block(r, "ip", ip, now)
            else:
                i_state = _get_state(_ip_state, ip)
                banned  = _local_record_block(i_state, now)

            if banned:
                logger.warning(
                    "rate_limiter_ip_banned",
                    ip=ip,
                    ban_duration_sec=_ban_duration_sec,
                    surface=surface,
                    backend="redis" if r else "local",
                )
                record_block("rate_limit_ip", surface)
                audit_decision(
                    surface=surface,
                    guard_type="rate_limit",
                    action="ban",
                    reason="ip_block_threshold_exceeded",
                    session_id=session_id,
                    correlation_id=correlation_id,
                    latency_ms=0.0,
                    extra={"ip": ip},
                )


def enforce(
    session_id: str,
    ip: Optional[str] = None,
    surface: str = "api",
    correlation_id: str = "",
) -> None:
    """Raise GuardrailBlocked if this session/IP is currently banned."""
    _load_policy()
    now = time.time()
    r   = _get_redis()

    with _lock:
        if session_id:
            if r is not None:
                is_banned, remaining = _redis_is_banned(r, "session", session_id, now)
            else:
                s_state  = _get_state(_session_state, session_id)
                is_banned = _local_is_banned(s_state, now)
                remaining = round(s_state.banned_until - now) if is_banned else 0.0

            if is_banned:
                logger.warning("rate_limiter_request_rejected",
                               session_id=session_id, remaining_sec=remaining, surface=surface)
                raise GuardrailBlocked(
                    reason="rate_limit_banned", surface=surface,
                    guard_type="rate_limit", correlation_id=correlation_id,
                    detail=f"session banned for {int(remaining)}s more",
                )

        if ip:
            if r is not None:
                is_banned, remaining = _redis_is_banned(r, "ip", ip, now)
            else:
                i_state  = _get_state(_ip_state, ip)
                is_banned = _local_is_banned(i_state, now)
                remaining = round(i_state.banned_until - now) if is_banned else 0.0

            if is_banned:
                logger.warning("rate_limiter_ip_rejected",
                               ip=ip, remaining_sec=remaining, surface=surface)
                raise GuardrailBlocked(
                    reason="rate_limit_banned", surface=surface,
                    guard_type="rate_limit", correlation_id=correlation_id,
                    detail=f"ip banned for {int(remaining)}s more",
                )


def reset(session_id: str = "", ip: str = "") -> None:
    """Reset ban/block state for a session or IP (admin/test use)."""
    r = _get_redis()
    with _lock:
        if session_id:
            if r:
                _redis_reset(r, "session", session_id)
            elif session_id in _session_state:
                del _session_state[session_id]
        if ip:
            if r:
                _redis_reset(r, "ip", ip)
            elif ip in _ip_state:
                del _ip_state[ip]
