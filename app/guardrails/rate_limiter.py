"""Guardrail block rate limiter.

Tracks consecutive GuardrailBlocked events per session_id and per IP.
After N consecutive blocks within a time window, the session/IP is
temporarily banned and all further requests are rejected without processing.

This prevents machine-speed probing to discover bypass patterns.

Policy (from policies.yaml):
  rate_limit.block_threshold       — blocks before temp-ban (default 5)
  rate_limit.window_sec            — rolling window in seconds (default 60)
  rate_limit.ban_duration_sec      — how long the ban lasts (default 300)
"""
from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import structlog

from app.guardrails.audit import audit_decision
from app.guardrails.exceptions import GuardrailBlocked
from app.guardrails.metrics import record_block

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Policy (loaded once, defaults are safe)
# ---------------------------------------------------------------------------
_block_threshold: int = 5       # consecutive blocks before ban
_window_sec: float = 60.0       # rolling window
_ban_duration_sec: float = 300.0  # 5 minute ban
_policy_loaded = False
_lock = threading.Lock()


def _load_policy() -> None:
    global _block_threshold, _window_sec, _ban_duration_sec, _policy_loaded
    if _policy_loaded:
        return
    try:
        from app.guardrails._policy_loader import get_policy
        rl = get_policy().get("rate_limit", {})
        _block_threshold = int(rl.get("block_threshold", 5))
        _window_sec = float(rl.get("window_sec", 60.0))
        _ban_duration_sec = float(rl.get("ban_duration_sec", 300.0))
    except Exception as e:
        logger.warning("rate_limiter_policy_load_failed", error=str(e))
    _policy_loaded = True


# ---------------------------------------------------------------------------
# In-memory state (per session_id and per IP)
# Production upgrade: swap _State store for Redis with TTL keys
# ---------------------------------------------------------------------------

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


def _is_banned(state: _State, now: float) -> bool:
    return state.banned_until > now


def _record_block_event(state: _State, now: float) -> bool:
    """Record a block event, prune old ones, return True if threshold exceeded."""
    _load_policy()
    # Prune events outside the rolling window
    while state.block_times and (now - state.block_times[0]) > _window_sec:
        state.block_times.popleft()
    state.block_times.append(now)
    return len(state.block_times) >= _block_threshold


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_and_record(
    session_id: str,
    ip: Optional[str] = None,
    surface: str = "api",
    correlation_id: str = "",
) -> None:
    """Call this AFTER a GuardrailBlocked is raised to record the block event.

    If the session or IP has exceeded the block threshold within the window,
    the ban is applied — subsequent calls to enforce() will raise GuardrailBlocked.
    """
    _load_policy()
    now = time.monotonic()

    with _lock:
        # Track per-session
        if session_id:
            s_state = _get_state(_session_state, session_id)
            if _record_block_event(s_state, now):
                s_state.banned_until = now + _ban_duration_sec
                logger.warning(
                    "rate_limiter_session_banned",
                    session_id=session_id,
                    block_count=len(s_state.block_times),
                    ban_duration_sec=_ban_duration_sec,
                    surface=surface,
                )
                record_block("rate_limit_session", surface)
                audit_decision(
                    surface=surface,
                    guard_type="rate_limit",
                    action="ban",
                    reason=f"session_block_threshold_exceeded(n={len(s_state.block_times)})",
                    session_id=session_id,
                    correlation_id=correlation_id,
                    latency_ms=0.0,
                )

        # Track per-IP
        if ip:
            i_state = _get_state(_ip_state, ip)
            if _record_block_event(i_state, now):
                i_state.banned_until = now + _ban_duration_sec
                logger.warning(
                    "rate_limiter_ip_banned",
                    ip=ip,
                    block_count=len(i_state.block_times),
                    ban_duration_sec=_ban_duration_sec,
                    surface=surface,
                )
                record_block("rate_limit_ip", surface)
                audit_decision(
                    surface=surface,
                    guard_type="rate_limit",
                    action="ban",
                    reason=f"ip_block_threshold_exceeded(n={len(i_state.block_times)})",
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
    """Call this at the START of every request before any processing.

    Raises GuardrailBlocked(reason="rate_limit_banned") if the session or IP
    is currently banned. Does nothing otherwise.
    """
    _load_policy()
    now = time.monotonic()

    with _lock:
        if session_id:
            s_state = _get_state(_session_state, session_id)
            if _is_banned(s_state, now):
                remaining = round(s_state.banned_until - now)
                logger.warning(
                    "rate_limiter_request_rejected",
                    session_id=session_id,
                    remaining_sec=remaining,
                    surface=surface,
                )
                raise GuardrailBlocked(
                    reason="rate_limit_banned",
                    surface=surface,
                    guard_type="rate_limit",
                    correlation_id=correlation_id,
                    detail=f"session banned for {remaining}s more",
                )

        if ip:
            i_state = _get_state(_ip_state, ip)
            if _is_banned(i_state, now):
                remaining = round(i_state.banned_until - now)
                logger.warning(
                    "rate_limiter_ip_rejected",
                    ip=ip,
                    remaining_sec=remaining,
                    surface=surface,
                )
                raise GuardrailBlocked(
                    reason="rate_limit_banned",
                    surface=surface,
                    guard_type="rate_limit",
                    correlation_id=correlation_id,
                    detail=f"ip banned for {remaining}s more",
                )


def reset(session_id: str = "", ip: str = "") -> None:
    """Reset state for a session or IP (admin/test use only)."""
    with _lock:
        if session_id and session_id in _session_state:
            del _session_state[session_id]
        if ip and ip in _ip_state:
            del _ip_state[ip]
