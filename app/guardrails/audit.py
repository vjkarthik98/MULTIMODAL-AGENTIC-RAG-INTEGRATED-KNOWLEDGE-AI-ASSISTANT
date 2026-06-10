"""Structured, tamper-evident guardrail audit log.

Every allow/block/scrub decision is logged with:
  - ISO timestamp
  - correlation_id (from X-Correlation-ID header or generated)
  - session_id
  - surface (input/output)
  - guard_type (injection, jailbreak, pii, ssrf, toxicity, groundedness, ...)
  - action (allow, block, scrub, warn)
  - reason (machine-readable slug)
  - query_prefix (first N chars — never full query)
  - latency_ms
  - HMAC-SHA256 signature (tamper-evident, uses AUDIT_HMAC_SECRET env var)

Output goes to structlog (JSON), which writes to the configured log file
and is picked up by the existing OTel pipeline.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

import structlog
from app.utils.logger import get_logger

logger = get_logger("guardrails.audit")

_HMAC_SECRET: Optional[bytes] = None


def _get_hmac_secret() -> bytes:
    global _HMAC_SECRET
    if _HMAC_SECRET is None:
        secret = os.environ.get("AUDIT_HMAC_SECRET", "")
        if not secret:
            # Derive from app secret key as fallback (still tamper-evident within session)
            secret = os.environ.get("SECRET_KEY", "guardrails-audit-default")
        _HMAC_SECRET = secret.encode("utf-8")
    return _HMAC_SECRET


def _sign(payload: Dict[str, Any]) -> str:
    """Return HMAC-SHA256 hex digest of the JSON-serialized payload."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hmac.new(
        _get_hmac_secret(),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def audit_decision(
    surface: str,
    guard_type: str,
    action: str,
    reason: str,
    session_id: str = "",
    correlation_id: str = "",
    query_prefix: str = "",
    latency_ms: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured audit log entry.

    Args:
        surface:        "input" or "output"
        guard_type:     e.g. "injection", "pii", "ssrf", "toxicity"
        action:         "allow", "block", "scrub", "warn"
        reason:         machine-readable slug
        session_id:     conversation session
        correlation_id: request trace ID
        query_prefix:   first N chars of query (never full text)
        latency_ms:     how long this check took
        extra:          additional context (never include raw PII)
    """
    try:
        payload: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "correlation_id": correlation_id or "",
            "session_id": session_id or "",
            "surface": surface,
            "guard_type": guard_type,
            "action": action,
            "reason": reason,
            "query_prefix": query_prefix[:80],
            "latency_ms": round(latency_ms, 2),
        }
        if extra:
            payload["extra"] = extra

        signature = _sign(payload)
        payload["signature"] = signature

        logger.info("guardrail_audit", **payload)
    except Exception as e:
        # Audit must never crash the main request path
        logger.warning("audit_emit_failed", error=str(e))
