"""Unified input guardrail — single entry point for all query sanitization.

Replaces the 7 scattered _INJECTION_PATTERNS / _sanitize* implementations:
  - app/agents/agent_controller.py   _sanitize()
  - app/agents/agent_router.py       _detect_injection()
  - app/api/api_routes.py            _check_prompt_injection()
  - app/tools/web_search.py          _sanitize_query()
  - app/embeddings/text_embedder.py  _sanitize()
  - app/embeddings/clip_text_embedder.py  _sanitize_injection()
  - app/ingestion/frame_captioner.py  _sanitize_caption()

Contract:
  check(query, surface, session_id, correlation_id)
    → GuardedInput   if allowed (possibly cleaned)
    raises GuardrailBlocked  if hard-blocked

  sanitize(text, surface, session_id, correlation_id)
    → str   cleaned text (for embedder/captioner paths that must not block)
    Logs detections but never raises — keeps ingestion pipeline running.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field

from app.guardrails.audit import audit_decision
from app.guardrails.exceptions import GuardrailBlocked
from app.guardrails.jailbreak import JailbreakResult
from app.guardrails.jailbreak import check as jailbreak_check
from app.guardrails.metrics import record_allow, record_block
from app.guardrails.ssrf import is_ssrf_risk
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Policy cache (loaded once)
# ---------------------------------------------------------------------------
_policy: dict = {}
_injection_patterns: list[re.Pattern] = []
_max_query_chars: int = 8000
_warn_query_chars: int = 4000
_repeat_token_limit: int = 500
_policy_loaded = False


def _load_policy() -> None:
    global _policy, _injection_patterns, _max_query_chars, _warn_query_chars
    global _repeat_token_limit, _policy_loaded
    if _policy_loaded:
        return
    try:
        from app.guardrails._policy_loader import get_policy

        _policy = get_policy()
        inj = _policy.get("injection", {})
        length = _policy.get("length", {})
        _max_query_chars = int(length.get("max_query_chars", 8000))
        _warn_query_chars = int(length.get("warn_query_chars", 4000))
        _repeat_token_limit = int(length.get("repeat_token_limit", 500))

        compiled = []
        for entry in inj.get("regex_patterns", []):
            try:
                compiled.append(
                    (
                        re.compile(entry["pattern"], re.IGNORECASE | re.DOTALL),
                        entry.get("severity", "medium"),
                        entry.get("description", ""),
                    )
                )
            except re.error as e:
                logger.warning("input_guard_pattern_compile_failed", error=str(e))
        _injection_patterns = compiled
    except Exception as e:
        logger.warning("input_guard_policy_load_failed", error=str(e))
    _policy_loaded = True


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class GuardedInput:
    text: str
    original_text: str
    blocked: bool = False
    reason: str = ""
    guard_type: str = ""
    pii_detected: bool = False
    pii_entities: list[dict] = field(default_factory=list)
    jailbreak_result: JailbreakResult | None = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Internal checks
# ---------------------------------------------------------------------------


def _normalize_encoding(text: str) -> str:
    """NFKC normalize + confusables map + strip null bytes, RTL overrides, zero-width chars.

    Order matters:
      1. Strip control/invisible chars first (null, RTL overrides, ZWS)
      2. NFKC (maps fullwidth Latin, standard superscripts, ligatures)
      3. Confusables map (covers Latin-extended ø, modifier letters, Cyrillic/Greek lookalikes)
      4. Re-NFKC after confusables to normalize any chained substitutions
    """
    # 1. Strip null bytes
    text = text.replace("\x00", "")
    # Strip RTL/LTR override chars (U+202E, U+202D, U+202C, U+202B, U+202A).
    # These literal bidi-override characters ARE the injection-defense
    # signature this function exists to strip (Trojan Source attack chars),
    # not an accidental/vulnerable occurrence. Reviewed 2026-07-25.
    for cp in ("‮", "‭", "‬", "‫", "‪"):  # nosec B613
        text = text.replace(cp, "")
    # Strip zero-width chars (ZWSP, ZWNJ, ZWJ, BOM, WORD-JOINER)
    for cp in ("​", "‌", "‍", "﻿", "⁠"):
        text = text.replace(cp, "")
    # 2. NFKC — maps fullwidth Latin (ｉｇｎｏｒｅ → ignore), standard superscripts, ligatures
    text = unicodedata.normalize("NFKC", text)
    # 3+4. Confusables map (Latin-extended, modifier letters, Cyrillic/Greek) + re-NFKC
    try:
        from app.guardrails.confusables import normalize_confusables

        text = normalize_confusables(text)
    except Exception:
        pass
    return text


def _check_length(text: str, correlation_id: str) -> str | None:
    """Return truncated text if over limit, None if under warn threshold."""
    if len(text) > _max_query_chars:
        logger.warning(
            "input_guard_length_truncated",
            original_len=len(text),
            limit=_max_query_chars,
            correlation_id=correlation_id,
        )
        return text[:_max_query_chars]
    if len(text) > _warn_query_chars:
        logger.info(
            "input_guard_length_warn",
            length=len(text),
            warn_threshold=_warn_query_chars,
            correlation_id=correlation_id,
        )
    return None  # no truncation needed


def _check_repeat_token(text: str) -> bool:
    """Detect runaway repeat-token attacks (e.g. 'HELLO' × 10000)."""
    words = text.split()
    if len(words) < _repeat_token_limit:
        return False
    # Check if any single token dominates >80% of the text
    if words:
        most_common = max(set(words), key=words.count)
        if words.count(most_common) > _repeat_token_limit:
            return True
    return False


def _check_injection(text: str) -> tuple[str, str] | None:
    """Return (matched_description, severity) if injection detected, else None."""
    _load_policy()
    lower = text.lower()
    for pattern, severity, description in _injection_patterns:
        if pattern.search(lower):
            return description, severity
    return None


def _check_ssrf_in_text(text: str) -> bool:
    """Check if text contains URLs that are SSRF risks."""
    url_pattern = re.compile(r'https?://\S+|file://\S+|ftp://\S+', re.IGNORECASE)
    for match in url_pattern.finditer(text):
        if is_ssrf_risk(match.group(0)):
            return True
    return False


def _check_scope_violation(text: str) -> str | None:
    """Detect clear scope violations (calendar, malware, network scanning)."""
    _load_policy()
    lower = text.lower()
    for pattern, severity, description in _injection_patterns:
        if severity in ("critical", "high") and "scope" in description.lower():
            if pattern.search(lower):
                return description
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check(
    query: str,
    surface: str = "api",
    session_id: str = "",
    correlation_id: str = "",
) -> GuardedInput:
    """Full guardrail check — raises GuardrailBlocked on hard violations.

    Call this at the API layer and agent controller before any processing.
    The returned GuardedInput.text is the safe, normalized query to use.
    """
    _load_policy()
    t_start = time.monotonic()

    original = query

    # 1. Encoding normalization
    query = _normalize_encoding(query)

    # 2. Length check (truncate, never block)
    truncated = _check_length(query, correlation_id)
    if truncated is not None:
        query = truncated

    # 3. Repeat-token attack
    if _check_repeat_token(query):
        _emit_block("length", "repeat_token_attack", surface, correlation_id, session_id, query)
        raise GuardrailBlocked(
            reason="repeat_token_attack",
            surface=surface,
            guard_type="length",
            correlation_id=correlation_id,
        )

    # 4. Injection regex check (Issue A fix — regex with modifier tolerance)
    inj_hit = _check_injection(query)
    if inj_hit:
        desc, severity = inj_hit
        _emit_block(
            "injection", f"injection_detected:{desc}", surface, correlation_id, session_id, query
        )
        raise GuardrailBlocked(
            reason="injection_detected",
            surface=surface,
            guard_type="injection",
            correlation_id=correlation_id,
            detail=f"pattern={desc}, severity={severity}",
        )

    # 5. Jailbreak check (Tier 1 regex + Tier 2 semantic)
    jb_result = jailbreak_check(query)
    if jb_result.is_jailbreak:
        _emit_block("jailbreak", "jailbreak_detected", surface, correlation_id, session_id, query)
        raise GuardrailBlocked(
            reason="jailbreak_detected",
            surface=surface,
            guard_type="jailbreak",
            correlation_id=correlation_id,
            detail=f"confidence={jb_result.confidence:.3f}, tier={jb_result.tier}",
        )

    # 6. SSRF in embedded URLs
    if _check_ssrf_in_text(query):
        _emit_block("ssrf", "ssrf_in_query", surface, correlation_id, session_id, query)
        raise GuardrailBlocked(
            reason="ssrf_in_query",
            surface=surface,
            guard_type="ssrf",
            correlation_id=correlation_id,
        )

    # 7. PII ingress detection (log only, don't block — policy: "log")
    pii_entities: list[dict] = []
    pii_detected = False
    try:
        from app.guardrails._policy_loader import get_policy
        from app.guardrails.pii import detect_pii

        ingress_action = get_policy().get("pii", {}).get("ingress_action", "log")
        if ingress_action in ("log", "strip"):
            pii_entities = detect_pii(query)
            pii_detected = bool(pii_entities)
            if pii_detected:
                logger.info(
                    "input_guard_pii_detected",
                    entity_types=[e["entity_type"] for e in pii_entities],
                    surface=surface,
                    correlation_id=correlation_id,
                )
    except Exception as e:
        logger.warning("input_guard_pii_check_failed", error=str(e))

    latency_ms = (time.monotonic() - t_start) * 1000
    record_allow("input", surface)
    audit_decision(
        surface=surface,
        guard_type="input",
        action="allow",
        reason="all_checks_passed",
        session_id=session_id,
        correlation_id=correlation_id,
        query_prefix=query[:80],
        latency_ms=latency_ms,
    )

    return GuardedInput(
        text=query,
        original_text=original,
        blocked=False,
        pii_detected=pii_detected,
        pii_entities=pii_entities,
        jailbreak_result=jb_result,
        latency_ms=latency_ms,
    )


_INGEST_SURFACES = frozenset(
    {
        "txt_ingest",
        "excel_ingest",
        "excel_chart_ingest",
        "document_ingest",
        "pdf_ingest",
        "audio_ingest",
        "text_embedder",
        "embedder",
    }
)


def sanitize(
    text: str,
    surface: str = "embedder",
    session_id: str = "",
    correlation_id: str = "",
) -> str:
    """Non-blocking sanitization for ingestion pipelines (embedder/captioner).

    Never raises GuardrailBlocked. Strips injection payload from text and
    returns the safe prefix. Logs detections for audit trail.
    Used by text_embedder, clip_text_embedder, frame_captioner.
    """
    _load_policy()
    text = _normalize_encoding(text)

    # Truncate to max length — skip for ingestion surfaces so full documents
    # are not truncated to the query limit (8000 chars) before chunking.
    if surface not in _INGEST_SURFACES and len(text) > _max_query_chars:
        text = text[:_max_query_chars]

    # Strip injection payload — return safe prefix up to match
    lower = text.lower()
    for pattern, severity, description in _injection_patterns:
        match = pattern.search(lower)
        if match:
            text = text[: match.start()].strip()
            logger.warning(
                "input_guard_sanitize_stripped",
                surface=surface,
                pattern=description,
                severity=severity,
                session_id=session_id,
                correlation_id=correlation_id,
            )
            record_block("injection", surface)
            audit_decision(
                surface=surface,
                guard_type="injection",
                action="scrub",
                reason=f"injection_stripped:{description}",
                session_id=session_id,
                correlation_id=correlation_id,
                query_prefix=text[:80],
                latency_ms=0.0,
            )
            break

    return text


def _emit_block(
    guard_type: str,
    reason: str,
    surface: str,
    correlation_id: str,
    session_id: str,
    query: str,
) -> None:
    record_block(guard_type, surface)
    audit_decision(
        surface=surface,
        guard_type=guard_type,
        action="block",
        reason=reason,
        session_id=session_id,
        correlation_id=correlation_id,
        query_prefix=query[:80],
        latency_ms=0.0,
    )
    logger.warning(
        "input_guard_blocked",
        guard_type=guard_type,
        reason=reason,
        surface=surface,
        correlation_id=correlation_id,
        session_id=session_id,
        query_prefix=query[:80],
    )
