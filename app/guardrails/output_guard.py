"""Unified output guardrail — runs before every AgentResponse is returned.

Checks (in order):
  1. Template artifact stripping (P1-7: [sic], Sources Used: N, llama tokens)
  2. Citation integrity — every [filename] token validated against sources[]
  3. PII egress scrub — Presidio anonymization on answer text
  4. Groundedness check — reuse Phase 25 hallucination detector
  5. Toxicity check — simple heuristic classifier (Detoxify optional)
  6. Output length enforcement
  7. Refusal template enforcement (never return empty string on block)

Returns GuardedOutput. Raises GuardrailBlocked only for toxicity (hard
block). All other checks result in warnings or in-place scrubbing.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from app.guardrails.audit import audit_decision
from app.guardrails.exceptions import GuardrailBlocked
from app.guardrails.metrics import record_block, record_scrub, record_allow
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Policy cache
# ---------------------------------------------------------------------------
_policy: dict = {}
_template_patterns: List[re.Pattern] = []
_toxicity_threshold: float = 0.75
_groundedness_threshold: float = 0.30
_max_answer_chars: int = 16000
_citation_validation: bool = True
_refusal_templates: dict = {}
_policy_loaded = False


def _load_policy() -> None:
    global _policy, _template_patterns, _toxicity_threshold
    global _groundedness_threshold, _max_answer_chars, _citation_validation
    global _refusal_templates, _policy_loaded
    if _policy_loaded:
        return
    try:
        from app.guardrails._policy_loader import get_policy
        _policy = get_policy()
        out = _policy.get("output", {})
        _toxicity_threshold = float(out.get("toxicity_threshold", 0.75))
        _groundedness_threshold = float(out.get("groundedness_threshold", 0.30))
        _max_answer_chars = int(out.get("max_answer_chars", 16000))
        _citation_validation = bool(out.get("citation_validation", True))
        _refusal_templates = out.get("refusal_templates", {})

        compiled = []
        for pat_str in out.get("template_artifact_patterns", []):
            try:
                compiled.append(re.compile(pat_str, re.IGNORECASE))
            except re.error:
                pass
        _template_patterns = compiled
    except Exception as e:
        logger.warning("output_guard_policy_load_failed", error=str(e))
    _policy_loaded = True


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GuardedOutput:
    text: str
    original_text: str
    blocked: bool = False
    block_reason: str = ""
    pii_scrubbed: bool = False
    template_artifacts_stripped: bool = False
    citations_validated: bool = False
    fabricated_citations: List[str] = field(default_factory=list)
    hallucination_warning: bool = False
    hallucination_detail: Optional[Dict[str, Any]] = None
    toxicity_score: float = 0.0
    repairs_applied: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Internal checks
# ---------------------------------------------------------------------------

def _strip_template_artifacts(text: str) -> tuple[str, bool]:
    """Remove llama/mistral token artifacts and prompt template leakage."""
    original = text
    for pat in _template_patterns:
        text = pat.sub("", text)
    # Collapse multiple spaces/newlines introduced by stripping
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, text != original


def _validate_citations(answer: str, sources: List[Dict]) -> tuple[str, List[str]]:
    """Validate [filename] citation tokens against retrieved sources.

    Any citation referencing a file not in sources[] is removed.
    Returns (cleaned_answer, fabricated_citations_removed).
    """
    citation_pattern = re.compile(r'\[([^\]]+)\]')
    source_filenames = set()
    for src in (sources or []):
        fn = src.get("filename") or src.get("source") or src.get("file") or ""
        if fn:
            source_filenames.add(fn.lower())
        # Also accept chunk IDs that embed the filename
        chunk_id = src.get("chunk_id") or src.get("id") or ""
        if "_" in chunk_id:
            source_filenames.add(chunk_id.split("_", 1)[-1].lower())

    fabricated = []
    def _check_citation(match: re.Match) -> str:
        token = match.group(1)
        # Allow numeric citations like [1], [2]
        if token.isdigit():
            return match.group(0)
        # Check if the token matches any known source file
        token_lower = token.lower()
        if any(
            token_lower in fn or fn in token_lower or token_lower == fn
            for fn in source_filenames
        ):
            return match.group(0)
        # No match found — fabricated citation
        fabricated.append(token)
        return ""

    if source_filenames:
        cleaned = citation_pattern.sub(_check_citation, answer)
    else:
        cleaned = answer

    return cleaned, fabricated


def _check_groundedness(
    answer: str,
    context_chunks: List[str],
    session_id: str,
) -> tuple[bool, Optional[Dict]]:
    """Reuse Phase 25 hallucination detector. Returns (flagged, detail)."""
    try:
        from app.eval.metrics.hallucination import hallucination_flag_single
        result = hallucination_flag_single(answer, context_chunks)
        flagged = result.get("confidence", 0.0) >= _groundedness_threshold
        return flagged, result
    except Exception as e:
        logger.warning("output_guard_groundedness_failed", error=str(e))
        return False, None


def _check_toxicity(answer: str) -> float:
    """Simple heuristic toxicity check. Returns score 0–1.

    Uses Detoxify if available; falls back to a keyword heuristic.
    """
    # Try Detoxify (GPU-accelerated)
    try:
        from detoxify import Detoxify
        results = Detoxify("original").predict(answer)
        return float(results.get("toxicity", 0.0))
    except ImportError:
        pass
    except Exception as e:
        logger.warning("output_guard_detoxify_failed", error=str(e))

    # Heuristic fallback — check for obvious harmful content patterns
    _TOXIC_PATTERNS = [
        r'\b(?:step[s]?\s+to\s+(?:make|build|create|synthesize)\s+(?:a\s+)?(?:bomb|weapon|explosive|poison|malware|virus))\b',
        r'\b(?:how\s+to\s+(?:kill|murder|harm|hurt|torture)\s+(?:someone|a\s+person|people))\b',
        r'\b(?:child\s+(?:abuse|exploitation|pornography|grooming))\b',
    ]
    text_lower = answer.lower()
    for pat in _TOXIC_PATTERNS:
        if re.search(pat, text_lower):
            return 0.95
    return 0.0


def _get_refusal(key: str) -> str:
    _load_policy()
    return _refusal_templates.get(key) or _refusal_templates.get("generic", "I'm unable to process that request.")


def _check_mojibake_repairs(sources: List[Dict]) -> List[str]:
    """Check if any source chunk was mojibake-repaired and return repair notes."""
    repairs = []
    for src in (sources or []):
        count = src.get("repaired_mojibake_count") or src.get("repairs_applied", 0)
        if count and int(count) > 0:
            fn = src.get("filename") or src.get("source") or "unknown"
            repairs.append(f"{fn}:mojibake_repaired(n={count})")
    return repairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check(
    answer: str,
    context_chunks: Optional[List[str]] = None,
    sources: Optional[List[Dict]] = None,
    session_id: str = "",
    correlation_id: str = "",
    surface: str = "output",
) -> GuardedOutput:
    """Run all output guardrail checks.

    Args:
        answer:         Raw LLM-generated answer text.
        context_chunks: List of retrieved context strings (for groundedness).
        sources:        List of source dicts from AgentResponse (for citation check).
        session_id:     Conversation session ID.
        correlation_id: Request trace ID.
        surface:        Always "output".

    Returns:
        GuardedOutput with cleaned text and metadata.

    Raises:
        GuardrailBlocked if toxicity threshold exceeded (hard block only).
    """
    _load_policy()
    t_start = time.monotonic()
    original = answer

    if not answer:
        return GuardedOutput(text="", original_text="", blocked=False)

    context_chunks = context_chunks or []
    sources = sources or []

    # 1. Groundedness check — must run FIRST on the raw answer before any
    #    citation stripping or PII scrubbing changes the text, otherwise the
    #    lexical scorer compares a mutated answer against the context and
    #    inflates hallucination_rate (the Phase 26 regression root cause).
    hallucination_warning = False
    hallucination_detail: Optional[Dict] = None
    if context_chunks:
        hallucination_warning, hallucination_detail = _check_groundedness(
            answer, context_chunks, session_id
        )
        if hallucination_warning:
            record_block("groundedness", surface)
            logger.warning(
                "output_guard_hallucination_flagged",
                detail=hallucination_detail,
                correlation_id=correlation_id,
                session_id=session_id,
            )

    # 2. Template artifact stripping (P1-7)
    answer, artifacts_stripped = _strip_template_artifacts(answer)
    if artifacts_stripped:
        record_scrub("template_artifact", surface)
        logger.info(
            "output_guard_template_artifacts_stripped",
            correlation_id=correlation_id,
            session_id=session_id,
        )

    # 3. Output length enforcement
    if len(answer) > _max_answer_chars:
        answer = answer[:_max_answer_chars]
        logger.info("output_guard_length_truncated", limit=_max_answer_chars)

    # 4. Citation integrity validation
    fabricated_citations: List[str] = []
    if _citation_validation:
        answer, fabricated_citations = _validate_citations(answer, sources)
        if fabricated_citations:
            record_scrub("citation", surface)
            logger.warning(
                "output_guard_fabricated_citations_removed",
                citations=fabricated_citations[:5],
                correlation_id=correlation_id,
                session_id=session_id,
            )

    # 5. PII egress scrub
    pii_scrubbed = False
    try:
        from app.guardrails.pii import scrub_pii
        answer, pii_scrubbed = scrub_pii(answer)
        if pii_scrubbed:
            record_scrub("pii", surface)
    except Exception as e:
        logger.warning("output_guard_pii_scrub_failed", error=str(e))

    # 6. Toxicity check (hard block if over threshold)
    toxicity_score = _check_toxicity(answer)
    if toxicity_score >= _toxicity_threshold:
        _emit_block("toxicity", "toxicity_threshold_exceeded", surface, correlation_id, session_id)
        raise GuardrailBlocked(
            reason="toxicity_threshold_exceeded",
            surface=surface,
            guard_type="toxicity",
            correlation_id=correlation_id,
            detail=f"score={toxicity_score:.3f}",
        )

    # 7. Mojibake repair disclosure (provenance transparency)
    repairs_applied = _check_mojibake_repairs(sources)

    latency_ms = (time.monotonic() - t_start) * 1000
    record_allow("output", surface)
    audit_decision(
        surface=surface,
        guard_type="output",
        action="allow",
        reason="output_checks_passed",
        session_id=session_id,
        correlation_id=correlation_id,
        query_prefix=answer[:80],
        latency_ms=latency_ms,
        extra={
            "pii_scrubbed": pii_scrubbed,
            "artifacts_stripped": artifacts_stripped,
            "fabricated_citations": len(fabricated_citations),
            "hallucination_warning": hallucination_warning,
            "toxicity_score": round(toxicity_score, 3),
        },
    )

    return GuardedOutput(
        text=answer,
        original_text=original,
        blocked=False,
        pii_scrubbed=pii_scrubbed,
        template_artifacts_stripped=artifacts_stripped,
        citations_validated=_citation_validation,
        fabricated_citations=fabricated_citations,
        hallucination_warning=hallucination_warning,
        hallucination_detail=hallucination_detail,
        toxicity_score=toxicity_score,
        repairs_applied=repairs_applied,
        latency_ms=latency_ms,
    )


def safe_refusal(reason: str) -> str:
    """Return an appropriate refusal message for a blocked response."""
    return _get_refusal(reason)


def _emit_block(
    guard_type: str,
    reason: str,
    surface: str,
    correlation_id: str,
    session_id: str,
) -> None:
    record_block(guard_type, surface)
    audit_decision(
        surface=surface,
        guard_type=guard_type,
        action="block",
        reason=reason,
        session_id=session_id,
        correlation_id=correlation_id,
        latency_ms=0.0,
    )
    logger.warning(
        "output_guard_blocked",
        guard_type=guard_type,
        reason=reason,
        correlation_id=correlation_id,
        session_id=session_id,
    )
