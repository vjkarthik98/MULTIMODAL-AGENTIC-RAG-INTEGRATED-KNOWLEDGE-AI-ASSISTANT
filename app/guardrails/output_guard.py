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

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.guardrails.audit import audit_decision
from app.guardrails.exceptions import GuardrailBlocked
from app.guardrails.metrics import record_allow, record_block, record_scrub
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Policy cache
# ---------------------------------------------------------------------------
_policy: dict = {}
_template_patterns: list[re.Pattern] = []
_toxicity_threshold: float = 0.75
_groundedness_threshold: float = 0.30
_max_answer_chars: int = 16000
_citation_validation: bool = True
_refusal_templates: dict = {}

# Last-resort PII pattern used only if app.guardrails.pii fails to import
# entirely — covers the highest-severity categories (email, phone, SSN,
# credit card) so a total module failure still doesn't ship raw PII.
_EMERGENCY_PII_RE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.-]+"  # email
    r"|\b\d{3}-\d{2}-\d{4}\b"  # US SSN
    r"|\b(?:\d[ -]*?){13,16}\b"  # credit card
    r"|\b(?:\+?\d{1,2}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"  # phone
)
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
    fabricated_citations: list[str] = field(default_factory=list)
    hallucination_warning: bool = False
    hallucination_detail: dict[str, Any] | None = None
    toxicity_score: float = 0.0
    repairs_applied: list[str] = field(default_factory=list)
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


# Numeric / multi-index citation token, e.g. [1], [2,3], [2, 3].
_NUMERIC_CITATION_RE = re.compile(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]')


def _detect_suspect_citations(answer: str, sources: list[dict]) -> list[str]:
    """Detect (but do NOT mutate) citation tokens for audit/metrics only.

    The answer body is cleaned downstream by
    ``app.core.response.strip_inline_citations`` (the single, canonical
    stripper), which removes every inline citation — numeric and structured —
    so the user-facing prose carries no references or filename leakage. This
    function therefore never edits the text; it only reports bracketed
    non-numeric tokens that reference a filename NOT in sources[], so the
    pipeline can log genuine provenance fabrication without the old
    destructive ``re.sub`` that left whitespace gaps ("net sales of  net
    sales") and wiped [2,3] before the source-chip filter could read it.
    """
    source_filenames = set()
    for src in sources or []:
        fn = src.get("filename") or src.get("source") or src.get("file") or ""
        if fn:
            source_filenames.add(fn.lower())
        chunk_id = src.get("chunk_id") or src.get("id") or ""
        if "_" in str(chunk_id):
            source_filenames.add(str(chunk_id).split("_", 1)[-1].lower())

    suspect: list[str] = []
    for match in re.finditer(r'\[([^\]]+)\]', answer):
        token = match.group(1).strip()
        # Numeric / multi-index citations are always legitimate.
        if _NUMERIC_CITATION_RE.fullmatch(match.group(0)):
            continue
        # Structured locator markers emitted by ingestion/reranker are not
        # fabrications (they are stripped downstream, not leaked).
        if re.match(
            r'(?i)^(sheet|page|pg|pages?|hyperlink|src|spk|t|para|paragraph|row|rows|section|figure|fig|table|caption|slide|frame|timestamp|error_markers)\b',
            token,
        ):
            continue
        token_lower = token.lower()
        if source_filenames and any(
            token_lower in fn or fn in token_lower for fn in source_filenames
        ):
            continue
        # Unknown bracketed token referencing no known source — report only.
        suspect.append(token)

    return suspect


def _check_groundedness(
    answer: str,
    context_chunks: list[str],
    session_id: str,
) -> tuple[bool, dict | None]:
    """Reuse Phase 25 hallucination detector. Returns (flagged, detail)."""
    try:
        from app.eval.metrics.hallucination import hallucination_flag_single

        result = hallucination_flag_single(answer, context_chunks)
        flagged = result.get("confidence", 0.0) >= _groundedness_threshold
        return flagged, result
    except Exception as e:
        logger.warning("output_guard_groundedness_failed", error=str(e))
        return False, None


# Detoxify availability + model are resolved once. A failed import is NOT
# cached by Python (sys.modules only caches successes), so the old per-call
# `from detoxify import Detoxify` re-scanned the path on every answer; and
# when installed, `Detoxify("original")` would reload the full model per call.
_detoxify_model: object | None = None
_detoxify_checked = False


def _get_detoxify():
    global _detoxify_model, _detoxify_checked
    if _detoxify_checked:
        return _detoxify_model
    _detoxify_checked = True
    try:
        # Detoxify fetches its checkpoint via torch.hub, which reads
        # TORCH_HOME (not HF_HOME — a separate cache mechanism from Hugging
        # Face Hub). start_server.py and download_all_models.py already set
        # this, but neither runs for a plain `pytest`/script/notebook
        # invocation, so torch.hub would otherwise fall back to its own
        # default cache dir, miss the checkpoint download_all_models.py
        # already fetched into .torch_cache/, and attempt a live re-download
        # — setdefault so an already-set TORCH_HOME (e.g. from
        # start_server.py) is never overridden.
        os.environ.setdefault("TORCH_HOME", settings.TORCH_HOME)
        from detoxify import Detoxify

        _detoxify_model = Detoxify("original")
    except ImportError:
        _detoxify_model = None
    except Exception as e:
        logger.warning("output_guard_detoxify_init_failed", error=str(e))
        _detoxify_model = None
    return _detoxify_model


_HARMFUL_CONTENT_PATTERNS = [
    r'\b(?:step[s]?\s+to\s+(?:make|build|create|synthesize)\s+(?:a\s+)?(?:bomb|weapon|explosive|poison|malware|virus))\b',
    r'\b(?:how\s+to\s+(?:kill|murder|harm|hurt|torture)\s+(?:someone|a\s+person|people))\b',
    r'\b(?:child\s+(?:abuse|exploitation|pornography|grooming))\b',
]


def _check_toxicity(answer: str) -> float:
    """Toxicity/harm score 0-1, combining two DIFFERENT threat categories:

    1. Detoxify — trained on toxic/abusive COMMENT-style language (insults,
       threats, hate speech). It does NOT reliably score calm, instructional
       text about dangerous topics as toxic (e.g. "Steps to build a bomb:
       step 1 is..." scores near 0 — there's no abusive tone for it to
       detect), so it must not be the only signal.
    2. A keyword/pattern heuristic for explicit harmful-instruction content
       (weapons/explosives instructions, violence instructions, child safety
       violations) — a different category Detoxify isn't designed to catch.

    These are complementary, not a primary/fallback pair: run both whenever
    possible and take the max, so an available Detoxify model can no longer
    suppress the harmful-content heuristic's genuine hits.
    """
    scores = []

    model = _get_detoxify()
    if model is not None:
        try:
            results = model.predict(answer)
            scores.append(float(results.get("toxicity", 0.0)))
        except Exception as e:
            logger.warning("output_guard_detoxify_failed", error=str(e))

    text_lower = answer.lower()
    for pat in _HARMFUL_CONTENT_PATTERNS:
        if re.search(pat, text_lower):
            scores.append(0.95)
            break

    return max(scores) if scores else 0.0


def _get_refusal(key: str) -> str:
    _load_policy()
    return _refusal_templates.get(key) or _refusal_templates.get(
        "generic", "I'm unable to process that request."
    )


def _check_mojibake_repairs(sources: list[dict]) -> list[str]:
    """Check if any source chunk was mojibake-repaired and return repair notes."""
    repairs = []
    for src in sources or []:
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
    context_chunks: list[str] | None = None,
    sources: list[dict] | None = None,
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
    hallucination_detail: dict | None = None
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

    # 4. Citation integrity — DETECT-ONLY (never mutate the answer here).
    #    The canonical cleaner strip_inline_citations() removes ALL inline
    #    citations (numeric + structured) so the prose has no references or
    #    filename leakage. Callers that need the cited-source indices MUST run
    #    extract_cited_indices() on the RAW answer (before this guard) — which
    #    is why this guard must not delete citation tokens.
    fabricated_citations: list[str] = []
    if _citation_validation:
        fabricated_citations = _detect_suspect_citations(answer, sources)
        if fabricated_citations:
            logger.info(
                "output_guard_suspect_citations_detected",
                citations=fabricated_citations[:5],
                correlation_id=correlation_id,
                session_id=session_id,
            )

    # 5. PII egress scrub — fail CLOSED: scrub_pii() itself never raises for
    # its own detection/anonymization failures (it falls back to a regex-only
    # scrub internally). This except only catches a total import/module
    # failure of app.guardrails.pii, in which case we still apply a minimal
    # inline regex scrub rather than shipping the raw answer.
    pii_scrubbed = False
    try:
        from app.guardrails.pii import scrub_pii

        answer, pii_scrubbed = scrub_pii(answer)
        if pii_scrubbed:
            record_scrub("pii", surface)
    except Exception as e:
        logger.error("output_guard_pii_scrub_module_failed", error=str(e))
        _fallback_scrubbed = _EMERGENCY_PII_RE.sub("<REDACTED>", answer)
        if _fallback_scrubbed != answer:
            answer = _fallback_scrubbed
            pii_scrubbed = True
            record_scrub("pii", surface)

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
