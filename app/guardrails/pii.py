"""PII detection and anonymization.

Wraps Microsoft Presidio (already in requirements.txt) into two clean
functions: detect_pii() for audit/logging, scrub_pii() for output.

Pre-warmed at server startup via warm_up() called from infra_registry.
"""
from __future__ import annotations

import re
import threading
from typing import List, Optional

import structlog

logger = structlog.get_logger(__name__)

# Lazy-loaded Presidio objects — initialized once via warm_up()
_analyzer: Optional[object] = None
_anonymizer: Optional[object] = None
_lock = threading.Lock()

# Extra regex-based PII patterns from policies.yaml
_EXTRA_PATTERNS: List[dict] = []
_POLICY_LOADED = False


def _load_extra_patterns() -> None:
    global _EXTRA_PATTERNS, _POLICY_LOADED
    if _POLICY_LOADED:
        return
    try:
        from app.guardrails._policy_loader import get_policy
        p = get_policy()
        _EXTRA_PATTERNS = p.get("pii", {}).get("extra_regex_patterns", [])
    except Exception:
        pass
    _POLICY_LOADED = True


def warm_up() -> None:
    """Pre-initialize Presidio engines. Call from startup to avoid cold-load latency."""
    global _analyzer, _anonymizer
    with _lock:
        if _analyzer is not None:
            return
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            _analyzer = AnalyzerEngine()
            _anonymizer = AnonymizerEngine()
            logger.info("pii_presidio_warmed_up")
        except Exception as e:
            logger.warning("pii_presidio_warmup_failed", error=str(e))


def _get_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        warm_up()
    return _analyzer, _anonymizer


def _get_entity_types() -> List[str]:
    try:
        from app.guardrails._policy_loader import get_policy
        return get_policy().get("pii", {}).get("entity_types", [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
        ])
    except Exception:
        return ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"]


def _apply_extra_regex(text: str) -> str:
    """Apply additional regex-based PII patterns from policies.yaml."""
    _load_extra_patterns()
    for pat in _EXTRA_PATTERNS:
        try:
            text = re.sub(pat["pattern"], pat["replacement"], text, flags=re.IGNORECASE)
        except re.error:
            pass
    return text


def detect_pii(text: str, language: str = "en") -> List[dict]:
    """Detect PII entities in text. Returns list of {entity_type, start, end, score}."""
    if not text:
        return []
    analyzer, _ = _get_engines()
    if analyzer is None:
        return []
    try:
        results = analyzer.analyze(
            text=text,
            language=language,
            entities=_get_entity_types(),
        )
        return [
            {
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 3),
                "text": text[r.start:r.end],
            }
            for r in results
        ]
    except Exception as e:
        logger.warning("pii_detect_failed", error=str(e))
        return []


def scrub_pii(text: str, language: str = "en") -> tuple[str, bool]:
    """Scrub PII from text using anonymization.

    Returns:
        (scrubbed_text, was_scrubbed: bool)
    """
    if not text:
        return text, False

    analyzer, anonymizer = _get_engines()
    if analyzer is None or anonymizer is None:
        # Presidio unavailable — fall back to regex-only
        scrubbed = _apply_extra_regex(text)
        return scrubbed, scrubbed != text

    try:
        results = analyzer.analyze(
            text=text,
            language=language,
            entities=_get_entity_types(),
        )
        if not results:
            # No Presidio hits — still apply regex
            scrubbed = _apply_extra_regex(text)
            return scrubbed, scrubbed != text

        from presidio_anonymizer.entities import OperatorConfig
        operators = {
            entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
            for entity in _get_entity_types()
        }
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )
        scrubbed = _apply_extra_regex(anonymized.text)
        was_scrubbed = scrubbed != text
        if was_scrubbed:
            logger.info(
                "pii_scrubbed",
                entity_types=[r.entity_type for r in results],
                text_length=len(text),
            )
        return scrubbed, was_scrubbed
    except Exception as e:
        logger.warning("pii_scrub_failed", error=str(e))
        return text, False


def has_pii(text: str, language: str = "en") -> bool:
    """Quick boolean check — True if any PII detected."""
    return bool(detect_pii(text, language))


def strip_pii_from_prompt(prompt: str, language: str = "en") -> tuple[str, bool]:
    """Strip PII from an LLM prompt before it reaches the model.

    Unlike scrub_pii() which anonymizes output, this replaces PII in the
    prompt with generic placeholders so the model never sees the raw values.
    Returns (cleaned_prompt, was_stripped: bool).

    Called by rag_pipeline and query_pipeline before llm.generate().
    """
    return scrub_pii(prompt, language)
