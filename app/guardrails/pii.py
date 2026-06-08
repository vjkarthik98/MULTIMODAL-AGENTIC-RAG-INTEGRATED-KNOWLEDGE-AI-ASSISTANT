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
            import logging as _logging
            import warnings as _warnings

            # Suppress Presidio's noisy multilingual warnings before any import.
            # These fire when the registry skips non-English recognizers — expected
            # behaviour, not errors. Both the Python warnings and logging channels
            # need to be silenced because Presidio uses both.
            _warnings.filterwarnings("ignore", category=UserWarning, module="presidio")
            for _plog in ("presidio-analyzer", "presidio_analyzer",
                          "presidio-anonymizer", "presidio_anonymizer"):
                _logging.getLogger(_plog).setLevel(_logging.ERROR)

            from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine

            # English-only NLP engine — no multilingual models loaded.
            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
            }
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()

            # English-only registry — non-en recognizers are skipped (warnings now suppressed).
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(languages=["en"], nlp_engine=nlp_engine)

            _analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                registry=registry,
                supported_languages=["en"],
            )
            _anonymizer = AnonymizerEngine()
            logger.info("pii_presidio_warmed_up", language="en")
        except Exception as e:
            logger.warning("pii_presidio_warmup_failed", error=str(e))


def _get_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        warm_up()
    return _analyzer, _anonymizer


def _get_entity_types() -> List[str]:
    # PERSON is intentionally excluded — CEO names, author names, and other
    # public figures are not PII in this context and cause false positives.
    _SAFE = ["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IP_ADDRESS"]
    try:
        from app.guardrails._policy_loader import get_policy
        policy_types = get_policy().get("pii", {}).get("entity_types", _SAFE)
        # Always strip PERSON even if policy re-adds it
        return [e for e in policy_types if e != "PERSON"]
    except Exception:
        return _SAFE


def _apply_extra_regex(text: str) -> str:
    """Apply additional regex-based PII patterns from policies.yaml."""
    _load_extra_patterns()
    for pat in _EXTRA_PATTERNS:
        try:
            text = re.sub(pat["pattern"], pat["replacement"], text, flags=re.IGNORECASE)
        except re.error:
            pass
    return text


# Minimum Presidio confidence to accept a PII match. Presidio defaults to 0.0
# (accepts everything), which causes false positives like "FY2023" matching
# US_DRIVER_LICENSE at score 0.3. Real PII (SSN, credit cards, email) scores
# 0.6-1.0, so 0.5 keeps genuine detections while dropping low-confidence noise.
_PII_SCORE_THRESHOLD = 0.35


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
            score_threshold=_PII_SCORE_THRESHOLD,
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
            score_threshold=_PII_SCORE_THRESHOLD,
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
