import asyncio
import hashlib
import re
import time
import unicodedata
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram

from app.core.config import settings

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_summary_duration = Histogram(
    "memory_summarizer_duration_seconds",
    "Memory summarizer duration",
    ["status"],
)
_summary_errors = Counter(
    "memory_summarizer_errors_total",
    "Memory summarizer errors by type",
    ["error_type"],
)
_summary_length = Histogram(
    "memory_summary_length_chars",
    "Length of generated summaries in characters",
)

# SEMAPHORE
_semaphore = asyncio.Semaphore(5)


# NORMALIZE TEXT


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(text.strip().split())


# SHA-256 HASH FOR DEDUP


def _hash(msg: dict) -> str:
    base = f"{msg.get('role')}|{str(msg.get('content'))[:200]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# DEDUP HISTORY


def _dedup(history: list[dict]) -> list[dict]:
    seen: set = set()
    out: list[dict] = []
    for msg in history:
        try:
            h = _hash(msg)
            if h in seen:
                continue
            seen.add(h)
            out.append(msg)
        except Exception:
            continue
    return out


# IMPORTANCE SORT


def _sort_by_importance(history: list[dict]) -> list[dict]:
    def _imp(m: dict) -> float:
        try:
            return float(m.get("importance", 0.5))
        except Exception:
            return 0.5

    return sorted(history, key=_imp, reverse=True)


# PII SCRUB BEFORE SUMMARIZATION


def _scrub_pii(text: str) -> str:
    if not settings.PII_DETECTION_ENABLED:
        return text
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        entities = getattr(
            settings,
            "PII_ENTITIES",
            [
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "US_SSN",
                "CREDIT_CARD",
                "LOCATION",
                "IP_ADDRESS",
            ],
        )
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        results = analyzer.analyze(text=text, entities=entities, language="en")
        if results:
            text = anonymizer.anonymize(text=text, analyzer_results=results).text
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pii_scrub_failed", error=str(exc))
    return text


# FORMAT CONVERSATION FOR SUMMARIZATION INPUT


def _format_for_summary(history: list[dict]) -> str:
    max_chars = settings.MEMORY_SUMMARY_INPUT_CHARS
    max_msgs = settings.MAX_HISTORY_MESSAGES

    history = _dedup(history[-max_msgs:])
    history = _sort_by_importance(history)

    parts: list[str] = []
    total: int = 0

    for msg in history:
        try:
            role = str(msg.get("role", "user")).upper()
            content = _clean(msg.get("content", ""))
            modality = msg.get("modality", "text")

            if len(content) < 5:
                continue

            # PII SCRUB BEFORE SENDING TO LLM
            content = _scrub_pii(content)
            content = content[: settings.MAX_PROMPT_CHARS]

            mod_tag = f"[{modality.upper()}] " if modality != "text" else ""
            line = f"{role}: {mod_tag}{content}"
            length = len(line)

            if total + length > max_chars:
                break

            parts.append(line)
            total += length

        except Exception:
            continue

    return "\n".join(parts)


# KEYWORD EXTRACTION FOR SUMMARY TAGS


def _extract_keywords(text: str, max_kw: int = 10) -> list[str]:
    try:
        import yake

        extractor = yake.KeywordExtractor(top=max_kw, stopwords=None)
        kws = extractor.extract_keywords(text)
        return [kw for kw, _ in kws]
    except ImportError:
        pass
    try:
        from keybert import KeyBERT

        kb = KeyBERT()
        kws = kb.extract_keywords(text, top_n=max_kw)
        return [kw for kw, _ in kws]
    except ImportError:
        pass
    return []


# COMPRESSION RATIO CHECK


def _compression_ratio(original: str, summary: str) -> float:
    if not original:
        return 0.0
    return round(len(summary) / max(len(original), 1), 3)


# VALIDATE SUMMARY OUTPUT


def _validate(summary: str) -> str:
    if not summary:
        return ""

    summary = _clean(summary)

    if len(summary) < settings.MIN_SUMMARY_LENGTH:
        return ""

    summary = summary[: settings.MEMORY_SUMMARY_MAX_CHARS]

    required = ["Key Facts", "User Intent"]

    if any(r in summary for r in required):
        return summary

    # ACCEPT SUFFICIENTLY LONG SUMMARY EVEN WITHOUT REQUIRED HEADERS
    if len(summary) >= settings.MIN_SUMMARY_LENGTH * 2:
        logger.warning("summary_missing_headers_accepted_as_fallback")
        return summary

    return ""


# BUILD SUMMARIZATION PROMPT

# Finance number preservation instruction — injected into both prompt builders
# (Plan Phase 6: finance number preservation in summaries).
_FINANCE_NUMBER_RULE = (
    "- Preserve ALL exact financial figures verbatim: dollar amounts, percentages, "
    "basis points, EPS values, dates, ticker symbols, and company names. "
    "Do NOT paraphrase or round any numbers.\n"
)


_FINANCE_NUM_RE = re.compile(
    r'(?:[$€£]\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:billion|million|thousand|B|M|K))?'
    r'|\d[\d,]*(?:\.\d+)?%'
    r'|\b\d[\d,]*(?:\.\d+)?\s*(?:billion|million|bps|basis points)'
    r'|\bQ[1-4]\s*(?:FY)?\d{2,4}'
    r'|\bFY\d{2,4}'
    r'|\b(?:EPS|eps)\s*(?:of\s*)?\$?\d[\d.]*)',
    re.IGNORECASE,
)


def _extract_finance_numbers(text: str) -> list[str]:
    return list(dict.fromkeys(_FINANCE_NUM_RE.findall(text)))


def _build_prompt(conv: str) -> str:
    instruction = (
        "Compress this conversation into structured memory.\n"
        "Rules:\n"
        "- Keep only factual, actionable information\n"
        "- No hallucination or inference beyond what is stated\n"
        "- No filler words\n"
        "- Be maximally concise\n" + _FINANCE_NUMBER_RULE + "\n"
    )

    format_block = (
        "Key Facts:\n- ...\n\n"
        "User Intent:\n- ...\n\n"
        "Preferences:\n- ...\n\n"
        "Tasks:\n- ...\n\n"
        "Context:\n- ..."
    )

    max_chars = settings.MAX_PROMPT_CHARS
    available = max_chars - len(instruction) - len(format_block) - 50
    body = f"Conversation:\n{conv[:max(available, 0)]}\n\n"

    return instruction + body + format_block


# INCREMENTAL SUMMARY — MERGE EXISTING SUMMARY WITH NEW TURNS


def _build_incremental_prompt(
    existing_summary: str,
    new_turns: str,
) -> str:
    instruction = (
        "Update this memory summary with new conversation turns.\n"
        "Rules:\n"
        "- Preserve existing facts unless contradicted\n"
        "- Add new facts from new turns\n"
        "- Remove outdated entries if new turns contradict them\n"
        "- Be maximally concise\n" + _FINANCE_NUMBER_RULE + "\n"
    )

    format_block = (
        "Key Facts:\n- ...\n\n"
        "User Intent:\n- ...\n\n"
        "Preferences:\n- ...\n\n"
        "Tasks:\n- ...\n\n"
        "Context:\n- ..."
    )

    max_chars = settings.MAX_PROMPT_CHARS
    available = max_chars - len(instruction) - len(format_block) - 100
    half = available // 2

    existing_block = f"EXISTING SUMMARY:\n{existing_summary[:half]}\n\n"
    new_block = f"NEW TURNS:\n{new_turns[:half]}\n\n"

    return instruction + existing_block + new_block + format_block


# PERSIST SUMMARY TO MONGO


def _persist_summary(
    summary: str,
    session_id: str,
    mongo_memory: Any,
    keywords: list[str] | None = None,
    user_id: str | None = None,
) -> None:
    try:
        if mongo_memory and hasattr(mongo_memory, "store_summary"):
            mongo_memory.store_summary(session_id, summary, user_id=user_id)
        if keywords:
            logger.debug(
                "summary_keywords_extracted",
                count=len(keywords),
                session_id=session_id,
            )
    except Exception as exc:
        logger.warning(
            "summary_persist_failed",
            error=str(exc),
            session_id=session_id,
        )


# MAIN SYNC SUMMARIZER


def summarize_conversation(
    llm: Any,
    history: list[dict],
    session_id: str = "default",
    mongo_memory: Any = None,
    existing_summary: str | None = None,
    user_id: str | None = None,
) -> str:

    if not history:
        return ""

    start = time.time()

    with tracer.start_as_current_span("summarize_conversation") as span:
        span.set_attribute("history.size", len(history))
        span.set_attribute("session.id", session_id)
        span.set_attribute("incremental", bool(existing_summary))

        try:
            conv = _format_for_summary(history)

            if not conv:
                logger.warning(
                    "summary_empty_conversation",
                    session_id=session_id,
                )
                span.set_status(Status(StatusCode.OK))
                return ""

            # INCREMENTAL OR FULL SUMMARIZATION
            if existing_summary and len(existing_summary.strip()) > settings.MIN_SUMMARY_LENGTH:
                prompt = _build_incremental_prompt(existing_summary, conv)
                span.set_attribute("summary.mode", "incremental")
            else:
                prompt = _build_prompt(conv)
                span.set_attribute("summary.mode", "full")

            # LLM GENERATION WITH HARD TIMEOUT GUARD
            t_llm = time.time()

            try:
                raw = llm.generate(
                    prompt,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=0.1,
                    session_id=session_id,
                )
            except Exception as exc:
                logger.error(
                    "summary_llm_generate_failed",
                    error=str(exc),
                    session_id=session_id,
                )
                return ""

            llm_latency = time.time() - t_llm

            if llm_latency > settings.MODEL_TIMEOUT_SEC:
                logger.warning(
                    "summary_llm_timeout",
                    llm_latency=round(llm_latency, 2),
                    session_id=session_id,
                )
                return ""

            summary = _validate(raw)

            if not summary:
                logger.warning(
                    "summary_invalid",
                    raw_length=len(raw) if raw else 0,
                    session_id=session_id,
                )
                span.set_status(Status(StatusCode.OK))
                return ""

            # FINANCE NUMBER PRESERVATION PASS
            # Extract numbers from original turns; append any dropped by LLM.
            original_numbers = _extract_finance_numbers(conv)
            if original_numbers:
                missing = [n for n in original_numbers if n not in summary]
                if missing:
                    summary = summary.rstrip() + "\n[KEY FIGURES: " + ", ".join(missing[:20]) + "]"
                    logger.debug(
                        "finance_figures_appended",
                        count=len(missing),
                        session_id=session_id,
                    )

            # KEYWORD EXTRACTION FOR TAGGING
            keywords = _extract_keywords(summary)

            # COMPRESSION RATIO LOG
            ratio = _compression_ratio(conv, summary)
            logger.debug(
                "summary_compression_ratio",
                ratio=ratio,
                session_id=session_id,
            )

            # AUTO-PERSIST TO MONGO
            if mongo_memory:
                _persist_summary(summary, session_id, mongo_memory, keywords, user_id=user_id)

            latency = round(time.time() - start, 2)

            _summary_duration.labels(status="success").observe(latency)
            _summary_length.observe(len(summary))

            span.set_attribute("summary.length", len(summary))
            span.set_attribute("summary.keywords", len(keywords))
            span.set_attribute("compression.ratio", ratio)
            span.set_status(Status(StatusCode.OK))

            logger.info(
                "summary_success",
                length=len(summary),
                input_messages=len(history),
                conv_chars=len(conv),
                keywords=keywords[:5],
                compression_ratio=ratio,
                llm_latency=round(llm_latency, 2),
                latency=latency,
                session_id=session_id,
            )

            return summary

        except Exception as exc:
            latency = round(time.time() - start, 2)
            error_type = type(exc).__name__

            _summary_duration.labels(status="error").observe(latency)
            _summary_errors.labels(error_type=error_type).inc()

            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)

            logger.error(
                "summary_failed",
                error=str(exc),
                error_type=error_type,
                session_id=session_id,
            )
            return ""


# ASYNC WRAPPER


async def summarize_conversation_async(
    llm: Any,
    history: list[dict],
    session_id: str = "default",
    mongo_memory: Any = None,
    existing_summary: str | None = None,
    user_id: str | None = None,
) -> str:

    async with _semaphore:
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: summarize_conversation(
                llm,
                history,
                session_id,
                mongo_memory,
                existing_summary,
                user_id,
            ),
        )


# GDPR PURGE — DELETE ALL SUMMARIES FOR A USER/SESSION


async def gdpr_purge_summaries(
    session_id: str,
    mongo_memory: Any,
) -> None:
    try:
        if mongo_memory and hasattr(mongo_memory, "delete"):
            await asyncio.get_event_loop().run_in_executor(
                None,
                mongo_memory.delete,
                session_id,
            )
            logger.info(
                "gdpr_summaries_purged",
                session_id=session_id,
            )
    except Exception as exc:
        logger.error(
            "gdpr_purge_summaries_failed",
            session_id=session_id,
            error=str(exc),
        )
