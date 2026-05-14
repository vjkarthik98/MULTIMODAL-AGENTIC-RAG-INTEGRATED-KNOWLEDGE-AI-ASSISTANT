import asyncio
import hashlib
import time
import unicodedata
from typing import Dict, List, Optional, Any

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram

from app.core.config import settings

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_format_duration = Histogram(
    "memory_formatter_duration_seconds",
    "Memory formatter duration",
    ["status"],
)
_format_errors = Counter(
    "memory_formatter_errors_total",
    "Memory formatter errors by type",
    ["error_type"],
)

# SEMAPHORE
_semaphore = asyncio.Semaphore(5)


# NORMALIZE TEXT

def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(text.strip().split())


# TRUNCATE

def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    return text[:max(limit, 0)]


# SHA-256 HASH FOR DEDUP

def _hash(msg: Dict) -> str:
    base = f"{msg.get('role')}|{str(msg.get('content'))[:200]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# RELATIVE TIME LABEL

def _relative_time(ts: Any) -> Optional[str]:
    try:
        age = time.time() - float(ts)
        if age < 60:
            return f"{int(age)}s ago"
        if age < 3600:
            return f"{int(age // 60)}m ago"
        if age < 86400:
            return f"{int(age // 3600)}h ago"
        return f"{int(age // 86400)}d ago"
    except Exception:
        return None


# PII SCRUB — STRIP SENSITIVE CONTENT BEFORE FORMATTING FOR PROMPT

def _scrub_pii(text: str) -> str:
    if not settings.PII_DETECTION_ENABLED:
        return text
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        entities   = getattr(settings, "PII_ENTITIES", [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
            "US_SSN", "CREDIT_CARD", "LOCATION", "IP_ADDRESS",
        ])
        analyzer   = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        results    = analyzer.analyze(text=text, entities=entities, language="en")
        if results:
            text = anonymizer.anonymize(text=text, analyzer_results=results).text
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pii_scrub_failed", error=str(exc))
    return text


# FORMAT SINGLE MESSAGE

def _format_message(msg: Dict, per_msg_limit: int, scrub: bool = False) -> str:
    try:
        role = str(msg.get("role", "user")).lower()
        if role not in {"user", "assistant", "system"}:
            role = "user"

        content = _clean(msg.get("content", ""))
        if len(content) < 3:
            return ""

        # PII SCRUB BEFORE INJECTING INTO PROMPT
        if scrub:
            content = _scrub_pii(content)

        modality  = msg.get("modality", "text")
        ts        = msg.get("timestamp")
        language  = msg.get("language")
        importance = msg.get("importance", 0.5)

        meta = f"[{role.upper()}]"

        if modality and modality != "text":
            meta += f"[{modality.upper()}]"

        if language and language != "en":
            meta += f"[{language.upper()}]"

        rel_time = _relative_time(ts) if ts else None
        if rel_time:
            meta += f"[{rel_time}]"

        # HIGH IMPORTANCE FLAG
        if importance and float(importance) >= 0.9:
            meta += "[IMPORTANT]"

        content = _truncate(content, per_msg_limit)

        return f"{meta}: {content}"

    except Exception:
        return ""


# DEDUP BY CONTENT HASH

def _dedup(messages: List[Dict]) -> List[Dict]:
    seen: set        = set()
    out:  List[Dict] = []
    for m in messages:
        try:
            h = _hash(m)
            if h in seen:
                continue
            seen.add(h)
            out.append(m)
        except Exception:
            continue
    return out


# IMPORTANCE SORT — HIGHER IMPORTANCE FLOATS TO TOP

def _sort_by_importance(messages: List[Dict]) -> List[Dict]:
    def _imp(m: Dict) -> float:
        try:
            return float(m.get("importance", 0.5))
        except Exception:
            return 0.5
    return sorted(messages, key=_imp, reverse=True)


# GROUP MESSAGES BY MODALITY FOR STRUCTURED OUTPUT

def _group_by_modality(messages: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for m in messages:
        mod = m.get("modality", "text")
        if mod not in groups:
            groups[mod] = []
        groups[mod].append(m)
    return groups


# LANGUAGE STATS — FOR MULTILINGUAL MEMORY CONTEXT

def _language_stats(messages: List[Dict]) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for m in messages:
        lang = m.get("language", "unknown")
        stats[lang] = stats.get(lang, 0) + 1
    return stats


# MAIN SYNC FORMAT HISTORY

def format_history(
    history: List[Dict],
    max_messages: Optional[int] = None,
    include_system: bool = True,
    session_id: str = "default",
    scrub_pii: bool = True,
    group_by_modality: bool = False,
) -> str:

    if not history:
        return ""

    start        = time.time()
    max_messages = max_messages or settings.MAX_HISTORY_MESSAGES

    with tracer.start_as_current_span("format_history") as span:
        span.set_attribute("history.size", len(history))
        span.set_attribute("session.id", session_id)

        try:
            # PER-MESSAGE CONTENT BUDGET
            per_msg_limit = max(
                settings.MEMORY_MAX_CONTEXT_CHARS // max(max_messages, 1),
                100,
            )

            history = _dedup(history)

            system_msgs: List[Dict] = []
            normal_msgs: List[Dict] = []

            for msg in history:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "system":
                    system_msgs.append(msg)
                else:
                    normal_msgs.append(msg)

            # TAKE MOST RECENT WINDOW THEN SORT BY IMPORTANCE WITHIN WINDOW
            normal_msgs = normal_msgs[-max_messages:]
            normal_msgs = _sort_by_importance(normal_msgs)

            parts: List[str] = ["[CONVERSATION MEMORY]"]

            # SYSTEM MESSAGES
            if include_system and system_msgs:
                parts.append("\n[SYSTEM]")
                for m in system_msgs[-settings.MAX_SYSTEM_MESSAGES:]:
                    fm = _format_message(m, per_msg_limit, scrub=scrub_pii)
                    if fm:
                        parts.append(fm)

            # GROUPED BY MODALITY OR FLAT CONVERSATION
            if group_by_modality and normal_msgs:
                groups = _group_by_modality(normal_msgs)
                for mod, msgs in groups.items():
                    parts.append(f"\n[{mod.upper()} CONTEXT]")
                    for m in msgs:
                        fm = _format_message(m, per_msg_limit, scrub=scrub_pii)
                        if fm:
                            parts.append(fm)
            else:
                if normal_msgs:
                    parts.append("\n[CONVERSATION]")
                    for m in normal_msgs:
                        fm = _format_message(m, per_msg_limit, scrub=scrub_pii)
                        if fm:
                            parts.append(fm)

            # LANGUAGE STATS FOOTER — USEFUL FOR MULTILINGUAL SESSIONS
            lang_stats = _language_stats(normal_msgs)
            if len(lang_stats) > 1:
                langs = ", ".join(
                    f"{k}:{v}" for k, v in
                    sorted(lang_stats.items(), key=lambda x: -x[1])
                )
                parts.append(f"\n[LANGUAGES: {langs}]")

            result = "\n".join(parts).strip()

            # SAFE TRUNCATION WITH HEAD + TAIL PRESERVATION
            if len(result) > settings.MAX_PROMPT_CHARS:
                split  = int(settings.MAX_PROMPT_CHARS * 0.7)
                result = (
                    result[:split] +
                    "\n...\n" +
                    result[-(settings.MAX_PROMPT_CHARS - split):]
                )
                logger.warning(
                    "formatter_truncated",
                    session_id=session_id,
                )

            latency = round(time.time() - start, 3)

            _format_duration.labels(status="success").observe(latency)

            span.set_attribute("output.size", len(result))
            span.set_attribute("system.count", len(system_msgs))
            span.set_attribute("normal.count", len(normal_msgs))
            span.set_status(Status(StatusCode.OK))

            logger.debug(
                "formatter_success",
                system_count=len(system_msgs),
                normal_count=len(normal_msgs),
                total_messages=len(history),
                output_size=len(result),
                latency=latency,
                session_id=session_id,
            )

            return result

        except Exception as exc:
            latency    = round(time.time() - start, 3)
            error_type = type(exc).__name__

            _format_duration.labels(status="error").observe(latency)
            _format_errors.labels(error_type=error_type).inc()

            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)

            logger.error(
                "formatter_failed",
                error=str(exc),
                error_type=error_type,
                session_id=session_id,
            )
            return ""


# ASYNC WRAPPER

async def format_history_async(
    history: List[Dict],
    max_messages: Optional[int] = None,
    include_system: bool = True,
    session_id: str = "default",
    scrub_pii: bool = True,
    group_by_modality: bool = False,
) -> str:

    async with _semaphore:
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: format_history(
                history,
                max_messages,
                include_system,
                session_id,
                scrub_pii,
                group_by_modality,
            ),
        )


