import hashlib
import time
import unicodedata
from typing import Dict, List, Optional

from app.core.config import settings
from app.memory.formatter import format_history
from app.utils.logger import get_logger

logger = get_logger(__name__)


# CLEAN

def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(text.strip().split())


# TRUNCATE

def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    return text[:limit]


# HASH PREFIX

def _prefix_hash(text: str, length: int = 200) -> str:
    return hashlib.sha256(text[:length].encode("utf-8")).hexdigest()


# DEDUP

def _dedup_summary_history(summary: str, history: str) -> tuple:
    if not summary or not history:
        return summary, history

    # REMOVE SUMMARY PREFIX FROM HISTORY IF OVERLAPPING
    if _prefix_hash(summary) == _prefix_hash(history):
        history = history.replace(summary[:200], "").strip()

    return summary, history


# FETCH MONGO SUMMARY

def _fetch_mongo_summary(session_id: str) -> str:
    try:
        from app.core.infra_registry import infra
        mongo = infra.get_mongo()
        if mongo and hasattr(mongo, "get_latest_summary"):
            return mongo.get_latest_summary(session_id) or ""
    except Exception:
        pass
    return ""


# MAIN

def build_memory_context(
    summary: str,
    filtered_history: List[Dict],
    max_total_chars: Optional[int] = None,
    session_id: str = "default",
) -> str:

    start           = time.time()
    max_total_chars = max_total_chars or settings.MEMORY_MAX_CONTEXT_CHARS

    try:
        # FETCH SUMMARY FROM MONGO IF NOT PROVIDED
        summary = _clean(summary)
        if not summary:
            summary = _clean(_fetch_mongo_summary(session_id))

        # FORMAT HISTORY
        history_str = format_history(
            filtered_history,
            max_messages=settings.MAX_HISTORY_MESSAGES,
            include_system=True,
            session_id=session_id,
        )
        history_str = _clean(history_str)

        # EARLY EXIT IF BOTH EMPTY
        if not summary and not history_str:
            logger.debug(
                event="memory_fusion_empty",
                session_id=session_id,
            )
            return ""

        # DEDUP OVERLAP
        summary, history_str = _dedup_summary_history(summary, history_str)

        # BUDGET ALLOCATION
        summary_budget = int(max_total_chars * 0.3)
        history_budget = int(max_total_chars * 0.6)

        parts = []

        # HEADER
        parts.append(
            "[MEMORY CONTEXT]\n"
            "Use only if relevant.\n"
            "Prefer recent interactions."
        )

        # LONG-TERM SUMMARY
        if summary:
            parts.append(
                "[Long-Term]\n" +
                _truncate(summary, summary_budget)
            )

        # RECENT HISTORY
        if history_str:
            parts.append(
                "[Recent]\n" +
                _truncate(history_str, history_budget)
            )

        # INSTRUCTION
        parts.append(
            "[Instruction]\n"
            "- Use only if relevant\n"
            "- Do not repeat\n"
            "- Prefer recent\n"
        )

        result = "\n\n".join(parts).strip()

        # SAFE TRUNCATION
        if len(result) > settings.MAX_PROMPT_CHARS:
            header      = parts[0]
            instruction = parts[-1]
            middle      = "\n\n".join(parts[1:-1])

            allowed = (
                settings.MAX_PROMPT_CHARS
                - len(header)
                - len(instruction)
                - 20
            )

            middle  = _truncate(middle, max(allowed, 0))
            result  = "\n\n".join([header, middle, instruction])

            logger.warning(
                event="memory_context_truncated",
                original_size=len("\n\n".join(parts)),
                truncated_size=len(result),
                session_id=session_id,
            )

        logger.debug(
            event="memory_fusion_success",
            size=len(result),
            has_summary=bool(summary),
            has_history=bool(history_str),
            latency=round(time.time() - start, 3),
            session_id=session_id,
        )

        return result

    except Exception as e:
        logger.error(
            event="memory_fusion_failed",
            error=str(e),
            session_id=session_id,
        )
        return ""