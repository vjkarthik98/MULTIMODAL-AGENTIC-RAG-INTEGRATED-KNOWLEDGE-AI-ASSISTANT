from typing import List, Dict
import time
import hashlib

from app.core.config import settings
from app.memory.formatter import format_history
from app.utils.logger import get_logger

logger = get_logger(__name__)


#  CLEAN 
def _clean(text: str) -> str:
    return " ".join(str(text or "").strip().split())


#  TRUNCATE 
def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    return text[:limit]


#  HASH 
def _hash(text: str) -> str:
    return hashlib.sha256(text[:200].encode()).hexdigest()


#  DEDUP 
def _dedup(summary: str, history: str):

    if not summary or not history:
        return summary, history

    if _hash(summary) == _hash(history[:200]):
        history = history.replace(summary[:200], "")

    return summary, history


#  MAIN 
def build_memory_context(
    summary: str,
    filtered_history: List[Dict],
    max_total_chars: int = None,
    session_id: str = "default"
) -> str:

    start = time.time()

    max_total_chars = max_total_chars or settings.MEMORY_MAX_CONTEXT_CHARS

    try:
        parts = []

        #  HEADER 
        parts.append(
            "[MEMORY CONTEXT]\n"
            "Use only if relevant.\n"
            "Prefer recent interactions."
        )

        #  INPUT 
        summary = _clean(summary)

        history_str = format_history(
            filtered_history,
            max_messages=settings.MAX_HISTORY_MESSAGES,
            include_system=True,
            session_id=session_id
        )

        history_str = _clean(history_str)

        #  DEDUP 
        summary, history_str = _dedup(summary, history_str)

        #  BUDGET 
        summary_budget = int(max_total_chars * 0.3)
        history_budget = int(max_total_chars * 0.6)

        #  SUMMARY 
        if summary:
            parts.append(
                "[Long-Term]\n" +
                _truncate(summary, summary_budget)
            )

        #  HISTORY 
        if history_str:
            parts.append(
                "[Recent]\n" +
                _truncate(history_str, history_budget)
            )

        #  INSTRUCTION 
        parts.append(
            "[Instruction]\n"
            "- Use only if relevant\n"
            "- Do not repeat\n"
            "- Prefer recent\n"
        )

        result = "\n\n".join(parts).strip()

        #  SAFE TRUNCATION 
        if len(result) > settings.MAX_PROMPT_CHARS:

            header = parts[0]
            instruction = parts[-1]

            middle = "\n\n".join(parts[1:-1])

            allowed = (
                settings.MAX_PROMPT_CHARS
                - len(header)
                - len(instruction)
                - 20
            )

            middle = _truncate(middle, allowed)

            result = "\n\n".join([header, middle, instruction])

            logger.warning(event="memory_truncated")

        logger.debug(
            event="memory_fusion_success",
            size=len(result),
            latency=round(time.time() - start, 3)
        )

        return result

    except Exception as e:
        logger.error(event="memory_fusion_failed", error=str(e))
        return ""