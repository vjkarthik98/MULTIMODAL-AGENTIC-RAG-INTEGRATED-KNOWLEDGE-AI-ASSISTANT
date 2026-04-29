from typing import List, Dict
import time

from app.core.config import settings
from app.memory.formatter import format_history
from app.utils.logger import get_logger


logger = get_logger(__name__)


#  CLEAN TEXT 
def _clean(text: str) -> str:
    return " ".join(str(text or "").strip().split())


#  SAFE TRUNCATION 
def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    return text[:max_chars].rstrip() + ("..." if len(text) > max_chars else "")


#  DEDUP 
def _deduplicate(summary: str, history: str):
    if not summary or not history:
        return summary, history

    if summary[:200] in history:
        history = history.replace(summary[:200], "")

    return summary, history


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
        header = (
            "[MEMORY CONTEXT]\n"
            "Use memory only when relevant.\n"
            "Prefer recent context over older summaries."
        )

        parts.append(header)

        #  CLEAN INPUT 
        summary = _clean(summary)
        history_str = format_history(
            filtered_history,
            max_messages=settings.MAX_HISTORY_MESSAGES,
            include_system=True,
            session_id=session_id
        )

        history_str = _clean(history_str)

        #  DEDUP 
        summary, history_str = _deduplicate(summary, history_str)

        #  SMART BUDGET 
        summary_budget = int(max_total_chars * 0.3)
        history_budget = int(max_total_chars * 0.6)

        #  SUMMARY 
        if summary:
            summary_block = "[Long-Term Memory]\n" + _truncate(
                summary,
                summary_budget
            )
            parts.append(summary_block)

        #  HISTORY 
        if history_str:
            history_block = "[Recent Context]\n" + _truncate(
                history_str,
                history_budget
            )
            parts.append(history_block)

        #  INSTRUCTION 
        instruction = (
            "[Instruction]\n"
            "- Use memory only if relevant\n"
            "- Do not repeat verbatim\n"
            "- Prefer recent context\n"
            "- Respect user intent\n"
        )

        parts.append(instruction)

        result = "\n\n".join(parts).strip()

        #  SAFE GLOBAL TRUNCATION 
        if len(result) > settings.MAX_PROMPT_CHARS:

            logger.warning("[MemoryFusion] safe truncation")

            # KEEP HEADER + INSTRUCTION ALWAYS
            header_part = parts[0]
            instruction_part = parts[-1]

            middle = "\n\n".join(parts[1:-1])

            allowed_middle = (
                settings.MAX_PROMPT_CHARS
                - len(header_part)
                - len(instruction_part)
                - 20
            )

            middle = _truncate(middle, allowed_middle)

            result = "\n\n".join([
                header_part,
                middle,
                instruction_part
            ])

        latency = round(time.time() - start, 2)

        logger.debug(
            "[MemoryFusion][SUCCESS] session_id=%s | size=%s | latency=%ss",
            session_id,
            len(result),
            latency
        )

        return result

    except Exception as e:
        logger.error("[MemoryFusion][FAILED] %s", str(e))
        return ""