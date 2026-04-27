from typing import List, Dict
import time

from app.core.config import settings
from app.memory.formatter import format_history
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""

    text = text.strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


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
        total_len = 0

        # HEADER
        parts.append("[MEMORY CONTEXT]")
        parts.append(
            "Use past interactions only when relevant. "
            "Prioritize recent context over older summaries."
        )

        # SUMMARY 
        if summary and isinstance(summary, str):
            summary_limit = settings.MEMORY_SUMMARY_MAX_CHARS

            summary_block = "[Long-Term Memory]\n" + _truncate(
                summary,
                summary_limit
            )

            if len(summary_block) <= max_total_chars:
                parts.append(summary_block)
                total_len += len(summary_block)

        # HISTORY 
        if filtered_history and isinstance(filtered_history, list):

            formatted_history = format_history(
                filtered_history,
                max_messages=settings.MAX_HISTORY_MESSAGES,
                include_system=True,
                session_id=session_id
            )

            history_limit = settings.MEMORY_HISTORY_MAX_CHARS

            history_block = "[Relevant Recent Context]\n" + _truncate(
                formatted_history,
                history_limit
            )

            remaining = max_total_chars - total_len

            if remaining > 0:
                if len(history_block) <= remaining:
                    parts.append(history_block)
                else:
                    # Trim safely
                    trimmed = history_block[:remaining]
                    parts.append(trimmed + "...")

        # INSTRUCTION 
        parts.append(
            "[Instruction]\n"
            "- Use memory only if relevant.\n"
            "- Do not repeat memory verbatim.\n"
            "- Prefer recent context.\n"
            "- Respect user preferences.\n"
        )

        result = "\n\n".join(parts).strip()

        # Final global safety
        if len(result) > settings.MAX_PROMPT_CHARS:
            logger.warning("[MemoryFusion] truncating final context")
            result = result[-settings.MAX_PROMPT_CHARS:]

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