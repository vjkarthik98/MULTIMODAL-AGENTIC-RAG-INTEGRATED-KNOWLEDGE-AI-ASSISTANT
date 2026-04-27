from typing import List, Dict
import time

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _truncate_text(text: str) -> str:
    max_chars = settings.MAX_PROMPT_CHARS

    if not text:
        return ""

    text = text.strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def _format_message(msg: Dict) -> str:
    try:
        role = str(msg.get("role", "unknown")).upper()
        content = str(msg.get("content", "")).strip()

        if not content:
            return ""

        modality = msg.get("modality")
        timestamp = msg.get("timestamp")

        meta_parts = []

        if modality:
            meta_parts.append(str(modality))

        if timestamp:
            try:
                ts = int(timestamp)
                meta_parts.append(f"t={ts}")
            except Exception:
                pass

        meta = f"[{role}]"
        if meta_parts:
            meta += " | " + " | ".join(meta_parts)
        meta += "]"

        content = _truncate_text(content)

        return f"{meta}: {content}"

    except Exception:
        return ""


def format_history(
    history: List[Dict],
    max_messages: int = None,
    include_system: bool = True,
    session_id: str = "default"
) -> str:

    if not history:
        return ""

    start = time.time()

    max_messages = max_messages or settings.MAX_HISTORY_MESSAGES

    system_msgs = []
    normal_msgs = []

    # Split messages
    for msg in history:
        if not isinstance(msg, dict):
            continue

        if msg.get("role") == "system":
            system_msgs.append(msg)
        else:
            normal_msgs.append(msg)

    # Take most recent messages
    normal_msgs = normal_msgs[-max_messages:]

    formatted_parts = []
    formatted_parts.append("[Conversation Memory]")

    # System summary
    if include_system and system_msgs:
        formatted_parts.append("\n[System Summary]")

        for msg in system_msgs[-settings.MAX_SYSTEM_MESSAGES:]:
            fm = _format_message(msg)
            if fm:
                formatted_parts.append(fm)

    # Normal conversation
    if normal_msgs:
        formatted_parts.append("\n[Recent Conversation]")

        for msg in normal_msgs:
            fm = _format_message(msg)
            if fm:
                formatted_parts.append(fm)

    result = "\n".join(formatted_parts).strip()

    # Final safety truncation
    if len(result) > settings.MAX_PROMPT_CHARS:
        logger.warning("[Formatter] truncating history")
        result = result[-settings.MAX_PROMPT_CHARS:]

    latency = round(time.time() - start, 2)

    logger.debug(
        "[Formatter][SUCCESS] session_id=%s | messages=%s | latency=%ss",
        session_id,
        len(history),
        latency
    )

    return result