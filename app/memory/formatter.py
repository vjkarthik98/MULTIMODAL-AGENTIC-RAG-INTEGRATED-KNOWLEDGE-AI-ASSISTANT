from typing import List, Dict
import time

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


#  CLEAN TEXT 
def _clean_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


#  SAFE TRUNCATION 
def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


#  FORMAT MESSAGE 
def _format_message(msg: Dict) -> str:
    try:
        role = str(msg.get("role", "user")).lower()

        # NORMALIZE ROLE
        if role not in {"user", "assistant", "system"}:
            role = "user"

        content = _clean_text(msg.get("content", ""))

        if len(content) < 3:
            return ""

        modality = msg.get("modality")
        timestamp = msg.get("timestamp")

        meta = f"[{role.upper()}]"

        if modality:
            meta += f" [{modality}]"

        if timestamp:
            try:
                ts = int(timestamp)
                meta += f" [t={ts}]"
            except Exception:
                pass

        content = _truncate_text(content, settings.MAX_PROMPT_CHARS // 4)

        return f"{meta}: {content}"

    except Exception:
        return ""


#  DEDUP 
def _deduplicate(messages: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []

    for m in messages:
        key = (m.get("role"), str(m.get("content"))[:200])

        if key not in seen:
            seen.add(key)
            unique.append(m)

    return unique


#  MAIN FORMATTER 
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

    # DEDUP FIRST
    history = _deduplicate(history)

    system_msgs = []
    normal_msgs = []

    for msg in history:
        if not isinstance(msg, dict):
            continue

        if msg.get("role") == "system":
            system_msgs.append(msg)
        else:
            normal_msgs.append(msg)

    # TAKE MOST RECENT
    normal_msgs = normal_msgs[-max_messages:]

    parts = ["[Conversation Memory]"]

    #  SYSTEM 
    if include_system and system_msgs:
        parts.append("\n[System Summary]")

        for msg in system_msgs[-settings.MAX_SYSTEM_MESSAGES:]:
            fm = _format_message(msg)
            if fm:
                parts.append(fm)

    #  CONVERSATION 
    if normal_msgs:
        parts.append("\n[Recent Conversation]")

        for msg in normal_msgs:
            fm = _format_message(msg)
            if fm:
                parts.append(fm)

    result = "\n".join(parts).strip()

    #  SMART TRUNCATION 
    if len(result) > settings.MAX_PROMPT_CHARS:
        logger.warning("[Formatter] truncating history safely")

        # KEEP SYSTEM + LAST PART
        split_point = int(settings.MAX_PROMPT_CHARS * 0.7)

        result = result[:split_point] + "\n...\n" + result[-(settings.MAX_PROMPT_CHARS - split_point):]

    latency = round(time.time() - start, 2)

    logger.debug(
        "[Formatter][SUCCESS] session_id=%s | messages=%s | latency=%ss",
        session_id,
        len(history),
        latency
    )

    return result