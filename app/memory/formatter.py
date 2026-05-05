from typing import List, Dict
import time
import hashlib

from app.core.config import settings
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
def _hash(msg: Dict) -> str:
    base = f"{msg.get('role')}|{str(msg.get('content'))[:200]}"
    return hashlib.sha256(base.encode()).hexdigest()


#  FORMAT MESSAGE 
def _format(msg: Dict) -> str:

    try:
        role = str(msg.get("role", "user")).lower()
        if role not in {"user", "assistant", "system"}:
            role = "user"

        content = _clean(msg.get("content", ""))
        if len(content) < 3:
            return ""

        modality = msg.get("modality")
        ts = msg.get("timestamp")

        meta = f"[{role.upper()}]"

        if modality:
            meta += f"[{modality}]"

        if ts:
            try:
                meta += f"[t={int(ts)}]"
            except Exception:
                pass

        content = _truncate(content, settings.MAX_PROMPT_CHARS // 4)

        return f"{meta}: {content}"

    except Exception:
        return ""


#  DEDUP 
def _dedup(messages: List[Dict]) -> List[Dict]:

    seen = set()
    out = []

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


#  MAIN 
def format_history(
    history: List[Dict],
    max_messages: int = None,
    include_system: bool = True,
    session_id: str = "default"
) -> str:

    if not history:
        return ""

    start = time.time()

    try:
        max_messages = max_messages or settings.MAX_HISTORY_MESSAGES

        history = _dedup(history)

        system_msgs = []
        normal_msgs = []

        for msg in history:
            if not isinstance(msg, dict):
                continue

            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                normal_msgs.append(msg)

        normal_msgs = normal_msgs[-max_messages:]

        parts = ["[Conversation Memory]"]

        #  SYSTEM 
        if include_system and system_msgs:
            parts.append("\n[System]")

            for m in system_msgs[-settings.MAX_SYSTEM_MESSAGES:]:
                fm = _format(m)
                if fm:
                    parts.append(fm)

        #  CONVERSATION 
        if normal_msgs:
            parts.append("\n[Conversation]")

            for m in normal_msgs:
                fm = _format(m)
                if fm:
                    parts.append(fm)

        result = "\n".join(parts).strip()

        #  SAFE TRUNCATION 
        if len(result) > settings.MAX_PROMPT_CHARS:

            split = int(settings.MAX_PROMPT_CHARS * 0.7)

            result = (
                result[:split] +
                "\n...\n" +
                result[-(settings.MAX_PROMPT_CHARS - split):]
            )

            logger.warning(event="formatter_truncated")

        logger.debug(
            event="formatter_success",
            messages=len(history),
            latency=round(time.time() - start, 3)
        )

        return result

    except Exception as e:
        logger.error(event="formatter_failed", error=str(e))
        return ""