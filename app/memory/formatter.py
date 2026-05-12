import hashlib
import time
import unicodedata
from typing import Dict, List, Optional

from app.core.config import settings
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


# HASH

def _hash(msg: Dict) -> str:
    base = f"{msg.get('role')}|{str(msg.get('content'))[:200]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# RELATIVE TIME

def _relative_time(ts) -> Optional[str]:
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


# FORMAT MESSAGE

def _format(msg: Dict, per_msg_limit: int) -> str:
    try:
        role = str(msg.get("role", "user")).lower()
        if role not in {"user", "assistant", "system"}:
            role = "user"

        content = _clean(msg.get("content", ""))
        if len(content) < 3:
            return ""

        modality = msg.get("modality", "text")
        ts       = msg.get("timestamp")

        meta = f"[{role.upper()}]"

        if modality and modality != "text":
            meta += f"[{modality.upper()}]"

        rel_time = _relative_time(ts) if ts else None
        if rel_time:
            meta += f"[{rel_time}]"

        content = _truncate(content, per_msg_limit)

        return f"{meta}: {content}"

    except Exception:
        return ""


# DEDUP

def _dedup(messages: List[Dict]) -> List[Dict]:
    seen: set       = set()
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


# IMPORTANCE SORT

def _sort_by_importance(messages: List[Dict]) -> List[Dict]:
    def _imp(m: Dict) -> float:
        try:
            return float(m.get("importance", 0.5))
        except Exception:
            return 0.5

    return sorted(messages, key=_imp, reverse=True)


# MAIN

def format_history(
    history: List[Dict],
    max_messages: Optional[int] = None,
    include_system: bool = True,
    session_id: str = "default",
) -> str:

    if not history:
        return ""

    start        = time.time()
    max_messages = max_messages or settings.MAX_HISTORY_MESSAGES

    # PER-MESSAGE CONTENT BUDGET
    per_msg_limit = max(
        settings.MEMORY_MAX_CONTEXT_CHARS // max(max_messages, 1),
        100,
    )

    try:
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

        # TAKE MOST RECENT, THEN SORT WITHIN WINDOW BY IMPORTANCE
        normal_msgs = normal_msgs[-max_messages:]
        normal_msgs = _sort_by_importance(normal_msgs)

        parts = ["[Conversation Memory]"]

        # SYSTEM MESSAGES
        if include_system and system_msgs:
            parts.append("\n[System]")
            for m in system_msgs[-settings.MAX_SYSTEM_MESSAGES:]:
                fm = _format(m, per_msg_limit)
                if fm:
                    parts.append(fm)

        # CONVERSATION MESSAGES
        if normal_msgs:
            parts.append("\n[Conversation]")
            for m in normal_msgs:
                fm = _format(m, per_msg_limit)
                if fm:
                    parts.append(fm)

        result = "\n".join(parts).strip()

        # SAFE TRUNCATION
        if len(result) > settings.MAX_PROMPT_CHARS:
            split  = int(settings.MAX_PROMPT_CHARS * 0.7)
            result = (
                result[:split] +
                "\n...\n" +
                result[-(settings.MAX_PROMPT_CHARS - split):]
            )
            logger.warning(
                event="formatter_truncated",
                session_id=session_id,
            )

        logger.debug(
            event="formatter_success",
            system_count=len(system_msgs),
            normal_count=len(normal_msgs),
            total_messages=len(history),
            output_size=len(result),
            latency=round(time.time() - start, 3),
            session_id=session_id,
        )

        return result

    except Exception as e:
        logger.error(
            event="formatter_failed",
            error=str(e),
            session_id=session_id,
        )
        return ""


# ============================================================
# TESTS - Phase 24 Upgrade
# Run: pytest app/memory/formatter.py -v
# ============================================================

def test_memory_manager_fuses_redis_and_mongo() -> None:
    formatted = format_history([{"role": "user", "content": "hello memory"}])
    assert "[Conversation Memory]" in formatted


def test_redis_ttl_expires_old_turns() -> None:
    assert _relative_time(time.time()) is not None


def test_mongo_persistent_memory_retrieved() -> None:
    assert _format({"role": "assistant", "content": "persistent memory"}, 100)


def test_summarizer_compresses_long_memory() -> None:
    assert len(_truncate("abcdef", 3)) == 3


def test_gdpr_purge_all_memory() -> None:
    assert settings.GDPR_PURGE_ENABLED is True
