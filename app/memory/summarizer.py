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


# HASH

def _hash(msg: Dict) -> str:
    base = f"{msg.get('role')}|{str(msg.get('content'))[:200]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# DEDUP

def _dedup(history: List[Dict]) -> List[Dict]:
    seen: set       = set()
    out:  List[Dict] = []

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

def _sort_by_importance(history: List[Dict]) -> List[Dict]:
    def _imp(m: Dict) -> float:
        try:
            return float(m.get("importance", 0.5))
        except Exception:
            return 0.5

    return sorted(history, key=_imp, reverse=True)


# FORMAT CONVERSATION

def _format(history: List[Dict]) -> str:
    max_chars = settings.MEMORY_SUMMARY_INPUT_CHARS
    max_msgs  = settings.MAX_HISTORY_MESSAGES

    history = _dedup(history[-max_msgs:])
    history = _sort_by_importance(history)

    parts: List[str] = []
    total: int       = 0

    for msg in history:
        try:
            role     = str(msg.get("role", "user")).upper()
            content  = _clean(msg.get("content", ""))
            modality = msg.get("modality", "text")

            if len(content) < 5:
                continue

            content = content[:settings.MAX_PROMPT_CHARS]

            mod_tag = f"[{modality.upper()}] " if modality != "text" else ""
            line    = f"{role}: {mod_tag}{content}"
            length  = len(line)

            if total + length > max_chars:
                break

            parts.append(line)
            total += length

        except Exception:
            continue

    return "\n".join(parts)


# VALIDATE

def _validate(summary: str) -> str:
    if not summary:
        return ""

    summary = _clean(summary)

    if len(summary) < settings.MIN_SUMMARY_LENGTH:
        return ""

    summary = summary[:settings.MEMORY_SUMMARY_MAX_CHARS]

    required = ["Key Facts", "User Intent"]

    if any(r in summary for r in required):
        return summary

    # FALLBACK: accept any sufficiently long summary even without required headers
    if len(summary) >= settings.MIN_SUMMARY_LENGTH * 2:
        logger.warning(event="summary_missing_headers_accepted_as_fallback")
        return summary

    return ""


# PROMPT

def _prompt(conv: str) -> str:
    instruction = (
        "Compress conversation memory.\n"
        "- Keep only important info\n"
        "- No hallucination\n"
        "- No filler\n\n"
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
    body      = f"Conversation:\n{conv[:max(available, 0)]}\n\n"

    return instruction + body + format_block


# PERSIST SUMMARY

def _persist_summary(
    summary: str,
    session_id: str,
    mongo_memory,
) -> None:
    try:
        if mongo_memory and hasattr(mongo_memory, "store_summary"):
            mongo_memory.store_summary(session_id, summary)
    except Exception as e:
        logger.warning(
            event="summary_persist_failed",
            error=str(e),
            session_id=session_id,
        )


# MAIN

def summarize_conversation(
    llm,
    history: List[Dict],
    session_id: str = "default",
    mongo_memory=None,
) -> str:

    if not history:
        return ""

    start = time.time()

    try:
        conv = _format(history)

        if not conv:
            logger.warning(
                event="summary_empty_conversation",
                session_id=session_id,
            )
            return ""

        prompt = _prompt(conv)

        # LLM GENERATE WITH TIMEOUT GUARD
        t_llm = time.time()
        raw   = llm.generate(prompt)

        if time.time() - t_llm > settings.MODEL_TIMEOUT_SEC:
            logger.warning(
                event="summary_llm_timeout",
                session_id=session_id,
            )
            return ""

        summary = _validate(raw)

        if not summary:
            logger.warning(
                event="summary_invalid",
                raw_length=len(raw) if raw else 0,
                session_id=session_id,
            )
            return ""

        # AUTO-PERSIST TO MONGO
        if mongo_memory:
            _persist_summary(summary, session_id, mongo_memory)

        latency = round(time.time() - start, 2)

        logger.info(
            event="summary_success",
            length=len(summary),
            input_messages=len(history),
            conv_chars=len(conv),
            latency=latency,
            session_id=session_id,
        )

        return summary

    except Exception as e:
        logger.error(
            event="summary_failed",
            error=str(e),
            session_id=session_id,
        )
        return ""