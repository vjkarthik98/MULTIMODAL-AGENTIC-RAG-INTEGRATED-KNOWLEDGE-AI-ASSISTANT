import time
import hashlib
from typing import List, Dict

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


#  CLEAN 
def _clean(text: str) -> str:
    return " ".join(str(text or "").strip().split())


#  HASH 
def _hash(msg: Dict) -> str:
    base = f"{msg.get('role')}|{str(msg.get('content'))[:200]}"
    return hashlib.sha256(base.encode()).hexdigest()


#  DEDUP 
def _dedup(history: List[Dict]) -> List[Dict]:

    seen = set()
    out = []

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


#  FORMAT 
def _format(history: List[Dict]) -> str:

    max_chars = settings.MEMORY_SUMMARY_INPUT_CHARS
    max_msgs = settings.MAX_HISTORY_MESSAGES

    history = _dedup(history[-max_msgs:])

    parts = []
    total = 0

    for msg in reversed(history):

        try:
            role = str(msg.get("role", "user")).upper()
            content = _clean(msg.get("content", ""))

            if len(content) < 5:
                continue

            content = content[:settings.MAX_PROMPT_CHARS]

            line = f"{role}: {content}"
            length = len(line)

            if total + length > max_chars:
                break

            parts.append(line)
            total += length

        except Exception:
            continue

    return "\n".join(reversed(parts))


#  VALIDATE 
def _validate(summary: str) -> str:

    if not summary:
        return ""

    summary = _clean(summary)

    if len(summary) < settings.MIN_SUMMARY_LENGTH:
        return ""

    summary = summary[:settings.MEMORY_SUMMARY_MAX_CHARS]

    required = ["Key Facts", "User Intent"]

    if not any(r in summary for r in required):
        return ""

    return summary


#  PROMPT 
def _prompt(conv: str) -> str:

    instruction = (
        "Compress conversation memory.\n"
        "- Keep only important info\n"
        "- No hallucination\n"
        "- No filler\n\n"
    )

    body = f"Conversation:\n{conv}\n\n"

    format_block = (
        "Key Facts:\n- ...\n\n"
        "User Intent:\n- ...\n\n"
        "Preferences:\n- ...\n\n"
        "Tasks:\n- ...\n\n"
        "Context:\n- ..."
    )

    max_chars = settings.MAX_PROMPT_CHARS
    available = max_chars - len(instruction) - len(format_block) - 50

    return instruction + body[:available] + format_block


#  MAIN 
def summarize_conversation(llm, history: List[Dict]) -> str:

    if not history:
        return ""

    start = time.time()

    try:
        conv = _format(history)

        if not conv:
            return ""

        prompt = _prompt(conv)

        summary = llm.generate(prompt)

        summary = _validate(summary)

        if not summary:
            logger.warning(event="summary_invalid")
            return ""

        logger.info(
            event="summary_success",
            length=len(summary),
            latency=round(time.time() - start, 2)
        )

        return summary

    except Exception as e:
        logger.error(event="summary_failed", error=str(e))
        return ""