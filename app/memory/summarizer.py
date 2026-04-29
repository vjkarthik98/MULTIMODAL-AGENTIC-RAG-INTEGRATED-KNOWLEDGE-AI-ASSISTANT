import time
from typing import List, Dict

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


#  CLEAN TEXT 
def _clean(text: str) -> str:
    return " ".join(str(text or "").strip().split())


#  DEDUP 
def _deduplicate(history: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []

    for msg in history:
        key = (msg.get("role"), str(msg.get("content"))[:200])
        if key not in seen:
            seen.add(key)
            unique.append(msg)

    return unique


#  FORMAT CONVERSATION 
def _format_conversation(history: List[Dict]) -> str:

    max_chars = settings.MEMORY_SUMMARY_INPUT_CHARS
    max_messages = settings.MAX_HISTORY_MESSAGES

    history = _deduplicate(history[-max_messages:])

    parts = []
    total_length = 0

    for msg in reversed(history):

        try:
            role = str(msg.get("role", "user")).upper()
            content = _clean(msg.get("content", ""))

            if len(content) < 5:
                continue

            content = content[:settings.MAX_PROMPT_CHARS]

            line = f"{role}: {content}"
            line_len = len(line)

            if total_length + line_len > max_chars:
                break

            parts.append(line)
            total_length += line_len

        except Exception:
            continue

    return "\n".join(reversed(parts))


#  VALIDATE SUMMARY 
def _validate_summary(text: str) -> str:

    if not text:
        return ""

    text = _clean(text)

    if len(text) < settings.MIN_SUMMARY_LENGTH:
        return ""

    if len(text) > settings.MEMORY_SUMMARY_MAX_CHARS:
        text = text[:settings.MEMORY_SUMMARY_MAX_CHARS]

    # BASIC STRUCTURE CHECK
    required_sections = ["Key Facts", "User Intent"]

    if not any(section in text for section in required_sections):
        return ""

    return text


#  SAFE PROMPT BUILDER 
def _build_prompt(conversation_text: str) -> str:

    instruction = (
        "You are a memory compression system for an AI assistant.\n\n"
        "Extract ONLY important information.\n\n"
        "Ignore:\n"
        "- greetings\n"
        "- small talk\n"
        "- repetition\n\n"
        "Focus on:\n"
        "- goals\n"
        "- facts\n"
        "- preferences\n"
        "- tasks\n"
        "- decisions\n\n"
    )

    body = f"Conversation:\n{conversation_text}\n\n"

    format_block = (
        "Return format:\n"
        "Key Facts:\n- ...\n\n"
        "User Intent:\n- ...\n\n"
        "Preferences:\n- ...\n\n"
        "Tasks:\n- ...\n\n"
        "Context:\n- ..."
    )

    # SAFE COMPOSITION
    max_chars = settings.MAX_PROMPT_CHARS

    available = max_chars - len(instruction) - len(format_block) - 50

    body = body[:available]

    return instruction + body + format_block


#  MAIN 
def summarize_conversation(llm, history: List[Dict]) -> str:

    if not history:
        return ""

    start_time = time.time()

    try:
        logger.info("[Summarizer][START]")

        # STEP 1: FORMAT
        t1 = time.time()
        conversation_text = _format_conversation(history)

        if not conversation_text:
            return ""

        # STEP 2: PROMPT
        prompt = _build_prompt(conversation_text)

        # STEP 3: GENERATE
        t2 = time.time()
        summary = llm.generate(prompt)

        # STEP 4: VALIDATE
        summary = _validate_summary(summary)

        if not summary:
            logger.warning("[Summarizer] invalid summary")
            return ""

        latency = round(time.time() - start_time, 2)

        logger.info(
            "[Summarizer][SUCCESS] len=%s latency=%ss format=%.2fs llm=%.2fs",
            len(summary),
            latency,
            t2 - t1,
            time.time() - t2
        )

        return summary

    except Exception as e:
        logger.error("[Summarizer][FAILED] %s", str(e))
        return ""