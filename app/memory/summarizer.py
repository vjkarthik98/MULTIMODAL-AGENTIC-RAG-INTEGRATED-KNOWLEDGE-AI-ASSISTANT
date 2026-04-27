import time
from typing import List, Dict

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


# FORMAT CONVERSATION 
def _format_conversation(history: List[Dict]) -> str:
    if not history:
        return ""

    max_chars = settings.MEMORY_SUMMARY_INPUT_CHARS
    max_messages = settings.MAX_HISTORY_MESSAGES

    parts = []
    total_length = 0

    # Use recent messages only
    history = history[-max_messages:]

    for msg in reversed(history):
        try:
            role = str(msg.get("role", "unknown")).upper()
            content = str(msg.get("content", "")).strip()

            if not content:
                continue

            # Truncate content per message
            if len(content) > settings.MAX_PROMPT_CHARS:
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


# VALIDATE SUMMARY 
def _validate_summary(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    if len(text) < settings.MIN_SUMMARY_LENGTH:
        return ""

    if len(text) > settings.MEMORY_SUMMARY_MAX_CHARS:
        text = text[:settings.MEMORY_SUMMARY_MAX_CHARS]

    return text


# MAIN 
def summarize_conversation(llm, history: List[Dict]) -> str:
    if not history:
        return ""

    start_time = time.time()

    try:
        logger.info("[Summarizer][START]")

        # STEP 1: FORMAT
        conversation_text = _format_conversation(history)

        if not conversation_text:
            return ""

        # STEP 2: PROMPT
        prompt = (
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
            "Conversation:\n"
            f"{conversation_text}\n\n"
            "Return format:\n"
            "Key Facts:\n- ...\n\n"
            "User Intent:\n- ...\n\n"
            "Preferences:\n- ...\n\n"
            "Tasks:\n- ...\n\n"
            "Context:\n- ..."
        )

        # Global safety
        if len(prompt) > settings.MAX_PROMPT_CHARS:
            logger.warning("[Summarizer] prompt truncated")
            prompt = prompt[-settings.MAX_PROMPT_CHARS:]

        # STEP 3: GENERATE
        summary = llm.generate(prompt)

        summary = _validate_summary(summary)

        if not summary:
            logger.warning("[Summarizer] weak or invalid summary")
            return ""

        latency = round(time.time() - start_time, 2)

        logger.info(
            "[Summarizer][SUCCESS] length=%s latency=%ss",
            len(summary),
            latency
        )

        return summary

    except Exception as e:
        logger.error("[Summarizer][FAILED] %s", str(e))
        return ""