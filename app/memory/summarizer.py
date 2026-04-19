from app.utils.logger import get_logger
from typing import List, Dict
import time

#Logger
logger = get_logger(__name__)

# HELPERS
def _format_conversation(history: List[Dict], max_chars: int = 4000) -> str:
    parts = []
    total_length = 0

    # Reverse -> keep recent messages first
    for msg in reversed(history):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "").strip()

        if not content:
            continue

        line = f"{role}: {content}"
        line_len = len(line)

        if total_length + line_len > max_chars:
            break

        parts.append(line)
        total_length += line_len

    # restore order
    return "\n".join(reversed(parts))


# MAIN SUMMARIZER
def summarize_conversation(llm, history: list[dict]) -> str:
    if not history:
        return ""
 
    try:
        start_time = time.time()

        logger.info("[Summarizer][START]")

        # STEP 1: FORMAT CONVERSATION 
        conversation_text = _format_conversation(history)

        # STEP 2: STRUCTURED PROMPT
        prompt = f"""
You are a memory compression system for an AI assistant.

Your job is to extract ONLY important information and compress it into structured memory.


Ignore:
- small talk
- greetings
- redundant phrases

Focus on:
- user goals
- facts
- preferences
- tasks
- decisions

-------------------------
Conversation:
{conversation_text}
-------------------------

Return STRICTLY in this format:

Key Facts:
- ...

User Intent:
- ...

Preferences:
- ...

Tasks / Actions:
- ...

Context:
- ...
"""
        
        # STEP 3: GENERATE SUMMARY
        summary = llm.generate(prompt)

        if not summary or len(summary.strip()) < 10:
            logger.warning("[Summarizer] Weak summary generated")
            return ""
        latency = time.time() - start_time

        logger.info(
            f"[Summarizer][SUCCESS] length={len(summary)} latency={latency: .2f}s"
        )
        return summary.strip()
    
    except Exception as e:
        logger.error(f"[Summarizer][FAIL] {str(e)}")
        raise
    