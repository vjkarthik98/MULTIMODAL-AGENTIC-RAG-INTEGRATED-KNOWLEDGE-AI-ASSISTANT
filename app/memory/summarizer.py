import logging

#Logger
logger = logging.getLogger(__name__)


def summarize_conversation(llm, history: list[dict]) -> str:
    """
    Summarize conversation history into a compact form.
    """

    try:
        logger.debug("[Summarizer] Starting summarization")

        # Convert history into text
        conversation_text = ""
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            conversation_text += f"{role.upper()}: {content}\n"

        prompt = f"""
Summarize the following conversation into concise key points.

Conversation:
{conversation_text}

Summary:
"""

        summary = llm.generate(prompt)

        logger.debug("[Summarizer] Summarization completed")

        return summary.strip()

    except Exception as e:
        logger.error(f"[Summarizer] Failed | error={str(e)}")
        raise