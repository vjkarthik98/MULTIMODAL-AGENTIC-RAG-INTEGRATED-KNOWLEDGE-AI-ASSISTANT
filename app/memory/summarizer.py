def summarize_conversation(llm, history: list[dict]) -> str:
    """
    Summarize conversation history into a compact form.
    
    Args:
        llm: Your GGUF model instance
        history (list): Chat history [
            {"role": "user", "content": "..."}],
            {"role": "assistant", "content": "..."}
        ]
        
    Returns:
        str: summarized memory
    """

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
    return summary.strip()
