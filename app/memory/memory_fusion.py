from typing import List, Dict
from app.memory.formatter import format_history

def build_memory_context(
    summary: str,
    filtered_history: List[Dict]
) -> str:
    """
    Combine summarized memory and filtered history.
    
    Args:
        Summary: summarized long-term memory
        filtered_history: relevant recent messges
        
    Returns:
        str: final memory context for LLM
    """

    parts = []

    # Step 1: Add summary (long-term memory)
    if summary and summary.strip():
        parts.append("Conversation Summary:\n" + summary.strip())

    # Step 2: Add filtered recent history
    if filtered_history:
        formatted_history = format_history(filtered_history)
        parts.append(formatted_history)

    # Step 3: Combine
    if not parts:
        return ""
    
    return "\n\n".join(parts)