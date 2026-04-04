from typing import List, Dict

def format_history(history: List[Dict]) -> str:
    """
    Convert chat history into readable conversation format.
    
    Args:
        history:[
                {"role: "user", "content": "..."},
                {"role: "assistant", "content": "..."}
            ]
    Returns:
        str: formatted conversation
    """

    if not history:
        return ""
    
    formatted = "Conversation History:\n"

    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            formatted += f"User: {content}\n"
        elif role == "assistant":
            formatted += f"Asssitant: {content}\n"

    return formatted.strip()
