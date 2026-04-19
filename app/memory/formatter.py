from typing import List, Dict
import time

def _truncate_text(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def _format_message(msg: Dict) -> str:
    role = msg.get("role", "unknown").upper()
    content = msg.get("content", "").strip()

    if not content:
        return ""
    
    # Optional Metadata
    modality = msg.get("modality")
    timestamp = msg.get("timestamp")

    meta = []

    if modality:
        meta.append(modality)

    if timestamp:
        try:
            ts = int(timestamp)
            meta.append(f"t={ts}")
        except Exception:
            pass
    
    meta_str = f"[{role}]"
    if meta:
        meta_str += " | " + " | ".join(meta)
    meta_str += "]"

    content = _truncate_text(content)

    return f"{meta_str}: {content}"


def format_history(
    history: List[Dict],
    max_messages: int = 10,
    include_system: bool = True
) -> str:
    
    if not history:
        return ""
    
    # STEP 1: SEPARATE SYSTEM VS NORMAL MESSAGES
    system_msgs = []
    normal_msgs = []

    for msg in history:
        if msg.get("role") == "system":
            system_msgs.append(msg)
        
        else:
            normal_msgs.append(msg)

    # STEP 2: TAKE MOST RECENT MESSAGES
    normal_msgs = normal_msgs[-max_messages:]

    # STEP 3: FORMAT MESSAGES
    formatted_parts = []

    formatted_parts.append("[Conversation Memory]")

    # system memory (summary)
    if include_system and system_msgs:
        formatted_parts.append("\n[System Summary]")
        for msg in system_msgs[-2:]: 
            fm = _format_message(msg)
            if fm:
                formatted_parts.append(fm)

    # STEP 4: JOIN
    return "\n".join(formatted_parts).strip()