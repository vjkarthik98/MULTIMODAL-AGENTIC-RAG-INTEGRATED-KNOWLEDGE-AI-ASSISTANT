from typing import List, Dict
from app.memory.formatter import format_history

def _truncate(text: str, max_chars: int = 2000) -> str:
    if not text:
        return ""
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def build_memory_context(
    summary: str,
    filtered_history: List[Dict],
    max_total_chars: int = 4000
) -> str:

    parts = []
    total_len = 0

    # HEADER (important for LLM grounding)

    parts.append("[MEMORY CONTEXT]")
    parts.append(
        "This memory contains relevant past interactions. "
        "Use it to improve answer accuracy, maintain continuity, "
        "and respect user preferences.\n"
    )

    # LONG TERM MEMORY (SUMMARY)
    if summary and summary.strip():
        summary_block = "[Long-Term Memory]\n" + _truncate(summary.strip(), 1500)

        parts.append(summary_block)
        total_len += len(summary_block)
        

    # RECENT HIGH-RELEVANCE HISTORY
    if filtered_history:
        formatted_history = format_history(filtered_history)

        history_block = "[Relevant Recent Context]\n" + _truncate(
            formatted_history, 2000
        )

        if total_len + len(history_block) <= max_total_chars:
            parts.append(history_block)

        else:
            # Trim aggressively if exceeding
            remaining = max_total_chars - total_len
            if remaining > 200:
                parts.append(history_block[:remaining] + "...")

    # FINAL INSTRUCTION
    parts.append(
        "\n[Instruction]\n"
        "- Use memory only if relevant to the current query.\n"
        "- Do NOT repeat memory verbatim.\n"
        "- Prioritize recent context over older summaries.\n"
        "- Respect user preferences and past decisions.\n"
    )
    
    return "\n\n".join(parts).strip()