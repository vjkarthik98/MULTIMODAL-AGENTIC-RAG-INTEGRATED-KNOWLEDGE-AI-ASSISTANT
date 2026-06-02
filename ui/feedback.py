"""
ui/feedback.py — Persist UI thumbs-up/down to the Phase-25 eval gold set.

Each entry is appended to app/eval/datasets/gold/feedback_gold.jsonl.
Schema matches the existing gold set format so the eval runner can pick
it up without changes.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path relative to the project root (one level up from ui/)
_GOLD_DIR = Path(__file__).parent.parent / "app" / "eval" / "datasets" / "gold"
_FEEDBACK_FILE = _GOLD_DIR / "feedback_gold.jsonl"


def save_feedback(
    query: str,
    answer: str,
    rating: str,                         # "positive" | "negative"
    route: str = "rag",
    sources: Optional[List[Any]] = None,
    session_id: str = "",
    confidence: float = 0.0,
) -> None:
    """
    Append one feedback entry to feedback_gold.jsonl.
    Safe to call concurrently — each call opens/appends/closes atomically.
    """
    _GOLD_DIR.mkdir(parents=True, exist_ok=True)

    source_names = _extract_source_names(sources or [])

    entry: Dict[str, Any] = {
        "id":                f"fb-{uuid.uuid4().hex[:8]}",
        "modality":          "ui_feedback",
        "source_file":       source_names[0] if source_names else "unknown",
        "query":             query,
        "relevant_chunk_ids": [],
        "reference_answer":  answer if rating == "positive" else "",
        "expected_route":    route,
        "tags":              [f"feedback_{rating}", "ui_feedback"],
        "added_by":          f"ui_feedback_{rating}",
        "added_at":          datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "feedback":          rating,
        "confidence":        confidence,
        "session_id":        session_id,
        "sources":           source_names[:5],
    }

    with _FEEDBACK_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _extract_source_names(sources: List[Any]) -> List[str]:
    names = []
    for s in sources:
        if isinstance(s, dict):
            name = s.get("source") or s.get("filename") or s.get("file") or ""
            if name:
                names.append(str(name))
        elif isinstance(s, str) and s:
            names.append(s)
    return names
