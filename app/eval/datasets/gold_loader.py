"""Load, validate, and filter gold-set JSONL files.

Only returns rows that have been human-curated (no TODO values).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.eval.config import GOLD_DIR


GOLD_FILES: Dict[str, str] = {
    "txt":     "text_gold.jsonl",
    "pdf":     "pdf_gold.jsonl",
    "docx":    "docx_gold.jsonl",
    "xlsx":    "xlsx_gold.jsonl",
    "image":   "image_gold.jsonl",
    "audio":   "audio_gold.jsonl",
    "video":   "video_gold.jsonl",
    "routing": "routing_gold.jsonl",
    "e2e":     "e2e_gold.jsonl",
}


def _is_curated(row: Dict[str, Any]) -> bool:
    """Return True only if this row has real (non-TODO) ground truth."""
    rel = row.get("relevant_chunk_ids")
    ans = row.get("reference_answer", "")
    behavior = row.get("expected_behavior")
    # Routing-only rows (no docs to retrieve) are always curated if they have expected_route
    if row.get("modality") == "routing":
        return bool(row.get("expected_route"))
    # Refusal / no-answer rows: correct behavior is abstention, so there is no
    # retrievable chunk — curated as long as they carry a reference abstention.
    if behavior == "abstain" or "NO_ANSWER_IN_KB" in str(ans):
        return bool(ans)
    # E2E rows that require web search have SEARCH_REQUIRED placeholder — still valid
    if "SEARCH_REQUIRED" in str(ans):
        return bool(row.get("expected_route"))
    # Injection probe rows
    if "INJECTION_PROBE" in str(ans):
        return True
    # Standard rows need real chunk IDs and a real answer
    if rel in (None, [], "TODO_ingest_then_fill", ["TODO_ingest_then_fill"]):
        return False
    if not ans or ans == "TODO":
        return False
    return True


def load_gold(
    modality: str,
    gold_dir: Path = GOLD_DIR,
    include_todos: bool = False,
) -> List[Dict[str, Any]]:
    """Load gold rows for a modality. By default skips unreviewed (TODO) rows."""
    fname = GOLD_FILES.get(modality)
    if not fname:
        raise ValueError(f"Unknown modality '{modality}'. Valid: {list(GOLD_FILES)}")

    path = gold_dir / fname
    if not path.exists():
        return []

    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}:{i+1}: {e}") from e
            if include_todos or _is_curated(row):
                rows.append(row)
    return rows


def load_all_gold(
    gold_dir: Path = GOLD_DIR,
    modalities: Optional[List[str]] = None,
    include_todos: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load gold sets for all (or specified) modalities."""
    targets = modalities or list(GOLD_FILES.keys())
    return {m: load_gold(m, gold_dir=gold_dir, include_todos=include_todos) for m in targets}


def gold_stats(gold_dir: Path = GOLD_DIR) -> Dict[str, Dict[str, int]]:
    """Return {modality: {total, curated, todo}} counts."""
    stats = {}
    for modality, fname in GOLD_FILES.items():
        path = gold_dir / fname
        if not path.exists():
            stats[modality] = {"total": 0, "curated": 0, "todo": 0}
            continue
        rows_all = load_gold(modality, gold_dir=gold_dir, include_todos=True)
        curated = [r for r in rows_all if _is_curated(r)]
        stats[modality] = {
            "total": len(rows_all),
            "curated": len(curated),
            "todo": len(rows_all) - len(curated),
        }
    return stats
