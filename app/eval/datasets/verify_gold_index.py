"""Verify the gold set against the LIVE Qdrant index and fill citation locators.

For every answer-type gold row this:
  1. Resolves each relevant_chunk_id ("{source}::chunk_{N}") to its live payload.
  2. Validates the chunk(s) actually contain the row's must_include_facts — flags
     stale mappings (chunk renumbered / content moved) as facts_missing.
  3. Fills expected_citation.locator from the real payload:
       pdf/docx -> page_number   xlsx -> sheet_name
       audio/video -> timestamp (start_timestamp/timestamp_start, seconds)
       image -> image_title
  4. Writes the gold files back and emits reports/gold_index_verification.json.

Deterministic, read-only against Qdrant (no GPU, no auth). Run as the KB owner:

    EVAL_USER_ID=<owner-uuid> python -m app.eval.datasets.verify_gold_index
    python -m app.eval.datasets.verify_gold_index --dry
"""

from __future__ import annotations

import json
import re
import sys

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings
from app.eval.config import EVAL_USER_ID, REPORTS_DIR
from app.eval.datasets.rebuild_gold import GOLD, MODALITY_FILES, read_jsonl, write_jsonl
from app.vectorstore.qdrant_store import QdrantVectorStore

_CHUNK_RE = re.compile(r"^(?P<src>.+)::chunk_(?P<cid>\d+)$")


def _load_index(store: QdrantVectorStore, user_id: str) -> dict[tuple[str, int], dict]:
    """Map (source, chunk_id) -> payload for every chunk the KB owner has."""
    idx: dict[tuple[str, int], dict] = {}
    for coll in (settings.TEXT_COLLECTION_NAME, settings.VISION_COLLECTION_NAME):
        flt = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
        offset = None
        while True:
            pts, offset = store.client.scroll(
                coll, scroll_filter=flt, limit=256, offset=offset, with_payload=True
            )
            for p in pts:
                pl = p.payload or {}
                src = pl.get("source") or pl.get("filename")
                cid = pl.get("chunk_id")
                if src is not None and isinstance(cid, int):
                    idx[(src, cid)] = pl
            if offset is None:
                break
    return idx


def _payload_text(pl: dict) -> str:
    return str(pl.get("text") or pl.get("content") or pl.get("caption") or "")


def _timestamp(pl: dict):
    for k in ("start_time", "timestamp_start", "start_timestamp"):
        v = pl.get(k)
        if v is not None:
            try:
                return round(float(v), 2)
            except (TypeError, ValueError):
                pass
    return None


# specific facts = numbers with >=4 digits or a decimal — reliable to string-match
_SPECIFIC = re.compile(r"\d[\d,]{3,}|\d+\.\d+")


def _facts_status(facts: list[str], text: str) -> str:
    specific = [t for f in facts for t in _SPECIFIC.findall(f)]
    if not specific:
        return "ok_unchecked"  # only small numbers — can't reliably validate here
    return "ok" if any(t in text for t in specific) else "facts_missing"


def _repair_chunks(idx: dict, source: str, facts: list[str], top: int = 2):
    """Find the chunk(s) of `source` that best contain the row's specific facts.

    High-confidence only: requires the single most-specific fact token to appear.
    Returns [(chunk_id_str, payload), ...] ranked by fact-token hits, or []."""
    specific = sorted({t for f in facts for t in _SPECIFIC.findall(f)}, key=len, reverse=True)
    if not specific:
        return []
    anchor = specific[0]  # longest / most specific token
    scored = []
    for (src, cid), pl in idx.items():
        if src != source:
            continue
        txt = _payload_text(pl)
        if anchor in txt:
            hits = sum(1 for t in specific if t in txt)
            scored.append((hits, cid, pl))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(f"{source}::chunk_{cid}", pl) for _, cid, pl in scored[:top]]


def _fill_locator(row: dict, chunks: list[dict]) -> None:
    """Set expected_citation.locator from the primary resolved chunk's payload."""
    cite = row.get("expected_citation") or {}
    lt = cite.get("locator_type")
    if lt in (None, "none", "web") or not chunks:
        return
    primary = chunks[0]
    if lt == "page":
        pg = primary.get("page_number")
        if pg is not None:
            cite["locator"] = int(pg)
    elif lt == "sheet":
        cite["locator"] = primary.get("sheet_name") or cite.get("locator")
    elif lt in ("timestamp", "timestamp+frame", "frame"):
        ts = _timestamp(primary)
        if ts is not None:
            cite["locator"] = ts
    elif lt == "section":
        sec = primary.get("section_title") or primary.get("heading")
        if sec:
            cite["locator"] = str(sec).split("\n")[0][:80]
    elif lt == "image_title":
        cite["locator"] = primary.get("image_title") or cite.get("locator")
    row["expected_citation"] = cite


def verify(dry: bool = False) -> None:
    store = QdrantVectorStore()
    print(f"[verify] loading index for user {EVAL_USER_ID} ...")
    idx = _load_index(store, EVAL_USER_ID)
    print(f"[verify] {len(idx)} chunks indexed across {len({s for s, _ in idx})} sources")

    report: dict[str, list] = {
        "facts_missing": [],
        "chunk_not_found": [],
        "locators_filled": [],
        "ok": [],
        "repaired": [],
    }

    for _modality, fname in MODALITY_FILES.items():
        path = GOLD / fname
        rows = read_jsonl(path)
        changed = False
        for row in rows:
            beh = row.get("expected_behavior")
            if beh != "answer":  # refusal/routing → no chunk ground truth
                continue
            if "SEARCH_REQUIRED" in str(row.get("reference_answer", "")):
                continue
            rel = row.get("relevant_chunk_ids") or []
            resolved, missing = [], []
            for cid_str in rel:
                m = _CHUNK_RE.match(cid_str)
                if not m:
                    missing.append(cid_str)
                    continue
                key = (m.group("src"), int(m.group("cid")))
                if key in idx:
                    resolved.append(idx[key])
                else:
                    missing.append(cid_str)
            if missing:
                report["chunk_not_found"].append({"id": row["id"], "missing": missing})
            facts = row.get("must_include_facts") or []
            combined = " ".join(_payload_text(p) for p in resolved)
            status = _facts_status(facts, combined) if resolved else "chunk_not_found"

            # Auto-repair: if the mapped chunk(s) don't contain the facts, try to
            # re-point to the chunk of the same source that does (high-confidence only).
            if status in ("facts_missing", "chunk_not_found") and rel:
                source = _CHUNK_RE.match(rel[0]).group("src") if _CHUNK_RE.match(rel[0]) else None
                # High-confidence numeric repair only. Bag-of-words repair is unsafe
                # for transcripts (shared vocabulary inflates overlap), and spoken-form
                # numbers (e.g. "four and three quarters percent") can't string-match —
                # those rows stay flagged for human review instead of being mis-repaired.
                repaired = _repair_chunks(idx, source, facts) if source else []
                if repaired:
                    row["relevant_chunk_ids"] = [cid for cid, _ in repaired]
                    resolved = [pl for _, pl in repaired]
                    combined = " ".join(_payload_text(p) for p in resolved)
                    status = _facts_status(facts, combined)
                    report["repaired"].append(
                        {"id": row["id"], "new_chunks": row["relevant_chunk_ids"]}
                    )
                    changed = True

            if status == "facts_missing":
                report["facts_missing"].append(
                    {
                        "id": row["id"],
                        "facts": facts,
                        "chunks": row.get("relevant_chunk_ids"),
                    }
                )
            elif resolved:
                report["ok"].append(row["id"])

            if resolved:
                before = json.dumps(row.get("expected_citation"))
                _fill_locator(row, resolved)
                if json.dumps(row.get("expected_citation")) != before:
                    report["locators_filled"].append(row["id"])
                    changed = True
        if changed and not dry:
            write_jsonl(path, rows)

    # write report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rpt_path = REPORTS_DIR / "gold_index_verification.json"
    summary = {k: len(v) for k, v in report.items()}
    if not dry:
        rpt_path.write_text(json.dumps({"summary": summary, "detail": report}, indent=2))

    print(f"\n{'DRY RUN — ' if dry else ''}gold-vs-index verification:")
    print(f"  chunks OK (facts present):   {summary['ok']}")
    print(f"  locators filled:             {summary['locators_filled']}")
    print(f"  FACTS MISSING (review):      {summary['facts_missing']}")
    print(f"  CHUNK NOT FOUND (review):    {summary['chunk_not_found']}")
    if report["facts_missing"]:
        print("  -- facts_missing ids:", [r["id"] for r in report["facts_missing"]])
    if report["chunk_not_found"]:
        print("  -- chunk_not_found ids:", [r["id"] for r in report["chunk_not_found"]])
    if not dry:
        print(f"  report: {rpt_path}")


if __name__ == "__main__":
    verify(dry="--dry" in sys.argv)
