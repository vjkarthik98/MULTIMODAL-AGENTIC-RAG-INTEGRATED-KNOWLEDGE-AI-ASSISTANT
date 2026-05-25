"""Bootstrap gold-set candidates from data/raw/finance/* and the real ingestion pipeline.

Usage:
    python -m app.eval.datasets.build_gold_set --modality txt
    python -m app.eval.datasets.build_gold_set --modality all
    python -m app.eval.datasets.build_gold_set --ingest      # ingest raw corpus into KB
    python -m app.eval.datasets.build_gold_set --validate    # check schema + update manifest

IMPORTANT: This script scaffolds CANDIDATES. Every row it outputs has:
  "relevant_chunk_ids": "TODO_ingest_then_fill"
  "reference_answer":   "TODO"   (if machine-proposed — must be human-verified)

A human MUST review each TODO row, fill in the correct values from the source document,
and remove the TODO before the eval harness will include the row in metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Project root so we can import app.*
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from app.core.config import get_settings
from app.eval.config import EVAL_USER_ID, GOLD_DIR, MANIFEST_PATH, RAW_CORPUS_DIR

_settings = get_settings()

MODALITY_TO_GOLD: Dict[str, str] = {
    "txt":   "text_gold.jsonl",
    "pdf":   "pdf_gold.jsonl",
    "docx":  "docx_gold.jsonl",
    "xlsx":  "xlsx_gold.jsonl",
    "image": "image_gold.jsonl",
    "audio": "audio_gold.jsonl",
    "video": "video_gold.jsonl",
}

MODALITY_EXTS: Dict[str, List[str]] = {
    "txt":   [".txt"],
    "pdf":   [".pdf"],
    "docx":  [".docx"],
    "xlsx":  [".xlsx"],
    "image": [".jpg", ".jpeg", ".png"],
    "audio": [".mp3", ".wav", ".m4a"],
    "video": [".mp4", ".mov", ".avi"],
}

QUESTION_TEMPLATES: Dict[str, List[str]] = {
    "txt": [
        "What was {company}'s total revenue in the most recent fiscal year?",
        "What were the key financial highlights reported?",
        "What risk factors were disclosed in this filing?",
        "What is the geographic breakdown of revenue?",
        "What was the net income reported?",
    ],
    "pdf": [
        "What is the main financial figure reported in this document?",
        "What period does this filing cover?",
        "What are the primary business segments described?",
        "What guidance or outlook is provided?",
    ],
    "docx": [
        "What are the key investment highlights mentioned?",
        "What revenue or financial projections are provided?",
    ],
    "xlsx": [
        "What is the primary financial metric shown in this spreadsheet?",
        "What time period does this data cover?",
        "What is the trend visible in this data?",
    ],
    "image": [
        "What financial figure is shown in this image?",
        "What company or document does this image relate to?",
        "What type of chart or graph is shown?",
    ],
    "audio": [
        "What financial results were announced?",
        "What guidance was provided?",
        "What questions did analysts ask?",
    ],
    "video": [
        "What financial figures are displayed in this video?",
        "What charts or graphs are shown?",
        "What company is being discussed?",
    ],
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _count_curated(rows: List[Dict[str, Any]]) -> int:
    return sum(
        1 for r in rows
        if r.get("relevant_chunk_ids") not in (["TODO_ingest_then_fill"], "TODO_ingest_then_fill", [], None)
        and r.get("reference_answer") not in ("TODO", None, "")
        and "SEARCH_REQUIRED" not in str(r.get("reference_answer", ""))
        and "INJECTION_PROBE" not in str(r.get("reference_answer", ""))
    )


def scaffold_candidates(modality: str) -> None:
    """Scan raw corpus for this modality and scaffold TODO candidates."""
    raw_dir = RAW_CORPUS_DIR / modality
    gold_path = GOLD_DIR / MODALITY_TO_GOLD[modality]
    exts = MODALITY_EXTS[modality]
    templates = QUESTION_TEMPLATES.get(modality, [])

    if not raw_dir.exists():
        print(f"[WARN] No raw corpus dir for {modality}: {raw_dir} — skipping scaffold")
        return

    files = [f for f in raw_dir.iterdir() if f.suffix.lower() in exts]
    if not files:
        print(f"[INFO] No {modality} files found in {raw_dir} — skipping scaffold")
        return

    existing = _load_jsonl(gold_path)
    existing_ids = {r["id"] for r in existing}

    new_rows = []
    for fpath in sorted(files):
        company = fpath.stem.split("_")[0].upper() if "_" in fpath.stem else fpath.stem.upper()
        for i, template in enumerate(templates):
            query = template.replace("{company}", company).replace("{ticker}", company)
            row_id = f"{modality[:3]}-scaffold-{fpath.stem}-{i:02d}"
            if row_id in existing_ids:
                continue
            row: Dict[str, Any] = {
                "id": row_id,
                "modality": modality,
                "source_file": fpath.name,
                "query": query,
                "relevant_chunk_ids": "TODO_ingest_then_fill",
                "reference_answer": "TODO",
                "expected_route": "rag",
                "tags": [f"{modality}-extraction", "machine-proposed"],
                "added_by": "machine",
                "added_at": time.strftime("%Y-%m-%d"),
            }
            new_rows.append(row)
            print(f"  [CANDIDATE] {row_id}: {query[:70]}")

    if new_rows:
        with open(gold_path, "a") as f:
            for row in new_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[OK] Added {len(new_rows)} candidates to {gold_path}")
    else:
        print(f"[INFO] No new candidates for {modality}")


def ingest_corpus() -> None:
    """Ingest all raw corpus files into the eval user's knowledge base."""
    try:
        from app.pipeline.ingestion_pipeline import process_file
    except ImportError as e:
        print(f"[ERROR] Cannot import ingestion pipeline: {e}")
        sys.exit(2)

    results = []
    for modality, exts in MODALITY_EXTS.items():
        raw_dir = RAW_CORPUS_DIR / modality
        if not raw_dir.exists():
            continue
        for fpath in sorted(raw_dir.iterdir()):
            if fpath.suffix.lower() not in exts:
                continue
            print(f"[INGEST] {fpath} → user_id={EVAL_USER_ID}")
            try:
                result = process_file(
                    str(fpath),
                    session_id="eval_ingest",
                    user_id=EVAL_USER_ID,
                )
                results.append({"file": str(fpath), **result})
                print(f"  status={result.get('status')} chunks={result.get('chunk_count', 0)}")
            except Exception as exc:
                print(f"  [ERROR] {exc}")
                results.append({"file": str(fpath), "status": "error", "error": str(exc)})

    ok = sum(1 for r in results if r.get("status") == "success")
    err = len(results) - ok
    print(f"\n[INGEST DONE] {ok}/{len(results)} files OK, {err} errors")
    if err:
        sys.exit(1)


def validate_and_update_manifest() -> None:
    """Validate all gold JSONL files and update manifest.yaml SHA-256 entries."""
    REQUIRED_FIELDS = ["id", "modality", "source_file", "query",
                       "relevant_chunk_ids", "reference_answer",
                       "expected_route", "added_by", "added_at"]
    VALID_ROUTES = {"rag", "search", "memory", "direct", "hybrid"}

    manifest: Dict[str, Any] = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = yaml.safe_load(f) or {}

    gold_files = manifest.get("gold_files", {})
    errors = 0

    for modality, fname in MODALITY_TO_GOLD.items():
        gold_path = GOLD_DIR / fname
        rows = _load_jsonl(gold_path)

        for i, row in enumerate(rows):
            for field in REQUIRED_FIELDS:
                if field not in row:
                    print(f"[ERROR] {fname}:{i} missing field '{field}'")
                    errors += 1
            route = row.get("expected_route", "")
            if route not in VALID_ROUTES:
                print(f"[ERROR] {fname}:{i} invalid expected_route='{route}'")
                errors += 1
            if not row.get("id", "").strip():
                print(f"[ERROR] {fname}:{i} empty id")
                errors += 1

        total = len(rows)
        curated = _count_curated(rows)

        # Recompute SHA-256
        if gold_path.exists():
            sha = _sha256_file(gold_path)
        else:
            sha = ""

        gold_files[fname] = {
            **gold_files.get(fname, {}),
            "sha256": sha,
            "triples_total": total,
            "triples_curated": curated,
        }
        print(f"[MANIFEST] {fname}: {curated}/{total} curated, sha256={sha[:16]}...")

    manifest["gold_files"] = gold_files
    with open(MANIFEST_PATH, "w") as f:
        yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)

    if errors:
        print(f"\n[VALIDATION FAILED] {errors} errors found")
        sys.exit(1)
    else:
        print("\n[VALIDATION OK] manifest.yaml updated")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gold evaluation dataset")
    parser.add_argument("--modality", choices=["all"] + list(MODALITY_TO_GOLD.keys()),
                        help="Scaffold candidates for this modality (or 'all')")
    parser.add_argument("--ingest", action="store_true",
                        help="Ingest raw corpus into eval user KB")
    parser.add_argument("--validate", action="store_true",
                        help="Validate gold files and update manifest")
    args = parser.parse_args()

    if args.ingest:
        ingest_corpus()
    elif args.validate:
        validate_and_update_manifest()
    elif args.modality:
        modalities = list(MODALITY_TO_GOLD.keys()) if args.modality == "all" else [args.modality]
        for m in modalities:
            print(f"\n=== Scaffolding {m} ===")
            scaffold_candidates(m)
        validate_and_update_manifest()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
