"""
Auto-fill relevant_chunk_ids in gold JSONL files by querying Qdrant.

For each gold triple with relevant_chunk_ids == ["TODO_ingest_then_fill"],
this script:
  1. Embeds the query using the same embedding model as the pipeline
  2. Searches Qdrant filtered to user_id=eval_default + source file match
  3. Takes the top-N chunk IDs as the relevant set
  4. Writes them back to the JSONL file

Usage:
    python scripts/fill_gold_chunk_ids.py
    python scripts/fill_gold_chunk_ids.py --top-n 3 --dry-run
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("ENV_FILE", ".env")

GOLD_DIR = Path("app/eval/datasets/gold")
USER_ID = "eval_default"
TODO_MARKER = "TODO_ingest_then_fill"


def get_source_key(source_file: str, all_sources: list[str]) -> str | None:
    """Find the Qdrant source string that ends with the given filename."""
    base = os.path.basename(source_file)
    for s in all_sources:
        if s.endswith(base) or s.endswith("_" + base):
            return s
    return None


def embed_query(query: str, embedder) -> list[float]:
    result = embedder.embed_query(query)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=3, help="Number of chunk IDs to fill per triple")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    parser.add_argument("--file", default=None, help="Only process this gold file (e.g. text_gold.jsonl)")
    args = parser.parse_args()

    from app.core.infra_registry import infra
    from app.core.config import get_settings
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Query
    from qdrant_client import models as qmodels
    from sentence_transformers import SentenceTransformer

    s = get_settings()
    qs = infra.get_vector_store()

    print(f"[fill_gold] Loading embedding model: {s.EMBEDDING_MODEL}")
    embedder = SentenceTransformer(s.EMBEDDING_MODEL)

    # Fetch all unique source strings for eval_default
    all_sources: set[str] = set()
    offset = None
    while True:
        batch, next_offset = qs.client.scroll(
            collection_name=s.TEXT_COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=USER_ID))]),
            limit=200,
            offset=offset,
            with_payload=["source", "chunk_id"],
            with_vectors=False,
        )
        for pt in batch:
            src = pt.payload.get("source", "")
            if src:
                all_sources.add(src)
        if next_offset is None:
            break
        offset = next_offset

    print(f"[fill_gold] Found {len(all_sources)} unique sources for user={USER_ID}")

    gold_files = sorted(GOLD_DIR.glob("*.jsonl"))
    if args.file:
        gold_files = [f for f in gold_files if f.name == args.file]

    total_filled = 0
    total_skipped = 0

    for gold_path in gold_files:
        lines = gold_path.read_text().strip().splitlines()
        updated_lines = []
        changed = False

        for line in lines:
            if not line.strip():
                updated_lines.append(line)
                continue

            row = json.loads(line)

            # Skip routing rows — they have no source file or chunk IDs
            if row.get("modality") == "routing":
                updated_lines.append(line)
                continue

            chunk_ids = row.get("relevant_chunk_ids", [])
            if TODO_MARKER not in chunk_ids:
                updated_lines.append(line)
                total_skipped += 1
                continue

            source_file = row.get("source_file", "")
            query = row.get("query", "")

            if not source_file or not query:
                print(f"  [skip] {row['id']}: missing source_file or query")
                updated_lines.append(line)
                continue

            # Find matching source string in Qdrant
            matched_source = get_source_key(source_file, list(all_sources))
            if not matched_source:
                print(f"  [WARN] {row['id']}: no Qdrant source matches '{source_file}' — leaving TODO")
                updated_lines.append(line)
                continue

            # Embed query and search within this source
            vec = embedder.encode(query).tolist()
            result = qs.client.query_points(
                collection_name=s.TEXT_COLLECTION_NAME,
                query=vec,
                query_filter=Filter(must=[
                    FieldCondition(key="user_id", match=MatchValue(value=USER_ID)),
                    FieldCondition(key="source",  match=MatchValue(value=matched_source)),
                ]),
                limit=args.top_n,
                with_payload=True,
                with_vectors=False,
            )
            hits = result.points

            if not hits:
                print(f"  [WARN] {row['id']}: no hits for query in source '{matched_source}'")
                updated_lines.append(line)
                continue

            filled_ids = [f"{matched_source}::chunk_{h.payload['chunk_id']}" for h in hits]
            row["relevant_chunk_ids"] = filled_ids

            # Print preview
            print(f"  [fill] {row['id']}: {len(filled_ids)} chunks")
            for h in hits:
                snippet = h.payload.get("text", "")[:80].replace("\n", " ")
                print(f"         chunk_{h.payload['chunk_id']} (score={h.score:.3f}): {snippet}")

            updated_lines.append(json.dumps(row))
            changed = True
            total_filled += 1

        if changed and not args.dry_run:
            gold_path.write_text("\n".join(updated_lines) + "\n")
            print(f"[fill_gold] Updated {gold_path.name}")
        elif changed:
            print(f"[fill_gold] DRY RUN — would update {gold_path.name}")

    print(f"\n[fill_gold] Done. Filled={total_filled}  Already-filled/skipped={total_skipped}")


if __name__ == "__main__":
    main()
