"""One-off diagnostic: list distinct `source` values + chunk counts in Qdrant
for the eval tenant, to check the corpus is the full 7-file canonical set
before trusting any tier1-retrieval gate number.

Usage (on the box, inside the running container):
    docker compose exec api python -m app.bin.check_eval_corpus
"""

from __future__ import annotations

import os
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings

EVAL_USER_ID = os.environ.get("EVAL_USER_ID", "36055d60-9099-4f51-81d2-08fe33916356")

CANONICAL_FILES = {
    "fomc_dec2024.txt",
    "apple_10k.pdf",
    "apple_investment_research_report.docx",
    "ctryprem.xlsx",
    "aapl-20240928_g2.jpg",
    "FOMC Press Conference September 18, 2024.mp3",
    "Q4 2025 Earnings Call.mp4",
}


def _count_sources(client: QdrantClient, collection: str) -> Counter:
    counts: Counter = Counter()
    offset = None
    flt = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=EVAL_USER_ID))])
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_payload=["source"],
            with_vectors=False,
        )
        for p in points:
            counts[(p.payload or {}).get("source", "<missing>")] += 1
        if offset is None:
            break
    return counts


def main() -> None:
    client = (
        QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=settings.QDRANT_TIMEOUT)
        if settings.QDRANT_URL
        else QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=settings.QDRANT_TIMEOUT)
    )

    print(f"EVAL_USER_ID = {EVAL_USER_ID}\n")

    all_sources: set[str] = set()
    for collection in (settings.TEXT_COLLECTION_NAME, settings.VISION_COLLECTION_NAME):
        print(f"=== {collection} ===")
        counts = _count_sources(client, collection)
        if not counts:
            print("  (no chunks found for this tenant)")
        for source, n in sorted(counts.items()):
            print(f"  {n:5d}  {source}")
        all_sources.update(counts.keys())
        print()

    missing = CANONICAL_FILES - all_sources
    extra = all_sources - CANONICAL_FILES

    print("=== Canonical corpus check ===")
    if not missing and not extra:
        print("OK: exactly the 7 canonical files are present, nothing extra.")
    else:
        if missing:
            print(f"MISSING ({len(missing)}): {sorted(missing)}")
        if extra:
            print(f"UNEXPECTED EXTRA ({len(extra)}): {sorted(extra)}")


if __name__ == "__main__":
    main()
