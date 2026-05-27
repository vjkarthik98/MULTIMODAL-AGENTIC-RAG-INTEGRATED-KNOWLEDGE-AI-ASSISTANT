"""
Rebuild BM25 index for a given user_id from all Qdrant text_collection chunks.

Usage:
    python -m scripts.rebuild_bm25_index --user_id eval_default
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.retrieval.bm25_retriever import BM25Document, BM25Retriever
from app.vectorstore.qdrant_store import QdrantVectorStore
from app.utils.logger import get_logger
from qdrant_client.models import FieldCondition, Filter, MatchValue

logger = get_logger(__name__)


def rebuild(user_id: str) -> int:
    qs = QdrantVectorStore()
    bm25 = BM25Retriever(user_id=user_id)

    scroll_filter = Filter(
        must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    ) if user_id else None

    points, _next = qs.client.scroll(
        collection_name=settings.TEXT_COLLECTION_NAME,
        with_payload=True,
        limit=5000,
        scroll_filter=scroll_filter,
    )
    logger.info("bm25_rebuild_fetched", user_id=user_id, count=len(points))

    doc_objects: list[BM25Document] = []
    for pt in points:
        p = pt.payload or {}
        obj = BM25Document.from_payload(p)
        if obj.text:
            doc_objects.append(obj)

    bm25.build_index(doc_objects, user_id=user_id)
    logger.info("bm25_rebuild_done", user_id=user_id, indexed=len(bm25.documents))
    return len(bm25.documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", default="eval_default")
    args = parser.parse_args()
    n = rebuild(args.user_id)
    print(f"Rebuilt BM25 index for user '{args.user_id}': {n} docs indexed.")
