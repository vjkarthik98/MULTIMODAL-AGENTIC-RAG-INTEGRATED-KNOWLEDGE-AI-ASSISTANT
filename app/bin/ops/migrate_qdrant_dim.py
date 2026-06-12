#!/usr/bin/env python3
"""
One-time migration: recreate the Qdrant collection at 1536-dim (BGE-large-en-v1.5).

Run AFTER upgrading EMBEDDING_MODEL to BAAI/bge-large-en-v1.5 and BEFORE
restarting the server or re-ingesting documents.

    QDRANT_ALLOW_RECREATE=true python app/bin/ops/migrate_qdrant_dim.py

WARNING: This DELETES the existing collection and all indexed vectors.
         Re-ingest all documents after running this script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(str(_project_root / ".env"))

if os.getenv("QDRANT_ALLOW_RECREATE", "").lower() not in ("true", "1", "yes"):
    print(
        "ERROR: Set QDRANT_ALLOW_RECREATE=true to confirm you want to drop and recreate "
        "the Qdrant collection.\n"
        "This operation is IRREVERSIBLE — all existing vectors will be deleted.\n"
        "Usage: QDRANT_ALLOW_RECREATE=true python app/bin/ops/migrate_qdrant_dim.py"
    )
    sys.exit(1)

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

OLD_DIM = 1024
NEW_DIM = settings.TEXT_EMBEDDING_DIM  # should be 1536 after config upgrade


def main() -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        HnswConfigDiff,
        OptimizersConfigDiff,
        VectorParams,
    )

    print(f"\n=== Qdrant Dimension Migration: {OLD_DIM} → {NEW_DIM} ===")
    print(f"Collection: {settings.COLLECTION_NAME}")
    print(f"Qdrant URL: {settings.QDRANT_URL or settings.QDRANT_HOST}")

    if NEW_DIM == OLD_DIM:
        print("WARNING: TEXT_EMBEDDING_DIM is still 1024. Update config first.")
        sys.exit(1)

    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        host=None if settings.QDRANT_URL else settings.QDRANT_HOST,
        port=None if settings.QDRANT_URL else settings.QDRANT_PORT,
        timeout=30,
    )

    collection = settings.COLLECTION_NAME
    vision_collection = settings.VISION_COLLECTION_NAME

    for col_name, dim in [(collection, NEW_DIM), (vision_collection, settings.VISION_EMBEDDING_DIM)]:
        print(f"\nProcessing collection: {col_name}  (dim={dim})")

        # Check if collection exists
        existing = [c.name for c in client.get_collections().collections]
        if col_name in existing:
            print(f"  Deleting existing collection '{col_name}'...")
            client.delete_collection(col_name)
            print(f"  Deleted.")
        else:
            print(f"  Collection '{col_name}' does not exist — creating fresh.")

        # Recreate with new dimension
        client.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=10000),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
        )
        print(f"  Created '{col_name}' with dim={dim}.")

        # Recreate payload indexes
        from qdrant_client.models import PayloadSchemaType
        for field, ftype in [
            ("user_id",    PayloadSchemaType.KEYWORD),
            ("session_id", PayloadSchemaType.KEYWORD),
            ("modality",   PayloadSchemaType.KEYWORD),
            ("source",     PayloadSchemaType.KEYWORD),
        ]:
            try:
                client.create_payload_index(col_name, field, ftype)
                print(f"  Index created: {field}")
            except Exception as exc:
                print(f"  Index {field} skipped: {exc}")

    print(
        "\nMigration complete.\n"
        "Next steps:\n"
        "  1. Restart the server\n"
        "  2. Re-ingest all documents (existing vectors are gone)\n"
        "  3. Optionally rebuild BM25 index: python app/bin/ops/rebuild_bm25_index.py"
    )


if __name__ == "__main__":
    main()
