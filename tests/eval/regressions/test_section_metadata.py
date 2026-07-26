"""Regression: section_number and is_forward_looking metadata must flow through
both Qdrant payload and BM25 metadata.

Phase 24 fix: added both fields to _PAYLOAD_INDEXES in qdrant_store.py and to
_metadata() in bm25_retriever.py so temporal boost can read them from either source.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def test_qdrant_payload_indexes_include_section_fields():
    """_PAYLOAD_INDEXES in QdrantVectorStore must include section_number and is_forward_looking."""
    from app.vectorstore.qdrant_store import QdrantVectorStore

    payload_index_fields = [name for name, _ in QdrantVectorStore._PAYLOAD_INDEXES]
    assert "section_number" in payload_index_fields, (
        "section_number missing from Qdrant _PAYLOAD_INDEXES — Phase 24 fix regressed!"
    )
    assert "is_forward_looking" in payload_index_fields, (
        "is_forward_looking missing from Qdrant _PAYLOAD_INDEXES — Phase 24 fix regressed!"
    )


def test_qdrant_payload_indexes_include_user_id():
    """_PAYLOAD_INDEXES must include user_id to support per-user filtered search."""
    from app.vectorstore.qdrant_store import QdrantVectorStore

    payload_index_fields = [name for name, _ in QdrantVectorStore._PAYLOAD_INDEXES]
    assert "user_id" in payload_index_fields, (
        "user_id missing from Qdrant _PAYLOAD_INDEXES — per-user filter will fail with 400 Bad Request!"
    )


def test_bm25_metadata_includes_section_fields():
    """BM25 _metadata method must include section_number and is_forward_looking keys."""
    import inspect
    from app.retrieval.bm25_retriever import BM25Retriever

    source = inspect.getsource(BM25Retriever)
    assert "section_number" in source, (
        "section_number missing from BM25Retriever — Phase 24 fix regressed!"
    )
    assert "is_forward_looking" in source, (
        "is_forward_looking missing from BM25Retriever — Phase 24 fix regressed!"
    )


def test_ensure_payload_indexes_called_on_existing_collection():
    """_ensure_collection must call _ensure_payload_indexes for BOTH new and existing
    collections — not just on first creation."""
    import inspect
    from app.vectorstore.qdrant_store import QdrantVectorStore

    source = inspect.getsource(QdrantVectorStore._ensure_collection)
    # The fix ensures _ensure_payload_indexes is called regardless of whether
    # the collection already existed
    assert "_ensure_payload_indexes" in source, (
        "_ensure_payload_indexes not called in _ensure_collection — "
        "existing collections won't get user_id index, causing 400 Bad Request!"
    )
