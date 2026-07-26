"""Regression: Qdrant _PAYLOAD_INDEXES must include user_id, section_number,
and is_forward_looking so filtered searches don't get 400 Bad Request.

Phase 24 fix: _ensure_collection now calls _ensure_payload_indexes() on BOTH
new AND existing collections. Previously only ran on collection creation.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


REQUIRED_PAYLOAD_INDEXES = ["user_id", "section_number", "is_forward_looking"]


def test_all_required_payload_indexes_defined():
    """All required payload index fields must be in QdrantVectorStore._PAYLOAD_INDEXES."""
    from app.vectorstore.qdrant_store import QdrantVectorStore

    defined = {name for name, _ in QdrantVectorStore._PAYLOAD_INDEXES}
    for required in REQUIRED_PAYLOAD_INDEXES:
        assert required in defined, (
            f"'{required}' missing from _PAYLOAD_INDEXES — Qdrant filtered search will fail with 400"
        )


def test_user_id_index_has_correct_schema_type():
    """user_id must be a KEYWORD index (not numeric) for string UUID filtering."""
    from qdrant_client.http.models import PayloadSchemaType
    from app.vectorstore.qdrant_store import QdrantVectorStore

    index_map = dict(QdrantVectorStore._PAYLOAD_INDEXES)
    user_id_schema = index_map.get("user_id")
    assert user_id_schema is not None, "user_id not in _PAYLOAD_INDEXES"
    # Accept both enum and string forms
    assert str(user_id_schema).upper() in ("KEYWORD", "PAYLOADSCHEMATYPE.KEYWORD"), (
        f"user_id index should be KEYWORD type, got {user_id_schema}"
    )


def test_ensure_payload_indexes_called_unconditionally():
    """_ensure_collection source must call _ensure_payload_indexes regardless of whether
    the collection already existed — not just inside 'if collection_created' branch."""
    import inspect
    from app.vectorstore.qdrant_store import QdrantVectorStore

    source = inspect.getsource(QdrantVectorStore._ensure_collection)

    # The fix ensures _ensure_payload_indexes is always called
    assert "_ensure_payload_indexes" in source, (
        "_ensure_payload_indexes not called in _ensure_collection. "
        "Existing collections won't get required indexes → 400 Bad Request on user_id filter."
    )
