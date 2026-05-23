import math
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.schema import IngestedDocument
from app.vectorstore.qdrant_store import _CircuitBreaker, QdrantVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store():
    """Build a QdrantVectorStore with __new__, bypassing the Qdrant connection."""
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.client            = MagicMock()
    store.text_collection   = "text_collection"
    store.vision_collection = "vision_collection"
    store.batch_size        = 64
    store.max_docs          = 1000
    store.text_dim          = 384
    store.vision_dim        = 512
    store._collection_cache = set()
    store.modality_filter   = None
    return store


def _make_doc(modality="text", embedding=None, structure=None):
    return IngestedDocument(
        text="Sample text for testing.",
        modality=modality,
        source_type=modality,
        source="test.txt",
        structure=structure or {"session_id": "s1", "doc_id": "doc_001"},
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# _CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:

    def test_starts_closed(self):
        cb = _CircuitBreaker("test_svc", fail_max=3, reset_timeout=60)
        assert cb.is_open() is False

    def test_opens_after_fail_max(self):
        cb = _CircuitBreaker("test_svc", fail_max=3, reset_timeout=60)
        with patch("app.vectorstore.qdrant_store._circuit_breaker_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            cb.record_failure()
            cb.record_failure()
            cb.record_failure()
        assert cb._open is True

    def test_record_success_closes(self):
        cb = _CircuitBreaker("test_svc", fail_max=2, reset_timeout=60)
        with patch("app.vectorstore.qdrant_store._circuit_breaker_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            cb.record_failure()
            cb.record_failure()
            assert cb._open is True
            cb.record_success()
        assert cb._open is False
        assert cb._failures == 0

    def test_half_open_after_timeout(self):
        cb = _CircuitBreaker("test_svc", fail_max=1, reset_timeout=1)
        with patch("app.vectorstore.qdrant_store._circuit_breaker_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            cb.record_failure()
            assert cb._open is True
            cb._opened_at = time.time() - 2  # fast-forward past reset_timeout
        assert cb.is_open() is False  # should be half-open (closed for one probe)

    def test_single_failure_below_threshold_stays_closed(self):
        cb = _CircuitBreaker("test_svc", fail_max=5, reset_timeout=60)
        with patch("app.vectorstore.qdrant_store._circuit_breaker_state") as mock_metric:
            mock_metric.labels.return_value = MagicMock()
            cb.record_failure()
        assert cb.is_open() is False


# ---------------------------------------------------------------------------
# _valid_vector
# ---------------------------------------------------------------------------

class TestValidVector:

    def test_correct_dim_valid(self):
        store = _make_store()
        assert store._valid_vector([0.1] * 384, 384) is True

    def test_wrong_dim_invalid(self):
        store = _make_store()
        assert store._valid_vector([0.1] * 256, 384) is False

    def test_non_list_invalid(self):
        store = _make_store()
        assert store._valid_vector("not a list", 384) is False

    def test_nan_invalid(self):
        store = _make_store()
        emb = [0.1] * 383 + [float("nan")]
        assert store._valid_vector(emb, 384) is False

    def test_inf_invalid(self):
        store = _make_store()
        emb = [0.1] * 383 + [float("inf")]
        assert store._valid_vector(emb, 384) is False

    def test_empty_list_wrong_dim(self):
        store = _make_store()
        assert store._valid_vector([], 384) is False

    def test_vision_dim_valid(self):
        store = _make_store()
        assert store._valid_vector([0.0] * 512, 512) is True


# ---------------------------------------------------------------------------
# _vector_id
# ---------------------------------------------------------------------------

class TestVectorId:

    def test_returns_valid_uuid_string(self):
        store = _make_store()
        vid = store._vector_id("doc_001", 0)
        uuid.UUID(vid)  # raises if not valid UUID

    def test_same_inputs_stable(self):
        store = _make_store()
        v1 = store._vector_id("doc_001", 0)
        v2 = store._vector_id("doc_001", 0)
        assert v1 == v2

    def test_different_doc_id_different_vector_id(self):
        store = _make_store()
        v1 = store._vector_id("doc_001", 0)
        v2 = store._vector_id("doc_002", 0)
        assert v1 != v2

    def test_different_chunk_id_different_vector_id(self):
        store = _make_store()
        v1 = store._vector_id("doc_001", 0)
        v2 = store._vector_id("doc_001", 1)
        assert v1 != v2

    def test_chunk_id_can_be_string(self):
        store = _make_store()
        vid = store._vector_id("doc_001", "chunk_0")
        assert isinstance(vid, str)


# ---------------------------------------------------------------------------
# _payload
# ---------------------------------------------------------------------------

class TestPayload:

    def test_returns_dict(self):
        store = _make_store()
        doc   = _make_doc()
        p     = store._payload(doc)
        assert isinstance(p, dict)

    def test_text_field_present(self):
        store = _make_store()
        doc   = _make_doc()
        p     = store._payload(doc)
        assert "text" in p
        assert "Sample text" in p["text"]

    def test_modality_preserved(self):
        store = _make_store()
        doc   = _make_doc(modality="image")
        p     = store._payload(doc)
        assert p["modality"] == "image"

    def test_user_id_injected(self):
        store = _make_store()
        doc   = _make_doc()
        p     = store._payload(doc, user_id="user_42")
        assert p["user_id"] == "user_42"

    def test_user_id_none_when_absent(self):
        store = _make_store()
        doc   = _make_doc(structure={"session_id": "s1", "doc_id": "d1"})
        p     = store._payload(doc, user_id=None)
        assert p["user_id"] is None

    def test_session_id_from_structure(self):
        store = _make_store()
        doc   = _make_doc(structure={"session_id": "sess_99", "doc_id": "d1"})
        p     = store._payload(doc)
        assert p["session_id"] == "sess_99"

    def test_text_truncated_to_max_chars(self):
        store = _make_store()
        long_text = "x" * 100_000
        doc = IngestedDocument(
            text=long_text,
            modality="text",
            source_type="text",
            source="big.txt",
            structure={"session_id": "s1", "doc_id": "d1"},
        )
        p = store._payload(doc)
        from app.core.config import settings
        assert len(p["text"]) <= settings.QDRANT_TEXT_MAX_CHARS

    def test_deleted_at_is_none(self):
        store = _make_store()
        doc   = _make_doc()
        p     = store._payload(doc)
        assert p["deleted_at"] is None

    def test_is_forward_looking_bool(self):
        store = _make_store()
        doc   = _make_doc(structure={
            "session_id": "s1",
            "doc_id": "d1",
            "is_forward_looking": True,
        })
        p = store._payload(doc)
        assert p["is_forward_looking"] is True
        assert isinstance(p["is_forward_looking"], bool)

    def test_source_truncated_to_200_chars(self):
        store = _make_store()
        doc = IngestedDocument(
            text="text",
            modality="text",
            source_type="text",
            source="s" * 300,
            structure={"session_id": "s1", "doc_id": "d1"},
        )
        p = store._payload(doc)
        assert len(p["source"]) <= 200


# ---------------------------------------------------------------------------
# _build_filter
# ---------------------------------------------------------------------------

class TestBuildFilter:

    def test_no_conditions_returns_none(self):
        store = _make_store()
        result = store._build_filter()
        assert result is None

    def test_user_id_produces_filter(self):
        store  = _make_store()
        result = store._build_filter(user_id="user_1")
        assert result is not None

    def test_session_id_produces_filter(self):
        store  = _make_store()
        result = store._build_filter(session_id="sess_1")
        assert result is not None

    def test_modality_filter_applied(self):
        store = _make_store()
        store.modality_filter = "image"
        result = store._build_filter()
        assert result is not None

    def test_combined_conditions(self):
        store  = _make_store()
        result = store._build_filter(user_id="u1", session_id="s1")
        assert result is not None

    def test_no_modality_filter_no_session_no_user_returns_none(self):
        store = _make_store()
        store.modality_filter = None
        assert store._build_filter(session_id=None, user_id=None) is None
