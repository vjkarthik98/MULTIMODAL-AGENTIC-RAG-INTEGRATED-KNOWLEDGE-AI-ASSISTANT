from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from app.ingestion.schema import IngestedDocument


def _make_doc(text="This is a sample text for embedding tests.", modality="text"):
    return IngestedDocument(
        text=text,
        modality=modality,
        source_type=modality,
        source="test.txt",
        structure={"session_id": "test"},
    )


def _make_embedder():
    """Build a TextEmbedder with mocked model, bypassing SentenceTransformer loading."""
    from app.embeddings.text_embedder import TextEmbedder

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1] * 384])

    embedder = TextEmbedder.__new__(TextEmbedder)
    embedder.model = mock_model
    embedder.model_name = "all-MiniLM-L6-v2"
    embedder.batch_size = 32
    embedder.device = "cpu"
    embedder.expected_dim = 384
    embedder.max_text_len = 4096
    return embedder


class TestTextEmbedderEmbedDocuments:

    def test_embed_documents_returns_list(self):
        embedder = _make_embedder()
        import app.embeddings.text_embedder as te_mod
        with patch.object(te_mod, "_cache") as mock_cache:
            mock_cache.get.return_value = None
            docs = [_make_doc()]
            result = embedder.embed_documents(docs, session_id="test")
        assert isinstance(result, list)

    def test_embed_documents_sets_embedding(self):
        embedder = _make_embedder()
        import app.embeddings.text_embedder as te_mod
        with patch.object(te_mod, "_cache") as mock_cache:
            mock_cache.get.return_value = None
            docs = [_make_doc()]
            result = embedder.embed_documents(docs, session_id="test")
        if result:
            assert result[0].embedding is not None

    def test_embed_text_returns_list_of_floats(self):
        embedder = _make_embedder()
        embedder.model.encode.return_value = np.array([0.1] * 384)
        import app.embeddings.text_embedder as te_mod
        with patch.object(te_mod, "_cache") as mock_cache:
            mock_cache.get.return_value = None
            result = embedder.embed_text("hello world", session_id="test")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_health_check_returns_dict(self):
        embedder = _make_embedder()
        result = embedder.health_check()
        assert isinstance(result, dict)
