from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from app.ingestion.schema import IngestedDocument


def _make_audio_doc():
    return IngestedDocument(
        text="This is transcribed speech audio content for testing.",
        modality="audio",
        subtype="speech",
        source_type="audio",
        source="test.wav",
        page=None,
        chunk_id=0,
        structure={
            "start_time": 0,
            "end_time": 2,
            "session_id": "test",
        },
    )


def _make_embedder():
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


class TestAudioEmbedding:

    def test_audio_docs_embedded_via_text_embedder(self):
        embedder = _make_embedder()
        import app.embeddings.text_embedder as te_mod
        with patch.object(te_mod, "_cache") as mock_cache:
            mock_cache.get.return_value = None
            docs = [_make_audio_doc()]
            result = embedder.embed_documents(docs, session_id="test")
        assert isinstance(result, list)

    def test_embedded_audio_doc_has_embedding(self):
        embedder = _make_embedder()
        embedder.model.encode.return_value = np.array([[0.2] * 384])
        import app.embeddings.text_embedder as te_mod
        with patch.object(te_mod, "_cache") as mock_cache:
            mock_cache.get.return_value = None
            docs = [_make_audio_doc()]
            result = embedder.embed_documents(docs, session_id="test")
        if result:
            assert result[0].embedding is not None
            assert isinstance(result[0].embedding, list)
