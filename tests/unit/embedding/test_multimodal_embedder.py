from unittest.mock import MagicMock, patch
import pytest

from app.ingestion.schema import IngestedDocument


def _make_doc(modality="image", embedding_space="vision"):
    return IngestedDocument(
        text="A white image showing test content.",
        modality=modality,
        source_type=modality,
        source="test.png",
        structure={"session_id": "test", "embedding_space": embedding_space},
    )


def _make_embedder():
    from app.embeddings.base_embedder import MultimodalEmbedder

    mock_text_embedder  = MagicMock()
    mock_image_embedder = MagicMock()

    embedder = MultimodalEmbedder.__new__(MultimodalEmbedder)
    embedder.text_embedder      = mock_text_embedder
    embedder.image_embedder     = mock_image_embedder
    embedder.max_docs           = 1000
    embedder.batch_size         = 32
    embedder._text_model_name   = "all-MiniLM-L6-v2"
    embedder._vision_model_name = "google/siglip-so400m-patch14-384"
    return embedder


class TestMultimodalEmbedder:

    def test_embed_documents_returns_tuple(self):
        embedder = _make_embedder()
        docs = [_make_doc()]
        text_docs, vision_docs = embedder.embed_documents(docs, session_id="test")
        assert isinstance(text_docs, list)
        assert isinstance(vision_docs, list)

    def test_text_modality_goes_to_text_embedder(self):
        embedder = _make_embedder()
        text_doc = IngestedDocument(
            text="Regular text content for testing embedding.",
            modality="text",
            source_type="text",
            source="test.txt",
            structure={"embedding_space": "text"},
        )

        embedder.text_embedder.embed_documents.return_value = [text_doc]
        text_docs, vision_docs = embedder.embed_documents([text_doc], session_id="test")
        assert isinstance(text_docs, list)
        assert isinstance(vision_docs, list)

    def test_health_check_returns_dict(self):
        embedder = _make_embedder()
        embedder.text_embedder.health_check.return_value = {"status": "ok"}
        embedder.image_embedder.health_check.return_value = {"status": "ok"}
        result = embedder.health_check()
        assert isinstance(result, dict)

    def test_empty_docs_returns_empty_lists(self):
        embedder = _make_embedder()
        text_docs, vision_docs = embedder.embed_documents([], session_id="test")
        assert text_docs == []
        assert vision_docs == []
