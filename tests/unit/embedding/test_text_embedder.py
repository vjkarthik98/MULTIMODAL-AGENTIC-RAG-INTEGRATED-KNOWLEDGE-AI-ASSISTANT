import pytest
from app.embeddings.text_embedder import TextEmbedder
from app.ingestion.schema import IngestedDocument


def test_embed_documents_basic():
    embedder = TextEmbedder()

    docs = [
        IngestedDocument(
            text="Sample text",
            modality="text",
            subtype="page",
            source_type="pdf",
            structure={"session_id": "test"}
        )
    ]

    result = embedder.embed_documents(docs, session_id="test")

    assert result[0].embedding is not None