import pytest
from app.embeddings.text_embedder import TextEmbedder
from app.ingestion.schema import IngestedDocument


def test_audio_embedding():
    docs = [
        IngestedDocument(
            text="hello world",
            modality="audio",
            subtype="speech",
            source_type="audio",
            source="test.wav",
            page=None,
            chunk_id=0,
            structure={
                "start_time": 0,
                "end_time": 2,
                "session_id": "test"
            }
        )
    ]

    embedder = TextEmbedder()

    embedded_docs = embedder.embed_documents(docs, session_id="test")

    assert isinstance(embedded_docs, list)
    assert len(embedded_docs) == 1

    doc = embedded_docs[0]

    assert doc.embedding is not None
    assert isinstance(doc.embedding, list)
    assert len(doc.embedding) > 0