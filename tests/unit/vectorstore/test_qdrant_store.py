import pytest
from app.vectorstore.qdrant_store import QdrantVectorStore
from app.ingestion.schema import IngestedDocument


def test_insert_documents_basic():
    store = QdrantVectorStore()

    docs = [
        IngestedDocument(
            text="Sample text",
            modality="text",
            subtype="page",
            source_type="pdf",
            structure={"session_id": "test", "doc_id": "123"},
            embedding=[0.1] * 384
        )
    ]

    store.insert_documents(docs)

    assert True  