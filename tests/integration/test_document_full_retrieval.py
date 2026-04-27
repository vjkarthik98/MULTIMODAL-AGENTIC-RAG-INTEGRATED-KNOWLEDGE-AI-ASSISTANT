import pytest
from app.retrieval.retriever import Retriever


def test_full_retrieval_pipeline():
    retriever = Retriever()

    results = retriever.retrieval(
        query="What is AI?",
        session_id="test",
        top_k=3
    )

    assert isinstance(results, list)

    if results:
        assert "text" in results[0]
        assert "score" in results[0]