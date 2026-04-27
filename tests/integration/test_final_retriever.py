from app.retrieval.retriever import Retriever


def test_retriever_basic():
    retriever = Retriever()

    query = "What is artificial intelligence?"

    results = retriever.retrieval(query, top_k=5)

    print("\n--- RETRIEVER OUTPUT-")

    for i, doc in enumerate(results):
        print(f"\nResult {i+1}")
        print("Score:", doc["score"])
        print("Text:", doc["text"][:200])
        print("Metadata:", doc.get("metadata", {}))

    # Basic assertions
    assert isinstance(results, list)
    assert len(results) > 0
    assert "text" in results[0]