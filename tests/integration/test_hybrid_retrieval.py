import os
from app.ingestion.text_ingest import ingest
from app.embeddings.text_embedder import TextEmbedder
from app.vectorstore.qdrant_store import QdrantVectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

def test_hybrid_retrieval():

    test_file = "test_hybrid.txt"
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Hybrid retrieval test document. " * 50)

    try:
        session_id = "hybrid_test"

        docs = ingest(test_file, session_id=session_id)

        embedder = TextEmbedder()

        docs = embedder.embed_documents(docs, session_id=session_id)

        store = QdrantVectorStore()
        store.insert_documents(docs)

        bm25 = BM25Retriever()
        bm25.build_index(docs)

        hybrid = HybridRetriever(bm25, store, embedder)

        results = hybrid.search(
            query="hybrid test",
            session_id=session_id,
            top_k=5
        )
        assert len(results) > 0
        assert "text" in results[0]

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)