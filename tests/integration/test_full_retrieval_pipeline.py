import os
from app.ingestion.text_ingest import ingest
from app.embeddings.text_embedder import TextEmbedder
from app.vectorstore.qdrant_store import QdrantVectorStore
from app.retrieval.retriever import Retriever 

def test_full_retrieval_pipeline():

    test_file = "test_retrieval.txt"

    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Retrieval pipeline test document." *50)

    try:
        session_id = "retrieval_test"


        # ingest
        docs = ingest(test_file, session_id=session_id)

        # embed
        embedder = TextEmbedder()
        docs = embedder.embed_documents(docs, session_id=session_id)
        
        # Store
        store = QdrantVectorStore()
        store.insert_documents(docs)

        # Retrieve (with reranking)
        retriever = Retriever()

        results = retriever.retrieval(
            query = "pipeline test",
            session_id = session_id,
            top_k=3
        )

        assert len(results) > 0

        first = results[0]

        assert "text" in first
        assert "metadata" in first
        assert "score" in first

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)