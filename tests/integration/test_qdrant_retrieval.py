import os
from app.ingestion.text_ingest import ingest
from app.embeddings.text_embedder import TextEmbedder
from app.vectorstore.qdrant_store import QdrantVectorStore


def test_qdrant_insert_and_search():

 
    # Create test file
 
    test_file = "test_qdrant_sample.txt"

    with open(test_file, "w", encoding="utf-8") as f:
        f.write("This is a test document for qdrant retrieval testing. " * 50)

    try:
        session_id = "qdrant_test"

     
        # Ingestion
     
        documents = ingest(test_file, session_id=session_id)
        assert len(documents) > 0

     
        # Embedding
     
        embedder = TextEmbedder()
        documents = embedder.embed_documents(documents, session_id=session_id)

        # Ensure embeddings exist
        for doc in documents:
            assert doc.embedding is not None

     
        # Store in Qdrant
     
        store = QdrantVectorStore()
        store.insert_documents(documents)

     
        # Query
     
        query_vector = embedder.embed_query(
            "test document", session_id=session_id
        )

        results = store.search_text(
            query_vector,
            limit=3,
            session_id=session_id
        )

     
        # Validate results
     
        assert len(results) > 0

        first_result = results[0]

        assert "text" in first_result
        assert "score" in first_result
        assert "metadata" in first_result

        # Check metadata integrity
        assert "session_id" in first_result["metadata"]

    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)