import os
from app.ingestion.text_ingest import ingest
from app.embeddings.text_embedder import TextEmbedder


def test_ingestion_to_embedding_pipeline():

 
    # Create temporary test file
 
    test_file = "test_sample.txt"

    with open(test_file, "w", encoding="utf-8") as f:
        f.write("This is a test document for integration testing. " * 50)

    try:
        session_id = "integration_test"

     
        # Ingestion
     
        documents = ingest(test_file, session_id=session_id)

        assert len(documents) > 0

        # Validate metadata exists
        assert documents[0].structure is not None
        assert "char_start" in documents[0].structure

     
        # Embedding
     
        embedder = TextEmbedder()
        documents = embedder.embed_documents(documents, session_id=session_id)

     
        # Validate embeddings
     
        for doc in documents:
            assert doc.embedding is not None
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) > 0

    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)