from app.utils.chunking import chunk_text
from app.ingestion.schema import IngestedDocument

import os
from datetime import datetime

from app.embeddings.text_embedder import TextEmbedder
from app.vectorstore.qdrant_store import QdrantVectorStore



def ingest(file_path: str) -> list[IngestedDocument]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    metadata = {
        "source": os.path.basename(file_path),
        "modality": "text",
        "ingestion_time": datetime.utcnow().isoformat()
    }

    chunks = chunk_text(text)

    return [
        IngestedDocument(
            text = chunk,
            metadata={
                **metadata,
                "chunk_id": i
            }
        )
        for i, chunk in enumerate(chunks)
    ]

def ingest_pipeline(file_path: str):
    # Step 1 : chunking (existing ingest function)
    documents = ingest(file_path)

    # Step 2 : embedding
    embedder = TextEmbedder()
    documents = embedder.embed_documents(documents)

    # Step 3 : convert to Qdrant format
    docs_for_qdrant = [
        {
            "text": doc.text,
            "embedding": doc.embedding,
            "metadata": doc.metadata
        }
        for doc in documents
    ]

    # Step 4: store in Qdrant
    store = QdrantVectorStore()
    store.insert_documents(docs_for_qdrant)

    return len(documents)

