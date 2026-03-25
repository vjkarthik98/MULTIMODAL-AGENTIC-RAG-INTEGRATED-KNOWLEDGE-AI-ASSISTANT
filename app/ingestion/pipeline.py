from app.ingestion.router import detect_modality

from app.ingestion.text_ingest import ingest as text_ingest
from app.ingestion.image_ingest import ingest as image_ingest
from app.ingestion.audio_ingest import ingest as audio_ingest
from app.ingestion.video_ingest import ingest as video_ingest

from app.ingestion.schema import IngestedDocument

from app.embeddings.text_embedder import TextEmbedder
from app.vectorstore.qdrant_store import QdrantVectorStore

from app.utils.logger import get_logger

logger = get_logger(__name__)


embedder = TextEmbedder()
vector_store = QdrantVectorStore()



def process_file(file_path: str):
    try:
        logger.info(f"Processing file: {file_path}")

        # Step 0: Basic Validation
        if not file_path:
            raise ValueError("Invalid file path")

        # Step 1: Detect modality
        modality = detect_modality(file_path)
        logger.info(f"Detected modality: {modality}")

        # Step 2: Ingestion(returns chunked documents)

        if modality == "text":
            documents = text_ingest(file_path)
        
        elif modality == "image": 
            documents = image_ingest(file_path)
        
        elif modality == "audio":
            documents = audio_ingest(file_path)
        
        elif modality == "video":
            documents = video_ingest(file_path)
        
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        # Validation after ingestion
        if not documents or len(documents) == 0:
            raise ValueError("No content extracted from file")
        
        logger.info(f"Chunks created: {len(documents)}")
        
        # Step 3: Embed all chunks (batch)
        documents = embedder.embed_documents(documents)

        # Validation after embedding
        for doc in documents:
            if not hasattr(doc, "embedding") or doc.embedding is None:
                raise ValueError("Embedding failed for one or more chunks")
            
            if not isinstance(doc.embedding, list):
                raise ValueError("Invalid embedding format")
            
        logger.info("Embedding completed")

        # Step 4: Prepare Qdrant payload
        docs_for_qdrant = [
            {
                "text": doc.text,
                "embedding": doc.embedding,
                "metadata": doc.metadata
            }
            for doc in documents
        ]

        # Step 5: Store in Qdrant
        vector_store.insert_documents(docs_for_qdrant)
        logger.info("Stored in Qdrant successfully")

        # Step 6: Structured response 
        return {
            "chunks": len(documents),
            "status": "success",
            "details": {
                "modality": modality,
                "embedding_done": True,
                "stored_in_qdrant": True
            }
        }
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")

        return {
            "chunks": 0,
            "status": "failed",
            "details": {
                "error": str(e)
            }
        }
