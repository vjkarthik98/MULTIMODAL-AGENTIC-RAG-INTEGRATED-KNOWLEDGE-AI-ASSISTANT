from app.ingestion.router import detect_modality

from app.ingestion.text_ingest import ingest as text_ingest
from app.ingestion.document_ingest import ingest as document_ingest
from app.ingestion.image_ingest import ingest as image_ingest
from app.ingestion.audio_ingest import ingest as audio_ingest
from app.ingestion.video_ingest import ingest as video_ingest

from app.core.model_loader import model_loader
from app.vectorstore.qdrant_store import QdrantVectorStore

from app.utils.logger import get_logger

logger = get_logger(__name__)

vector_store = QdrantVectorStore()


def process_file(file_path, session_id="default"):
    try:
        logger.info(f"[IngestionPipeline] session_id={session_id} | Processing file: {file_path}")

        # Step 0: Basic Validation
        if not file_path:
            raise ValueError("Invalid file path")

        # Step 1: Detect modality
        modality = detect_modality(file_path)
        logger.info(f"[IngestionPipeline] session_id={session_id} | Detected modality: {modality}")

        # Step 2: Ingestion
        if modality == "text":
            documents = text_ingest(file_path)

        elif modality == "document":
            documents = document_ingest(file_path)

        elif modality == "image":
            documents = image_ingest(file_path, session_id=session_id)

        elif modality == "audio":
            documents = audio_ingest(file_path)

        elif modality == "video":
            documents = video_ingest(file_path)

        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # Validation after ingestion
        if not documents or len(documents) == 0:
            raise ValueError("No content extracted from file")

        logger.info(f"[IngestionPipeline] session_id={session_id} | Chunks created: {len(documents)}")

        # Step 3: Embedding handling
        text_docs = []
        pre_embedded_docs = []

        for doc in documents:
            if getattr(doc, "embedding", None) is not None:
                pre_embedded_docs.append(doc)
            else:
                text_docs.append(doc)

        logger.debug(
            f"[IngestionPipeline] session_id={session_id} | Docs needing embedding: {len(text_docs)}"
        )

        # Embed only text-based docs
        if text_docs:
            embedder = model_loader.get_embedder()
            text_docs = embedder.embed_documents(text_docs)

        documents = pre_embedded_docs + text_docs

        # Validation after embedding
        for doc in documents:
            if not hasattr(doc, "embedding") or doc.embedding is None:
                raise ValueError("Embedding failed for one or more chunks")

            if not isinstance(doc.embedding, list):
                raise ValueError("Invalid embedding format")

        logger.info(f"[IngestionPipeline] session_id={session_id} | Embedding completed")

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

        logger.info(f"[IngestionPipeline] session_id={session_id} | Stored in Qdrant successfully")

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
        logger.error(
            f"[IngestionPipeline] session_id={session_id} | Pipeline failed: {str(e)}"
        )

        return {
            "chunks": 0,
            "status": "failed",
            "details": {
                "error": str(e)
            }
        }