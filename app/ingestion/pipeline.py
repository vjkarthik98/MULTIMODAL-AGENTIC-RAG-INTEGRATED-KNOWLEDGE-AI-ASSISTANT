from app.embeddings.multimodal_embedder import MultimodalEmbedder
from app.ingestion.chunking import chunk_documents
from app.ingestion.router import route_ingestion
from app.retrieval.bm25_retriever import BM25Retriever
from app.utils.logger import get_logger
from app.vectorstore.qdrant_store import QdrantVectorStore


logger = get_logger(__name__)


class _UnavailableVectorStore:
    def insert_documents(self, documents):
        raise RuntimeError("Qdrant vector store is unavailable")

    def insert_vision_documents(self, documents):
        raise RuntimeError("Qdrant vector store is unavailable")


try:
    vector_store = QdrantVectorStore()
except Exception as exc:  # pragma: no cover - depends on local infra
    logger.warning("[IngestionPipeline] Vector store unavailable during import | error=%s", exc)
    vector_store = _UnavailableVectorStore()


bm25 = BM25Retriever()
embedder = MultimodalEmbedder()


def process_file(file_path: str, session_id: str = "default"):
    try:
        logger.info("[IngestionPipeline][START] session_id=%s | file=%s", session_id, file_path)

        documents = route_ingestion(file_path, session_id=session_id)
        if not documents:
            raise ValueError("No documents returned from ingestion")

        chunked_documents = chunk_documents(documents)
        logger.info(
            "[IngestionPipeline] session_id=%s | docs_chunked=%s",
            session_id,
            len(chunked_documents),
        )

        text_documents, vision_documents = embedder.embed_documents(
            chunked_documents,
            session_id=session_id,
        )

        if not text_documents and not vision_documents:
            raise ValueError("Embedding pipeline produced no documents")

        if text_documents:
            if not hasattr(bm25, "all_documents"):
                bm25.all_documents = []
            bm25.all_documents.extend(text_documents)
            bm25.build_index(bm25.all_documents)
            vector_store.insert_documents(text_documents)

        if vision_documents:
            vector_store.insert_vision_documents(vision_documents)

        stored_count = len(text_documents) + len(vision_documents)
        logger.info(
            "[IngestionPipeline][SUCCESS] session_id=%s | stored=%s",
            session_id,
            stored_count,
        )

        return {
            "chunks": len(chunked_documents),
            "status": "success",
            "details": {
                "session_id": session_id,
                "text_documents": len(text_documents),
                "vision_documents": len(vision_documents),
                "stored": stored_count,
                "bm25_indexed": len(text_documents),
            },
        }

    except Exception as exc:
        logger.error("[IngestionPipeline][ERROR] session_id=%s | error=%s", session_id, exc)
        return {
            "chunks": 0,
            "status": "failed",
            "details": {
                "error": str(exc),
                "session_id": session_id,
            },
        }
