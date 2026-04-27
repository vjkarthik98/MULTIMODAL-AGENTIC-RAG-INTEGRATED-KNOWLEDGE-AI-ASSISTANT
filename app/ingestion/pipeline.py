import time
from typing import Dict

from app.core.config import settings
from app.embeddings.multimodal_embedder import MultimodalEmbedder
from app.ingestion.chunking import chunk_documents
from app.ingestion.router import route_ingestion
from app.retrieval.bm25_retriever import BM25Retriever
from app.utils.logger import get_logger
from app.vectorstore.qdrant_store import QdrantVectorStore


logger = get_logger(__name__)


class _UnavailableVectorStore:
    def insert_documents(self, documents):
        raise RuntimeError("Vector store unavailable")

    def insert_vision_documents(self, documents):
        raise RuntimeError("Vector store unavailable")


class IngestionPipeline:
    def __init__(self):
        try:
            self.vector_store = QdrantVectorStore()
        except Exception as e:
            logger.warning("[IngestionPipeline] vector store unavailable | %s", str(e))
            self.vector_store = _UnavailableVectorStore()

        self.bm25 = BM25Retriever()
        self.embedder = MultimodalEmbedder()

        self.max_chunks = settings.MAX_CHUNKS
        self.batch_size = settings.INGESTION_BATCH_SIZE

    def _deduplicate(self, documents):
        seen = set()
        unique = []

        for d in documents:
            key = (
                getattr(d, "text", "")[:100],
                str(getattr(d, "structure", {}).get("doc_id")),
                str(getattr(d, "chunk_id", "")),
            )

            if key not in seen:
                seen.add(key)
                unique.append(d)

        return unique
    
    def _validate_embeddings(self, docs):
        valid = []
        invalid = 0

        for d in docs:
            emb = getattr(d, "embedding", None)

            if isinstance(emb, list) and len(emb) > 0:
                valid.append(d)

            else:
                invalid += 1

        if invalid > 0:
            logger.warning(
                "[IngestionPipeline] invalid embeddings skipped=%s",
                invalid
            )

        return valid 

    def process_file(self, file_path: str, session_id: str = "default") -> Dict:
        if not session_id:
            raise ValueError("session_id required")

        start = time.time()

        try:
            logger.info(
                "[IngestionPipeline][START] session_id=%s | file=%s",
                session_id,
                file_path
            )

            # STEP 1: INGEST
            documents = route_ingestion(file_path, session_id=session_id)
            if not documents:
                raise ValueError("No documents from ingestion")

            # STEP 2: CHUNK
            chunked_documents = chunk_documents(documents)

            if len(chunked_documents) > self.max_chunks:
                logger.warning(
                    "[IngestionPipeline] chunk limit applied %s -> %s",
                    len(chunked_documents),
                    self.max_chunks
                )
                chunked_documents = chunked_documents[:self.max_chunks]

            # Deduplicate early
            chunked_documents = self._deduplicate(chunked_documents)

            logger.info(
                "[IngestionPipeline] chunked=%s",
                len(chunked_documents)
            )

            # STEP 3: EMBEDDING
            text_docs, vision_docs = self.embedder.embed_documents(
                chunked_documents,
                session_id=session_id
            )

            if not text_docs and not vision_docs:
                raise ValueError("No embeddings generated")
            
            # Validate embeddings
            text_docs = self._validate_embeddings(text_docs)
            vision_docs = self._validate_embeddings(vision_docs)

            total_embeddings = len(text_docs) + len(vision_docs)

            if total_embeddings == 0:
                raise ValueError("All embeddings invalid")
            
            logger.info(
                "[IngestionPipeline] embeddings | text=%s vision=%s",
                len(text_docs),
                len(vision_docs)
            )

            # STEP 4: BUILD BM25
            try:
                self.bm25.build_index(chunked_documents)
                logger.info(
                    "[IngestionPiepline] BM25 index built | docs=%s",
                    len(chunked_documents)
                )

            except Exception as e:
                logger.error("[IngestedPipeline] BM25 build failed | %s", str(e))

            # STEP 5: STORE TEXT (BATCHED)
            if text_docs:
                for i in range(0, len(text_docs), self.batch_size):
                    batch = text_docs[i:i + self.batch_size]

                    try:
                        self.vector_store.insert_documents(batch)
                    except Exception as e:
                        logger.error("[IngestionPipeline] text insert failed | %s, str(e)")

            # STEP 6: STORE VISION
            if vision_docs:
                for i in range(0, len(vision_docs), self.batch_size):
                    batch = vision_docs[i:i + self.batch_size]

                    try:
                        self.vector_store.insert_documents(batch)
                    except Exception as e:
                        logger.error("[IngestionPipeline] vision insert failed | %s, str(e)")

            latency = round(time.time() - start, 2)

            total = len(text_docs) + len(vision_docs)

            logger.info(
                "[IngestionPipeline][SUCCESS] session_id=%s | stored=%s | latency=%ss",
                session_id,
                total,
                latency
            )

            return {
                "chunks": len(chunked_documents),
                "status": "success",
                "details": {
                    "session_id": session_id,
                    "text_documents": len(text_docs),
                    "vision_documents": len(vision_docs),
                    "stored": total,
                    "latency": latency
                }
            }

        except Exception as e:
            logger.error(
                "[IngestionPipeline][FAILED] session_id=%s | error=%s",
                session_id,
                str(e)
            )

            return {
                "chunks": 0,
                "status": "failed",
                "details": {
                    "error": str(e),
                    "session_id": session_id
                }
            }


# Singleton 
pipeline = IngestionPipeline()


def process_file(file_path: str, session_id: str = "default"):
    return pipeline.process_file(file_path, session_id)