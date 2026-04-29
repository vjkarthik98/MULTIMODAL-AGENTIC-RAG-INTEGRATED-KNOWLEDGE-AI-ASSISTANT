from typing import Dict, List
import time

from app.core.config import settings
from app.core.model_loader import model_loader
from app.chunking.chunker import chunk_documents
from app.ingestion.router import route_ingestion
from app.utils.logger import get_logger
from app.core.infra_registry import infra


logger = get_logger(__name__)


# FALLBACK VECTOR STORE
class _UnavailableVectorStore:
    def insert_documents(self, documents):
        raise RuntimeError("VECTOR STORE UNAVAILABLE")



# BATCHED EMBEDDING
def _batched_embedding(embedder, docs, session_id):
    batch_size = 16  

    results = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]

        try:
            embedded = embedder.embed_documents(batch, session_id=session_id)
            results.extend(embedded)
        except Exception as e:
            logger.warning("[Embedding][BatchFailed] %s", str(e))

    return results


# INGESTION PIPELINE
class IngestionPipeline:

    def __init__(self):

        # SHARED INFRA
        try:
            self.vector_store = infra.get_vector_store()
        except Exception as e:
            logger.warning("[IngestionPipeline] VECTOR STORE UNAVAILABLE | %s", str(e))
            self.vector_store = _UnavailableVectorStore()

        self.bm25 = infra.get_bm25()

        # CONFIG
        self.max_chunks = settings.MAX_CHUNKS
        self.batch_size = settings.INGESTION_BATCH_SIZE

    
    # DEDUPLICATION
    def _deduplicate(self, documents: List):
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

    
    # FILTER EMPTY CHUNKS
    def _filter_valid_chunks(self, documents: List):
        return [d for d in documents if getattr(d, "text", "").strip()]

    
    # VALIDATE EMBEDDINGS
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
            logger.warning("[IngestionPipeline] INVALID EMBEDDINGS SKIPPED=%s", invalid)

        return valid

    
    # MAIN PIPELINE
    def process_file(self, file_path: str, session_id: str = "default") -> Dict:

        if not session_id:
            raise ValueError("SESSION_ID REQUIRED")

        start = time.time()

        logger.info(
            "[IngestionPipeline][START] session_id=%s | file=%s",
            session_id,
            file_path
        )

        try:
            # STEP 1: INGESTION
            documents = route_ingestion(file_path, session_id=session_id)

            if not documents:
                raise ValueError("NO DOCUMENTS FROM INGESTION")

            logger.info("[IngestionPipeline] INGESTED=%s", len(documents))

            # STEP 2: CHUNKING
            chunked_documents = chunk_documents(documents)
            chunked_documents = self._filter_valid_chunks(chunked_documents)

            if not chunked_documents:
                raise ValueError("CHUNKING FAILED: NO VALID CHUNKS")

            if len(chunked_documents) > self.max_chunks:
                logger.warning(
                    "[IngestionPipeline] CHUNK LIMIT %s -> %s",
                    len(chunked_documents),
                    self.max_chunks
                )
                chunked_documents = chunked_documents[:self.max_chunks]

            chunked_documents = self._deduplicate(chunked_documents)

            logger.info("[IngestionPipeline] CHUNKED=%s", len(chunked_documents))

            # STEP 3: EMBEDDING 
            embedder = model_loader.get_embedder()

            embedded_docs = _batched_embedding(
                embedder,
                chunked_documents,
                session_id
            )

            embedded_docs = self._validate_embeddings(embedded_docs)

            if not embedded_docs:
                raise ValueError("EMBEDDING FAILED: NO VALID EMBEDDINGS")

            logger.info("[IngestionPipeline] EMBEDDINGS=%s", len(embedded_docs))

            # STEP 4: BM25 
            try:
                if chunked_documents:
                    # use incremental update instead of overwrite
                    if hasattr(self.bm25, "add_documents"):
                        self.bm25.add_documents(chunked_documents)
                    else:
                        # fallback 
                        logger.warning("[IngestionPipeline] BM25 fallback to build_index")
                        self.bm25.build_index(chunked_documents)

                    logger.info("[IngestionPipeline] BM25 UPDATED | docs=%s", len(chunked_documents))

                else:
                    logger.warning("[IngestionPipeline] BM25 skipped (no docs)")

            except Exception as e:
                logger.error("[IngestionPipeline] BM25 FAILED | %s", str(e))

            # STEP 5: STORE IN QDRANT
            total_inserted = 0

            for i in range(0, len(embedded_docs), self.batch_size):
                batch = embedded_docs[i:i + self.batch_size]

                try:
                    self.vector_store.insert_documents(batch)
                    total_inserted += len(batch)
                except Exception as e:
                    logger.error("[IngestionPipeline] INSERT FAILED | %s", str(e))
                    raise RuntimeError("VECTOR STORE INSERT FAILED")

            latency = round(time.time() - start, 2)

            logger.info(
                "[IngestionPipeline][SUCCESS] session_id=%s | stored=%s | latency=%ss",
                session_id,
                total_inserted,
                latency
            )

            return {
                "chunks": len(chunked_documents),
                "status": "success",
                "details": {
                    "session_id": session_id,
                    "stored": total_inserted,
                    "latency": latency,
                }
            }

        except Exception as e:
            logger.error(
                "[IngestionPipeline][FAILED] session_id=%s | error=%s",
                session_id,
                str(e)
            )
            raise RuntimeError(f"Ingestion failed: {str(e)}")


# SINGLETON
pipeline = IngestionPipeline()


def process_file(file_path: str, session_id: str = "default"):
    return pipeline.process_file(file_path, session_id)

