from typing import Dict, List
import time
import hashlib

from app.core.config import settings
from app.core.model_loader import model_loader
from app.chunking.chunker import chunk_documents
from app.ingestion.router import route_ingestion
from app.utils.logger import get_logger
from app.core.infra_registry import infra

logger = get_logger(__name__)


#  FALLBACK VECTOR STORE 
class _UnavailableVectorStore:
    def insert_documents(self, documents):
        raise RuntimeError("VECTOR_STORE_UNAVAILABLE")


#  HASH 
def _hash(text: str, doc_id: str, chunk_id: str):
    base = f"{text[:100]}|{doc_id}|{chunk_id}"
    return hashlib.sha256(base.encode()).hexdigest()


#  BATCHED EMBEDDING 
def _batched_embedding(embedder, docs, session_id):

    batch_size = settings.INGESTION_BATCH_SIZE
    results = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]

        try:
            # PASS DOCS 
            embedded_docs = embedder.embed_documents(batch, session_id=session_id)

            if not embedded_docs:
                continue

            results.extend(embedded_docs)

        except Exception as e:
            logger.warning(event="embedding_batch_failed", error=str(e))

    return results

#  PIPELINE 
class IngestionPipeline:

    def __init__(self):

        try:
            self.vector_store = infra.get_vector_store()
        except Exception as e:
            logger.warning(event="vector_store_unavailable", error=str(e))
            self.vector_store = _UnavailableVectorStore()

        self.bm25 = infra.get_bm25()

        self.max_chunks = settings.MAX_CHUNKS
        self.batch_size = settings.INGESTION_BATCH_SIZE

    #  DEDUP 
    def _deduplicate(self, docs: List):

        seen = set()
        unique = []

        for d in docs:
            text = getattr(d, "text", "")
            structure = getattr(d, "structure", {}) or {}

            h = _hash(
                text,
                structure.get("doc_id"),
                str(getattr(d, "chunk_id", ""))
            )

            if h in seen:
                continue

            seen.add(h)
            unique.append(d)

        return unique

    #  FILTER 
    def _valid_chunks(self, docs: List):
        return [d for d in docs if getattr(d, "text", "").strip()]

    #  EMB VALID 
    def _valid_embeddings(self, docs):

        valid = []
        invalid = 0

        for d in docs:
            emb = getattr(d, "embedding", None)

            if isinstance(emb, list) and len(emb) == settings.TEXT_EMBEDDING_DIM:
                valid.append(d)
            else:
                invalid += 1

        if invalid:
            logger.warning(event="invalid_embeddings", count=invalid)

        return valid
    #  MAIN 
    def process_file(self, file_path: str, session_id: str = "default") -> Dict:

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        start = time.time()

        try:
            #  INGEST 
            docs = route_ingestion(file_path, session_id=session_id)

            if not docs:
                raise ValueError("INGESTION_EMPTY")

            #  CHUNK 
            chunks = chunk_documents(docs)
            chunks = self._valid_chunks(chunks)

            if not chunks:
                raise ValueError("NO_VALID_CHUNKS")

            if len(chunks) > self.max_chunks:
                chunks = chunks[:self.max_chunks]
                logger.warning(event="chunk_limit_applied")

            chunks = self._deduplicate(chunks)

            #  EMBED 
            embedder = model_loader.get_embedder()

            embedded = _batched_embedding(embedder, chunks, session_id)
            embedded = self._valid_embeddings(embedded)

            if not embedded:
                raise ValueError("NO_VALID_EMBEDDINGS")

            #  BM25 
            try:
                if hasattr(self.bm25, "add_documents"):
                    self.bm25.add_documents(chunks)
                else:
                    self.bm25.build_index(chunks)
            except Exception as e:
                logger.error(event="bm25_failed", error=str(e))

            #  STORE 
            total = 0

            for i in range(0, len(embedded), self.batch_size):
                batch = embedded[i:i + self.batch_size]

                try:
                    self.vector_store.insert_documents(batch)
                    total += len(batch)
                except Exception as e:
                    logger.error(event="vector_insert_failed", error=str(e))
                    raise RuntimeError("VECTOR_INSERT_FAILED")

            latency = round(time.time() - start, 2)

            logger.info(
                event="ingestion_success",
                chunks=len(chunks),
                stored=total,
                latency=latency
            )

            return {
                "status": "success",
                "chunks": len(chunks),
                "stored": total,
                "latency": latency,
                "session_id": session_id
            }

        except Exception as e:
            logger.error(event="ingestion_failed", error=str(e))
            raise RuntimeError(f"INGESTION_FAILED: {str(e)}")


#  SINGLETON 
pipeline = IngestionPipeline()


def process_file(file_path: str, session_id: str = "default"):
    return pipeline.process_file(file_path, session_id)