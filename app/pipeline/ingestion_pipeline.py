import hashlib
import os
import time
from typing import Dict, List

from app.chunking.chunker import chunk_documents
from app.core.config import settings
from app.core.infra_registry import infra
from app.core.model_loader import model_loader
from app.ingestion.router import route_ingestion
from app.utils.logger import get_logger

logger = get_logger(__name__)


# FALLBACK VECTOR STORE

class _UnavailableVectorStore:
    def insert_documents(self, documents, session_id: str = "") -> None:
        logger.warning(event="vector_store_unavailable_skipping_insert")


# HASH

def _hash(text: str, doc_id: str, chunk_id: str) -> str:
    base = f"{text[:100]}|{doc_id}|{chunk_id}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# MODALITY BREAKDOWN

def _modality_counts(docs: List) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for d in docs:
        m = getattr(d, "modality", "unknown")
        counts[m] = counts.get(m, 0) + 1
    return counts


# BATCHED TEXT EMBEDDING

def _batched_text_embedding(embedder, docs: List, session_id: str) -> List:
    batch_size = settings.INGESTION_BATCH_SIZE
    results    = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        try:
            embedded = embedder.embed_documents(batch, session_id=session_id)
            if embedded:
                results.extend(embedded)
        except Exception as e:
            logger.warning(
                event="text_embedding_batch_failed",
                batch_start=i,
                error=str(e),
                session_id=session_id,
            )

    return results


# PIPELINE

class IngestionPipeline:

    def __init__(self) -> None:
        self.vector_store = infra.get_vector_store() or _UnavailableVectorStore()
        self.bm25         = infra.get_bm25()
        self.max_chunks   = settings.MAX_CHUNKS
        self.batch_size   = settings.INGESTION_BATCH_SIZE

    # DEDUP

    def _deduplicate(self, docs: List) -> List:
        seen:   set  = set()
        unique: List = []

        for d in docs:
            text      = getattr(d, "text", "")
            structure = getattr(d, "structure", {}) or {}

            h = _hash(
                text,
                structure.get("doc_id", ""),
                str(getattr(d, "chunk_id", "")),
            )

            if h in seen:
                continue

            seen.add(h)
            unique.append(d)

        return unique

    # FILTER VALID CHUNKS

    def _valid_chunks(self, docs: List) -> List:
        return [
            d for d in docs
            if getattr(d, "text", "").strip()
            and len(getattr(d, "text", "").strip()) >= settings.CHUNK_MIN_SIZE
        ]

    # FILTER VALID EMBEDDINGS

    def _valid_embeddings(self, docs: List) -> List:
        valid   = []
        invalid = 0

        for d in docs:
            emb = getattr(d, "embedding", None)
            if isinstance(emb, list) and len(emb) in (
                settings.TEXT_EMBEDDING_DIM,
                settings.VISION_EMBEDDING_DIM,
            ):
                valid.append(d)
            else:
                invalid += 1

        if invalid:
            logger.warning(
                event="invalid_embeddings_skipped",
                count=invalid,
            )

        return valid

    # SEPARATE BY MODALITY

    def _split_by_modality(self, docs: List):
        text_docs:   List = []
        vision_docs: List = []

        for d in docs:
            space = (getattr(d, "structure", {}) or {}).get("embedding_space", "text")
            if space == "vision":
                vision_docs.append(d)
            else:
                text_docs.append(d)

        return text_docs, vision_docs

    # MAIN

    def process_file(
        self,
        file_path: str,
        session_id: str = "default",
    ) -> Dict:

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        file_name = os.path.basename(file_path)
        start     = time.time()

        try:
            # INGEST
            t_ingest = time.time()
            docs     = route_ingestion(file_path, session_id=session_id)

            if not docs:
                raise ValueError("INGESTION_EMPTY")

            ingest_latency = round(time.time() - t_ingest, 2)

            logger.info(
                event="ingest_complete",
                file=file_name,
                docs=len(docs),
                modality_breakdown=_modality_counts(docs),
                ingest_latency=ingest_latency,
                session_id=session_id,
            )

            # CHUNK
            t_chunk = time.time()
            chunks  = chunk_documents(docs)
            chunks  = self._valid_chunks(chunks)

            if not chunks:
                raise ValueError("NO_VALID_CHUNKS")

            if len(chunks) > self.max_chunks:
                chunks = chunks[:self.max_chunks]
                logger.warning(
                    event="chunk_limit_applied",
                    limit=self.max_chunks,
                    session_id=session_id,
                )

            chunks = self._deduplicate(chunks)
            chunk_latency = round(time.time() - t_chunk, 2)

            # EMBED
            t_embed      = time.time()
            text_chunks, vision_chunks = self._split_by_modality(chunks)

            text_embedded:   List = []
            vision_embedded: List = []

            if text_chunks:
                embedder      = model_loader.get_embedder()
                text_embedded = _batched_text_embedding(embedder, text_chunks, session_id)

            if vision_chunks:
                try:
                    multimodal       = model_loader.get_multimodal_embedder()
                    _, vis_embedded  = multimodal.embed_documents(vision_chunks, session_id=session_id)
                    vision_embedded  = vis_embedded
                except Exception as e:
                    logger.warning(
                        event="vision_embedding_failed",
                        error=str(e),
                        session_id=session_id,
                    )

            all_embedded = text_embedded + vision_embedded
            all_embedded = self._valid_embeddings(all_embedded)

            if not all_embedded:
                raise ValueError("NO_VALID_EMBEDDINGS")

            embed_latency = round(time.time() - t_embed, 2)

            # BM25 INDEX UPDATE
            try:
                self.bm25.add_documents(chunks, session_id=session_id)
            except Exception as e:
                logger.error(
                    event="bm25_update_failed",
                    error=str(e),
                    session_id=session_id,
                )

            # VECTOR STORE INSERT
            t_store = time.time()
            total   = 0

            for i in range(0, len(all_embedded), self.batch_size):
                batch = all_embedded[i:i + self.batch_size]
                try:
                    self.vector_store.insert_documents(batch, session_id=session_id)
                    total += len(batch)
                except Exception as e:
                    logger.error(
                        event="vector_insert_batch_failed",
                        batch_start=i,
                        error=str(e),
                        session_id=session_id,
                    )

            store_latency = round(time.time() - t_store, 2)
            total_latency = round(time.time() - start, 2)

            logger.info(
                event="ingestion_pipeline_success",
                file=file_name,
                chunks=len(chunks),
                embedded=len(all_embedded),
                stored=total,
                modality_breakdown=_modality_counts(chunks),
                ingest_latency=ingest_latency,
                chunk_latency=chunk_latency,
                embed_latency=embed_latency,
                store_latency=store_latency,
                total_latency=total_latency,
                session_id=session_id,
            )

            return {
                "status":     "success",
                "chunks":     len(chunks),
                "embedded":   len(all_embedded),
                "stored":     total,
                "latency":    total_latency,
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(
                event="ingestion_pipeline_failed",
                file=file_name,
                error=str(e),
                latency=round(time.time() - start, 2),
                session_id=session_id,
            )
            raise RuntimeError(f"INGESTION_FAILED: {str(e)}")


# SINGLETON

pipeline = IngestionPipeline()


def process_file(file_path: str, session_id: str = "default") -> Dict:
    return pipeline.process_file(file_path, session_id)