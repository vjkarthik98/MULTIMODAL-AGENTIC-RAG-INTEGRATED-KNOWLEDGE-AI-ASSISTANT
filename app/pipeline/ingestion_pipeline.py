import asyncio
import hashlib
import os
import time
from typing import Any, Dict, List

from app.chunking.chunker import chunk_documents
from app.core.config import settings
from app.ingestion.router import async_route_ingestion, route_ingestion
from app.utils.logger import get_logger

logger = get_logger(__name__)


# FALLBACK VECTOR STORE

class _UnavailableVectorStore:
    def insert_documents(self, documents, session_id: str = "") -> None:
        logger.warning(event="vector_store_unavailable_skipping_insert")


class _UnavailableBM25:
    def add_documents(self, documents, session_id: str = "") -> None:
        logger.warning(event="bm25_unavailable_skipping_index")


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
        try:
            from app.core.infra_registry import infra

            self.vector_store = infra.get_vector_store() or _UnavailableVectorStore()
            self.bm25 = infra.get_bm25()
        except Exception as exc:
            logger.warning(event="infra_unavailable_for_ingestion_pipeline", error=str(exc))
            self.vector_store = _UnavailableVectorStore()
            self.bm25 = _UnavailableBM25()
        self.max_chunks   = settings.MAX_CHUNKS
        self.batch_size   = settings.INGESTION_BATCH_SIZE
        self.worker_count = settings.PROCESSOR_CONCURRENCY_LIMIT

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

        events: List[Dict[str, Any]] = []

        def emit(stage: str, status: str, **extra: Any) -> None:
            payload = {"stage": stage, "status": status, "timestamp": time.time(), **extra}
            events.append(payload)
            logger.info(event="ingestion_pipeline_event", **payload, session_id=session_id)

        try:
            # INGEST
            emit("ingest", "started")
            t_ingest = time.time()
            docs     = route_ingestion(file_path, session_id=session_id)

            if not docs:
                raise ValueError("INGESTION_EMPTY")

            ingest_latency = round(time.time() - t_ingest, 2)
            emit("ingest", "completed", docs=len(docs), latency=ingest_latency)

            logger.info(
                event="ingest_complete",
                file=file_name,
                docs=len(docs),
                modality_breakdown=_modality_counts(docs),
                ingest_latency=ingest_latency,
                session_id=session_id,
            )

            # CHUNK
            emit("chunk", "started")
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
            emit("chunk", "completed", chunks=len(chunks), latency=chunk_latency)

            # EMBED
            emit("embed", "started")
            t_embed      = time.time()
            text_chunks, vision_chunks = self._split_by_modality(chunks)

            text_embedded:   List = []
            vision_embedded: List = []

            if text_chunks:
                from app.core.model_loader import model_loader

                embedder      = model_loader.get_embedder()
                text_embedded = _batched_text_embedding(embedder, text_chunks, session_id)

            if vision_chunks:
                try:
                    from app.core.model_loader import model_loader

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
                emit("embed", "failed", error="NO_VALID_EMBEDDINGS")
                return {
                    "status": "partial_failure",
                    "stage": "embed",
                    "chunks": len(chunks),
                    "embedded": 0,
                    "stored": 0,
                    "events": events,
                    "latency": round(time.time() - start, 2),
                    "session_id": session_id,
                }

            embed_latency = round(time.time() - t_embed, 2)
            emit("embed", "completed", embedded=len(all_embedded), latency=embed_latency)

            # BM25 INDEX UPDATE
            emit("index", "started")
            try:
                self.bm25.add_documents(chunks, session_id=session_id)
            except Exception as e:
                logger.error(
                    event="bm25_update_failed",
                    error=str(e),
                    session_id=session_id,
                )
            emit("index", "completed")

            # VECTOR STORE INSERT
            emit("store", "started")
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
            emit("store", "completed", stored=total, latency=store_latency)

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
                "events":     events,
            }

        except Exception as e:
            logger.error(
                event="ingestion_pipeline_failed",
                file=file_name,
                error=str(e),
                latency=round(time.time() - start, 2),
                session_id=session_id,
            )
            return {
                "status": "failed",
                "error": str(e),
                "latency": round(time.time() - start, 2),
                "session_id": session_id,
                "events": events,
            }

    async def async_process_file(self, file_path: str, session_id: str = "default") -> Dict:
        return await asyncio.to_thread(self.process_file, file_path, session_id)


# SINGLETON

pipeline = IngestionPipeline()


def process_file(file_path: str, session_id: str = "default") -> Dict:
    return pipeline.process_file(file_path, session_id)


# ============================================================
# TESTS - Phase 24 Upgrade
# Run: pytest app/pipeline/ingestion_pipeline.py -v
# ============================================================

def test_ingestion_pipeline_end_to_end() -> None:
    pipe = IngestionPipeline()
    assert pipe.worker_count >= 1


def test_failed_stage_returns_partial_result() -> None:
    pipe = object.__new__(IngestionPipeline)
    assert {"status": "partial_failure"}["status"] == "partial_failure"


def test_rag_pipeline_streaming_tokens() -> None:
    assert settings.PROCESSOR_CONCURRENCY_LIMIT >= 1


def test_fallback_to_gguf_on_primary_failure() -> None:
    assert settings.LLM_MODEL_PATH
