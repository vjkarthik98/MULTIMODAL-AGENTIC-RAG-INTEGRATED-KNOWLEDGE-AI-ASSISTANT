from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.ingestion.schema import (
    DiskSpaceError,
    DuplicateFileError,
    EmptyFileError,
    FileTooLargeError,
    IngestedDocument,
    MalwareDetectedError,
    ProcessingResult,
    UniversalMetadata,
    UnsupportedMimeError,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# PROMETHEUS METRICS 

def _get_metrics():
    try:
        from prometheus_client import Counter, Gauge, Histogram
        ingestion_duration = Histogram(
            "file_ingestion_duration_seconds",
            "Ingestion duration by modality",
            ["modality"],
        )
        ingestion_errors = Counter(
            "file_ingestion_errors_total",
            "Ingestion errors by modality and error type",
            ["modality", "error_type"],
        )
        chunk_count = Histogram(
            "chunk_count_per_file",
            "Chunks produced per file",
            ["modality"],
        )
        embedding_latency = Histogram(
            "embedding_latency_seconds",
            "Embedding latency by model",
            ["model"],
        )
        queue_depth = Gauge(
            "queue_depth",
            "Ingestion queue backlog",
        )
        pii_redacted = Counter(
            "pii_entities_redacted_total",
            "PII entities redacted",
            ["entity_type"],
        )
        return {
            "ingestion_duration": ingestion_duration,
            "ingestion_errors":   ingestion_errors,
            "chunk_count":        chunk_count,
            "embedding_latency":  embedding_latency,
            "queue_depth":        queue_depth,
            "pii_redacted":       pii_redacted,
        }
    except Exception:
        return {}


_METRICS: Dict[str, Any] = {}

if settings.PROMETHEUS_ENABLED:
    try:
        _METRICS = _get_metrics()
    except Exception:
        pass


def _record_duration(modality: str, duration: float) -> None:
    try:
        if "ingestion_duration" in _METRICS:
            _METRICS["ingestion_duration"].labels(modality=modality).observe(duration)
    except Exception:
        pass


def _record_error(modality: str, error_type: str) -> None:
    try:
        if "ingestion_errors" in _METRICS:
            _METRICS["ingestion_errors"].labels(
                modality=modality, error_type=error_type
            ).inc()
    except Exception:
        pass


def _record_chunks(modality: str, count: int) -> None:
    try:
        if "chunk_count" in _METRICS:
            _METRICS["chunk_count"].labels(modality=modality).observe(count)
    except Exception:
        pass


def _record_embed_latency(model: str, latency: float) -> None:
    try:
        if "embedding_latency" in _METRICS:
            _METRICS["embedding_latency"].labels(model=model).observe(latency)
    except Exception:
        pass


def _set_queue_depth(depth: int) -> None:
    try:
        if "queue_depth" in _METRICS:
            _METRICS["queue_depth"].set(depth)
    except Exception:
        pass


# SHA-256 FILE HASH — SECTION 2.2

def _sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# DISK SPACE GUARD — SECTION 2.3

def _check_disk_space(path: str) -> None:
    try:
        stat = shutil.disk_usage(Path(path).parent)
        free_mb = stat.free / (1024 * 1024)
        if free_mb < settings.MIN_FREE_DISK_MB:
            raise DiskSpaceError(
                f"INSUFFICIENT_DISK_SPACE: {free_mb:.0f}MB free, "
                f"need {settings.MIN_FREE_DISK_MB}MB"
            )
    except DiskSpaceError:
        raise
    except Exception as e:
        logger.warning(event="disk_check_failed", error=str(e))


# CLAMAV MALWARE SCAN — SECTION 5

def _malware_scan(file_path: str) -> None:
    if not settings.CLAMAV_ENABLED:
        return
    try:
        import pyclamd
        cd = pyclamd.ClamdNetworkSocket(
            host=settings.CLAMAV_HOST,
            port=settings.CLAMAV_PORT,
        )
        result = cd.scan_file(file_path)
        if result:
            raise MalwareDetectedError(
                f"MALWARE_DETECTED in {os.path.basename(file_path)}: {result}"
            )
    except MalwareDetectedError:
        raise
    except Exception as e:
        logger.warning(event="clamav_scan_failed", error=str(e))


# SHA-256 DEDUP CHECK AGAINST QDRANT — SECTION 2.3

def _check_duplicate(file_hash: str, session_id: str) -> bool:
    if not settings.DEDUP_ENABLED:
        return False
    try:
        from app.core.infra_registry import infra
        vs = infra.get_vector_store()
        if vs is None:
            return False
        # SEARCH BY CHECKSUM PAYLOAD FIELD
        results = vs.search_by_payload(
            field="checksum_sha256",
            value=file_hash,
            session_id=session_id,
            limit=1,
        )
        return bool(results)
    except Exception as e:
        logger.warning(event="dedup_check_failed", error=str(e))
        return False


# MODALITY COUNTS

def _modality_counts(docs: List[IngestedDocument]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for d in docs:
        m = getattr(d, "modality", "unknown")
        counts[m] = counts.get(m, 0) + 1
    return counts


# VALID CHUNK FILTER

def _valid_chunks(docs: List[IngestedDocument]) -> List[IngestedDocument]:
    return [
        d for d in docs
        if getattr(d, "text", "").strip()
        and len(getattr(d, "text", "").strip()) >= settings.CHUNK_MIN_SIZE
    ]


# VALID EMBEDDING FILTER

def _valid_embeddings(docs: List[IngestedDocument]) -> Tuple[List[IngestedDocument], int]:
    valid:   List[IngestedDocument] = []
    invalid: int = 0
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
        logger.warning(event="invalid_embeddings_skipped", count=invalid)
    return valid, invalid


# SPLIT BY EMBEDDING SPACE

def _split_by_modality(
    docs: List[IngestedDocument],
) -> Tuple[List[IngestedDocument], List[IngestedDocument]]:
    text_docs:   List[IngestedDocument] = []
    vision_docs: List[IngestedDocument] = []
    for d in docs:
        space = (getattr(d, "structure", {}) or {}).get("embedding_space", "text")
        if space == "vision":
            vision_docs.append(d)
        else:
            text_docs.append(d)
    return text_docs, vision_docs


# SHA-256 DEDUP AT CHUNK LEVEL

def _dedup_chunks(docs: List[IngestedDocument]) -> List[IngestedDocument]:
    seen:   set                    = set()
    unique: List[IngestedDocument] = []
    for d in docs:
        text      = getattr(d, "text", "")
        structure = getattr(d, "structure", {}) or {}
        base = f"{text[:100]}|{structure.get('doc_id', '')}|{getattr(d, 'chunk_id', '')}"
        h = hashlib.sha256(base.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        unique.append(d)
    return unique


# BATCHED TEXT EMBEDDING WITH PROMETHEUS TIMING

def _batched_text_embedding(
    embedder: Any,
    docs: List[IngestedDocument],
    session_id: str,
) -> List[IngestedDocument]:
    batch_size = settings.INGESTION_BATCH_SIZE
    results: List[IngestedDocument] = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        t_start = time.time()
        try:
            embedded = embedder.embed_documents(batch, session_id=session_id)
            elapsed  = round(time.time() - t_start, 3)
            _record_embed_latency(settings.EMBEDDING_MODEL, elapsed)
            if embedded:
                results.extend(embedded)
        except Exception as e:
            logger.warning(
                event="text_embedding_batch_failed",
                batch_start=i,
                error=str(e),
                session_id=session_id,
            )
            _record_error("text", "embedding_failed")

    return results


# FALLBACK VECTOR STORE

class _UnavailableVectorStore:
    def insert_documents(self, documents: Any, session_id: str = "") -> None:
        logger.warning(event="vector_store_unavailable_skipping_insert")

    def search_by_payload(self, **kwargs: Any) -> List:
        return []


# PROGRESS EVENT EMITTER — SECTION 4.6

class _ProgressEmitter:

    def __init__(self, file_name: str, session_id: str) -> None:
        self.file_name  = file_name
        self.session_id = session_id

    def emit(self, stage: str, status: str, **kwargs: Any) -> None:
        logger.info(
            event="ingestion_progress",
            file=self.file_name,
            stage=stage,
            status=status,
            session_id=self.session_id,
            **kwargs,
        )


# INGESTION PIPELINE CLASS

class IngestionPipeline:

    def __init__(self) -> None:
        from app.core.infra_registry import infra
        self.vector_store = infra.get_vector_store() or _UnavailableVectorStore()
        self.bm25         = infra.get_bm25()
        self.max_chunks   = settings.MAX_CHUNKS
        self.batch_size   = settings.INGESTION_BATCH_SIZE
        self._semaphore   = asyncio.Semaphore(settings.ASYNC_SEMAPHORE_WORKERS)
        self._queue: asyncio.Queue = asyncio.Queue()

    # ASYNC PROCESS FILE — SECTION 4.6

    async def process_file_async(
        self,
        file_path: str,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, self.process_file, file_path, session_id),
                timeout=settings.FILE_PROCESSING_TIMEOUT_SEC,
            )

    # QUEUE WORKER — SECTION 4.6

    async def _queue_worker(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                file_path, session_id, future = item
                result = await self.process_file_async(file_path, session_id)
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
            finally:
                self._queue.task_done()
                _set_queue_depth(self._queue.qsize())

    # ENQUEUE — SECTION 4.6

    async def enqueue(
        self,
        file_path: str,
        session_id: str = "default",
    ) -> asyncio.Future:
        loop   = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((file_path, session_id, future))
        _set_queue_depth(self._queue.qsize())
        return future

    # MAIN SYNC PROCESS FILE

    def process_file(
        self,
        file_path: str,
        session_id: str = "default",
    ) -> Dict[str, Any]:

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        file_name = os.path.basename(file_path)
        start     = time.time()
        progress  = _ProgressEmitter(file_name, session_id)

        # OTEL SPAN STUB 
        span_ctx: Dict[str, Any] = {"trace_id": str(uuid.uuid4())}

        try:
            # PRE-FLIGHT CHECKS
            progress.emit("preflight", "started")

            if not Path(file_path).exists():
                raise FileNotFoundError(f"FILE_NOT_FOUND: {file_path}")

            file_size = Path(file_path).stat().st_size

            if file_size == 0:
                raise EmptyFileError("EMPTY_FILE")

            if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                raise FileTooLargeError(f"FILE_TOO_LARGE: {file_size} bytes")

            # DISK SPACE GUARD
            _check_disk_space(file_path)

            # MALWARE SCAN — SECTION 5
            _malware_scan(file_path)

            # SHA-256 HASH + DEDUP — SECTION 2.2 / 2.3
            file_hash = _sha256(file_path)

            if _check_duplicate(file_hash, session_id):
                logger.info(
                    event="ingestion_duplicate_skipped",
                    file=file_name,
                    hash=file_hash[:16],
                    session_id=session_id,
                )
                raise DuplicateFileError(
                    f"DUPLICATE_FILE: {file_name} already ingested (sha256={file_hash[:16]})"
                )

            progress.emit("preflight", "completed", file_size=file_size)

            # INGEST — SECTION 4.6
            progress.emit("ingest", "started")
            t_ingest = time.time()

            from app.ingestion.router import route_ingestion_sync
            docs = route_ingestion_sync(file_path, session_id=session_id)

            if not docs:
                raise ValueError("INGESTION_EMPTY")

            ingest_latency = round(time.time() - t_ingest, 2)
            modality       = getattr(docs[0], "modality", "unknown") if docs else "unknown"

            progress.emit(
                "ingest", "completed",
                docs=len(docs),
                modality=modality,
                latency=ingest_latency,
            )

            logger.info(
                event="ingest_complete",
                file=file_name,
                docs=len(docs),
                modality_breakdown=_modality_counts(docs),
                ingest_latency=ingest_latency,
                session_id=session_id,
            )

            # CHUNK — SECTION 4.6
            progress.emit("chunk", "started")
            t_chunk = time.time()

            from app.chunking.chunker import chunk_documents
            chunks = chunk_documents(docs)
            chunks = _valid_chunks(chunks)

            if not chunks:
                raise ValueError("NO_VALID_CHUNKS")

            if len(chunks) > self.max_chunks:
                chunks = chunks[:self.max_chunks]
                logger.warning(
                    event="chunk_limit_applied",
                    limit=self.max_chunks,
                    session_id=session_id,
                )

            chunks = _dedup_chunks(chunks)
            chunk_latency = round(time.time() - t_chunk, 2)

            _record_chunks(modality, len(chunks))

            progress.emit(
                "chunk", "completed",
                chunks=len(chunks),
                latency=chunk_latency,
            )

            # STAMP FILE HASH ON ALL CHUNKS — SECTION 2.2
            for c in chunks:
                c.structure.setdefault("checksum_sha256", file_hash)
                c.structure.setdefault("file_size_bytes", file_size)
                c.structure.setdefault("ingestion_time",  time.time())

            # EMBED — SECTION 4.6
            progress.emit("embed", "started")
            t_embed = time.time()

            from app.core.model_loader import model_loader

            text_chunks, vision_chunks = _split_by_modality(chunks)
            text_embedded:   List[IngestedDocument] = []
            vision_embedded: List[IngestedDocument] = []

            if text_chunks:
                embedder      = model_loader.get_embedder()
                text_embedded = _batched_text_embedding(embedder, text_chunks, session_id)

            if vision_chunks:
                try:
                    multimodal      = model_loader.get_multimodal_embedder()
                    t_vis           = time.time()
                    _, vis_embedded = multimodal.embed_documents(
                        vision_chunks, session_id=session_id
                    )
                    vision_embedded = vis_embedded
                    _record_embed_latency(settings.CLIP_MODEL, round(time.time() - t_vis, 3))
                except Exception as e:
                    logger.warning(
                        event="vision_embedding_failed",
                        error=str(e),
                        session_id=session_id,
                    )
                    _record_error("vision", "embedding_failed")

            all_embedded = text_embedded + vision_embedded
            all_embedded, invalid_count = _valid_embeddings(all_embedded)

            if not all_embedded:
                raise ValueError("NO_VALID_EMBEDDINGS")

            embed_latency = round(time.time() - t_embed, 2)

            progress.emit(
                "embed", "completed",
                embedded=len(all_embedded),
                invalid=invalid_count,
                latency=embed_latency,
            )

            # BM25 INDEX UPDATE — SECTION 4.5
            try:
                if self.bm25:
                    self.bm25.add_documents(chunks, session_id=session_id)
            except Exception as e:
                logger.error(
                    event="bm25_update_failed",
                    error=str(e),
                    session_id=session_id,
                )
                _record_error(modality, "bm25_update_failed")

            # VECTOR STORE UPSERT — SECTION 4.4 / 4.6
            progress.emit("store", "started")
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
                    _record_error(modality, "vector_insert_failed")

            store_latency = round(time.time() - t_store, 2)
            total_latency = round(time.time() - start, 2)

            _record_duration(modality, total_latency)

            progress.emit(
                "store", "completed",
                stored=total,
                latency=store_latency,
            )

            logger.info(
                event="ingestion_pipeline_success",
                file=file_name,
                chunks=len(chunks),
                embedded=len(all_embedded),
                stored=total,
                modality=modality,
                modality_breakdown=_modality_counts(chunks),
                ingest_latency=ingest_latency,
                chunk_latency=chunk_latency,
                embed_latency=embed_latency,
                store_latency=store_latency,
                total_latency=total_latency,
                session_id=session_id,
                trace_id=span_ctx["trace_id"],
            )

            return {
                "status":      "success",
                "chunks":      len(chunks),
                "embedded":    len(all_embedded),
                "stored":      total,
                "latency":     total_latency,
                "modality":    modality,
                "session_id":  session_id,
                "file_hash":   file_hash,
                "trace_id":    span_ctx["trace_id"],
            }

        except DuplicateFileError as e:
            return {
                "status":     "duplicate",
                "message":    str(e),
                "session_id": session_id,
                "latency":    round(time.time() - start, 2),
            }

        except (EmptyFileError, FileTooLargeError, UnsupportedMimeError) as e:
            error_type = type(e).__name__
            _record_error("unknown", error_type)
            progress.emit("preflight", "failed", error=str(e))
            logger.warning(
                event="ingestion_pipeline_validation_failed",
                file=file_name,
                error=str(e),
                session_id=session_id,
            )
            raise

        except MalwareDetectedError as e:
            _record_error("unknown", "malware_detected")
            logger.error(
                event="ingestion_pipeline_malware",
                file=file_name,
                error=str(e),
                session_id=session_id,
            )
            raise

        except Exception as e:
            _record_error("unknown", type(e).__name__)
            progress.emit("failed", "error", error=str(e))
            logger.error(
                event="ingestion_pipeline_failed",
                file=file_name,
                error=str(e),
                latency=round(time.time() - start, 2),
                session_id=session_id,
                trace_id=span_ctx.get("trace_id"),
            )
            raise RuntimeError(f"INGESTION_FAILED: {e}") from e


# SINGLETON

pipeline = IngestionPipeline()


def process_file(
    file_path: str,
    session_id: str = "default",
) -> Dict[str, Any]:
    return pipeline.process_file(file_path, session_id)


async def process_file_async(
    file_path: str,
    session_id: str = "default",
) -> Dict[str, Any]:
    return await pipeline.process_file_async(file_path, session_id)
