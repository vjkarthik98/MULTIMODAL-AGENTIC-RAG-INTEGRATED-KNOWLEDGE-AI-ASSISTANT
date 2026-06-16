from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.ingestion.schema import (
    CorruptFileError,
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


# TEXT CORRUPTION PREFLIGHT — protects against corpus poisoning by
# silently ingesting unreadable / binary-tainted "text" files.
#
# Only scans extensions that are supposed to be plain text. Binary
# formats (.pdf, .docx, .xlsx, images, audio, video) legitimately
# contain non-printable bytes and are validated by their own parsers
# downstream.

_TEXT_LIKE_EXTS = {".txt", ".md", ".markdown", ".csv", ".json", ".log"}

# Read at most this many bytes for the scan. Enough to catch BOM,
# null bytes, and trailing-binary tails on the docs we ingest;
# bounded so a giant file does not blow memory in preflight.
_CORRUPTION_SCAN_MAX_BYTES = 2 * 1024 * 1024  # 2 MiB


def _scan_corruption(file_path: str) -> List[str]:
    """Return a list of corruption reason codes for a text-like file.

    An empty list means the file looks clean. A non-empty list is
    treated by the pipeline as a hard fail (CORRUPTED_FILE).
    """
    ext = Path(file_path).suffix.lower()
    if ext not in _TEXT_LIKE_EXTS:
        return []

    try:
        with open(file_path, "rb") as f:
            raw = f.read(_CORRUPTION_SCAN_MAX_BYTES)
    except OSError as e:
        logger.warning(event="corruption_scan_read_failed", error=str(e))
        return []

    if not raw:
        return []

    reasons: List[str] = []

    # NULL BYTES — strongest signal of "this is not text"
    if b"\x00" in raw:
        reasons.append("null_bytes")

    # ANSI ESCAPE SEQUENCES — never present in legitimate plain text
    # written by humans; appears here as fake "[31mERROR[0m" markers
    # left in the corrupted sample.
    if b"\x1b[" in raw:
        reasons.append("ansi_escape_sequences")

    # BINARY TAIL — last 64 bytes contain a high ratio of non-printable,
    # non-whitespace bytes. Catches truncation artefacts and trailing
    # garbage even when null bytes are absent.
    tail = raw[-64:] if len(raw) >= 64 else raw
    if tail:
        non_printable = sum(
            1 for b in tail
            if b < 0x20 and b not in (0x09, 0x0A, 0x0D)
        )
        if non_printable / len(tail) >= 0.10:
            reasons.append("binary_tail")

    # DECODE TEST — try utf-8 strict, fall back to utf-8-sig (strips BOM).
    # If neither works, the file is not valid UTF-8 text.
    decoded: Optional[str] = None
    for enc in ("utf-8", "utf-8-sig"):
        try:
            decoded = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        reasons.append("invalid_utf8")
        return reasons  # cannot run char-level checks on undecodable bytes

    # REPLACEMENT-CHAR RATIO — U+FFFD is the standard "this byte could
    # not be decoded" marker. A real document should not contain any;
    # >0.5% means a meaningful chunk of the text is unrecoverable.
    if decoded:
        ufffd_ratio = decoded.count("�") / len(decoded)
        if ufffd_ratio >= 0.005:
            reasons.append(f"replacement_char_ratio={ufffd_ratio:.4f}")

        # CONTROL-CHAR RATIO — non-printable, non-whitespace control
        # codepoints in decoded text.
        ctrl = sum(
            1 for ch in decoded
            if ord(ch) < 0x20 and ch not in ("\t", "\n", "\r")
        )
        ctrl_ratio = ctrl / len(decoded)
        if ctrl_ratio >= 0.01:
            reasons.append(f"control_char_ratio={ctrl_ratio:.4f}")

    return reasons


# SHA-256 DEDUP CHECK AGAINST QDRANT — SECTION 2.3

def _check_duplicate(file_hash: str, session_id: str) -> bool:
    if not settings.DEDUP_ENABLED:
        return False
    try:
        from app.core.infra_registry import infra
        vs = infra.get_vector_store()
        if vs is None:
            return False
        # GLOBAL dedup — no session_id filter so cross-session re-uploads are caught
        results = vs.search_by_payload(
            field="checksum_sha256",
            value=file_hash,
            session_id=None,
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
    result = []
    for d in docs:
        text = getattr(d, "text", "").strip()
        space = (getattr(d, "structure", {}) or {}).get("embedding_space", "text")
        # Vision chunks (video frames, images) are kept even with short captions —
        # their value is the SigLIP image embedding, not the caption text length.
        if space == "vision":
            result.append(d)
        elif text and len(text) >= settings.CHUNK_MIN_SIZE:
            result.append(d)
    return result


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

def _stream_embed_and_store(
    text_chunks: List[IngestedDocument],
    vision_chunks: List[IngestedDocument],
    embedder: Any,
    vector_store: Any,
    session_id: str,
    user_id: str,
    micro_batch: int = 1,
) -> Tuple[int, int]:
    """Embed and store one micro-batch at a time, clearing GPU cache between batches.

    Returns (total_embedded, total_stored).
    """
    import gc
    try:
        import torch
        _cuda = torch.cuda.is_available()
    except ImportError:
        _cuda = False

    def _clear_cache() -> None:
        if _cuda:
            torch.cuda.empty_cache()
        gc.collect()

    total_embedded = 0
    total_stored   = 0
    # empty_cache()+gc.collect() after EVERY micro-batch was costing more than
    # the embeds themselves (and forced micro_batch=1 era behavior). Clearing
    # periodically + on failure keeps the OOM protection without the tax.
    clear_every = max(int(settings.INGESTION_CACHE_CLEAR_EVERY), 1)
    batches_done = 0

    # TEXT CHUNKS — micro-batched embed + batched Qdrant upsert
    for i in range(0, len(text_chunks), micro_batch):
        batch = text_chunks[i : i + micro_batch]
        t_start = time.time()
        try:
            embedded = embedder.embed_documents(batch, session_id=session_id)
            _record_embed_latency(settings.EMBEDDING_MODEL, round(time.time() - t_start, 3))
            if embedded:
                valid, _ = _valid_embeddings(embedded)
                if valid:
                    vector_store.insert_documents(valid, session_id=session_id, user_id=user_id)
                    total_stored   += len(valid)
                    total_embedded += len(valid)
        except Exception as e:
            logger.warning(
                event="stream_text_embed_failed",
                batch_start=i,
                error=str(e),
                session_id=session_id,
            )
            _record_error("text", "embedding_failed")
            _clear_cache()
        batches_done += 1
        if batches_done % clear_every == 0:
            _clear_cache()

    # VISION CHUNKS — micro-batched embed + batched Qdrant upsert
    if vision_chunks:
        from app.core.model_loader import model_loader as _ml
        for i in range(0, len(vision_chunks), micro_batch):
            batch = vision_chunks[i : i + micro_batch]
            t_vis = time.time()
            try:
                multimodal = _ml.get_multimodal_embedder()
                txt_from_vis, vis_embedded = multimodal.embed_documents(batch, session_id=session_id)
                _record_embed_latency(settings.SIGLIP_MODEL, round(time.time() - t_vis, 3))
                combined = vis_embedded + txt_from_vis
                valid, _ = _valid_embeddings(combined)
                if valid:
                    vector_store.insert_documents(valid, session_id=session_id, user_id=user_id)
                    total_stored   += len(valid)
                    total_embedded += len(valid)
            except Exception as e:
                logger.warning(
                    event="stream_vision_embed_failed",
                    batch_start=i,
                    error=str(e),
                    session_id=session_id,
                )
                _record_error("vision", "embedding_failed")
                _clear_cache()
            batches_done += 1
            if batches_done % clear_every == 0:
                _clear_cache()

    _clear_cache()
    return total_embedded, total_stored


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


# INGEST JOB — PROGRESS TRACKING (Phase 8)

@dataclass
class IngestJob:
    job_id:       str
    filename:     str
    modality:     str
    status:       str          # "queued"|"extracting"|"chunking"|"embedding"|"done"|"error"
    progress:     float = 0.0  # 0.0–1.0
    chunks_done:  int   = 0
    chunks_total: int   = 0
    error:        Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_JOB_TTL_SECONDS = 3600   # 1 hour


def _job_key(job_id: str) -> str:
    return f"ingest_job:{job_id}"


def _store_job(job: IngestJob) -> None:
    """Persist IngestJob to Redis with a 1-hour TTL. Silently skips if Redis unavailable."""
    try:
        from app.core.infra_registry import infra
        redis = infra.get_memory()
        if redis and hasattr(redis, "client"):
            redis.client.setex(_job_key(job.job_id), _JOB_TTL_SECONDS, json.dumps(job.to_dict()))
    except Exception:
        pass


def get_ingest_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch IngestJob dict from Redis. Returns None if not found."""
    try:
        from app.core.infra_registry import infra
        redis = infra.get_memory()
        if redis and hasattr(redis, "client"):
            raw = redis.client.get(_job_key(job_id))
            if raw:
                return json.loads(raw)
    except Exception:
        pass
    return None


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
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, self.process_file, file_path, session_id, user_id),
                timeout=settings.FILE_PROCESSING_TIMEOUT_SEC,
            )

    # QUEUE WORKER — SECTION 4.6

    async def _queue_worker(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                file_path, session_id, future = item[:3]
                user_id = item[3] if len(item) > 3 else None
                result = await self.process_file_async(file_path, session_id, user_id)
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
        user_id: Optional[str] = None,
    ) -> asyncio.Future:
        loop   = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((file_path, session_id, future, user_id))
        _set_queue_depth(self._queue.qsize())
        return future

    # BACKGROUND JOB — return job_id immediately; poll /api/ingestion/status/{job_id}
    # Used for audio/video ingestion which can take minutes. (Phase 8)

    _BACKGROUND_MODALITIES = {"mp3", "mp4", "audio", "video"}

    async def process_file_background(
        self,
        file_path: str,
        session_id: str = "default",
        user_id: Optional[str] = None,
    ) -> str:
        """Start ingestion in background. Returns job_id immediately."""
        job_id   = str(uuid.uuid4())
        filename = Path(file_path).name
        ext      = Path(file_path).suffix.lstrip(".").lower()
        modality = {
            "mp3": "mp3", "wav": "mp3", "m4a": "mp3",
            "mp4": "mp4", "mov": "mp4", "avi": "mp4",
        }.get(ext, ext)

        job = IngestJob(
            job_id=job_id,
            filename=filename,
            modality=modality,
            status="queued",
        )
        _store_job(job)

        async def _run() -> None:
            try:
                job.status = "extracting"
                _store_job(job)
                result = await self.process_file_async(file_path, session_id, user_id)
                job.status       = "done"
                job.progress     = 1.0
                job.chunks_done  = result.get("chunks", 0) if isinstance(result, dict) else 0
                job.chunks_total = job.chunks_done
            except Exception as exc:
                job.status = "error"
                job.error  = str(exc)
                logger.warning(event="background_ingest_failed", job_id=job_id, error=str(exc))
            finally:
                _store_job(job)

        asyncio.create_task(_run())
        return job_id

    # MAIN SYNC PROCESS FILE

    def process_file(
        self,
        file_path: str,
        session_id: str = "default",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        file_name = os.path.basename(file_path)
        start     = time.time()
        progress  = _ProgressEmitter(file_name, session_id)

        # OTEL SPAN STUB
        span_ctx: Dict[str, Any] = {"trace_id": str(uuid.uuid4())}

        # SET ACTIVE USER FOR ALL DOWNSTREAM STORAGE — every ingestor's
        # resolved_*_dir() helpers read this contextvar so PDF images,
        # frames, OCR thumbs etc. all land under data/users/{user_id}/.
        from app.utils.paths import set_current_user, reset_current_user
        effective_user_id = user_id or settings.DEFAULT_DEV_USER_ID
        _user_token = set_current_user(effective_user_id)

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

            # CORPUS-POISONING GUARD — reject text files containing null
            # bytes, ANSI escapes, undecodable UTF-8, high U+FFFD ratio,
            # or binary trailing garbage. Prevents corrupted content
            # from reaching the chunker / embedder / vector store.
            corruption_reasons = _scan_corruption(file_path)
            if corruption_reasons:
                logger.warning(
                    event="ingestion_corruption_detected",
                    file=file_name,
                    reasons=corruption_reasons,
                    session_id=session_id,
                )
                _record_error("text", "corrupted_file")
                raise CorruptFileError(
                    f"CORRUPTED_FILE: {file_name} failed text-integrity checks "
                    f"({', '.join(corruption_reasons)})"
                )

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
            docs = route_ingestion_sync(file_path, session_id=session_id, user_id=user_id)

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

            # docs from route_ingestion_sync are already per-modality chunked;
            # use them directly to avoid the legacy re-splitter bypass.
            chunks = list(docs)
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

            # STAMP FILE HASH AND USER_ID ON ALL CHUNKS — SECTION 2.2
            for c in chunks:
                if not isinstance(c.structure, dict):
                    c.structure = {}
                c.structure.setdefault("checksum_sha256", file_hash)
                c.structure.setdefault("file_size_bytes", file_size)
                c.structure.setdefault("ingestion_time",  time.time())
                if user_id:
                    c.structure["user_id"] = user_id

            # EMBED + STORE (streaming micro-batches) — SECTION 4.6
            # One micro-batch at a time: embed → validate → store → clear GPU cache.
            # Prevents CUDA OOM on large chunks across all modalities.
            progress.emit("embed", "started")
            t_embed = time.time()

            from app.embeddings import get_embedder as _get_embedder_for_modality

            text_chunks, vision_chunks = _split_by_modality(chunks)
            embedder = _get_embedder_for_modality(modality) if text_chunks else None

            micro = getattr(settings, "INGESTION_MICRO_BATCH", 1)

            total_embedded, total = _stream_embed_and_store(
                text_chunks,
                vision_chunks,
                embedder,
                self.vector_store,
                session_id,
                user_id,
                micro_batch=micro,
            )

            # CLEAN UP PERSISTENT FRAME STAGING DIR — done after all embeddings stored.
            if modality in ("video", "mp4"):
                import shutil as _shutil
                for chunk in chunks:
                    asset = (getattr(chunk, "structure", {}) or {}).get("asset_path", "")
                    if asset:
                        frame_stage = Path(asset).parent
                        if frame_stage.name.startswith("frames_") and frame_stage.exists():
                            _shutil.rmtree(frame_stage, ignore_errors=True)
                            break

            if total_embedded == 0:
                raise ValueError("NO_VALID_EMBEDDINGS")

            embed_latency = round(time.time() - t_embed, 2)

            progress.emit(
                "embed", "completed",
                embedded=total_embedded,
                latency=embed_latency,
            )

            # VECTOR STORE UPSERT already completed inside _stream_embed_and_store
            progress.emit("store", "started")
            t_store = time.time()

            # BM25 INDEX UPDATE — after Qdrant upsert succeeds
            if total > 0:
                try:
                    if self.bm25:
                        self.bm25.add_documents(chunks, session_id=session_id, user_id=user_id)
                except Exception as e:
                    logger.error(
                        event="bm25_update_failed",
                        error=str(e),
                        session_id=session_id,
                    )
                    _record_error(modality, "bm25_update_failed")

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
                embedded=total_embedded,
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

            # Phase 24.4 — collect doc_id, pages, warnings from ingested chunks
            _doc_id   = chunks[0].doc_id() if chunks else ""
            _pages    = chunks[0].structure.get("total_pages") if chunks else None
            _warnings: List[str] = []
            for _c in chunks:
                for _w in (getattr(_c, "warnings", None) or []):
                    if _w not in _warnings:
                        _warnings.append(_w)

            return {
                "status":            "success",
                "doc_id":            _doc_id,
                "filename":          file_name,
                "modality":          modality,
                "chunks":            len(chunks),
                "stored":            total,
                "session_id":        session_id,
                "ingestion_time_sec": total_latency,
                "pages":             _pages,
                "warnings":          _warnings,
                "file_hash":         file_hash,
                "embedded":          total_embedded,
                "user_id":           user_id,
                "trace_id":          span_ctx["trace_id"],
            }

        except DuplicateFileError as e:
            return {
                "status":     "duplicate",
                "message":    str(e),
                "session_id": session_id,
                "latency":    round(time.time() - start, 2),
            }

        except (EmptyFileError, FileTooLargeError, UnsupportedMimeError, CorruptFileError) as e:
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

        finally:
            reset_current_user(_user_token)


# SINGLETON

pipeline = IngestionPipeline()


def process_file(
    file_path: str,
    session_id: str = "default",
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    return pipeline.process_file(file_path, session_id, user_id)


async def process_file_async(
    file_path: str,
    session_id: str = "default",
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    return await pipeline.process_file_async(file_path, session_id, user_id)
