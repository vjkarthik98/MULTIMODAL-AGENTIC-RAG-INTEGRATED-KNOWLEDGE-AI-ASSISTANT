import asyncio
import contextvars
import hashlib
import mimetypes
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import magic
import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.response import ErrorCode, Modality, Severity, UniversalErrorResponse
from app.ingestion.audio_ingest import ingest as audio_ingest
from app.ingestion.audio_ingest import AudioIngestor
from app.ingestion.audio_ingest import ingest_audio_full
from app.ingestion.docx_ingest import ingest as word_ingest  # kept for backward compat
from app.ingestion.docx_ingest import ingest_docx_full
from app.ingestion.docx_ingest import DocxIngestor
from app.ingestion.image_ingest import ingest as image_ingest  # backward-compat
from app.ingestion.image_ingest import ImageIngestor
from app.ingestion.image_ingest import ingest_image_full
from app.ingestion.pdf_ingest import ingest as pdf_ingest  # kept for backward compat
from app.ingestion.pdf_ingest import ingest_pdf_full
from app.ingestion.pdf_ingest import PdfIngestor
from app.ingestion.schema import IngestedDocument
from app.ingestion.txt_ingest import ingest as text_ingest
from app.ingestion.txt_ingest import TxtIngestor
from app.ingestion.video_ingest import ingest as video_ingest
from app.ingestion.video_ingest import ingest_video_full
from app.ingestion.video_ingest import VideoIngestor
from app.ingestion.xlsx_ingest import ingest as excel_ingest  # backward-compat
from app.ingestion.xlsx_ingest import XlsxIngestor
from app.ingestion.xlsx_ingest import ingest_xlsx_full
from app.utils.logger import get_logger
from app.guardrails.exceptions import GuardrailBlocked

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_ingestion_duration = Histogram(
    "file_ingestion_duration_seconds",
    "Ingestion duration by modality",
    ["modality"],
)
_ingestion_errors = Counter(
    "file_ingestion_errors_total",
    "Ingestion errors by modality and error type",
    ["modality", "error_type"],
)
_ingestion_docs = Histogram(
    "chunk_count_per_file",
    "Chunks produced per file",
    ["modality"],
)

# EXTENSION MAPS
TEXT_EXTENSIONS: Set[str]  = {".txt", ".md", ".rst", ".csv", ".log", ".json", ".yaml", ".yml"}
PDF_EXTENSIONS: Set[str]   = {".pdf"}
WORD_EXTENSIONS: Set[str]  = {".docx", ".doc"}
EXCEL_EXTENSIONS: Set[str] = {".xlsx", ".xls"}
IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".heic", ".heif", ".gif", ".svg"}
AUDIO_EXTENSIONS: Set[str] = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".aiff", ".opus"}
VIDEO_EXTENSIONS: Set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".ts"}

# "document" kept as alias so any callers using old string still work
DOCUMENT_EXTENSIONS: Set[str] = PDF_EXTENSIONS | WORD_EXTENSIONS | EXCEL_EXTENSIONS

EXT_TO_MODALITY: Dict[str, str] = {
    **{ext: "text"  for ext in TEXT_EXTENSIONS},
    **{ext: "pdf"   for ext in PDF_EXTENSIONS},
    **{ext: "word"  for ext in WORD_EXTENSIONS},
    **{ext: "excel" for ext in EXCEL_EXTENSIONS},
    **{ext: "image" for ext in IMAGE_EXTENSIONS},
    **{ext: "audio" for ext in AUDIO_EXTENSIONS},
    **{ext: "video" for ext in VIDEO_EXTENSIONS},
}

# MAGIC-BYTE MIME TO MODALITY MAP
MIME_TO_MODALITY: Dict[str, str] = {
    "text/plain":                  "text",
    "text/markdown":               "text",
    "text/csv":                    "text",
    "text/x-rst":                  "text",
    "application/json":            "text",
    "application/x-yaml":          "text",
    "application/pdf":             "pdf",
    "application/msword":          "word",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "word",
    "application/vnd.ms-excel":    "excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":       "excel",
    "image/jpeg":                  "image",
    "image/png":                   "image",
    "image/gif":                   "image",
    "image/webp":                  "image",
    "image/tiff":                  "image",
    "image/bmp":                   "image",
    "image/heic":                  "image",
    "image/heif":                  "image",
    "image/svg+xml":               "image",
    "audio/mpeg":                  "audio",
    "audio/wav":                   "audio",
    "audio/x-wav":                 "audio",
    "audio/flac":                  "audio",
    "audio/ogg":                   "audio",
    "audio/aac":                   "audio",
    "audio/mp4":                   "audio",
    "audio/x-m4a":                 "audio",
    "audio/opus":                  "audio",
    "video/mp4":                   "video",
    "video/x-msvideo":             "video",
    "video/quicktime":             "video",
    "video/x-matroska":            "video",
    "video/webm":                  "video",
    "video/x-flv":                 "video",
    "video/x-ms-wmv":              "video",
    "video/mp2t":                  "video",
}

MODALITY_LABEL: Dict[str, str] = {
    "text":  Modality.TEXT,
    "pdf":   Modality.PDF,
    "word":  Modality.PDF,
    "excel": Modality.PDF,
    "image": Modality.IMAGE,
    "audio": Modality.AUDIO,
    "video": Modality.VIDEO,
}

MODALITY_SIZE_LIMITS: Dict[str, int] = {
    "text":  settings.MAX_FILE_SIZE_TEXT,
    "pdf":   settings.MAX_FILE_SIZE_PDF,
    "word":  settings.MAX_FILE_SIZE_DOCX,
    "excel": settings.MAX_FILE_SIZE_XLSX,
    "image": settings.MAX_FILE_SIZE_IMAGE,
    "audio": settings.MAX_FILE_SIZE_AUDIO,
    "video": settings.MAX_FILE_SIZE_VIDEO,
}

INGESTION_HANDLERS: Dict[str, Callable] = {
    "text":  text_ingest,
    "pdf":   ingest_pdf_full,   # full path: PdfIngestor → PdfChunker (rich metadata)
    "word":  ingest_docx_full,   # full path: DocxIngestor → DocxChunker (rich metadata)
    "excel": ingest_xlsx_full,   # full path: XlsxIngestor → XlsxChunker (rich metadata)
    "image": ingest_image_full,   # full path: ImageIngestor → ImageChunker (rich metadata)
    "audio": ingest_audio_full,  # full path: AudioIngestor → AudioChunker (rich metadata)
    "video": ingest_video_full,  # full path: VideoIngestor → VideoChunker (rich metadata)
}

# Per-modality ingestor classes exposing .extract(path, meta) -> List[RawExtract].
# Used by the new per-modality pipeline chain (ingestion_pipeline.py Step 4).
INGESTOR_MAP: Dict[str, type] = {
    "text":  TxtIngestor,
    "txt":   TxtIngestor,
    "pdf":   PdfIngestor,
    "word":  DocxIngestor,
    "docx":  DocxIngestor,
    "excel": XlsxIngestor,
    "xlsx":  XlsxIngestor,
    "image": ImageIngestor,
    "audio": AudioIngestor,
    "mp3":   AudioIngestor,
    "video": VideoIngestor,
    "mp4":   VideoIngestor,
}

MAX_INGESTED_DOCS: int = 5000

# SEMAPHORE — CAP CONCURRENT INGESTION WORKERS
_semaphore = asyncio.Semaphore(5)


# SHA-256 HASH

def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# FILE SHA-256 FOR DEDUP CHECK

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# MAGIC-BYTE MIME DETECTION — NEVER TRUST EXTENSION

def _detect_mime_magic(path: Path) -> str:
    try:
        mime = magic.from_file(str(path), mime=True)
        return mime or ""
    except Exception as exc:
        logger.warning("mime_magic_failed", path=str(path), error=str(exc))
        # FALLBACK TO MIMETYPES STDLIB
        guessed, _ = mimetypes.guess_type(path.name)
        return guessed or "application/octet-stream"


# MIME VALIDATION AGAINST ALLOWLIST

def _validate_mime(path: Path, mime: str) -> None:
    allowed = settings.ALLOWED_MIME_TYPES
    if allowed and mime and mime not in allowed:
        raise ValueError(f"MIME_NOT_ALLOWED: {mime}")


# NULL-BYTE DETECTION

def _check_null_bytes(path: Path) -> int:
    count = 0
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                count += chunk.count(b"\x00")
    except Exception:
        pass
    return count


# PATH TRAVERSAL GUARD

def _guard_path(file_path: str) -> Path:

    resolved = Path(file_path).expanduser().resolve()

    allowed_roots = [
        Path(settings.DATA_DIR).resolve(),
        Path(tempfile.gettempdir()).resolve(),
        Path("C:/temp").resolve(),
    ]

    for root in allowed_roots:
        root = root.resolve()

        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            pass

    raise ValueError(f"PATH_TRAVERSAL_DETECTED: {file_path}")


# DISK SPACE GUARD

def _check_disk_space(path: Path) -> None:
    try:
        import shutil
        min_free_mb = getattr(settings, "MIN_FREE_DISK_MB", 500)
        stat = shutil.disk_usage(str(path.parent))
        free_mb = stat.free / (1024 * 1024)
        if free_mb < min_free_mb:
            raise IOError(
                f"INSUFFICIENT_DISK_SPACE: {free_mb:.1f} MB free, "
                f"minimum required {min_free_mb} MB"
            )
    except IOError:
        raise
    except Exception as exc:
        logger.warning("disk_space_check_failed", error=str(exc))


# CLAMAV MALWARE SCAN — PHASE 25

def _clamav_scan(path: Path) -> None:
    if not getattr(settings, "CLAMAV_ENABLED", False):
        return
    try:
        import clamd
        host    = getattr(settings, "CLAMAV_HOST", "localhost")
        port    = getattr(settings, "CLAMAV_PORT", 3310)
        timeout = getattr(settings, "CLAMAV_TIMEOUT", 30)
        cd      = clamd.ClamdNetworkSocket(host=host, port=port, timeout=timeout)
        result  = cd.scan(str(path))
        if result:
            status = result.get(str(path), ("OK", ""))[0]
            if status == "FOUND":
                virus = result[str(path)][1]
                raise GuardrailBlocked(
                    reason=f"MALWARE_DETECTED: {virus} in {path.name}",
                    surface="ingestion",
                    guard_type="malware",
                )
    except ValueError:
        raise
    except Exception as exc:
        logger.warning("clamav_scan_failed", path=str(path), error=str(exc))


# MODALITY DETECTION — MAGIC-BYTE FIRST, EXTENSION FALLBACK

def detect_modality(file_path: str) -> Tuple[str, str]:
    """
    Returns (modality, mime_type).
    Uses magic-byte MIME detection as primary source.
    Extension used only as fallback when MIME is ambiguous.
    """
    if not file_path:
        raise ValueError("FILE_PATH_REQUIRED")

    path = _guard_path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {file_path}")

    stat = path.stat()

    if stat.st_size == 0:
        raise ValueError("EMPTY_FILE")

    # DISK SPACE CHECK BEFORE ANY PROCESSING
    _check_disk_space(path)

    # MAGIC-BYTE MIME DETECTION
    mime          = _detect_mime_magic(path)
    mime_modality = MIME_TO_MODALITY.get(mime)
    ext           = path.suffix.lower()
    ext_modality  = EXT_TO_MODALITY.get(ext)

    # MISMATCH: magic bytes contradict the file extension — reject immediately
    if mime_modality and ext_modality and mime_modality != ext_modality:
        raise ValueError(
            f"INVALID_FILE_TYPE: file extension {ext!r} claims {ext_modality} "
            f"but magic bytes indicate {mime_modality} (mime={mime})"
        )

    modality = mime_modality or ext_modality

    if not modality:
        raise ValueError(f"UNSUPPORTED_TYPE: mime={mime}, ext={path.suffix.lower()}")

    # PER-MODALITY SIZE CHECK
    limit = MODALITY_SIZE_LIMITS.get(modality, settings.MAX_FILE_SIZE_MB * 1024 * 1024)
    if stat.st_size > limit:
        raise ValueError(
            f"FILE_TOO_LARGE: {stat.st_size} bytes exceeds "
            f"{limit} bytes for {modality}"
        )

    # ALLOWED EXTENSIONS CHECK
    allowed_types = settings.ALLOWED_FILE_TYPES
    ext           = path.suffix.lower()
    if allowed_types and ext.lstrip(".") not in allowed_types:
        raise ValueError(f"EXT_NOT_ALLOWED: {ext}")

    # MIME ALLOWLIST CHECK
    _validate_mime(path, mime)

    return modality, mime


# DOCUMENT VALIDATION

def _validate_documents(
    documents: List[IngestedDocument],
    session_id: str,
    modality: str,
) -> List[IngestedDocument]:

    validated: List[IngestedDocument] = []
    seen:      Set[str]               = set()
    skipped    = 0

    for i, doc in enumerate(documents):
        try:
            if isinstance(doc, dict):
                doc = IngestedDocument(**doc)

            if not isinstance(doc, IngestedDocument):
                skipped += 1
                continue

            doc = doc.finalize()
            doc.structure["session_id"] = session_id

            h = _stable_hash(doc.text)
            if h in seen:
                continue

            seen.add(h)
            validated.append(doc)

        except Exception as exc:
            skipped += 1
            logger.warning(
                "doc_validation_failed",
                index=i,
                modality=modality,
                error=str(exc),
            )

    if skipped:
        logger.warning(
            "docs_skipped",
            skipped=skipped,
            accepted=len(validated),
            modality=modality,
        )

    return validated



# MAIN ASYNC ROUTE INGESTION

async def route_ingestion(
    file_path: str,
    session_id: str,
    user_id: Optional[str] = None,
) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    if not file_path:
        raise ValueError("FILE_PATH_REQUIRED")

    # Re-set the contextvar inside this coroutine — covers the case where the
    # caller spun up a fresh event loop via asyncio.run() in a worker thread,
    # which would otherwise have an empty context.
    _user_token = None
    from app.utils.paths import set_current_user, get_current_user
    effective_uid = user_id or settings.DEFAULT_DEV_USER_ID
    if get_current_user() != effective_uid:
        _user_token = set_current_user(effective_uid)

    start = time.time()
    path  = Path(file_path)

    with tracer.start_as_current_span("route_ingestion") as span:
        span.set_attribute("file.name", path.name)
        span.set_attribute("session.id", session_id)

        try:
            async with _semaphore:

                # MALWARE SCAN — BEFORE ANYTHING ELSE
                await asyncio.get_event_loop().run_in_executor(
                    None, _clamav_scan, path
                )

                # NULL BYTE CHECK
                null_count = await asyncio.get_event_loop().run_in_executor(
                    None, _check_null_bytes, path
                )
                if null_count:
                    logger.warning(
                        "null_bytes_detected",
                        count=null_count,
                        file=path.name,
                        session_id=session_id,
                    )

                # SHA-256 DEDUP CHECK
                checksum = await asyncio.get_event_loop().run_in_executor(
                    None, _file_sha256, path
                )
                span.set_attribute("file.sha256", checksum)

                # MODALITY + MIME DETECTION
                modality, mime = await asyncio.get_event_loop().run_in_executor(
                    None, detect_modality, file_path
                )

                span.set_attribute("file.modality", modality)
                span.set_attribute("file.mime", mime)

                logger.info(
                    "ingestion_start",
                    modality=modality,
                    mime=mime,
                    file=path.name,
                    checksum=checksum,
                    session_id=session_id,
                )

                # PER-MODALITY MODEL WARM — load ONLY the models this
                # modality needs (e.g. .txt → text embedder; .mp3 →
                # whisper + text embedder; .png → BLIP/SigLIP/text embedder).
                # Other models stay unloaded so VRAM/RAM stay free.
                try:
                    from app.core.model_registry import model_registry
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        model_registry.ensure_for_modality,
                        modality,
                    )
                except Exception as warm_exc:
                    logger.warning(
                        "modality_warmup_failed",
                        modality=modality,
                        error=str(warm_exc),
                        session_id=session_id,
                    )

                handler = INGESTION_HANDLERS.get(modality)
                if not handler:
                    raise ValueError(f"HANDLER_NOT_FOUND: {modality}")

                # RUN HANDLER — async handlers awaited directly, sync handlers run in thread pool.
                # user_id is read from the contextvar set in IngestionPipeline.process_file
                # so individual ingestor signatures don't need to change.
                if asyncio.iscoroutinefunction(handler):
                    docs = await handler(file_path, session_id)
                else:
                    ctx = contextvars.copy_context()
                    docs = await asyncio.get_event_loop().run_in_executor(
                        None, ctx.run, handler, file_path, session_id
                    )

                if not docs:
                    raise ValueError("NO_DOCS_RETURNED")

                docs = _validate_documents(docs, session_id, modality)

                if not docs:
                    raise ValueError("NO_VALID_DOCS")

                if len(docs) > MAX_INGESTED_DOCS:
                    docs = docs[:MAX_INGESTED_DOCS]
                    logger.warning(
                        "doc_limit_applied",
                        limit=MAX_INGESTED_DOCS,
                        modality=modality,
                    )

                latency = round(time.time() - start, 2)

                # PROMETHEUS
                _ingestion_duration.labels(modality=modality).observe(latency)
                _ingestion_docs.labels(modality=modality).observe(len(docs))

                span.set_attribute("docs.count", len(docs))
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "ingestion_success",
                    modality=modality,
                    docs=len(docs),
                    latency=latency,
                    session_id=session_id,
                )

                return docs

        except Exception as exc:
            latency   = round(time.time() - start, 2)
            error_type = type(exc).__name__
            modality_label = "unknown"

            try:
                modality_label = detect_modality(file_path)[0]
            except Exception:
                pass

            _ingestion_errors.labels(
                modality=modality_label,
                error_type=error_type,
            ).inc()

            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)

            logger.error(
                "ingestion_failed",
                file=path.name,
                session_id=session_id,
                error=str(exc),
                error_type=error_type,
                latency=latency,
            )
            raise

        finally:
            if _user_token is not None:
                from app.utils.paths import reset_current_user
                reset_current_user(_user_token)


# SYNC WRAPPER FOR BACKWARD COMPATIBILITY WITH SYNC CALLERS

def route_ingestion_sync(
    file_path: str,
    session_id: str,
    user_id: Optional[str] = None,
) -> List[IngestedDocument]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    route_ingestion(file_path, session_id, user_id=user_id),
                )
                return future.result()
        return loop.run_until_complete(route_ingestion(file_path, session_id, user_id=user_id))
    except RuntimeError:
        return asyncio.run(route_ingestion(file_path, session_id, user_id=user_id))


