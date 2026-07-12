from __future__ import annotations

import asyncio
import json
import os
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.auth.dependencies import get_current_user
from app.auth.models import UserPublic, UserRole
from app.core.config import settings
from app.core.infra_registry import infra
from app.retrieval.bm25_retriever import BM25AggregatorRetriever
from app.utils.logger import get_logger
from app.utils.paths import user_knowledge_base_dir, user_staging_dir

logger = get_logger(__name__)

router = APIRouter()

# Strong references for fire-and-forget model pre-warm tasks (see
# _prewarm_models_for_upload below) — asyncio does not keep an unreferenced
# Task alive, so without this set a GC pass could cancel warm-up mid-load.
_PENDING_PREWARM_TASKS: set = set()


def _prewarm_models_for_upload(ext: str) -> None:
    """Kick off model loading for the likely modality the instant the
    filename is known, instead of waiting for the file to be fully written,
    malware-scanned, hashed, and re-detected via magic bytes in
    route_ingestion(). That whole chain previously gated the first
    model_registry.ensure_for_modality() call, so a 45MB audio upload paid
    for Whisper/pyannote/NER loading (tens of seconds) only AFTER the upload
    had already finished — this runs the load concurrently with it instead.

    Extension-based, so it's a guess (a .doc misdetected as corrupted Word
    or a container mismatch could differ from the magic-byte result) — fire
    and forget. route_ingestion()'s own ensure_for_modality() call is
    idempotent (already-loaded models are skipped), so a wrong guess just
    means the correct models load at their normal (later) time — never
    slower than before, only sometimes faster.
    """
    try:
        from app.ingestion.router import EXT_TO_MODALITY
        from app.core.model_registry import model_registry

        modality = EXT_TO_MODALITY.get(ext)
        if not modality:
            return

        loop = asyncio.get_event_loop()
        # run_in_executor() submits to the thread pool immediately and
        # returns an already-scheduled Future — NOT a coroutine, so it must
        # not be passed through create_task() (raises TypeError). Track the
        # Future itself for the GC-safety/cleanup purpose described above.
        future = loop.run_in_executor(None, model_registry.ensure_for_modality, modality)
        _PENDING_PREWARM_TASKS.add(future)
        future.add_done_callback(_PENDING_PREWARM_TASKS.discard)
    except Exception as exc:
        logger.debug(event="upload_prewarm_skip", ext=ext, error=str(exc))


def get_current_user_id(request: Request, explicit_user_id: Optional[str] = None) -> str:
    """
    Resolve current user id from request.state.user (set by AuthMiddleware from JWT).
    The explicit_user_id form-field parameter is ignored — user_id is always
    sourced from the verified JWT to prevent spoofing.
    Falls back to DEFAULT_DEV_USER_ID only when AUTH_ENABLED=false (dev mode).
    """
    if request.state.user is not None:
        return request.state.user.user_id
    return settings.DEFAULT_DEV_USER_ID

# ALLOWED EXTENSIONS
ALLOWED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".docx", ".xlsx", ".xls",
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif",
    ".tiff", ".tif", ".heic", ".heif",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus",
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ".svg", ".cr2", ".nef", ".arw",
}

# EXTENSION TO MODALITY MAP
_EXT_MODALITY: Dict[str, str] = {
    ".txt": "text", ".md": "text", ".markdown": "text",
    ".csv": "text", ".json": "text", ".log": "text",
    ".pdf": "pdf", ".docx": "docx", ".doc": "docx",
    ".xlsx": "xlsx", ".xls": "xlsx",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".bmp": "image", ".webp": "image", ".gif": "image",
    ".tiff": "image", ".tif": "image",
    ".heic": "image", ".heif": "image",
    ".svg": "image", ".cr2": "image", ".nef": "image", ".arw": "image",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio",
    ".flac": "audio", ".ogg": "audio", ".aac": "audio",
    ".opus": "audio", ".wma": "audio", ".aiff": "audio",
    ".mp4": "video", ".avi": "video", ".mov": "video",
    ".mkv": "video", ".webm": "video", ".flv": "video",
    ".wmv": "video", ".ts": "video",
}

# MODALITY TO SIZE LIMIT MAP
_MODALITY_SIZE_LIMITS: Dict[str, int] = {
    "text":   settings.MAX_FILE_SIZE_TEXT,
    "pdf":    settings.MAX_FILE_SIZE_PDF,
    "docx":   settings.MAX_FILE_SIZE_DOCX,
    "xlsx":   settings.MAX_FILE_SIZE_XLSX,
    "image":  settings.MAX_FILE_SIZE_IMAGE,
    "audio":  settings.MAX_FILE_SIZE_AUDIO,
    "video":  settings.MAX_FILE_SIZE_VIDEO,
}

# LAZY SINGLETONS
_rag_pipeline = None
_query_pipeline_fn = None
_audit_log_enabled: bool = settings.AUDIT_LOG_ENABLED

# IN-MEMORY FILE DEDUP — maps SHA-256 hex → doc_id of first successful ingest
# Bounded LRU so it never grows without limit on long-running servers
_INGEST_DEDUP_MAX = 10_000
_INGEST_DEDUP: Dict[str, str] = {}


def _dedup_set(file_hash: str, request_id: str) -> None:
    if len(_INGEST_DEDUP) >= _INGEST_DEDUP_MAX:
        # evict oldest entry
        try:
            oldest = next(iter(_INGEST_DEDUP))
            del _INGEST_DEDUP[oldest]
        except StopIteration:
            pass
    _INGEST_DEDUP[file_hash] = request_id


def _get_rag_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        from app.pipeline.rag_pipeline import RAGPipeline
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def _get_query_pipeline():
    global _query_pipeline_fn
    if _query_pipeline_fn is None:
        from app.pipeline.query_pipeline import query_pipeline
        _query_pipeline_fn = query_pipeline
    return _query_pipeline_fn


# REQUEST MODELS

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(default="default", max_length=128)
    # Optional: identify which user's knowledge base to query against.
    # Overrides X-User-ID header. Falls back to header → dev default if omitted.
    user_id: Optional[str] = Field(default=None, max_length=128)
    # Optional: restrict retrieval to chunks whose `source` filename
    # contains any of these substrings. Use this to scope a query to a
    # specific uploaded file when the session has many ingested documents.
    sources: Optional[List[str]] = Field(default=None, max_length=20)
    no_cache: bool = Field(default=False)
    force_web: bool = Field(default=False)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = unicodedata.normalize("NFC", v.strip())
        if not v:
            raise ValueError("Query cannot be empty or whitespace")
        return v

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            return "default"
        return v[:128]

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        cleaned = [s.strip()[:256] for s in v if s and s.strip()]
        return cleaned or None


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    vote: str = Field(..., pattern="^(up|down|none)$")
    message_id: Optional[str] = Field(default=None, max_length=128)
    query: Optional[str] = Field(default=None, max_length=500)
    response_snippet: Optional[str] = Field(default=None, max_length=500)


class ClearMemoryRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)


class GDPRPurgeRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)


class SessionUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    pinned: Optional[bool] = None
    archived: Optional[bool] = None

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("Title cannot be empty")
        return v


# HELPERS

def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(text.strip().split())


def _size_limit(ext: str) -> int:
    modality = _EXT_MODALITY.get(ext.lower(), "text")
    return _MODALITY_SIZE_LIMITS.get(modality, settings.MAX_FILE_SIZE_MB * 1024 * 1024)


def _request_id() -> str:
    return str(uuid.uuid4())


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_disk_space(path: Path) -> None:
    try:
        import shutil
        usage = shutil.disk_usage(path)
        min_bytes = getattr(settings, "MIN_FREE_DISK_MB", 500) * 1024 * 1024
        if usage.free < min_bytes:
            logger.warning(
                event="low_disk_space_before_upload",
                free_mb=round(usage.free / 1024 / 1024, 1),
            )
    except Exception:
        pass


def _audit_log(
    event: str,
    request_id: str,
    session_id: str,
    **kwargs: Any,
) -> None:
    if not _audit_log_enabled:
        return
    try:
        logger.info(
            event=f"audit_{event}",
            request_id=request_id,
            session_id=session_id,
            timestamp=time.time(),
            **kwargs,
        )
    except Exception:
        pass


def _rate_limit_check(request: Request) -> None:
    pass


# MALWARE SCAN — CLAMAV

def _malware_scan(file_path: str) -> bool:
    if not getattr(settings, "MALWARE_SCAN_ENABLED", False):
        return True
    try:
        import clamd
        socket_path = getattr(settings, "CLAMAV_SOCKET", "/var/run/clamav/clamd.ctl")
        cd = clamd.ClamdUnixSocket(path=socket_path)
        result = cd.scan(file_path)
        if result and file_path in result:
            status = result[file_path][0]
            if status == "FOUND":
                logger.warning(
                    event="malware_detected",
                    file=os.path.basename(file_path),
                    virus=result[file_path][1],
                )
                return False
        return True
    except ImportError:
        logger.debug(event="clamd_not_installed_skipping_scan")
        return True
    except Exception as exc:
        logger.warning(event="malware_scan_failed", error=str(exc))
        return True


# PROMPT INJECTION CHECK — delegates to unified guardrail (Phase 26)

def _check_prompt_injection(query: str, correlation_id: str = "", session_id: str = "") -> str:
    from app.guardrails.input_guard import sanitize as _guard_sanitize
    return _guard_sanitize(query, surface="api", session_id=session_id, correlation_id=correlation_id)


# PII SCRUB ON RAW QUERY — strip PII before the query enters the pipeline.
# Applied on the short query string (better Presidio accuracy than on the full prompt).

def _scrub_query_pii(query: str, session_id: str = "") -> str:
    try:
        from app.guardrails.pii import scrub_pii as _sp
        cleaned, changed = _sp(query)
        if changed:
            logger.warning(event="query_pii_scrubbed", session_id=session_id,
                           original_len=len(query), scrubbed_len=len(cleaned))
        return cleaned
    except Exception as _e:
        logger.warning(event="query_pii_scrub_failed", error=str(_e))
        return query


# SSRF URL DETECTION IN QUERY TEXT
# Extracts URLs from a query and blocks if any resolve to a private/SSRF-risk address.

_URL_RE = re.compile(r'https?://[^\s\'"<>]+', re.IGNORECASE)

def _check_query_ssrf(query: str) -> bool:
    """Return True if the query contains a URL that is an SSRF risk."""
    try:
        from app.guardrails.ssrf import is_ssrf_risk
        for url in _URL_RE.findall(query):
            if is_ssrf_risk(url):
                return True
    except Exception:
        pass
    return False


# CLEANUP TEMP FILE

def _cleanup_file(file_path: Optional[Path]) -> None:
    if file_path and file_path.exists():
        try:
            file_path.unlink()
        except Exception as exc:
            logger.warning(
                event="upload_cleanup_failed",
                path=str(file_path),
                error=str(exc),
            )


# SHA-256 FILE HASH FOR DEDUP

def _compute_file_hash(file_path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _qdrant_has_checksum(user_id: str, file_hash: str) -> bool:
    """True if chunks with this checksum already exist in Qdrant for this user.

    Qdrant is the source of truth — the in-memory dedup map alone is not, because
    a user can delete a file (purging its vectors) and re-upload it. We only honor
    a dedup hit when the vectors actually still exist for the user.
    """
    try:
        vs = infra.get_vector_store()
        if vs is None:
            return False
        return bool(vs.search_by_payload(
            field="checksum_sha256", value=file_hash, user_id=user_id, limit=1,
        ))
    except Exception:
        return False


# ERROR CODE → HTTP 422 DETAIL PREFIXES THAT INDICATE STRUCTURED INGESTION ERRORS
_INGEST_422_PREFIXES = (
    "EMPTY_FILE",
    "PASSWORD_PROTECTED",
    "CORRUPTED_FILE",
    "EMPTY_DOCUMENT",
    "NO_CONTENT_EXTRACTED",
    "FILE_TOO_LARGE",
    "INVALID_DIMENSIONS",
    "IMAGE_TOO_SMALL",
    "AUDIO_TOO_SHORT",
    "EMPTY_CONTENT",
    "SILENT_AUDIO",
    "FFMPEG_NOT_FOUND",
    "DRM_PROTECTED",
    "SVG_RASTERIZATION_FAILED",
    "PILLOW_HEIF_REQUIRED",
    "INVALID_FILE_TYPE",
    "UNSUPPORTED_TYPE",
    "MIME_NOT_ALLOWED",
    "EXT_NOT_ALLOWED",
    "CORRUPT_IMAGE",
    "IMAGE_LOAD_FAILED",
    "INVALID_IMAGE_DIMENSIONS",
    "MAGIC_BYTE_MIME_MISMATCH",
    "UNSUPPORTED_IMAGE_FORMAT",
    "RAW_DECODE_FAILED",
    # Pipeline processing failures — file was received but produced no usable content
    "NO_VALID_EMBEDDINGS",
    "NO_VALID_AUDIO_SEGMENTS",
    "NO_VALID_CHUNKS",
    "INGESTION_EMPTY",
)


# INGEST

@router.post("/ingest")
async def ingest_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    current_user: UserPublic = Depends(get_current_user),
) -> JSONResponse:
    start = time.time()
    request_id = _request_id()
    file_path: Optional[Path] = None
    user_id = current_user.user_id

    _rate_limit_check(request)

    # Guest upload limit — enforced atomically in Redis (Lua script, no race condition)
    if current_user.role == UserRole.GUEST:
        from app.auth.guest_service import check_and_increment_uploads
        allowed = await asyncio.to_thread(check_and_increment_uploads, user_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Guest upload limit reached. Sign up free to continue.",
                headers={"X-Guest-Upgrade-Required": "true", "X-Guest-Limit-Type": "uploads"},
            )

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file: missing filename")

        filename = Path(file.filename).name
        ext = Path(filename).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}",
            )

        # Fire model pre-warm now, in parallel with the disk write / malware
        # scan / hashing below — see _prewarm_models_for_upload docstring.
        _prewarm_models_for_upload(ext)

        max_size = _size_limit(ext)
        staging_dir = user_staging_dir(user_id)
        _check_disk_space(staging_dir)

        # Write to user staging first (temp), then copy to knowledge_base on success.
        # NOTE: the uniqueness token lives in the DIRECTORY name, not the filename —
        # os.path.basename(file_path) becomes the chunk's `source` / citation tag
        # downstream, so a prefixed filename would leak the random token into every
        # answer's citations and source chips (e.g. "[3f9a1b..._report.pdf]").
        upload_dir = staging_dir / uuid.uuid4().hex
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / filename
        size = 0

        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(settings.UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large for type {ext}: "
                            f"max {max_size // (1024 * 1024)}MB"
                        ),
                    )
                f.write(chunk)

        if size == 0:
            return JSONResponse(
                status_code=422,
                content={"request_id": request_id, "error_code": "EMPTY_FILE", "detail": "Uploaded file is empty"},
            )

        if not _malware_scan(str(file_path)):
            raise HTTPException(status_code=400, detail="File rejected: malware detected")

        # DEDUP CHECK — per-user (tenant-isolated) AND Qdrant-verified.
        # The in-memory map is keyed by user_id:hash so one user's upload never
        # masks another user's identical file. A dedup hit is only honored when the
        # vectors actually still exist in Qdrant for this user — otherwise (e.g. the
        # user deleted the file and re-uploaded it) the stale entry is dropped and
        # the file is re-ingested normally.
        file_hash = await asyncio.to_thread(_compute_file_hash, file_path)
        dedup_key = f"{user_id}:{file_hash}"
        if dedup_key in _INGEST_DEDUP:
            if await asyncio.to_thread(_qdrant_has_checksum, user_id, file_hash):
                latency = round(time.time() - start, 3)
                # Copy to kb_dir so the file shows in the sidebar list even though
                # the vectors are already in Qdrant from a prior upload.
                import shutil as _shutil
                kb_dir = user_knowledge_base_dir(user_id)
                kb_path = kb_dir / filename
                if not kb_path.exists():
                    _shutil.copy2(str(file_path), str(kb_path))
                return JSONResponse(
                    status_code=200,
                    content={
                        "request_id":  request_id,
                        "status":      "duplicate",
                        "duplicate":   True,
                        "file_hash":   file_hash,
                        "filename":    filename,
                        "ext":         ext,
                        "modality":    _EXT_MODALITY.get(ext, "unknown"),
                        "file_size":   size,
                        "latency":     latency,
                    },
                )
            # Stale entry — vectors no longer in Qdrant. Drop it and re-ingest.
            _INGEST_DEDUP.pop(dedup_key, None)
            logger.info(event="dedup_stale_entry_dropped", file=filename, user_id=user_id)

        _audit_log(
            "ingest_received",
            request_id=request_id,
            session_id=session_id,
            file=filename,
            ext=ext,
            size=size,
            ip=_client_ip(request),
        )

        modality = _EXT_MODALITY.get(ext, "unknown")

        # kb_path is the intended destination — the pipeline copies here only on success.
        # We never copy before the pipeline finishes so the file never appears in the
        # sidebar before embeddings are in Qdrant.
        kb_dir  = user_knowledge_base_dir(user_id)
        kb_path = kb_dir / filename

        # REGISTER DEDUP NOW (per-user key) — prevents duplicate submissions
        # while the background job is still processing.
        _dedup_set(dedup_key, request_id)

        # INGEST THREAD READINESS GATE — ensure the gpu_ingest_executor thread
        # has been seeded with PyTorch CUDA context before dispatching any GPU
        # work.  Without this, a lazy-loaded LLM (llama.cpp) might claim the
        # CUDA device first and cause a SIGSEGV when the ingest thread later
        # tries to initialise its own CUBLAS handle.
        from app.core.startup_optimizer import is_ingest_thread_ready, seed_gpu_ingest_thread
        if not is_ingest_thread_ready():
            await asyncio.to_thread(seed_gpu_ingest_thread)

        # LAUNCH BACKGROUND PIPELINE — returns job_id in milliseconds.
        # The background job owns the staging file, copies to kb_path on success,
        # and cleans up staging when done.
        from app.pipeline.ingestion_pipeline import IngestionPipeline
        pipeline_instance = IngestionPipeline()
        job_id = await pipeline_instance.process_file_background(
            str(file_path), session_id, user_id, kb_path=str(kb_path)
        )

        # Hand the staging file to the background job — don't let the finally block delete it.
        file_path = None

        latency = round(time.time() - start, 3)

        _audit_log(
            "ingest_queued",
            request_id=request_id,
            session_id=session_id,
            file=filename,
            job_id=job_id,
            ip=_client_ip(request),
        )

        logger.info(
            event="api_ingest_queued",
            request_id=request_id,
            file=filename,
            ext=ext,
            modality=modality,
            size=size,
            job_id=job_id,
            latency=latency,
            session_id=session_id,
        )

        return JSONResponse(
            status_code=202,
            content={
                "request_id": request_id,
                "status":     "processing",
                "job_id":     job_id,
                "filename":   filename,
                "modality":   modality,
                "file_size":  size,
                "ext":        ext,
                "session_id": session_id,
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            event="api_ingest_failed",
            request_id=request_id,
            error=str(exc),
            session_id=session_id,
        )
        raise HTTPException(status_code=500, detail="Ingestion failed")

    finally:
        # Only cleans up when file_path is still set (i.e. an error occurred before
        # the background job was launched — the job sets file_path=None on success).
        background_tasks.add_task(_cleanup_file, file_path)


# DIRECT LLM GENERATE — eval judge endpoint (no RAG, no guardrails, raw LLM output)
# Only accessible internally; used by GGUFJudge to score Ragas metrics.

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=16000)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)


@router.post("/llm/generate")
async def llm_generate(
    request_body: GenerateRequest,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """Direct LLM generation — bypasses RAG pipeline. Used by eval judge."""
    try:
        from app.core.model_loader import model_loader
        llm = model_loader.get_llm()
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM not loaded")
        result = await asyncio.to_thread(
            llm.generate,
            request_body.prompt,
            max_tokens=request_body.max_tokens,
            temperature=request_body.temperature,
        )
        return {"text": result or "", "status": "ok"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# HEALTH

@router.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status":    "ok",
        "service":   settings.APP_NAME,
        "version":   settings.APP_VERSION,
        "timestamp": time.time(),
    }


# INFRA HEALTH

@router.get("/infra/health")
def infra_health() -> Dict[str, Any]:
    try:
        return {
            "status": "ok",
            "infra":  infra.health_check(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# TOOLS LIST

@router.get("/tools")
def list_tools() -> Dict[str, Any]:
    try:
        from app.agents.tool_registry import ToolRegistry
        registry = ToolRegistry()
        return {
            "status": "ok",
            "tools":  registry.list_tools(),
        }
    except Exception as exc:
        logger.error(event="list_tools_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# QUERY

@router.post("/query")
async def query_rag(
    request_body: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    start = time.time()
    request_id = _request_id()
    session_id = request_body.session_id
    user_id = current_user.user_id

    _rate_limit_check(request)

    try:
        query = _clean(request_body.query)

        if not query:
            raise HTTPException(status_code=400, detail="Empty query after cleaning")

        # GIBBERISH GUARD
        if _is_gibberish(query):
            logger.info(event="query_gibberish_rejected", session_id=session_id)
            return {
                "request_id": request_id, "answer": _gibberish_msg(query),
                "confidence": 0.0, "sources": [], "cache_hit": False,
                "decision": "gibberish_rejected", "latency": round(time.time() - start, 3),
            }

        # PROMPT INJECTION CHECK
        _pre_sanitize = query
        query = _check_prompt_injection(query)

        if not query and _pre_sanitize:
            _blocked_msg = (
                "⚠️ Your message was blocked by the security guardrail. "
                "It matched a restricted pattern (prompt injection or policy violation). "
                "Please rephrase your query."
            )
            logger.warning(event="query_blocked_by_guardrail",
                           request_id=request_id, session_id=session_id)
            return {
                "request_id": request_id,
                "answer":     _blocked_msg,
                "confidence": 0.0,
                "sources":    [],
                "cache_hit":  False,
                "decision":   "blocked",
                "source":     "input_guard",
                "latency":    round(time.time() - start, 3),
            }

        if _check_query_ssrf(query):
            _ssrf_msg = (
                "⚠️ Your message was blocked by the SSRF guardrail. "
                "It contains a URL pointing to a private or restricted network address. "
                "Please remove the URL and try again."
            )
            logger.warning(event="query_blocked_ssrf",
                           request_id=request_id, session_id=session_id)
            return {
                "request_id": request_id,
                "answer":     _ssrf_msg,
                "confidence": 0.0,
                "sources":    [],
                "cache_hit":  False,
                "decision":   "blocked",
                "source":     "ssrf_guard",
                "latency":    round(time.time() - start, 3),
            }

        # PII SCRUB — strip PII from the raw query before it enters the pipeline
        query = _scrub_query_pii(query, session_id=session_id)

        query = query[:settings.MAX_PROMPT_CHARS]

        _audit_log(
            "query_received",
            request_id=request_id,
            session_id=session_id,
            query_len=len(query),
            ip=_client_ip(request),
        )

        pipeline_fn = _get_query_pipeline()

        # IN-FLIGHT JOIN — if /rag/query/stream is mid-generation for this
        # exact (session, query), await its completion instead of running a
        # duplicate 7B generation that contends for the same GPU. The stream
        # writes its guarded answer to the shared query cache before
        # signalling, so the pipeline's cache-hit branch returns the identical
        # text in ~15ms. On stream refusal/error nothing is cached and we
        # compute normally — the fallback path is unchanged, just serialized.
        _stream_joined = False
        try:
            from app.pipeline.query_pipeline import _cache_key as _qck
            _join_ev = _INFLIGHT_STREAMS.get(_qck(session_id, query))
            if _join_ev is not None:
                await asyncio.wait_for(_join_ev.wait(), timeout=120.0)
                _stream_joined = True
                logger.info(event="meta_joined_inflight_stream",
                            request_id=request_id, session_id=session_id)
        except asyncio.TimeoutError:
            logger.warning(event="meta_inflight_stream_timeout",
                           request_id=request_id, session_id=session_id)
        except Exception:
            pass

        result = await asyncio.to_thread(
            pipeline_fn, query, session_id, request_body.sources, user_id,
            # A just-joined stream result IS the fresh regeneration the user
            # asked for — read it from cache even when no_cache was requested.
            request_body.no_cache and not _stream_joined,
        )

        if not isinstance(result, dict) or "answer" not in result:
            raise RuntimeError("Invalid pipeline response")

        latency = round(time.time() - start, 3)

        # PHASE 24.8 — build flat standardised response and validate with AgentResponse
        from app.agents.agent_schema import AgentResponse
        from app.core.config import settings as _cfg

        raw_conf = result.get("confidence", 0.0)
        try:
            raw_conf = float(raw_conf)
        except (TypeError, ValueError):
            raw_conf = 0.0
        confidence = round(max(0.0, min(raw_conf, 1.0)), 6)

        sources = result.get("sources") or []
        if not isinstance(sources, list):
            sources = []

        hallucination_warning = (
            result.get("hallucination_warning", False)
            or (not sources)
            or confidence < _cfg.AGENT_LOW_CONFIDENCE
        )

        validated = AgentResponse(
            response=result.get("answer", ""),
            source=result.get("source", "agent"),
            decision=result.get("decision", "rag"),
            confidence=confidence,
            request_id=result.get("request_id", request_id),
            session_id=session_id,
            latency=latency,
            hallucination_warning=hallucination_warning,
            is_fallback=result.get("is_fallback", False),
        )

        flat_response = {
            "answer":               validated.response,
            "confidence":           validated.confidence,
            "decision":             validated.decision,
            "source":               validated.source,
            "session_id":           session_id,
            "request_id":           request_id,
            "latency":              latency,
            "sources":              sources,
            "is_fallback":          validated.is_fallback,
            "hallucination_warning": validated.hallucination_warning,
            "memory_injected":      bool(result.get("memory_injected", False)),
            "cache_hit":            bool(result.get("cache_hit", False)),
        }

        _audit_log(
            "query_completed",
            request_id=request_id,
            session_id=session_id,
            latency=latency,
        )

        logger.info(
            event="api_query_success",
            request_id=request_id,
            latency=latency,
            session_id=session_id,
            confidence=confidence,
            sources_count=len(sources),
        )

        return flat_response

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            event="api_query_failed",
            request_id=request_id,
            error=str(exc),
            session_id=session_id,
        )
        raise HTTPException(status_code=500, detail="Query processing failed")


# STREAM

# Live LLM tokens from rag.stream() are forwarded as-is (true streaming).
# _stream_chunks is only used to pace CANNED messages — guardrail blocks,
# gibberish rejections, cache hits — so they render with the same typing
# feel as a real generation instead of appearing all at once.
#
# Pacing is ADAPTIVE: the replay always completes in <= _STREAM_MAX_TICKS
# ticks. The old fixed 1-char/8ms replay added ~3s of artificial delay on a
# 400-char answer AFTER generation had already finished (the guarded stream
# yields the complete answer as one token). Short answers keep the smooth
# 1-char feel; long ones grow the chunk instead of the wait.
_STREAM_CHUNK_DELAY_SEC = 0.008
_STREAM_MAX_TICKS = 160          # worst-case replay ≈ 160 × 8ms ≈ 1.3s


def _stream_chunks(text: str):
    chunk = max(1, len(text) // _STREAM_MAX_TICKS)
    for i in range(0, len(text), chunk):
        yield text[i:i + chunk]


# IN-FLIGHT STREAM REGISTRY — the frontend fires /rag/query/stream and
# /rag/query in PARALLEL for every message. Without coordination both run a
# full pipeline, i.e. two concurrent 7B generations contending for the same
# GPU. The stream registers itself here; the meta endpoint awaits the event
# and then serves the stream's cache-written answer (identical text, ~15ms)
# instead of generating a duplicate. Single-worker uvicorn ⇒ in-process dict
# is sufficient.
_INFLIGHT_STREAMS: Dict[str, asyncio.Event] = {}


_VOWELS = frozenset('aeiou')
def _gibberish_msg(query: str) -> str:
    return (
        f'It looks like you typed “{query}”, which doesn’t appear to be '
        "a complete question or message. "
        "Could you please clarify what you need help with?"
    )
# Stale fallback messages produced by old code versions — treat as cache misses
_STALE_CACHE_PHRASES = [
    "The provided documents do not contain the information needed to answer this question.",
]

def _is_gibberish(query: str) -> bool:
    """Return True for random-character or number-jumble inputs that can't be answered."""
    q = query.strip().lower()
    words = q.split()
    if not words or len(q) < 3:
        return False
    # Multi-word queries are almost never gibberish
    if len(words) > 1:
        return False
    token = words[0]
    alpha = re.sub(r'[^a-z]', '', token)
    # Short single tokens are fine (acronyms, tickers: "AAPL", "RAG", "AI")
    if len(alpha) < 4:
        return False
    # Digits mixed with random letters (e.g. "12348fdgsfdg")
    if re.search(r'\d', token):
        return True
    vowel_count = sum(1 for c in alpha if c in _VOWELS)
    # Zero vowels of any length → definite gibberish ("hhshs", "qwrty", "pfft")
    if vowel_count == 0:
        return True
    # Very low vowel ratio — English averages ~38%; gibberish drops below 18%
    if len(alpha) >= 5 and vowel_count / len(alpha) < 0.18:
        return True
    # Long consonant run (5+) — extremely rare in real English words
    max_run = max((len(m.group()) for m in re.finditer(r'[^aeiou]+', alpha)), default=0)
    if max_run >= 5:
        return True
    return False


def _sse(payload: str) -> str:
    # JSON-encode so embedded newlines/quotes can't be mistaken for SSE
    # framing ("\n\n" terminates an event, breaking `data: <raw text>`).
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/query/stream")
async def stream_query(
    request_body: QueryRequest,
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> StreamingResponse:
    request_id = _request_id()
    session_id = request_body.session_id

    _rate_limit_check(request)

    # Guest query limit — atomic Lua check in Redis, fails open on Redis outage
    if current_user.role == UserRole.GUEST:
        from app.auth.guest_service import check_and_increment_queries
        allowed = await asyncio.to_thread(check_and_increment_queries, current_user.user_id)
        if not allowed:
            async def _guest_limit_stream():
                import json as _json
                payload = _json.dumps({"__type__": "guest_limit", "limit_type": "queries"})
                yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(
                _guest_limit_stream(),
                media_type="text/event-stream",
                headers={
                    "X-Guest-Upgrade-Required": "true",
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                },
            )

    try:
        query = _clean(request_body.query)

        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        _pre_sanitize_stream = query
        query = _check_prompt_injection(query)

        if not query and _pre_sanitize_stream:
            _blocked_msg = (
                "⚠️ Your message was blocked by the security guardrail. "
                "It matched a restricted pattern (prompt injection or policy violation). "
                "Please rephrase your query."
            )
            logger.warning(event="stream_query_blocked_by_guardrail", session_id=session_id)

            async def blocked_stream():
                for piece in _stream_chunks(_blocked_msg):
                    yield _sse(piece)
                    await asyncio.sleep(_STREAM_CHUNK_DELAY_SEC)
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                blocked_stream(),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
            )

        if _check_query_ssrf(query):
            _ssrf_msg = (
                "⚠️ Your message was blocked by the SSRF guardrail. "
                "It contains a URL pointing to a private or restricted network address. "
                "Please remove the URL and try again."
            )
            logger.warning(event="stream_query_blocked_ssrf", session_id=session_id)

            async def ssrf_blocked_stream():
                for piece in _stream_chunks(_ssrf_msg):
                    yield _sse(piece)
                    await asyncio.sleep(_STREAM_CHUNK_DELAY_SEC)
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                ssrf_blocked_stream(),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
            )

        # PII SCRUB — strip PII from the raw query before it enters the pipeline
        query = _scrub_query_pii(query, session_id=session_id)

        query = query[:settings.MAX_PROMPT_CHARS]

        # GIBBERISH GUARD — reject random-character inputs before any pipeline work
        if _is_gibberish(query):
            logger.info(event="stream_gibberish_rejected", session_id=session_id)

            _gmsg = _gibberish_msg(query)

            async def gibberish_stream():
                for piece in _stream_chunks(_gmsg):
                    yield _sse(piece)
                    await asyncio.sleep(_STREAM_CHUNK_DELAY_SEC)
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                gibberish_stream(),
                media_type="text/event-stream",
                headers={"X-Request-ID": request_id, "Cache-Control": "no-cache",
                         "X-Accel-Buffering": "no"},
            )

        # WEB SEARCH DETECTION — must run BEFORE the cache check so stale
        # KB answers never mask live web results. Triggers on:
        #   • explicit user intent ("from web", "search online", etc.)
        #   • real-time signals ("today", "stock price", "latest", etc.)
        # When triggered the web tool is called directly and ONLY web source
        # chips are emitted — no KB file sources are shown at all.
        _EXPLICIT_WEB_PHRASES = frozenset({
            "from web", "from the web", "search web", "search the web",
            "get from web", "get it from web", "web search", "find online",
            "search online", "look online", "from internet", "from the internet",
            "find on the internet", "look it up",
        })
        # NOTE: bare "current" and "share price" were removed — both are
        # ordinary terms inside static finance documents ("current rating",
        # "current price target", "12-month price target ... share price of
        # $207.00"), not exclusively live-data requests. They forced ANY
        # matching question to skip the knowledge base entirely ("no KB file
        # sources shown at all" — see below), silently dropping ingested
        # DOCX/PDF context for common document questions. The remaining
        # signals are specific enough to real-time intent to keep.
        _REALTIME_SIGNALS = {
            "today", "now", "latest", "live", "right now",
            "stock price", "price today", "news today",
            "this week", "this month", "currently trading",
        }
        _q_lower = query.lower()
        _is_web_request = (
            request_body.force_web
            or any(phrase in _q_lower for phrase in _EXPLICIT_WEB_PHRASES)
            or any(sig in _q_lower for sig in _REALTIME_SIGNALS)
        )

        if _is_web_request:
            try:
                from app.pipeline.query_pipeline import _get_tool_registry
                _reg = _get_tool_registry()
                _search_tool = _reg.get_optional("search")
                if _search_tool is not None:
                    _tool_out = await asyncio.to_thread(
                        _search_tool.handler, query, {}, session_id
                    ) or {}
                    _web_answer = (_tool_out.get("answer") or "").strip()
                    if _web_answer:
                        logger.info(event="stream_web_search", session_id=session_id)

                        _web_sources = _tool_out.get("sources") or []
                        _web_titles  = _tool_out.get("titles") or []

                        async def web_event_stream():
                            for piece in _stream_chunks(_web_answer):
                                yield _sse(piece)
                                await asyncio.sleep(_STREAM_CHUNK_DELAY_SEC)
                            # Emit the web sources (URL + article title) as a typed
                            # sources event so the client renders them as title+domain
                            # cards below the answer (same contract as the KB path).
                            if _web_sources:
                                _payload = [
                                    {"source": u, "modality": "web",
                                     "title": (_web_titles[_i] if _i < len(_web_titles) else ""),
                                     "page_number": None, "start_time": None}
                                    for _i, u in enumerate(_web_sources) if u
                                ]
                                yield f'data: {{"__type__":"sources","data":{json.dumps(_payload)}}}\n\n'
                            yield "data: [DONE]\n\n"

                        return StreamingResponse(
                            web_event_stream(),
                            media_type="text/event-stream",
                            headers={
                                "X-Request-ID":      request_id,
                                "Cache-Control":     "no-cache",
                                "X-Accel-Buffering": "no",
                            },
                        )
            except Exception as _web_err:
                logger.warning(
                    event="stream_web_search_failed",
                    error=str(_web_err),
                    session_id=session_id,
                )
                # Fall through to cache check / normal RAG if web search fails

        # REDIS CACHE CHECK — skip for web requests and explicit no_cache (regenerate).
        # On a cache hit we stream the cached answer immediately.
        try:
            from app.pipeline.query_pipeline import _cache_get
            cached = None if request_body.no_cache else _cache_get(session_id, query)
        except Exception:
            cached = None

        # Invalidate stale cache entries that contain old fallback message text
        if cached and any(p in (cached.get("answer") or "") for p in _STALE_CACHE_PHRASES):
            logger.info(event="stream_stale_cache_evicted", session_id=session_id)
            cached = None

        if cached and cached.get("answer"):
            cached_answer = str(cached["answer"])
            cached_sources = cached.get("sources") or []
            logger.debug(event="stream_cache_hit", session_id=session_id)

            async def cached_event_stream():
                for piece in _stream_chunks(cached_answer):
                    yield _sse(piece)
                    await asyncio.sleep(_STREAM_CHUNK_DELAY_SEC)
                if cached_sources:
                    import json as _json
                    yield f'data: {{"__type__":"sources","data":{_json.dumps(cached_sources)}}}\n\n'
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                cached_event_stream(),
                media_type="text/event-stream",
                headers={
                    "X-Request-ID":      request_id,
                    "X-Cache":           "HIT",
                    "Cache-Control":     "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # Cache miss — run the full LLM pipeline
        rag = _get_rag_pipeline()
        _stream_user_id = current_user.user_id

        # Register as the in-flight generation for this (session, query) so the
        # parallel /rag/query call joins this result instead of duplicating it.
        from app.pipeline.query_pipeline import _cache_key as _qck
        _inflight_key = _qck(session_id, query)
        _inflight_ev = asyncio.Event()
        _INFLIGHT_STREAMS[_inflight_key] = _inflight_ev

        generator = await asyncio.to_thread(
            rag.stream,
            query,
            session_id,
            _stream_user_id,
            request_body.sources,
        )

        async def event_stream():
            _answer_parts: list = []
            _final_answer: Optional[str] = None
            _sources_payload: Optional[str] = None
            _refused = False
            _loop = asyncio.get_running_loop()
            _STOP = object()

            def _next_token():
                # generator.__next__ blocks on llama.cpp between tokens — it must
                # never run on the event loop or it stalls every other request.
                try:
                    return next(generator)
                except StopIteration:
                    return _STOP

            try:
                while True:
                    token = await _loop.run_in_executor(None, _next_token)
                    if token is _STOP:
                        break
                    if not token:
                        continue
                    if token.startswith("\x00REFUSAL\x00"):
                        _refused = True
                        yield 'data: {"__type__":"refusal"}\n\n'
                        continue
                    if token.startswith("\x00REPLACE\x00"):
                        # Canonical guarded answer — supersedes the streamed tokens
                        # for both the client bubble and persistence below.
                        _final_answer = token[9:]
                        import json as _json
                        yield f'data: {{"__type__":"replace","data":{_json.dumps(_final_answer)}}}\n\n'
                        continue
                    if token.startswith("\x00SOURCES\x00"):
                        _sources_payload = token[9:]
                        yield f'data: {{"__type__":"sources","data":{token[9:]}}}\n\n'
                        continue
                    _answer_parts.append(token)
                    yield _sse(token)

                # PERSIST BEFORE [DONE] — the client patches/reads the stored turn
                # the moment it sees [DONE]; writing after it would race that read.
                _full_answer = (_final_answer or "".join(_answer_parts)).strip()
                _src: list = []
                if _sources_payload:
                    import json as _json
                    try:
                        _src = _json.loads(_sources_payload)
                    except Exception:
                        _src = []
                if _full_answer and not _refused:
                    # Single canonical persistence: Redis short-term memory +
                    # Mongo chat transcript + auto-summarize trigger. This used
                    # to be the parallel /rag/query (meta) call's job — that
                    # call is now only made by the client on refusal fallback.
                    try:
                        from app.pipeline.query_pipeline import _store_interaction
                        await asyncio.to_thread(
                            _store_interaction,
                            session_id,
                            query,
                            _full_answer,
                            None,
                            user_id=_stream_user_id,
                            sources=_src,
                        )
                    except Exception as _se:
                        logger.warning(event="stream_save_turn_failed",
                                       session_id=session_id, error=str(_se))
                    try:
                        # ALWAYS write the fresh result — including no_cache
                        # (regenerate): no_cache means "don't READ stale", not
                        # "don't share fresh". The joined /rag/query waiter and
                        # the next page-reload both need this entry; on
                        # regenerate we intentionally OVERWRITE the old answer.
                        from app.pipeline.query_pipeline import _cache_get, _cache_set
                        if request_body.no_cache or not await asyncio.to_thread(_cache_get, session_id, query):
                            await asyncio.to_thread(_cache_set, session_id, query, {
                                    "answer":                _full_answer,
                                    "sources":               _src,
                                    "sources_used":          len(_src),
                                    "confidence":            0.9,
                                    "decision":              "rag",
                                    "source":                "stream",
                                    "session_id":            session_id,
                                    "request_id":            request_id,
                                    "is_fallback":           False,
                                    "hallucination_warning": False,
                                    "metadata":              {"from_stream": True},
                                })
                    except Exception as _ce:
                        logger.debug(event="stream_cache_write_failed",
                                     session_id=session_id, error=str(_ce))

                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error(
                    event="api_stream_failed",
                    request_id=request_id,
                    error=str(exc),
                    session_id=session_id,
                )
                yield "data: [Stream interrupted]\n\n"
            finally:
                # Wake the parallel /rag/query waiter AFTER the cache write so
                # it reads the freshly stored answer. Always signal — on
                # refusal, error, or client disconnect the waiter just computes
                # its own answer instead of waiting out the timeout.
                _INFLIGHT_STREAMS.pop(_inflight_key, None)
                _inflight_ev.set()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "X-Request-ID":      request_id,
                "Cache-Control":     "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        # If we registered as in-flight but failed before streaming began,
        # release any parallel /rag/query waiter immediately.
        try:
            _INFLIGHT_STREAMS.pop(_inflight_key, None)  # type: ignore[has-type]
            _inflight_ev.set()                           # type: ignore[has-type]
        except NameError:
            pass
        logger.error(
            event="api_stream_error",
            request_id=request_id,
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail="Streaming failed")


# UPLOAD

@router.post("/upload")
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    start = time.time()
    request_id = _request_id()
    file_path: Optional[Path] = None
    user_id = current_user.user_id

    _rate_limit_check(request)

    # Guest upload limit — same atomic Lua check as /ingest
    if current_user.role == UserRole.GUEST:
        from app.auth.guest_service import check_and_increment_uploads
        allowed = await asyncio.to_thread(check_and_increment_uploads, user_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Guest upload limit reached. Sign up free to continue.",
                headers={"X-Guest-Upgrade-Required": "true", "X-Guest-Limit-Type": "uploads"},
            )

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file: missing filename")

        filename = Path(file.filename).name
        ext = Path(filename).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}",
            )

        # Fire model pre-warm now, in parallel with the disk write / malware
        # scan / hashing below — see _prewarm_models_for_upload docstring.
        _prewarm_models_for_upload(ext)

        max_size = _size_limit(ext)

        # DISK SPACE GUARD
        staging_dir = user_staging_dir(user_id)
        _check_disk_space(staging_dir)

        # See ingest(): uniqueness token goes in the directory, not the filename,
        # so the clean basename flows through to citations / source chips.
        upload_dir = staging_dir / uuid.uuid4().hex
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / filename
        size = 0

        # STREAM SAVE WITH SIZE CHECK
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(settings.UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large for type {ext}: "
                            f"max {max_size // (1024 * 1024)}MB"
                        ),
                    )
                f.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # MALWARE SCAN
        if not _malware_scan(str(file_path)):
            raise HTTPException(
                status_code=400,
                detail="File rejected: malware detected",
            )

        # PERSIST TO KNOWLEDGE BASE — copy original file before pipeline may clean staging
        import shutil as _shutil
        kb_dir = user_knowledge_base_dir(user_id)
        kb_path = kb_dir / filename
        _shutil.copy2(str(file_path), str(kb_path))

        _audit_log(
            "upload_received",
            request_id=request_id,
            session_id=session_id,
            file=filename,
            ext=ext,
            size=size,
            ip=_client_ip(request),
        )

        # INGEST THREAD READINESS GATE — mirrors the gate in /ingest.
        from app.core.startup_optimizer import is_ingest_thread_ready, seed_gpu_ingest_thread
        if not is_ingest_thread_ready():
            await asyncio.to_thread(seed_gpu_ingest_thread)

        # INGEST — run on the dedicated gpu_ingest_executor whose single thread
        # was pre-seeded with PyTorch CUDA context during startup warmup
        # (or by seed_gpu_ingest_thread() above).  Using asyncio.to_thread()
        # here dispatches to a random OS thread that has no CUDA context; after
        # llama.cpp initialises its CUBLAS handle that causes a SIGSEGV.
        from app.pipeline.ingestion_pipeline import process_file
        from app.core.startup_optimizer import get_gpu_ingest_executor
        _ingest_loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                _ingest_loop.run_in_executor(
                    get_gpu_ingest_executor(),
                    process_file, str(file_path), session_id, user_id,
                ),
                timeout=settings.FILE_PROCESSING_TIMEOUT_SEC,
            )
        except Exception as exc:
            err_str = str(exc).removeprefix("INGESTION_FAILED: ").strip()
            if any(err_str.startswith(p) for p in _INGEST_422_PREFIXES):
                error_code = err_str.split(":")[0].strip()
                return JSONResponse(
                    status_code=422,
                    content={
                        "request_id": request_id,
                        "error_code": error_code,
                        "detail": err_str,
                    },
                )
            raise

        if not result or result.get("status") not in ("success", "partial_failure"):
            error_detail = result.get("error", "Ingestion failed") if result else "Ingestion failed"
            raise RuntimeError(error_detail)

        chunks = result.get("chunks", 0)
        stored = result.get("stored", 0)

        if chunks <= 0:
            raise RuntimeError("No usable content extracted from file")

        latency = round(time.time() - start, 3)

        _audit_log(
            "upload_completed",
            request_id=request_id,
            session_id=session_id,
            file=filename,
            chunks=chunks,
            stored=stored,
            latency=latency,
        )

        logger.info(
            event="api_upload_success",
            request_id=request_id,
            file=filename,
            ext=ext,
            size=size,
            chunks=chunks,
            stored=stored,
            latency=latency,
            session_id=session_id,
        )

        return {
            "request_id":     request_id,
            "status":         result.get("status", "success"),
            "filename":       filename,
            "file_size":      size,
            "file_size_mb":   round(size / (1024 * 1024), 2),
            "ext":            ext,
            "modality":       _EXT_MODALITY.get(ext, "unknown"),
            "chunks":         chunks,
            "stored":         stored,
            "latency":        latency,
            "pipeline_events": result.get("events", []),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            event="api_upload_failed",
            request_id=request_id,
            error=str(exc),
            session_id=session_id,
        )
        raise HTTPException(status_code=500, detail="File upload failed")

    finally:
        background_tasks.add_task(_cleanup_file, file_path)


# CLEAR MEMORY

@router.post("/memory/clear")
async def clear_memory(
    request_body: ClearMemoryRequest,
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    request_id = _request_id()
    session_id = request_body.session_id
    user_id = current_user.user_id

    try:
        from app.memory.memory_manager import MemoryManager
        manager = MemoryManager()
        await asyncio.to_thread(manager.clear, session_id)

        _audit_log(
            "memory_cleared",
            request_id=request_id,
            session_id=session_id,
            ip=_client_ip(request),
        )

        logger.info(
            event="api_memory_clear_success",
            request_id=request_id,
            session_id=session_id,
        )

        return {
            "request_id": request_id,
            "status":     "ok",
            "session_id": session_id,
            "message":    "Session memory cleared",
        }

    except Exception as exc:
        logger.error(
            event="api_memory_clear_failed",
            request_id=request_id,
            error=str(exc),
            session_id=session_id,
        )
        raise HTTPException(status_code=500, detail="Memory clear failed")


# CLEAR QUERY RESPONSE CACHE

@router.post("/cache/clear")
async def clear_query_cache(
    request: Request,
) -> Dict[str, Any]:
    """Flush all query-response cache entries (qresp:* keys in Redis).
    Use this when cached answers are stale or were generated by MockLLM."""
    request_id = _request_id()

    try:
        from app.core.infra_registry import infra
        memory = infra.get_memory()
        deleted = 0
        if memory and hasattr(memory, "cache_flush_query_cache"):
            deleted = await asyncio.to_thread(memory.cache_flush_query_cache)

        logger.info(
            event="api_query_cache_cleared",
            request_id=request_id,
            deleted=deleted,
            ip=_client_ip(request),
        )

        return {
            "request_id": request_id,
            "status":     "ok",
            "deleted":    deleted,
            "message":    f"Query response cache cleared ({deleted} entries removed)",
        }

    except Exception as exc:
        logger.error(
            event="api_query_cache_clear_failed",
            request_id=request_id,
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail="Cache clear failed")


# GDPR PURGE

@router.post("/memory/purge")
async def gdpr_purge(
    request_body: GDPRPurgeRequest,
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    request_id = _request_id()
    # Admins may purge any user_id; regular users may only purge their own data
    if current_user.role.value == "admin":
        user_id = request_body.user_id
    else:
        user_id = current_user.user_id

    if not getattr(settings, "GDPR_PURGE_ENABLED", True):
        raise HTTPException(status_code=403, detail="GDPR purge not enabled")

    try:
        from app.memory.memory_manager import MemoryManager
        manager = MemoryManager()
        await asyncio.to_thread(manager.gdpr_purge, user_id)

        # BM25 PURGE — clear all per-modality indexes for this user
        try:
            agg = BM25AggregatorRetriever(user_id=user_id)
            await asyncio.to_thread(agg.clear, user_id)
        except Exception as exc:
            logger.warning(event="gdpr_bm25_purge_failed", error=str(exc))

        _audit_log(
            "gdpr_purge",
            request_id=request_id,
            session_id=user_id,
            ip=_client_ip(request),
        )

        logger.info(
            event="api_gdpr_purge_success",
            request_id=request_id,
            user_id=user_id,
        )

        return {
            "request_id": request_id,
            "status":     "ok",
            "user_id":    user_id,
            "message":    "All user data purged",
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            event="api_gdpr_purge_failed",
            request_id=request_id,
            error=str(exc),
            user_id=user_id,
        )
        raise HTTPException(status_code=500, detail="GDPR purge failed")


# SESSION HISTORY

@router.get("/memory/history/{session_id}")
async def get_history(
    session_id: str,
    limit: int = 20,
) -> Dict[str, Any]:
    request_id = _request_id()

    try:
        from app.memory.memory_manager import MemoryManager
        manager = MemoryManager()
        history = await asyncio.to_thread(manager.get_history, session_id, limit)

        return {
            "request_id": request_id,
            "status":     "ok",
            "session_id": session_id,
            "count":      len(history),
            "history":    [
                {
                    "role":      msg.get("role"),
                    "content":   msg.get("content", "")[:500],
                    "modality":  msg.get("modality", "text"),
                    "timestamp": msg.get("timestamp"),
                }
                for msg in history
            ],
        }

    except Exception as exc:
        logger.error(
            event="api_history_failed",
            request_id=request_id,
            error=str(exc),
            session_id=session_id,
        )
        raise HTTPException(status_code=500, detail="History fetch failed")


# METRICS

@router.get("/metrics")
def metrics() -> Dict[str, Any]:
    if not getattr(settings, "PROMETHEUS_ENABLED", False):
        return {
            "status":  "disabled",
            "message": "Set PROMETHEUS_ENABLED=true to enable metrics",
        }

    try:
        from app.core.model_loader import model_loader
        return {
            "status": "ok",
            "models": model_loader.health_check(),
            "infra":  infra.health_check(),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# MODEL HEALTH

@router.get("/models/health")
def model_health() -> Dict[str, Any]:
    try:
        from app.core.model_loader import model_loader
        return {
            "status": "ok",
            "models": model_loader.health_check(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))




# ─── KNOWLEDGE BASE ───────────────────────────────────────────────────────────

def _qdrant_embedded_sources(user_id: str) -> set:
    """Return the set of source filenames that have at least one point in Qdrant for this user.

    Uses a single scroll (limit=1000) — fast and sufficient for any realistic KB size.
    Returns empty set on any error so callers degrade gracefully.
    """
    try:
        vs = infra.get_vector_store()
        if vs is None or not hasattr(vs, "client"):
            return set()
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        sources: set = set()
        for collection in (vs.text_collection, vs.vision_collection):
            if collection not in getattr(vs, "_collection_cache", {}):
                continue
            offset = None
            while True:
                points, next_offset = vs.client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]),
                    with_payload=True,
                    with_vectors=False,
                    limit=500,
                    offset=offset,
                )
                for pt in points:
                    src = (pt.payload or {}).get("source")
                    if src:
                        sources.add(src)
                if next_offset is None:
                    break
                offset = next_offset
        return sources
    except Exception:
        return set()


@router.get("/knowledge-base")
async def list_knowledge_base(
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """List all files in the authenticated user's knowledge base."""
    user_id = current_user.user_id
    from app.utils.paths import user_knowledge_base_dir
    kb_dir = user_knowledge_base_dir(user_id)

    # Get the set of filenames that actually have embeddings in Qdrant.
    # Files on disk but absent from Qdrant are orphaned (upload failed mid-run).
    # We delete them silently so they stop showing in the sidebar.
    embedded = await asyncio.to_thread(_qdrant_embedded_sources, user_id)

    files = []
    for f in sorted(kb_dir.iterdir()):
        if not f.is_file():
            continue
        # Prune orphans: file on disk but no Qdrant points.
        if embedded and f.name not in embedded:
            try:
                f.unlink(missing_ok=True)
                logger.info(event="kb_orphan_removed", user_id=user_id, file=f.name)
            except Exception:
                pass
            continue
        stat = f.stat()
        files.append({
            "filename":      f.name,
            "size_bytes":    stat.st_size,
            "size_mb":       round(stat.st_size / (1024 * 1024), 3),
            "uploaded_at":   stat.st_mtime,
            "modality":      _EXT_MODALITY.get(f.suffix.lower(), "unknown"),
        })
    return {
        "user_id":    user_id,
        "file_count": len(files),
        "files":      files,
    }


@router.delete("/knowledge-base/{filename}")
async def delete_knowledge_base_file(
    filename: str,
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Permanently delete a file from the authenticated user's knowledge base AND
    purge its vectors from Qdrant and BM25 so it no longer appears in query results.
    """
    user_id = current_user.user_id
    request_id = _request_id()

    # Sanitize filename — prevent path traversal
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    from app.utils.paths import user_knowledge_base_dir
    kb_dir = user_knowledge_base_dir(user_id)
    file_path = kb_dir / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {safe_name}")

    # PURGE FROM QDRANT — must succeed before the file is removed from disk.
    # Failing silently here would leave orphaned vectors that keep appearing in
    # query results for a file the user believes is deleted.
    qdrant_purged = False
    try:
        vs = infra.get_vector_store()
        if vs:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            # Include user_id so one user's delete never touches another user's vectors.
            _filter = Filter(
                must=[
                    FieldCondition(key="source",  match=MatchValue(value=safe_name)),
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                ]
            )
            vs.client.delete(collection_name=vs.text_collection, points_selector=_filter)
            # Vision collection is best-effort — missing collection is not fatal
            try:
                vs.client.delete(collection_name=vs.vision_collection, points_selector=_filter)
            except Exception:
                pass
            qdrant_purged = True
        else:
            # No vector store configured — treat as purged so deletion can proceed
            qdrant_purged = True
    except Exception as exc:
        logger.error(event="kb_delete_qdrant_failed", file=safe_name, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail="Failed to purge vectors from the knowledge base index. File was NOT deleted. Please try again.",
        )

    # PURGE FROM BM25 — best-effort; missing BM25 index entry does not block deletion.
    # Use BM25AggregatorRetriever so all per-modality pkl files (txt, pdf, docx, xlsx,
    # image, audio, video) are cleaned up — infra.get_bm25() only covers the legacy
    # single-index path and would silently miss the per-modality indexes.
    bm25_purged = False
    try:
        agg = BM25AggregatorRetriever(user_id=user_id)
        agg.delete_by_source(safe_name, user_id=user_id)
        bm25_purged = True
    except Exception as exc:
        # BM25 failure is non-fatal — Qdrant is the primary store. Log and continue.
        logger.warning(event="kb_delete_bm25_failed", file=safe_name, error=str(exc))

    # FLUSH QUERY CACHE — stale cached answers that referenced this file must not
    # be served after deletion. Flush all qresp:* entries for this user's session.
    cache_flushed = 0
    try:
        memory = infra.get_memory()
        if memory and hasattr(memory, "cache_flush_query_cache"):
            cache_flushed = await asyncio.to_thread(memory.cache_flush_query_cache)
            logger.info(event="kb_delete_cache_flushed", file=safe_name, entries=cache_flushed)
    except Exception as exc:
        logger.warning(event="kb_delete_cache_flush_failed", file=safe_name, error=str(exc))

    # CLEAR DEDUP ENTRY — so a re-upload of the same file re-ingests instead of
    # being skipped as a duplicate. Compute the hash before unlinking.
    try:
        _del_hash = _compute_file_hash(file_path)
        _INGEST_DEDUP.pop(f"{user_id}:{_del_hash}", None)
    except Exception:
        pass

    # DELETE FILE — only reached if Qdrant purge succeeded
    file_path.unlink()

    _audit_log(
        "knowledge_base_delete",
        request_id=request_id,
        session_id=user_id,
        file=safe_name,
        ip=_client_ip(request),
    )

    logger.info(event="knowledge_base_file_deleted", user_id=user_id, file=safe_name,
                qdrant_purged=qdrant_purged, bm25_purged=bm25_purged,
                cache_flushed=cache_flushed)

    return {
        "request_id":    request_id,
        "status":        "deleted",
        "filename":      safe_name,
        "user_id":       user_id,
        "qdrant_purged": qdrant_purged,
        "bm25_purged":   bm25_purged,
        "cache_flushed": cache_flushed,
    }


# ─── CHAT SESSIONS (RECENTS) ──────────────────────────────────────────────────

@router.get("/sessions")
async def list_chat_sessions(
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """List the authenticated user's saved chats for the Recents sidebar."""
    user_id = current_user.user_id
    request_id = _request_id()

    try:
        mongo = infra.get_mongo()
        sessions = await asyncio.to_thread(mongo.list_chat_sessions, user_id) if mongo else []
        return {
            "request_id": request_id,
            "status":     "ok",
            "count":      len(sessions),
            "sessions":   sessions,
        }
    except Exception as exc:
        logger.error(event="api_sessions_list_failed", request_id=request_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to load chat history")


@router.get("/sessions/{session_id}")
async def get_chat_session(
    session_id: str,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """Fetch one chat's full transcript — used when a Recents entry is opened."""
    user_id = current_user.user_id
    request_id = _request_id()

    try:
        mongo = infra.get_mongo()
        session = await asyncio.to_thread(mongo.get_chat_session, user_id, session_id) if mongo else None
        if not session:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {"request_id": request_id, "status": "ok", **session}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(event="api_session_fetch_failed", request_id=request_id, session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to load chat")


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """Permanently delete a chat session — removes the session record, all
    messages, all summaries from MongoDB, and the live Redis context."""
    user_id = current_user.user_id
    request_id = _request_id()

    try:
        # 1. Wipe MongoDB: sessions + messages + summaries.
        # Treat as idempotent: if the sessions doc was already gone but messages
        # remain, we still clean everything and return 200 — never 404.
        mongo = infra.get_mongo()
        if mongo:
            await asyncio.to_thread(mongo.delete_chat_session_all, user_id, session_id)

        # 2. Clear Redis live context so the session can't be resumed from cache
        try:
            redis_mem = infra.get_memory()
            if redis_mem is not None:
                await asyncio.to_thread(redis_mem.delete, session_id)
        except Exception as exc:
            logger.warning(event="session_delete_redis_failed", session_id=session_id, error=str(exc))

        _audit_log(
            "chat_session_delete",
            request_id=request_id,
            session_id=user_id,
            chat_session_id=session_id,
            ip=_client_ip(request),
        )

        return {"request_id": request_id, "status": "ok", "deleted": True, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(event="api_session_delete_failed", request_id=request_id, session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to delete chat")


@router.delete("/sessions")
async def delete_all_chat_sessions(
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """Permanently delete ALL chat sessions for the current user."""
    user_id = current_user.user_id
    request_id = _request_id()
    try:
        mongo = infra.get_mongo()
        count = await asyncio.to_thread(mongo.delete_all_chat_sessions, user_id) if mongo else 0

        # Also flush Redis short-term memory for this user
        try:
            redis_mem = infra.get_redis_memory()
            if redis_mem:
                await asyncio.to_thread(redis_mem.delete_all_user_sessions, user_id)
        except Exception:
            pass

        _audit_log("all_sessions_deleted", request_id=request_id,
                   session_id=user_id, count=count, ip=_client_ip(request))
        return {"request_id": request_id, "status": "ok", "deleted": count}
    except Exception as exc:
        logger.error(event="api_delete_all_sessions_failed", request_id=request_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to delete history")


class LastMessagePatchRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=50000)
    sources: Optional[List[Dict[str, Any]]] = Field(default=None)
    msg_id: Optional[str] = Field(default=None, max_length=128)


@router.patch("/sessions/{session_id}/last-message")
async def patch_last_message(
    session_id: str,
    body: LastMessagePatchRequest,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """Overwrite the last assistant message content + sources so reload shows
    exactly what the user saw during streaming."""
    user_id = current_user.user_id
    try:
        mongo = infra.get_mongo()
        if mongo:
            await asyncio.to_thread(
                mongo.patch_last_assistant_message,
                session_id,
                user_id,
                body.content,
                body.sources or [],
                body.msg_id,
            )
        return {"status": "ok"}
    except Exception as exc:
        logger.warning(event="api_patch_last_msg_failed", session_id=session_id, error=str(exc))
        return {"status": "error"}


@router.patch("/sessions/{session_id}")
async def update_chat_session(
    session_id: str,
    body: SessionUpdateRequest,
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """Rename, pin/unpin, or archive/unarchive a chat in Recents."""
    user_id = current_user.user_id
    request_id = _request_id()

    fields = body.model_dump(exclude_unset=True, exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    try:
        mongo = infra.get_mongo()
        updated = await asyncio.to_thread(mongo.update_chat_session, user_id, session_id, fields) if mongo else False
        if not updated:
            raise HTTPException(status_code=404, detail="Chat not found")

        _audit_log(
            "chat_session_update",
            request_id=request_id,
            session_id=user_id,
            chat_session_id=session_id,
            fields=list(fields.keys()),
            ip=_client_ip(request),
        )

        return {"request_id": request_id, "status": "ok", "session_id": session_id, **fields}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(event="api_session_update_failed", request_id=request_id, session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to update chat")


@router.post("/feedback")
async def submit_feedback(
    body: FeedbackRequest,
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """Record a thumbs-up / thumbs-down rating for an assistant message."""
    user_id = current_user.user_id
    request_id = _request_id()

    try:
        mongo = infra.get_mongo()
        if mongo:
            _vote_val = None if body.vote == "none" else body.vote
            saved, _ = await asyncio.gather(
                asyncio.to_thread(
                    mongo.save_feedback,
                    user_id, body.session_id, body.vote,
                    body.message_id, body.query, body.response_snippet,
                ) if body.vote != "none" else asyncio.sleep(0),
                asyncio.to_thread(
                    mongo.patch_message_vote,
                    user_id, body.session_id, body.message_id or "", _vote_val,
                ) if body.message_id else asyncio.sleep(0),
            )
        else:
            saved = False

        _audit_log(
            "feedback_submitted",
            request_id=request_id,
            session_id=user_id,
            chat_session_id=body.session_id,
            vote=body.vote,
            ip=_client_ip(request),
        )

        return {"request_id": request_id, "status": "ok", "saved": saved}

    except Exception as exc:
        logger.error(event="api_feedback_failed", request_id=request_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to save feedback")


# ─────────────────────────────────────────────────────────────────
# PHASE 8 NEW ENDPOINTS
# ─────────────────────────────────────────────────────────────────

# GET /api/ingestion/status/{job_id} — poll IngestJob progress

@router.get("/ingestion/status/{job_id}")
async def get_ingestion_status(
    job_id: str,
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """Poll ingestion job progress (for audio/video uploads)."""
    from app.pipeline.ingestion_pipeline import get_ingest_job
    job = get_ingest_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # Audio/video transcription (Whisper + diarization) is the long phase and
    # reports no incremental progress on its own, so the upload bar would stall.
    # Synthesise a smooth estimate from elapsed time vs an expected duration
    # derived from file size (~6s per MB observed for large-v3 on GPU), capped
    # below 1.0 so it never shows "done" before the pipeline actually finishes.
    try:
        if (job.get("status") in ("extracting", "queued")
                and str(job.get("modality")) in ("mp3", "mp4")
                and float(job.get("progress") or 0.0) < 0.9
                and float(job.get("started_at") or 0.0) > 0.0):
            elapsed  = time.time() - float(job["started_at"])
            size_mb  = max(1.0, float(job.get("size_bytes") or 0) / (1024 * 1024))
            expected = max(20.0, size_mb * 6.0)
            est      = min(0.90, elapsed / expected)
            job["progress"] = max(float(job.get("progress") or 0.0), round(est, 3))
    except Exception:
        pass
    return job


# GET /api/sources/{chunk_id} — return full chunk metadata for TranscriptViewer

@router.get("/api/sources/{chunk_id}")
async def get_source_chunk(
    chunk_id: str,
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return full metadata for a single chunk (used by TranscriptViewer)."""
    user_id = current_user.user_id
    try:
        vs = infra.get_vector_store()
        if vs is None:
            raise HTTPException(status_code=503, detail="Vector store unavailable")
        results = vs.search_by_payload(
            field="chunk_id",
            value=chunk_id,
            user_id=user_id,
            session_id=None,
            limit=1,
        )
        if not results:
            raise HTTPException(status_code=404, detail="Chunk not found")
        chunk = results[0]
        meta  = chunk.get("metadata", {}) or {}
        if meta.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Forbidden")
        return {
            "chunk_id":   chunk_id,
            "text":       chunk.get("text", ""),
            "metadata":   meta,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(event="api_source_chunk_failed", chunk_id=chunk_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to fetch chunk")


# GET /api/kb/files — list all files in user's KB

@router.get("/api/kb/files")
async def list_kb_files(
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """List all ingested files in the user's knowledge base."""
    user_id = current_user.user_id
    try:
        vs = infra.get_vector_store()
        if vs is None:
            return {"files": [], "count": 0}
        results = vs.search_by_payload(
            field="user_id",
            value=user_id,
            user_id=user_id,
            session_id=None,
            limit=1000,
        )
        # Group by source filename
        seen:  Dict[str, Dict] = {}
        for r in results:
            meta   = r.get("metadata", {}) or {}
            src    = meta.get("source") or meta.get("source_file") or ""
            fhash  = meta.get("checksum_sha256", "") or meta.get("file_hash", "")
            if src not in seen:
                seen[src] = {
                    "filename":    src,
                    "modality":    meta.get("modality", "unknown"),
                    "file_hash":   fhash,
                    "chunk_count": 0,
                    "ingested_at": meta.get("ingestion_time", ""),
                }
            seen[src]["chunk_count"] += 1
        files = sorted(seen.values(), key=lambda x: x["ingested_at"], reverse=True)
        return {"files": files, "count": len(files)}
    except Exception as exc:
        logger.error(event="api_kb_files_failed", user_id=user_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to list KB files")


# DELETE /api/kb/files/{file_hash} — delete a specific file from KB

@router.delete("/api/kb/files/{file_hash}")
async def delete_kb_file(
    file_hash: str,
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete all chunks for a file from the user's knowledge base."""
    user_id = current_user.user_id
    try:
        vs = infra.get_vector_store()
        if vs is None:
            raise HTTPException(status_code=503, detail="Vector store unavailable")
        results = vs.search_by_payload(
            field="checksum_sha256",
            value=file_hash,
            user_id=user_id,
            session_id=None,
            limit=1000,
        )
        if not results:
            raise HTTPException(status_code=404, detail="File not found in KB")
        ids_to_delete = []
        for r in results:
            meta = r.get("metadata", {}) or {}
            if meta.get("user_id") == user_id:
                point_id = r.get("id")
                if point_id:
                    ids_to_delete.append(point_id)
        if not ids_to_delete:
            raise HTTPException(status_code=403, detail="No matching chunks for this user")
        vs.delete_by_ids(ids_to_delete)
        return {"deleted_chunks": len(ids_to_delete), "file_hash": file_hash, "status": "ok"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(event="api_kb_delete_failed", file_hash=file_hash, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to delete file")


# GET /api/transcript/{file_hash} — full ordered transcript for audio/video

@router.get("/api/transcript/{file_hash}")
async def get_transcript(
    file_hash: str,
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return full ordered transcript chunks for an audio/video file."""
    user_id = current_user.user_id
    try:
        vs = infra.get_vector_store()
        if vs is None:
            raise HTTPException(status_code=503, detail="Vector store unavailable")
        results = vs.search_by_payload(
            field="checksum_sha256",
            value=file_hash,
            user_id=user_id,
            session_id=None,
            limit=2000,
        )
        chunks = []
        for r in results:
            meta = r.get("metadata", {}) or {}
            if meta.get("user_id") != user_id:
                continue
            modality = meta.get("modality", "")
            if modality not in ("mp3", "mp4", "audio", "video"):
                continue
            chunks.append({
                "chunk_id":       r.get("id", ""),
                "text":           r.get("text", ""),
                "start":          meta.get("start_timestamp") or meta.get("timestamp_start"),
                "end":            meta.get("end_timestamp") or meta.get("timestamp_end"),
                "speaker_name":   meta.get("speaker_name"),
                "speaker_role":   meta.get("speaker_role"),
                "call_section":   meta.get("call_section"),
                "chunk_index":    meta.get("chunk_index", 0),
            })
        chunks.sort(key=lambda x: (x.get("chunk_index") or 0, x.get("start") or 0))
        return {"transcript": chunks, "count": len(chunks)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(event="api_transcript_failed", file_hash=file_hash, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to fetch transcript")
