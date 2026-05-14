from __future__ import annotations

import asyncio
import os
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.core.infra_registry import infra
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# UPLOAD STAGING DIR
UPLOAD_DIR: Path = settings.UPLOAD_STAGING_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# TEMP DIR
TEMP_DIR: Path = settings.TEMP_DIR
TEMP_DIR.mkdir(parents=True, exist_ok=True)

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
_audit_log_enabled: bool = getattr(settings, "AUDIT_LOGGING_ENABLED", True)


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


class ClearMemoryRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)


class GDPRPurgeRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)


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


# PROMPT INJECTION CHECK

def _check_prompt_injection(query: str) -> str:
    patterns = getattr(settings, "PROMPT_INJECTION_PATTERNS", [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard the above",
        "forget everything",
        "you are now",
        "act as",
        "jailbreak",
    ])
    lower = query.lower()
    for pattern in patterns:
        if pattern in lower:
            logger.warning(
                event="prompt_injection_detected",
                pattern=pattern,
            )
            idx = lower.find(pattern)
            query = query[:idx].strip()
            break
    return query


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
) -> Dict[str, Any]:
    start = time.time()
    request_id = _request_id()
    session_id = request_body.session_id

    _rate_limit_check(request)

    try:
        query = _clean(request_body.query)

        if not query:
            raise HTTPException(status_code=400, detail="Empty query after cleaning")

        # PROMPT INJECTION CHECK
        query = _check_prompt_injection(query)
        query = query[:settings.MAX_PROMPT_CHARS]

        _audit_log(
            "query_received",
            request_id=request_id,
            session_id=session_id,
            query_len=len(query),
            ip=_client_ip(request),
        )

        pipeline_fn = _get_query_pipeline()

        result = await asyncio.to_thread(pipeline_fn, query, session_id)

        if not isinstance(result, dict) or "answer" not in result:
            raise RuntimeError("Invalid pipeline response")

        latency = round(time.time() - start, 3)

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
        )

        return {
            "request_id": request_id,
            "status":     "success",
            "data":       result,
            "latency":    latency,
        }

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

@router.post("/query/stream")
async def stream_query(
    request_body: QueryRequest,
    request: Request,
) -> StreamingResponse:
    request_id = _request_id()
    session_id = request_body.session_id

    _rate_limit_check(request)

    try:
        query = _clean(request_body.query)

        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        query = _check_prompt_injection(query)
        query = query[:settings.MAX_PROMPT_CHARS]

        rag = _get_rag_pipeline()

        generator = await asyncio.to_thread(
            rag.stream,
            query,
            session_id,
        )

        async def event_stream():
            try:
                for token in generator:
                    if token:
                        yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error(
                    event="api_stream_failed",
                    request_id=request_id,
                    error=str(exc),
                    session_id=session_id,
                )
                yield "data: [Stream interrupted]\n\n"

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
    session_id: str = "default",
) -> Dict[str, Any]:
    start = time.time()
    request_id = _request_id()
    file_path: Optional[Path] = None

    _rate_limit_check(request)

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

        max_size = _size_limit(ext)

        # DISK SPACE GUARD
        _check_disk_space(UPLOAD_DIR)

        file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
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

        _audit_log(
            "upload_received",
            request_id=request_id,
            session_id=session_id,
            filename=filename,
            ext=ext,
            size=size,
            ip=_client_ip(request),
        )

        # INGEST
        from app.pipeline.ingestion_pipeline import process_file
        result = await asyncio.to_thread(process_file, str(file_path), session_id)

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
            filename=filename,
            chunks=chunks,
            stored=stored,
            latency=latency,
        )

        logger.info(
            event="api_upload_success",
            request_id=request_id,
            filename=filename,
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
) -> Dict[str, Any]:
    request_id = _request_id()
    session_id = request_body.session_id

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


# GDPR PURGE

@router.post("/memory/purge")
async def gdpr_purge(
    request_body: GDPRPurgeRequest,
    request: Request,
) -> Dict[str, Any]:
    request_id = _request_id()
    user_id = request_body.user_id

    if not getattr(settings, "GDPR_PURGE_ENABLED", True):
        raise HTTPException(status_code=403, detail="GDPR purge not enabled")

    try:
        from app.memory.memory_manager import MemoryManager
        manager = MemoryManager()
        await asyncio.to_thread(manager.purge_user, user_id)

        # BM25 PURGE
        try:
            from app.core.infra_registry import infra
            bm25 = infra.get_bm25()
            if bm25 and hasattr(bm25, "purge_by_session"):
                await asyncio.to_thread(bm25.purge_by_session, user_id)
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


