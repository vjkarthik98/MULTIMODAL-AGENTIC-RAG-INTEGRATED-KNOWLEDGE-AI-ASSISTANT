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
from app.auth.models import UserPublic
from app.core.config import settings
from app.core.infra_registry import infra
from app.utils.logger import get_logger
from app.utils.paths import user_knowledge_base_dir, user_staging_dir

logger = get_logger(__name__)

router = APIRouter()


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

        # DEDUP CHECK
        file_hash = await asyncio.to_thread(_compute_file_hash, file_path)
        if file_hash in _INGEST_DEDUP:
            latency = round(time.time() - start, 3)
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

        # PERSIST TO KNOWLEDGE BASE — copy original file before pipeline may clean staging
        kb_dir = user_knowledge_base_dir(user_id)
        kb_path = kb_dir / filename
        import shutil as _shutil
        _shutil.copy2(str(file_path), str(kb_path))

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

        # RUN FULL PIPELINE — ingest → chunk → embed → store
        try:
            from app.pipeline.ingestion_pipeline import process_file
            result = await asyncio.to_thread(process_file, str(file_path), session_id, user_id)

        except RuntimeError as exc:
            # Pipeline wraps ingestor errors as: RuntimeError("INGESTION_FAILED: <original>")
            inner = str(exc).removeprefix("INGESTION_FAILED: ").strip()
            if any(inner.startswith(p) for p in _INGEST_422_PREFIXES):
                error_code = inner.split(":")[0].strip()
                return JSONResponse(
                    status_code=422,
                    content={"request_id": request_id, "error_code": error_code, "detail": inner},
                )
            raise HTTPException(status_code=500, detail=str(exc))

        except Exception as exc:
            err_str = str(exc)
            if any(err_str.startswith(p) for p in _INGEST_422_PREFIXES):
                error_code = err_str.split(":")[0].strip()
                return JSONResponse(
                    status_code=422,
                    content={"request_id": request_id, "error_code": error_code, "detail": err_str},
                )
            raise HTTPException(status_code=500, detail=err_str)

        # PIPELINE-LEVEL DUPLICATE (detected by pipeline's own hash check)
        if result.get("status") == "duplicate":
            latency = round(time.time() - start, 3)
            return JSONResponse(
                status_code=200,
                content={
                    "request_id": request_id,
                    "status":     "duplicate",
                    "duplicate":  True,
                    "filename":   filename,
                    "ext":        ext,
                    "modality":   modality,
                    "file_size":  size,
                    "latency":    latency,
                },
            )

        chunks = result.get("chunks", 0)
        stored = result.get("stored", 0)

        if chunks <= 0:
            return JSONResponse(
                status_code=422,
                content={
                    "request_id": request_id,
                    "error_code": "NO_CONTENT_EXTRACTED",
                    "detail":     "No usable content extracted from file",
                },
            )

        # REGISTER DEDUP AFTER SUCCESSFUL PIPELINE
        pipeline_hash = result.get("file_hash", file_hash)
        _dedup_set(pipeline_hash, request_id)

        latency = round(time.time() - start, 3)

        _audit_log(
            "ingest_completed",
            request_id=request_id,
            session_id=session_id,
            file=filename,
            chunks=chunks,
            stored=stored,
            latency=latency,
        )

        logger.info(
            event="api_ingest_success",
            request_id=request_id,
            file=filename,
            ext=ext,
            modality=modality,
            size=size,
            chunks=chunks,
            stored=stored,
            latency=latency,
            session_id=session_id,
        )

        return JSONResponse(
            status_code=200,
            content={
                "request_id":        request_id,
                "status":            "success",
                "duplicate":         False,
                "doc_id":            result.get("doc_id", ""),
                "filename":          filename,
                "modality":          modality,
                "chunks":            chunks,
                "stored":            stored,
                "session_id":        session_id,
                "ingestion_time_sec": latency,
                "pages":             result.get("pages"),
                "warnings":          result.get("warnings", []),
                "file_hash":         pipeline_hash,
                "ext":               ext,
                "file_size":         size,
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

        result = await asyncio.to_thread(
            pipeline_fn, query, session_id, request_body.sources, user_id
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

# The guarded answer arrives from rag.stream() as a single completed string
# (output_guard must see the whole thing before anything reaches the client —
# see the comment in RAGPipeline.stream). To still give the UI a live,
# token-by-token feel, we re-chunk it here and pace the chunks with
# asyncio.sleep (non-blocking, so other requests keep flowing on the loop).
_STREAM_CHUNK_CHARS = 1
_STREAM_CHUNK_DELAY_SEC = 0.008


def _stream_chunks(text: str):
    for i in range(0, len(text), _STREAM_CHUNK_CHARS):
        yield text[i:i + _STREAM_CHUNK_CHARS]


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

    try:
        query = _clean(request_body.query)

        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        query = _check_prompt_injection(query)
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

        # REAL-TIME QUERY DETECTION — must run BEFORE the cache check so stale
        # KB answers don't mask live Tavily results. Real-time queries should
        # never be served from cache (stock prices, earnings, news change daily).
        _REALTIME_SIGNALS = {
            "today", "now", "current", "latest", "live", "right now",
            "stock price", "share price", "price today", "news today",
            "this week", "this month", "currently trading",
        }
        _q_lower = query.lower()
        _is_realtime = any(sig in _q_lower for sig in _REALTIME_SIGNALS)

        if _is_realtime:
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
                        logger.info(event="stream_realtime_web_search", session_id=session_id)

                        _web_sources = _tool_out.get("sources") or []

                        async def web_event_stream():
                            for piece in _stream_chunks(_web_answer):
                                yield _sse(piece)
                                await asyncio.sleep(_STREAM_CHUNK_DELAY_SEC)
                            # Emit the web source URLs as a typed sources event so
                            # the client renders them as clickable web chips below
                            # the answer (same contract as the KB path's sentinel).
                            if _web_sources:
                                _payload = [
                                    {"source": u, "modality": "web",
                                     "page_number": None, "start_time": None}
                                    for u in _web_sources if u
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
                    event="stream_realtime_web_search_failed",
                    error=str(_web_err),
                    session_id=session_id,
                )
                # Fall through to cache check / normal RAG if web search fails

        # REDIS CACHE CHECK — only for non-realtime queries (realtime bypassed above).
        # On a cache hit we stream the cached answer immediately.
        try:
            from app.pipeline.query_pipeline import _cache_get
            cached = _cache_get(session_id, query)
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

        generator = await asyncio.to_thread(
            rag.stream,
            query,
            session_id,
            _stream_user_id,
        )

        async def event_stream():
            try:
                for token in generator:
                    if not token:
                        continue
                    if token.startswith("\x00REFUSAL\x00"):
                        # The model refused despite relevant docs. Do NOT stream
                        # the refusal text (that is the flash bug). Signal the
                        # client to stream the accurate meta-path answer instead.
                        yield 'data: {"__type__":"refusal"}\n\n'
                        continue
                    if token.startswith("\x00SOURCES\x00"):
                        # Emit sources as a typed JSON event so the client can
                        # display them immediately without a second round-trip.
                        # Sentinel is 9 bytes (\x00SOURCES\x00) — slice at [9:]
                        yield f'data: {{"__type__":"sources","data":{token[9:]}}}\n\n'
                        continue
                    for piece in _stream_chunks(token):
                        yield _sse(piece)
                        await asyncio.sleep(_STREAM_CHUNK_DELAY_SEC)
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
    session_id: str = Form("default"),
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    start = time.time()
    request_id = _request_id()
    file_path: Optional[Path] = None
    user_id = current_user.user_id

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

        # INGEST
        from app.pipeline.ingestion_pipeline import process_file
        try:
            result = await asyncio.to_thread(process_file, str(file_path), session_id, user_id)
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




# ─── KNOWLEDGE BASE ───────────────────────────────────────────────────────────

@router.get("/knowledge-base")
async def list_knowledge_base(
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> Dict[str, Any]:
    """List all files in the authenticated user's knowledge base."""
    user_id = current_user.user_id
    from app.utils.paths import user_knowledge_base_dir
    kb_dir = user_knowledge_base_dir(user_id)
    files = []
    for f in sorted(kb_dir.iterdir()):
        if f.is_file():
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

    # PURGE FROM BM25 — best-effort; missing BM25 index entry does not block deletion
    bm25_purged = False
    try:
        bm25 = infra.get_bm25()
        if bm25 and hasattr(bm25, "delete_by_source"):
            bm25.delete_by_source(safe_name, user_id=user_id)
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
