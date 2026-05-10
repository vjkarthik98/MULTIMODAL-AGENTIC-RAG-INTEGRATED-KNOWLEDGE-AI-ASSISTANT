import os
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.infra_registry import infra
from app.pipeline.ingestion_pipeline import process_file
from app.pipeline.query_pipeline import query_pipeline
from app.pipeline.rag_pipeline import RAGPipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

rag_pipeline = RAGPipeline()

# UPLOAD STAGING DIR
UPLOAD_DIR = settings.UPLOAD_STAGING_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ALLOWED EXTENSIONS
ALLOWED_EXTENSIONS = {
    ".pdf", ".txt", ".md",
    ".docx", ".xlsx", ".xls",
    ".png", ".jpg", ".jpeg", ".bmp", ".webp",
    ".mp3", ".wav", ".m4a", ".flac",
    ".mp4", ".avi", ".mov", ".mkv",
}

# EXTENSION TO MODALITY MAP (for size limit lookup)
_EXT_MODALITY = {
    ".txt": "text", ".md": "text",
    ".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx", ".xls": "xlsx",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".bmp": "image", ".webp": "image",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".flac": "audio",
    ".mp4": "video", ".avi": "video", ".mov": "video", ".mkv": "video",
}


# REQUEST MODELS

class QueryRequest(BaseModel):
    query:      str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(default="default", max_length=128)


class ClearMemoryRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


# HELPERS

def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(text.strip().split())


def _size_limit(ext: str) -> int:
    modality = _EXT_MODALITY.get(ext, "text")
    return settings.FILE_SIZE_LIMITS.get(modality, settings.MAX_FILE_SIZE_MB * 1024 * 1024)


# HEALTH

@router.get("/health")
def health_check():
    return {
        "status":    "ok",
        "service":   "rag-api",
        "version":   settings.APP_VERSION,
        "timestamp": time.time(),
    }


# INFRA HEALTH

@router.get("/infra/health")
def infra_health():
    return {
        "status": "ok",
        "infra":  infra.health_check(),
    }


# TOOLS

@router.get("/tools")
def list_tools():
    try:
        from app.agents.tool_registry import ToolRegistry
        registry = ToolRegistry()
        return {
            "status": "ok",
            "tools":  registry.list_tools(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# QUERY

@router.post("/query")
def query_rag(request: QueryRequest):

    start      = time.time()
    request_id = str(uuid.uuid4())

    try:
        query = _clean(request.query)

        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        query = query[:settings.MAX_PROMPT_CHARS]

        result = query_pipeline(
            query=query,
            session_id=request.session_id,
        )

        if not isinstance(result, dict) or "answer" not in result:
            raise RuntimeError("Invalid pipeline response")

        latency = round(time.time() - start, 3)

        logger.info(
            event="api_query_success",
            request_id=request_id,
            latency=latency,
            session_id=request.session_id,
        )

        return {
            "request_id": request_id,
            "status":     "success",
            "data":       result,
            "latency":    latency,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            event="api_query_failed",
            request_id=request_id,
            error=str(e),
            session_id=request.session_id,
        )
        raise HTTPException(status_code=500, detail="Query processing failed")


# STREAM

@router.post("/query/stream")
def stream_query(request: QueryRequest):

    request_id = str(uuid.uuid4())

    try:
        query = _clean(request.query)

        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        generator = rag_pipeline.stream(
            query[:settings.MAX_PROMPT_CHARS],
            session_id=request.session_id,
        )

        def event_stream():
            try:
                for token in generator:
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(
                    event="api_stream_failed",
                    request_id=request_id,
                    error=str(e),
                    session_id=request.session_id,
                )
                yield "data: [Stream interrupted]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"X-Request-ID": request_id},
        )

    except Exception as e:
        logger.error(
            event="api_stream_error",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Streaming failed")


# UPLOAD

@router.post("/upload")
async def upload_file(
    request:    Request,
    file:       UploadFile = File(...),
    session_id: str = "default",
):

    start      = time.time()
    request_id = str(uuid.uuid4())
    file_path: Optional[Path] = None

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file")

        ext      = Path(file.filename).suffix.lower()
        filename = Path(file.filename).name

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}",
            )

        max_size  = _size_limit(ext)
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{filename}"
        size      = 0

        # SAVE FILE WITH SIZE CHECK
        with open(file_path, "wb") as f:
            while chunk := await file.read(settings.UPLOAD_CHUNK_SIZE):
                size += len(chunk)

                if size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large for type {ext}: max {max_size // (1024*1024)}MB",
                    )

                f.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # INGEST
        result = process_file(str(file_path), session_id=session_id)

        if not result or result.get("status") != "success":
            raise RuntimeError("Ingestion failed")

        chunks = result.get("chunks", 0)

        if chunks <= 0:
            raise RuntimeError("No usable content extracted from file")

        latency = round(time.time() - start, 3)

        logger.info(
            event="api_upload_success",
            request_id=request_id,
            filename=filename,
            ext=ext,
            size=size,
            chunks=chunks,
            latency=latency,
            session_id=session_id,
        )

        return {
            "request_id": request_id,
            "status":     "success",
            "filename":   filename,
            "file_size":  size,
            "chunks":     chunks,
            "stored":     result.get("stored", 0),
            "latency":    latency,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            event="api_upload_failed",
            request_id=request_id,
            error=str(e),
            session_id=session_id,
        )
        raise HTTPException(status_code=500, detail="File upload failed")

    finally:
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(
                    event="api_cleanup_failed",
                    request_id=request_id,
                    error=str(e),
                )


# CLEAR MEMORY

@router.post("/memory/clear")
def clear_memory(request: ClearMemoryRequest):
    try:
        from app.memory.memory_manager import MemoryManager
        manager = MemoryManager()
        manager.clear(request.session_id)

        return {
            "status":     "ok",
            "session_id": request.session_id,
            "message":    "Session memory cleared",
        }

    except Exception as e:
        logger.error(
            event="api_memory_clear_failed",
            error=str(e),
            session_id=request.session_id,
        )
        raise HTTPException(status_code=500, detail="Memory clear failed")