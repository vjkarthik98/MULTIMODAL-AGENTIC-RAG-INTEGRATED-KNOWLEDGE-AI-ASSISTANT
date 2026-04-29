from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pathlib import Path
import time
import uuid
import os

from app.core.config import settings
from app.utils.logger import get_logger

from app.pipeline.ingestion_pipeline import process_file
from app.pipeline.query_pipeline import query_pipeline
from app.pipeline.rag_pipeline import RAGPipeline


logger = get_logger(__name__)
router = APIRouter()

rag_pipeline = RAGPipeline()

UPLOAD_DIR = settings.DATA_DIR / "raw"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# REQUEST MODEL
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str = Field(default="default")


# UTIL
def _clean(text: str) -> str:
    return " ".join(text.strip().split())


# HEALTH
@router.get("/health")
def health_check():
    return {"status": "ok", "service": "rag-api"}


# QUERY
@router.post("/query")
def query_rag(request: QueryRequest):

    start = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(
            "[RAGRoute][QUERY] request_id=%s session_id=%s",
            request_id,
            request.session_id
        )

        query = _clean(request.query)

        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        query = query[:settings.MAX_PROMPT_CHARS]

        result = query_pipeline(
            query=query,
            session_id=request.session_id
        )

        if not result:
            raise RuntimeError("Empty response from query pipeline")

        return {
            "request_id": request_id,
            "status": "success",
            "data": result,
            "latency": round(time.time() - start, 2)
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "[RAGRoute][QUERY_FAIL] request_id=%s error=%s",
            request_id,
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Query processing failed"
        )


# STREAM
@router.post("/query/stream")
def stream_query(request: QueryRequest):

    request_id = str(uuid.uuid4())

    try:
        logger.info(
            "[RAGRoute][STREAM] request_id=%s session_id=%s",
            request_id,
            request.session_id
        )

        query = _clean(request.query)

        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        generator = rag_pipeline.stream(
            query[:settings.MAX_PROMPT_CHARS],
            session_id=request.session_id
        )

        def event_stream():
            try:
                for token in generator:
                    yield f"data: {token}\n\n"
            except Exception as e:
                logger.error(
                    "[RAGRoute][STREAM_FAIL] request_id=%s error=%s",
                    request_id,
                    str(e)
                )
                yield "data: [Stream interrupted]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(
            "[RAGRoute][STREAM_ERROR] request_id=%s error=%s",
            request_id,
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Streaming failed"
        )


# UPLOAD
@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = "default"
):

    start = time.time()
    request_id = str(uuid.uuid4())

    file_path = None

    try:
        logger.info(
            "[RAGRoute][UPLOAD] request_id=%s session_id=%s file=%s",
            request_id,
            session_id,
            file.filename
        )

        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file")

        # FILE TYPE CHECK
        allowed_ext = {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".mp3", ".wav", ".mp4"}
        ext = Path(file.filename).suffix.lower()

        if ext not in allowed_ext:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        filename = Path(file.filename).name
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{filename}"

        size = 0

        # SAVE FILE
        with open(file_path, "wb") as f:
            while chunk := await file.read(settings.UPLOAD_CHUNK_SIZE):
                size += len(chunk)

                if size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="File too large")

                f.write(chunk)

        # PROCESS FILE
        result = process_file(str(file_path), session_id=session_id)

        if not result or result.get("status") != "success":
            raise RuntimeError("Ingestion pipeline failed")

        chunks = result.get("chunks", 0)

        if chunks == 0:
            raise RuntimeError("No chunks generated from file")

        latency = round(time.time() - start, 2)

        return {
            "request_id": request_id,
            "status": "success",
            "filename": filename,
            "chunks": chunks,
            "latency": latency
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "[RAGRoute][UPLOAD_FAIL] request_id=%s error=%s",
            request_id,
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )

    finally:
        # CLEANUP (ALWAYS REMOVE TEMP FILE)
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(
                    "[RAGRoute][CLEANUP_FAIL] request_id=%s error=%s",
                    request_id,
                    str(e)
                )