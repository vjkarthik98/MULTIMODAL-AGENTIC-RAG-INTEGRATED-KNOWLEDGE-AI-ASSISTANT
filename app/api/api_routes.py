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


#  REQUEST MODEL 
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str = Field(default="default")


#  CLEAN 
def _clean(text: str) -> str:
    return " ".join(str(text or "").strip().split())


#  HEALTH 
@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "rag-api",
        "timestamp": time.time()
    }


#  QUERY 
@router.post("/query")
def query_rag(request: QueryRequest):

    start = time.time()
    request_id = str(uuid.uuid4())

    try:
        query = _clean(request.query)

        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        query = query[:settings.MAX_PROMPT_CHARS]

        result = query_pipeline(
            query=query,
            session_id=request.session_id
        )

        if not isinstance(result, dict) or "answer" not in result:
            raise RuntimeError("Invalid pipeline response")

        latency = round(time.time() - start, 3)

        return {
            "request_id": request_id,
            "status": "success",
            "data": result,
            "latency": latency
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            event="api_query_failed",
            request_id=request_id,
            error=str(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Query processing failed"
        )


#  STREAM 
@router.post("/query/stream")
def stream_query(request: QueryRequest):

    request_id = str(uuid.uuid4())

    try:
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
                    event="api_stream_failed",
                    request_id=request_id,
                    error=str(e)
                )
                yield "data: [Stream interrupted]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(
            event="api_stream_error",
            request_id=request_id,
            error=str(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Streaming failed"
        )


#  UPLOAD 
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
        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file")

        allowed_ext = {
            ".pdf", ".txt", ".png", ".jpg",
            ".jpeg", ".mp3", ".wav", ".mp4"
        }

        ext = Path(file.filename).suffix.lower()

        if ext not in allowed_ext:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        filename = Path(file.filename).name
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{filename}"

        size = 0
        max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024

        #  SAVE FILE 
        with open(file_path, "wb") as f:
            while chunk := await file.read(settings.UPLOAD_CHUNK_SIZE):
                size += len(chunk)

                if size > max_size:
                    raise HTTPException(status_code=400, detail="File too large")

                f.write(chunk)

        #  INGEST 
        result = process_file(str(file_path), session_id=session_id)

        if not result or result.get("status") != "success":
            raise RuntimeError("Ingestion failed")

        chunks = result.get("chunks", 0)

        if chunks <= 0:
            raise RuntimeError("No usable content extracted")

        latency = round(time.time() - start, 3)

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
            event="api_upload_failed",
            request_id=request_id,
            error=str(e)
        )

        raise HTTPException(
            status_code=500,
            detail="File upload failed"
        )

    finally:
        #  CLEANUP 
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(
                    event="api_cleanup_failed",
                    request_id=request_id,
                    error=str(e)
                )