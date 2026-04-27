from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pathlib import Path
import time
import uuid

from app.core.config import settings
from app.utils.logger import get_logger

from app.ingestion.pipeline import process_file
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


# HEALTH 
@router.get("/health")
def health_check():
    return {"status": "ok", "service": "rag-api"}


# QUERY 
@router.post("/query")
def query_rag(request: QueryRequest):
    start = time.time()

    try:
        request_id = str(uuid.uuid4())

        logger.info(
            "[RAGRoute][QUERY] request_id=%s session_id=%s",
            request_id,
            request.session_id
        )

        query = request.query[:settings.MAX_PROMPT_CHARS]

        result = query_pipeline(
            query=query,
            session_id=request.session_id
        )

        result["api_latency"] = round(time.time() - start, 2)
        result["request_id"] = request_id

        return result

    except Exception as e:
        logger.error(
            "[RAGRoute][QUERY_FAIL] session_id=%s | %s",
            request.session_id,
            str(e)
        )

        raise HTTPException(status_code=500, detail="Query processing failed")


# STREAM 
@router.post("/query/stream")
def stream_query(request: QueryRequest):

    try:
        request_id = str(uuid.uuid4())

        logger.info(
            "[RAGRoute][STREAM] request_id=%s session_id=%s",
            request_id,
            request.session_id
        )

        generator = rag_pipeline.stream(
            request.query[:settings.MAX_PROMPT_CHARS],
            session_id=request.session_id
        )

        def safe_generator():
            try:
                for token in generator:
                    yield str(token)
            except Exception as e:
                logger.error("[RAGRoute][STREAM_FAIL] %s", str(e))
                yield "\n[Stream interrupted]\n"

        return StreamingResponse(
            safe_generator(),
            media_type="text/plain"
        )

    except Exception as e:
        logger.error("[RAGRoute][STREAM_ERROR] %s", str(e))
        raise HTTPException(status_code=500, detail="Streaming failed")


# UPLOAD 
@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = "default"
):
    start = time.time()

    try:
        request_id = str(uuid.uuid4())

        logger.info(
            "[RAGRoute][UPLOAD] request_id=%s session_id=%s file=%s",
            request_id,
            session_id,
            file.filename
        )

        # Validate filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file")

        filename = Path(file.filename).name
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{filename}"

        size = 0

        # Stream write (critical fix)
        with open(file_path, "wb") as f:
            while chunk := await file.read(settings.UPLOAD_CHUNK_SIZE):
                size += len(chunk)

                if size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=400,
                        detail="File too large"
                    )

                f.write(chunk)

        # Run ingestion
        result = process_file(str(file_path), session_id=session_id)

        latency = round(time.time() - start, 2)

        logger.info(
            "[RAGRoute][UPLOAD_SUCCESS] request_id=%s latency=%ss",
            request_id,
            latency
        )

        return {
            "filename": filename,
            "status": result.get("status", "unknown"),
            "chunks": result.get("chunks", 0),
            "details": result.get("details", {}),
            "latency": latency,
            "request_id": request_id
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error("[RAGRoute][UPLOAD_FAIL] %s", str(e))
        raise HTTPException(status_code=500, detail="File upload failed")