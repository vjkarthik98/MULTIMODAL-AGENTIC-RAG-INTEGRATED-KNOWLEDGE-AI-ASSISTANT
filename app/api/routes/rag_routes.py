from fastapi import APIRouter, UploadFile, File
import os
from app.utils.logger import get_logger


from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.ingestion.pipeline import process_file
from app.pipeline.query_pipeline import query_pipeline
from app.pipeline.rag_pipeline import RAGPipeline

# Logger
logger = get_logger(__name__)

router = APIRouter()

# Pipelines
rag_pipeline = RAGPipeline()

UPLOAD_DIR = "data/raw"

# REQUEST SCHEMA
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"


# HEALTH CHECK
@router.get("/health")
def health_check():
    return {"status": "ok", "service": "rag-api"}


# UNIFIED QUERY
@router.post("/query")
def query_rag(request: QueryRequest):
    try:
        logger.info(f"[RAGRoute] session_id={request.session_id} | Query received")

        result = query_pipeline(
            query=request.query,
            session_id=request.session_id
        )

        logger.info(f"[RAGRoute] session_id={request.session_id} | Query completed")

        return result

    except Exception as e:
        logger.error(f"[RAGRoute] session_id={request.session_id} | Error: {str(e)}")
        return {
            "answer": "Something went wrong,",
            "error": str(e)
        }


# STREAMING QUERY
@router.post("/query/stream")
def stream_query(request: QueryRequest):
    try:
        logger.info(f"[RAGRoute] session_id={request.session_id} | Stream started")

        generator = rag_pipeline.stream(request.query, session_id=request.session_id)

        return StreamingResponse(
            generator,
            media_type="text/plain"
        )

    except Exception as e:
        logger.error(f"[RAGRoute] Stream error | {str(e)}")
        return {"error": str(e)}

# FILE UPLOAD
@router.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = "default"):
    try:
        logger.info(f"[RAGRoute] session_id={session_id} | Upload started: {file.filename}")

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # run ingestion pipeline
        result = process_file(file_path, session_id=session_id)

        logger.info(f"[RAGRoute] session_id={session_id} | Upload + Ingestion completed")

        return {
            "filename": file.filename,
            "status": result.get("status", "unknown"),
            "chunks": result.get("chunks", 0),
            "details": result.get("details", {})
        }

    except Exception as e:
        logger.error(f"[RAGRoute]  Upload error: {str(e)}")
        return {"error": str(e)}
