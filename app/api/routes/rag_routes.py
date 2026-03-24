from fastapi import APIRouter, UploadFile, File
import os

from app.pipeline.rag_pipeline import RAGPipeline
from app.ingestion.text_ingest import ingest_pipeline

from fastapi.responses import StreamingResponse

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query:str


router = APIRouter()
pipeline = RAGPipeline()

UPLOAD_DIR = "data/raw"

# Health check
@router.get("/test")
def test_route():
    return{"message" : "RAG route working"}

# Query endpoint (RAG)
@router.post("/query")
def query_rag(request:QueryRequest):
    result = pipeline.run(request.query)
    return result

# Stream Endpoint 
@router.post("/query/stream")
def stream_query(request: QueryRequest):
    generator = pipeline.stream(request.query)

    return StreamingResponse(
        generator,
        media_type = "text/plain"
    )

# Upload + Ingestion endpoint
@router.post("/upload/file", response_model=dict)
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # run ingestion pipeline
        count = ingest_pipeline(file_path)

        return {
            "filename": file.filename,
            "chunks_inserted": count
        }

    except Exception as e:
        return {"error": str(e)}