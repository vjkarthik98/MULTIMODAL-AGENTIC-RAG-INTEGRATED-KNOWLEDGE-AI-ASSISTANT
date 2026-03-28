from fastapi import APIRouter, UploadFile, File
import os

from app.pipeline.rag_pipeline import RAGPipeline
from app.ingestion.text_ingest import ingest_pipeline

from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from app.ingestion.pipeline import process_file

from app.retrieval.query_pipeline import query_text, query_image

from faster_whisper import WhisperModel

class QueryRequest(BaseModel):
    query:str


router = APIRouter()
pipeline = RAGPipeline()

UPLOAD_DIR = "data/raw"

# Load audio model once
audio_model = WhisperModel("base", compute_type="int8")

# Health check
@router.get("/test")
def test_route():
    return{"message" : "RAG route working"}

@router.get("/rag/query/text")
def rag_text_query(q: str):
    return {"results": query_text(q)}

@router.get("/rag/query/image")
def rag_image_query(q: str):
    return {"results": query_image(q)}

# Query endpoint (RAG)
@router.post("/query")
def query_rag(request:QueryRequest):
    return pipeline.run(request.query)
    

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
        result = process_file(file_path)

        return {
            "filename": file.filename,
            "chunks_inserted": result.get("chunks", 0),
            "status": result.get("status", "unknown"),
            "details": result.get("details", {})
        }

    except Exception as e:
        return {"error": str(e)}
    
# Audio Query
@router.post("/rag/query/audio")
async def rag_audio_query(file: UploadFile = File(...)):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save uploaded audio
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # step 1: Transcribe audio
        segments, _ = audio_model.transcribe(file_path)

        query_text_data = ""
        for segment in segments:
            query_text_data += segment.text + " "

        query_text_data = query_text_data.strip()

        if not query_text_data:
            return {"error": "Empty transcription"}
        print(f"\n Transcribed Query: {query_text_data}\n")

        # Step 2: Run Full RAG pipeline
        result = pipeline.run(query_text_data)

        return result
    
    except Exception as e:
        return {"error": str(e)}
