from fastapi import APIRouter, UploadFile, File
import os

from app.pipeline.rag_pipeline import RAGPipeline


from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from app.ingestion.pipeline import process_file

from app.retrieval.query_pipeline import query_text, query_image

from faster_whisper import WhisperModel

from app.retrieval.query_pipeline import query_audio

from app.retrieval.query_pipeline import query_video

class QueryRequest(BaseModel):
    query:str
    session_id: str = "default"


router = APIRouter()
pipeline = RAGPipeline()

UPLOAD_DIR = "data/raw"

# Health check
@router.get("/test")
def test_route():
    return{"message" : "RAG route working"}

@router.get("/rag/query/text")
def rag_text_query(q: str, session_id: str = "default"):
    return {"results": query_text(q, session_id=session_id)}

@router.get("/rag/query/image")
def rag_image_query(q: str):
    return {"results": query_image(q)}

# Query endpoint (RAG)
@router.post("/query")
def query_rag(request: QueryRequest):
    try:
        result = pipeline.run(
            request.query,
            session_id = request.session_id
        )
        return result
    
    except Exception as e:
        return {"error": str(e)}
    

# Stream Endpoint 
@router.post("/query/stream")
def stream_query(request: QueryRequest):
    generator = pipeline.stream(request.query, session_id=request.session_id)

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

        # step 1: Convert audio -> text(via query pipeline)
        query_text_data = query_audio(file_path)

        if not query_text_data:
            return {"error": "Empty transcription"}
        
        print(f"\n Transcibed Query: {query_text_data}\n")

        # Step 2: Run RAG
        result = pipeline.run(query_text_data, session_id="audio_" + file.filename)

        return {
            "transcribed_query": query_text_data,
            "answer": result
        }
       
    except Exception as e:
        return {"error": str(e)}

# Video Query
@router.post("/rag/query/video")
async def rag_video_query(file: UploadFile = File(...)):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok = True)

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save uploaded video
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Step 1: Video -> text
        query_text_data = query_video(file_path)

        if not query_text_data:
            return {"error": "Empty transcription"}
        
        print(f"\n Transcribed Video Query: {query_text_data}\n")

        # Step 2: Run RAG
        result = pipeline.run(query_text_data, session_id="video_" + file.filename)

        return {
            "transcribed_query": query_text_data,
            "answer": result
        }
    
    except Exception as e:
        return {"error": str(e)}

