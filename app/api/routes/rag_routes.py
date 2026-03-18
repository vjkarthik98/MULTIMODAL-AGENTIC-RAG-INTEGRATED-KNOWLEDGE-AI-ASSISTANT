from fastapi import APIRouter

from src.rag_system.pipeline.rag_pipeline import RAGPipeline

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query:str


router = APIRouter()
pipeline = RAGPipeline()

@router.get("/test")
def test_route():
    return{"message" : "RAG route working"}

@router.post("/query")
def query_rag(request:QueryRequest):
    result = pipeline.run(request.query)
    return result