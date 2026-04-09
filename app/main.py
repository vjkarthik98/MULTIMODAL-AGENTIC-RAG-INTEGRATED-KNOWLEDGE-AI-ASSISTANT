import logging
from fastapi import FastAPI
from app.api.routes.rag_routes import router as rag_router

logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
app= FastAPI(
    title = "Multimodal RAG Assistant",
    version = "0.18.0"
)

# Include RAG routes
app.include_router(rag_router, prefix="/rag")

# Root endpoint
@app.get("/")
def root():
    return {"message": "RAG API is running"}