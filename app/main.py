from fastapi import FastAPI
from app.api.routes.rag_routes import router as rag_router

app= FastAPI(
    title = "Multimodal RAG Assistant",
    version = "0.5.0"
)

# Include RAG routes
app.include_router(rag_router, prefix="/rag")

# Root endpoint
@app.get("/")
def root():
    return {"message": "RAG API is running"}