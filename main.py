from fastapi import FastAPI
from app.api.routes.rag_routes import router as rag_router

app = FastAPI(title = "Mutlimodal RAG API")

app.include_router(rag_router, prefix="/rag", tags=["RAG"])

@app.get("/")
def root():
    return{"message": "API is running"}