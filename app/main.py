from fastapi import FastAPI
from app.api.routes.rag_routes import router as rag_router
from app.utils.logger import get_logger

# Logger
logger = get_logger(__name__)

# APP INIT
app = FastAPI(
    tile="Multimodal RAG Assistant",
    version="0.19.0",
    description = "Agent-based multimodal RAG system with memory and hybrid retrieval",
)

# ROUTES
app.include_router(rag_router, prefix="/rag", tags=["RAG"])

# STARTUP EVENT
@app.on_event("startup")
def start_event():
    logger.info("[Startup] Application is starting...")

    # Warmup LLM
    try:
        from app.core.model_loader import model_loader

        llm = model_loader.get_llm()
        if hasattr(llm, "warmup"):
            llm.warmup()

        logger.info("[Startup] Model Warmup completed")

    except Exception as e:
        logger.warning(f"[Startup] Warmup skipped | {str(e)}")

    logger.info("[Startup] Application ready")


# ROOT ENDPOINT
@app.get("/")
def root():
    return {
        "message": "Multimodal RAG API is running",
        "version": "0.19.0",
        "status": "healthy"
    }

# HEALTH CHECK
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "rag-api"
    }