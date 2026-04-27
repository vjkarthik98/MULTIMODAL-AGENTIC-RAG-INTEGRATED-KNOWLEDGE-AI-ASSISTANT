from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import time
import uuid

from app.api.routes.rag_routes import router as rag_router
from app.core.config import settings
from app.utils.logger import get_logger
from scripts.init_qdrant import initialize_qdrant


logger = get_logger(__name__)
initialize_qdrant() 


# LIFESPAN 
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("[Startup] initializing application")

    # Warmup models
    try:
        from app.core.model_loader import model_loader

        llm = model_loader.get_llm()

        if hasattr(llm, "warmup"):
            llm.warmup()

        logger.info("[Startup] model warmup completed")

    except Exception as e:
        logger.warning("[Startup] warmup skipped | %s", str(e))

    logger.info("[Startup] application ready")

    yield

    logger.info("[Shutdown] shutting down application")


# APP 
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan
)


# MIDDLEWARE 
@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.time()
    request_id = str(uuid.uuid4())

    try:
        response = await call_next(request)

        latency = round(time.time() - start, 2)

        logger.info(
            "[Request] id=%s method=%s path=%s status=%s latency=%ss",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency
        )

        response.headers["X-Request-ID"] = request_id

        return response

    except Exception as e:
        logger.error(
            "[Request][FAILED] id=%s path=%s | %s",
            request_id,
            request.url.path,
            str(e)
        )
        raise


# ROUTES 
app.include_router(rag_router, prefix="/rag", tags=["RAG"])


# ROOT 
@app.get("/")
def root():
    return {
        "message": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


# HEALTH 
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.APP_NAME
    }


# READINESS 
@app.get("/ready")
def readiness():
    try:
        from app.core.model_loader import model_loader

        llm = model_loader.get_llm()

        return {
            "status": "ready",
            "model_loaded": llm is not None
        }

    except Exception as e:
        logger.error("[Readiness] %s", str(e))
        return {
            "status": "not_ready",
            "error": str(e)
        }