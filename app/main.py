from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import uuid
import asyncio

from app.api.api_routes import router as rag_router
from app.core.config import settings
from app.utils.logger import get_logger

from scripts.init_qdrant import initialize_qdrant
from app.core.infra_registry import infra

logger = get_logger(__name__)


# CONCURRENCY
semaphore = asyncio.Semaphore(settings.MAX_PARALLEL_REQUESTS)


# LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(event="startup_begin", env=settings.ENV)


    #  VECTOR DB INIT 
    try:
        initialize_qdrant()
        logger.info(event="qdrant_ready")
    except Exception as e:
        logger.warning(event="qdrant_init_failed", error=str(e))

    #  MODEL + INFRA WARMUP 
    try:
        from app.core.model_loader import model_loader

        logger.info(event="warmup_complete")

        model_loader.get_embedder()

        logger.info(event="warmup_complete")

    except Exception as e:
        logger.warning(event="warmup_failed", error=str(e))

    logger.info(event="app_ready")

    yield

    logger.info(event="shutdown")


# APP
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan
)


# GLOBAL ERROR
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):

    logger.error(
        event="global_error",
        path=request.url.path,
        error=str(exc)
    )

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error"
        }
    )


# REQUEST LOGGER
@app.middleware("http")
async def request_logger(request: Request, call_next):

    start = time.time()
    request_id = str(uuid.uuid4())

    try:
        response = await call_next(request)

        latency = round(time.time() - start, 3)

        log_data = {
            "event": "request",
            "id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency": latency
        }

        if latency > settings.SLOW_REQUEST_THRESHOLD:
            logger.warning(**log_data)
        else:
            logger.info(**log_data)

        response.headers["X-Request-ID"] = request_id

        return response

    except Exception as e:
        logger.error(
            event="request_failed",
            id=request_id,
            path=request.url.path,
            error=str(e)
        )
        raise


# CONCURRENCY LIMIT
@app.middleware("http")
async def limit_concurrency(request: Request, call_next):

    async with semaphore:
        return await call_next(request)


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

        health = model_loader.health_check()

        return {
            "status": "ready",
            "models": health
        }

    except Exception as e:
        logger.error(event="readiness_failed", error=str(e))
        return {
            "status": "not_ready",
            "error": str(e)
        }