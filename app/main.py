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


# CONCURRENCY CONTROL 
semaphore = asyncio.Semaphore(settings.MAX_PARALLEL_REQUESTS)



# LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("[Startup] environment=%s", settings.ENV)

    # QDRANT INIT
    try:
        initialize_qdrant()
        logger.info("[Startup] qdrant initialized")
    except Exception as e:
        logger.warning("[Startup] qdrant init failed | %s", str(e))

    # MODEL WARMUP
    try:
        from app.core.model_loader import model_loader

        model_loader.warmup()

        infra.warmup()

        logger.info("[Startup] model warmup completed")

    except Exception as e:
        logger.warning("[Startup] model warmup failed | %s", str(e))

    logger.info("[Startup] application ready")

    yield

    logger.info("[Shutdown] application shutting down")



# APP
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan
)



# GLOBAL ERROR HANDLER
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):

    logger.error("[GlobalError] %s", str(exc))

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
        }
    )



# CONCURRENCY LIMIT MIDDLEWARE
@app.middleware("http")
async def limit_concurrency(request: Request, call_next):
    async with semaphore:
        return await call_next(request)



# REQUEST LOGGER
@app.middleware("http")
async def request_logger(request: Request, call_next):

    start = time.time()
    request_id = str(uuid.uuid4())

    try:
        response = await call_next(request)

        latency = round(time.time() - start, 2)

        if latency > 2:
            logger.warning(
                "[SlowRequest] id=%s path=%s latency=%ss",
                request_id,
                request.url.path,
                latency
            )
        else:
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

        health = model_loader.health_check()

        return {
            "status": "ready",
            "models": health
        }

    except Exception as e:
        logger.error("[Readiness] %s", str(e))
        return {
            "status": "not_ready",
            "error": str(e)
        }