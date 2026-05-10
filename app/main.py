import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.api_routes import router as rag_router
from app.core.config import settings
from app.core.infra_registry import infra
from app.utils.logger import get_logger, bind_request_context

logger = get_logger(__name__)

# CONCURRENCY SEMAPHORE
semaphore = asyncio.Semaphore(settings.MAX_PARALLEL_REQUESTS)


# LIFESPAN

@asynccontextmanager
async def lifespan(app: FastAPI):

    startup_start = time.time()

    logger.info(event="startup_begin", env=settings.ENV, version=settings.APP_VERSION)

    # QDRANT INIT
    try:
        from scripts.init_qdrant import initialize_qdrant
        initialize_qdrant()
        logger.info(event="qdrant_ready")
    except Exception as e:
        logger.warning(event="qdrant_init_failed", error=str(e))

    # INFRA WARMUP (Qdrant + BM25 + Mongo)
    try:
        await infra.warmup()
        logger.info(event="infra_warmup_complete")
    except Exception as e:
        logger.warning(event="infra_warmup_failed", error=str(e))

    # EMBEDDER WARMUP (minimum required for query path)
    try:
        from app.core.model_loader import model_loader
        model_loader.get_embedder()
        logger.info(event="embedder_ready")
    except Exception as e:
        logger.warning(event="embedder_warmup_failed", error=str(e))

    startup_latency = round(time.time() - startup_start, 2)
    logger.info(event="app_ready", startup_latency=startup_latency)

    yield

    # SHUTDOWN
    logger.info(event="shutdown_begin")
    logger.info(event="shutdown_complete")


# APP

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs"    if settings.ENV != "production" else None,
    redoc_url="/redoc"  if settings.ENV != "production" else None,
)


# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# GZIP

app.add_middleware(GZipMiddleware, minimum_size=1000)


# GLOBAL ERROR HANDLER

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        event="global_error",
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "status":  "error",
            "message": "Internal server error",
        },
    )


# REQUEST LOGGER MIDDLEWARE

@app.middleware("http")
async def request_logger(request: Request, call_next):

    start      = time.time()
    request_id = str(uuid.uuid4())

    # BIND CONTEXT FOR STRUCTURED LOGGING
    bind_request_context(request_id=request_id)

    # CLIENT IP (proxy-aware)
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip     = forwarded_for.split(",")[0].strip() if forwarded_for else (
        request.client.host if request.client else "unknown"
    )

    try:
        response = await call_next(request)
        latency  = round(time.time() - start, 3)

        log_kwargs = dict(
            event=  "http_request",
            id=     request_id,
            method= request.method,
            path=   request.url.path,
            status= response.status_code,
            latency=latency,
            ip=     client_ip,
        )

        if latency > settings.SLOW_REQUEST_THRESHOLD:
            logger.warning(**log_kwargs)
        else:
            logger.info(**log_kwargs)

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        logger.error(
            event="request_failed",
            id=    request_id,
            path=  request.url.path,
            error= str(e),
        )
        raise


# CONCURRENCY LIMIT MIDDLEWARE

@app.middleware("http")
async def limit_concurrency(request: Request, call_next):
    # Wrap the request handling in a helper function
    async def _handle_request():
        async with semaphore:
            return await call_next(request)

    try:
        # Use the older, backward-compatible wait_for method
        return await asyncio.wait_for(_handle_request(), timeout=settings.REQUEST_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning(
            event="request_timeout",
            path=request.url.path,
            timeout=settings.REQUEST_TIMEOUT_SEC,
        )
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Server busy, please retry"},
        )


# ROUTES

app.include_router(rag_router, prefix="/rag", tags=["RAG"])


# ROOT

@app.get("/", tags=["System"])
def root():
    return {
        "message": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env":     settings.ENV,
        "status":  "running",
    }


# HEALTH

@app.get("/health", tags=["System"])
def health():
    return {
        "status":  "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


# READINESS

@app.get("/ready", tags=["System"])
def readiness():
    try:
        from app.core.model_loader import model_loader
        models = model_loader.health_check()
        infra_status = infra.health_check()

        all_ready = models.get("embedder", False)

        return {
            "status": "ready" if all_ready else "degraded",
            "models": models,
            "infra":  infra_status,
        }

    except Exception as e:
        logger.error(event="readiness_failed", error=str(e))
        return {
            "status": "not_ready",
            "error":  str(e),
        }


# METRICS STUB

@app.get("/metrics", tags=["System"])
def metrics():
    if not settings.PROMETHEUS_ENABLED:
        return {"status": "disabled", "message": "Set PROMETHEUS_ENABLED=true to enable metrics"}

    try:
        from app.core.model_loader import model_loader
        return {
            "status":  "ok",
            "models":  model_loader.health_check(),
            "infra":   infra.health_check(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}