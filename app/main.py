# APP/MAIN.PY — PHASE 24 UPGRADE
# FASTAPI APPLICATION ENTRY POINT — WIRES EVERYTHING TOGETHER
# SECTION 4.6 — LIFESPAN, MIDDLEWARE, OTEL, PROMETHEUS, CORS, RATE LIMIT

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response

from app.core.config import settings
from app.utils.logger import get_logger, bind_request_context

logger = get_logger(__name__)

# Set CUDA TF32 + cuDNN benchmark flags before any model code runs
try:
    from app.core.startup_optimizer import set_cuda_performance_flags
    set_cuda_performance_flags()
except Exception:
    pass

# CONCURRENCY SEMAPHORE — SECTION 2.1
semaphore = asyncio.Semaphore(settings.MAX_PARALLEL_REQUESTS)


# PROMETHEUS SETUP — SECTION 6

def _setup_prometheus() -> None:
    if not settings.PROMETHEUS_ENABLED:
        return
    try:
        from prometheus_client import start_http_server
        start_http_server(settings.PROMETHEUS_PORT)
        logger.info(
            event="prometheus_started",
            port=settings.PROMETHEUS_PORT,
        )
    except Exception as e:
        logger.warning(event="prometheus_start_failed", error=str(e))


# OPENTELEMETRY SETUP — SECTION 2.1

def _setup_otel() -> None:
    if not settings.OTEL_ENABLED:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        resource = Resource.create({"service.name": settings.OTEL_SERVICE_NAME})
        sampler  = TraceIdRatioBased(settings.OTEL_SAMPLING_RATIO)
        provider = TracerProvider(resource=resource, sampler=sampler)

        exporter  = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)

        logger.info(
            event="otel_initialized",
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            service=settings.OTEL_SERVICE_NAME,
            sampling_ratio=settings.OTEL_SAMPLING_RATIO,
        )
    except Exception as e:
        logger.warning(event="otel_init_failed", error=str(e))


# QDRANT INIT — runs in background so it doesn't block Uvicorn ready signal

async def _init_qdrant_async() -> None:
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _init_qdrant_sync)
        logger.info(event="qdrant_ready")
    except Exception as e:
        logger.warning(event="qdrant_init_failed", error=str(e))


def _init_qdrant_sync() -> None:
    from scripts.init_qdrant import initialize_qdrant
    initialize_qdrant()


# INFRA WARMUP — fires as a background task so Uvicorn is ready immediately.
# Qdrant/Redis/Mongo connections establish concurrently while the first request
# is being served; circuit breakers handle any transient failures gracefully.

async def _warmup_infra_background() -> None:
    try:
        from app.core.infra_registry import infra
        await infra.warmup()
        logger.info(event="infra_warmup_complete")
    except Exception as e:
        logger.warning(event="infra_warmup_failed", error=str(e))


# MODEL WARMUP — parallel GPU preload via ThreadPoolExecutor.
# All GPU models are loaded concurrently into VRAM so the first query
# has zero cold-start penalty. Uses model_registry._ensure() which is
# already thread-safe and parallel-load aware.

async def _warmup_models_async() -> None:
    if not settings.WARMUP_AT_STARTUP:
        logger.info(
            event="startup_warmup_skipped",
            reason="lazy_mode",
            hint="models load on first ingest/query via model_registry",
        )
        return
    try:
        from app.core.startup_optimizer import preload_gpu_models
        await preload_gpu_models()
    except Exception as e:
        logger.warning(event="startup_warmup_failed", error=str(e))

    # Guardrail warm-up — pre-initializes Presidio PII engine + jailbreak
    # corpus embeddings so first request doesn't pay cold-start latency.
    try:
        import asyncio as _asyncio
        loop = _asyncio.get_running_loop()
        await loop.run_in_executor(None, _warmup_guardrails)
    except Exception as e:
        logger.warning(event="guardrail_warmup_failed", error=str(e))


def _warmup_guardrails() -> None:
    try:
        from app.guardrails import warm_up as guardrail_warm_up
        guardrail_warm_up()
        logger.info(event="guardrail_warmup_complete")
    except Exception as e:
        logger.warning(event="guardrail_warmup_failed", error=str(e))


# AUDIT LOG SETUP — SECTION 5

def _setup_audit_log() -> None:
    try:
        settings.AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not settings.AUDIT_LOG_PATH.exists():
            settings.AUDIT_LOG_PATH.touch()
        logger.info(event="audit_log_ready", path=str(settings.AUDIT_LOG_PATH))
    except Exception as e:
        logger.warning(event="audit_log_setup_failed", error=str(e))


# TEMP DIR CLEANUP ON STARTUP — SECTION 5

def _cleanup_temp_dirs() -> None:
    try:
        import shutil
        from pathlib import Path

        sweeps = []

        # Per-user temp/temp_frames/staging — sweep on startup so a crashed
        # ingestion run doesn't leave orphan frame dirs lying around.
        users_root = Path("data/users")
        if users_root.exists():
            for udir in users_root.iterdir():
                if not udir.is_dir():
                    continue
                for sub in ("temp", "temp_frames", "staging"):
                    p = udir / sub
                    if p.exists():
                        sweeps.append(p)

        for temp_dir in sweeps:
            if temp_dir.exists():
                for item in temp_dir.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                    except Exception:
                        pass
        logger.info(event="temp_dirs_cleaned")
    except Exception as e:
        logger.warning(event="temp_dir_cleanup_failed", error=str(e))


# LIFESPAN — SECTION 4.6

@asynccontextmanager
async def lifespan(app: FastAPI):

    startup_start = time.time()

    logger.info(
        event="startup_begin",
        env=settings.ENV,
        version=settings.APP_VERSION,
        app=settings.APP_NAME,
    )

    # SYNC FAST-PATH — only cheap, local operations that must finish before
    # Uvicorn signals ready. Target: <500 ms total.
    _cleanup_temp_dirs()
    _setup_audit_log()
    _setup_otel()
    _setup_prometheus()

    # BACKGROUND TASKS — network I/O and GPU model loads run concurrently
    # after Uvicorn is already accepting requests. Circuit breakers and lazy
    # getters mean the app degrades gracefully until each task completes.
    background_tasks = []

    # Qdrant collection init (network + retry — must not block ready signal)
    background_tasks.append(asyncio.create_task(_init_qdrant_async()))

    # Infra warmup: Qdrant client + BM25 + Redis + Mongo — all concurrent
    background_tasks.append(asyncio.create_task(_warmup_infra_background()))

    # GPU model preload: all models load in parallel into VRAM
    background_tasks.append(asyncio.create_task(_warmup_models_async()))

    startup_latency = round(time.time() - startup_start, 2)
    logger.info(
        event="app_ready",
        startup_latency=startup_latency,
        version=settings.APP_VERSION,
        note="infra+models loading in background",
    )

    yield

    # SHUTDOWN
    logger.info(event="shutdown_begin")
    for task in background_tasks:
        if not task.done():
            task.cancel()
    _cleanup_temp_dirs()
    logger.info(event="shutdown_complete")


# FASTAPI APP

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs"   if settings.ENV != "production" else None,
    redoc_url="/redoc" if settings.ENV != "production" else None,
    root_path=settings.ROOT_PATH,
)


# CORS MIDDLEWARE — SECTION 5

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# GZIP MIDDLEWARE

app.add_middleware(GZipMiddleware, minimum_size=1000)


# AUTH MIDDLEWARE — Phase 27
# Must be added AFTER CORSMiddleware and GZipMiddleware so it runs first on ingress.
# Populates request.state.user from the Bearer JWT on every request.

from app.api.middleware import AuthMiddleware
app.add_middleware(AuthMiddleware)


# GLOBAL EXCEPTION HANDLER

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error(
        event="global_error",
        path=str(request.url.path),
        method=request.method,
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=500,
        content={
            "status":     "error",
            "message":    "Internal server error",
            "request_id": request_id,
        },
    )


# REQUEST LOGGER + CORRELATION ID MIDDLEWARE — SECTION 2.1

@app.middleware("http")
async def request_logger(request: Request, call_next) -> Response:

    start      = time.time()
    request_id = request.headers.get(
        settings.CORRELATION_ID_HEADER,
        str(uuid.uuid4()),
    )
    request.state.request_id = request_id

    # BIND STRUCTURED LOG CONTEXT
    bind_request_context(request_id=request_id)

    # CLIENT IP — PROXY AWARE
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip     = (
        forwarded_for.split(",")[0].strip()
        if forwarded_for
        else (request.client.host if request.client else "unknown")
    )

    # AUDIT LOGGING — SECTION 5
    if settings.AUDIT_LOG_ENABLED:
        _write_audit_log(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            client_ip=client_ip,
        )

    try:
        response = await call_next(request)
        latency  = round(time.time() - start, 3)

        path_str = str(request.url.path)
        # Skip noisy probe paths — /health is hit by load balancers, IDE probes,
        # browser tabs, and Swagger UI; logging every hit drowns out real traffic.
        _SKIP_LOG_PATHS = ("/health", "/infra/health", "/models/health", "/metrics")
        skip_log = path_str in _SKIP_LOG_PATHS

        log_kwargs = dict(
            event=   "http_request",
            id=      request_id,
            method=  request.method,
            path=    path_str,
            status=  response.status_code,
            latency= latency,
            ip=      client_ip,
        )

        if latency > settings.SLOW_REQUEST_THRESHOLD:
            logger.warning(**log_kwargs)
        elif not skip_log:
            logger.info(**log_kwargs)

        response.headers[settings.CORRELATION_ID_HEADER] = request_id
        response.headers["X-Request-ID"]                 = request_id
        return response

    except Exception as e:
        logger.error(
            event="request_failed",
            id=    request_id,
            path=  str(request.url.path),
            error= str(e),
        )
        raise


# CONCURRENCY LIMIT + REQUEST TIMEOUT MIDDLEWARE — SECTION 2.1

@app.middleware("http")
async def limit_concurrency(request: Request, call_next) -> Response:

    async def _handle() -> Response:
        async with semaphore:
            return await call_next(request)

    # Ingest endpoints process large files (embed 100+ chunks) — give them
    # more time than regular API calls.
    is_ingest = "/ingest" in request.url.path or "/upload" in request.url.path
    timeout = settings.FILE_PROCESSING_TIMEOUT_SEC if is_ingest else settings.REQUEST_TIMEOUT_SEC

    try:
        return await asyncio.wait_for(
            _handle(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(
            event="request_timeout",
            path=str(request.url.path),
            timeout=timeout,
            request_id=request_id,
        )
        return JSONResponse(
            status_code=503,
            content={
                "status":     "error",
                "message":    "Server busy — please retry",
                "request_id": request_id,
            },
        )


# RATE LIMIT MIDDLEWARE — SECTION 5

_rate_limit_store: Dict[str, Any] = {}


@app.middleware("http")
async def rate_limit(request: Request, call_next) -> Response:
    if not settings.RATE_LIMIT_RPM:
        return await call_next(request)

    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip     = (
        forwarded_for.split(",")[0].strip()
        if forwarded_for
        else (request.client.host if request.client else "unknown")
    )

    now    = time.time()
    window = 60.0
    key    = f"ratelimit:{client_ip}"

    entry = _rate_limit_store.get(key, {"count": 0, "window_start": now})

    if now - entry["window_start"] > window:
        entry = {"count": 0, "window_start": now}

    entry["count"] += 1
    _rate_limit_store[key] = entry

    if entry["count"] > settings.RATE_LIMIT_RPM:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.warning(
            event="rate_limit_exceeded",
            client_ip=client_ip,
            count=entry["count"],
            limit=settings.RATE_LIMIT_RPM,
        )
        return JSONResponse(
            status_code=429,
            content={
                "status":     "error",
                "message":    "Rate limit exceeded — please slow down",
                "request_id": request_id,
            },
        )

    return await call_next(request)


# AUDIT LOG WRITER — SECTION 5

def _write_audit_log(
    request_id: str,
    method:     str,
    path:       str,
    client_ip:  str,
) -> None:
    try:
        import json
        entry = json.dumps({
            "request_id": request_id,
            "method":     method,
            "path":       path,
            "client_ip":  client_ip,
            "timestamp":  time.time(),
        })
        with open(settings.AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception as e:
        logger.warning(event="audit_log_write_failed", error=str(e))


# ROUTES

from app.api.api_routes import router as rag_router
app.include_router(rag_router, prefix="/rag", tags=["RAG"])

from app.auth.router import router as auth_router
app.include_router(auth_router)  # mounts at /auth (prefix defined in router.py)

from app.auth.admin_router import router as admin_router
app.include_router(admin_router)  # mounts at /admin — requires role=admin JWT


# ROOT

@app.get("/", tags=["System"])
def root() -> Dict[str, Any]:
    return {
        "message": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env":     settings.ENV,
        "status":  "running",
    }


# HEALTH — SECTION 6

@app.get("/health", tags=["System"])
def health() -> Dict[str, Any]:
    return {
        "status":  "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env":     settings.ENV,
    }


# READINESS — SECTION 6

@app.get("/ready", tags=["System"])
def readiness() -> Dict[str, Any]:
    try:
        from app.core.model_loader import model_loader
        from app.core.infra_registry import infra

        models       = model_loader.health_check()
        infra_status = infra.health_check()
        all_ready    = models.get("embedder", False)

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


# METRICS — SECTION 6

@app.get("/metrics", tags=["System"])
def metrics() -> Dict[str, Any]:
    if not settings.PROMETHEUS_ENABLED:
        return {
            "status":  "disabled",
            "message": "Set PROMETHEUS_ENABLED=true to enable metrics",
        }

    try:
        from app.core.model_loader import model_loader
        from app.core.infra_registry import infra

        return {
            "status": "ok",
            "models": model_loader.health_check(),
            "infra":  infra.health_check(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# INFRA HEALTH — SECTION 6

@app.get("/infra/health", tags=["System"])
def infra_health() -> Dict[str, Any]:
    try:
        from app.core.infra_registry import infra
        return {
            "status": "ok",
            "infra":  infra.health_check(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# TOOLS LIST — SECTION 4.9

@app.get("/tools", tags=["Agents"])
def list_tools() -> Dict[str, Any]:
    try:
        from app.agents.tool_registry import ToolRegistry
        registry = ToolRegistry()
        return {
            "status": "ok",
            "tools":  registry.list_tools(),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )