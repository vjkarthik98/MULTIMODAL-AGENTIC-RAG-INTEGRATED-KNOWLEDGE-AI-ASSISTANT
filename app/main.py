from __future__ import annotations

# APP/MAIN.PY — MAGIK FINANCE RAG v0.30.0
# FASTAPI APPLICATION ENTRY POINT — WIRES EVERYTHING TOGETHER
# SECTION 4.6 — LIFESPAN, MIDDLEWARE, OTEL, PROMETHEUS, CORS, RATE LIMIT
# ── HF CACHE MUST BE SET BEFORE ANY TRANSFORMERS/TORCH IMPORT ──────────────
import os as _os
from pathlib import Path as _Path

_hf_home = _os.getenv("HF_HOME", str(_Path(__file__).parents[1] / ".hf_cache"))
_os.environ["HF_HOME"] = _hf_home
_os.environ["TRANSFORMERS_CACHE"] = _hf_home + "/hub"
_os.environ["HF_DATASETS_CACHE"] = _hf_home + "/datasets"
_os.environ["HF_HUB_CACHE"] = _hf_home + "/hub"
# ────────────────────────────────────────────────────────────────────────────

import asyncio
import faulthandler
import time
import uuid

# Print full Python + C stack to stderr on SIGSEGV/SIGFPE/SIGABRT so crashes
# are diagnosable even when uvicorn swallows the output.
faulthandler.enable()
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from opentelemetry import trace

from app.core.config import settings
from app.utils.logger import _inject_otel_context, bind_request_context, get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

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
        sampler = TraceIdRatioBased(settings.OTEL_SAMPLING_RATIO)
        provider = TracerProvider(resource=resource, sampler=sampler)

        exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
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
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _init_qdrant_sync)
        logger.info(event="qdrant_ready")
    except Exception as e:
        logger.warning(event="qdrant_init_failed", error=str(e))


def _init_qdrant_sync() -> None:
    from app.vectorstore.qdrant_store import initialize_qdrant

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

        # preload_gpu_models() runs guardrail warmup internally (before LLM)
        # to guarantee all PyTorch CUDA ops finish before llama.cpp CUBLAS init.
        await preload_gpu_models()
    except Exception as e:
        logger.warning(event="startup_warmup_failed", error=str(e))


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

    # Auth config check — fail fast if AUTH_ENABLED=False in production
    from app.core.startup_validator import validate_auth_config, validate_model_manifest

    validate_auth_config()

    # Model manifest check — fail fast if required models are not cached
    try:
        validate_model_manifest()
    except RuntimeError as exc:
        logger.critical(event="model_manifest_invalid", error=str(exc))
        raise

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

    # Online eval (Phase 31) — no-op unless ONLINE_EVAL_ENABLED and
    # ONLINE_EVAL_SAMPLE_RATE > 0. Must run in THIS process: it pushes into
    # the same prometheus_client default registry _setup_prometheus() above
    # just started serving on settings.PROMETHEUS_PORT — a separate CLI/cron
    # process would have its own registry and never reach the scrape target.
    from app.eval.jobs.online_eval import run_online_eval_loop

    background_tasks.append(asyncio.create_task(run_online_eval_loop()))

    # Drift eval (monitoring Phase 3) — same "no-op unless enabled, must run
    # in this process" reasoning as online eval directly above; pushes into
    # the same Prometheus registry. No-op unless DRIFT_ENABLED and
    # ONLINE_EVAL_SAMPLE_RATE > 0 (it reads the same sampled-traffic
    # collection online eval does).
    from app.eval.jobs.drift_eval import run_drift_eval_loop

    background_tasks.append(asyncio.create_task(run_drift_eval_loop()))

    # VRAM reaper — evicts ingestion-only models (Whisper, BLIP2, Qwen2-VL,
    # TrOCR, diarizer, NER, FinBERT) once they go idle, or sooner under VRAM
    # pressure. The query hot set (llm/text_embedder/reranker/SigLIP) is
    # pinned and never touched. No-op unless MODEL_IDLE_EVICTION_ENABLED.
    # Must run in THIS process — it operates on this process's model_loader
    # singleton and its Prometheus gauges share the registry _setup_prometheus()
    # is already serving.
    from app.core.model_reaper import run_model_reaper_loop

    background_tasks.append(asyncio.create_task(run_model_reaper_loop()))

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
    docs_url="/docs" if settings.ENV != "production" else None,
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

from app.api.middleware import AuthMiddleware, CSRFMiddleware

app.add_middleware(AuthMiddleware)
# Runs before AuthMiddleware on ingress (Starlette applies middleware in
# reverse registration order) — a forged request gets rejected before it
# even reaches token verification.
app.add_middleware(CSRFMiddleware)


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
            "status": "error",
            "message": "Internal server error",
            "request_id": request_id,
        },
    )


# REQUEST LOGGER + CORRELATION ID MIDDLEWARE — SECTION 2.1


@app.middleware("http")
async def request_logger(request: Request, call_next) -> Response:

    start = time.time()
    request_id = request.headers.get(
        settings.CORRELATION_ID_HEADER,
        str(uuid.uuid4()),
    )
    request.state.request_id = request_id

    # BIND STRUCTURED LOG CONTEXT
    bind_request_context(request_id=request_id)

    # CLIENT IP — PROXY AWARE
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = (
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

    # Root span for the request — gives every log line emitted while handling
    # it (via _inject_otel_context) a real trace_id, which is what lets
    # Grafana's Loki->Tempo "derived field" correlation actually match.
    with tracer.start_as_current_span("http_request") as span:
        _inject_otel_context()
        try:
            response = await call_next(request)
            latency = round(time.time() - start, 3)

            path_str = str(request.url.path)
            # Skip noisy probe paths — /health is hit by load balancers, IDE probes,
            # browser tabs, and Swagger UI; logging every hit drowns out real traffic.
            _SKIP_LOG_PATHS = ("/health", "/infra/health", "/models/health", "/metrics")
            skip_log = path_str in _SKIP_LOG_PATHS

            log_kwargs = dict(
                event="http_request",
                id=request_id,
                method=request.method,
                path=path_str,
                status=response.status_code,
                latency=latency,
                ip=client_ip,
            )

            if latency > settings.SLOW_REQUEST_THRESHOLD:
                logger.warning(**log_kwargs)
            elif not skip_log:
                logger.info(**log_kwargs)

            response.headers[settings.CORRELATION_ID_HEADER] = request_id
            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as e:
            span.record_exception(e)
            logger.error(
                event="request_failed",
                id=request_id,
                path=str(request.url.path),
                error=str(e),
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
                "status": "error",
                "message": "Server busy — please retry",
                "request_id": request_id,
            },
        )


# RATE LIMIT MIDDLEWARE — SECTION 5

_rate_limit_store: dict[str, Any] = {}


from app.utils.net import resolve_client_ip as _resolve_client_ip


def _rate_limit_user_id(request: Request) -> str | None:
    """Independently verify the Bearer token to get a user_id, without
    depending on AuthMiddleware having already run.

    Empirically verified (not assumed) that this middleware executes BEFORE
    AuthMiddleware: FastAPI/Starlette's add_middleware() prepends, so the
    LAST middleware registered in app/main.py ends up OUTERMOST — and this
    `rate_limit` middleware is registered (via @app.middleware("http")) after
    AuthMiddleware's app.add_middleware(AuthMiddleware) call below, which
    means it actually runs first on ingress despite the auth block appearing
    earlier in the file. request.state.user is therefore always None here.
    Small, accepted cost: a valid token gets decoded + blacklist-checked
    twice per request (once here, once in AuthMiddleware) rather than once —
    reordering to share the work would make rate limiting depend on Auth
    running first, losing its current cheap, outermost early-rejection
    property for anonymous/malformed-token traffic.
    """
    from app.api.middleware import _extract_bearer

    token = _extract_bearer(request)
    if not token:
        return None
    try:
        from app.auth.jwt_handler import verify_token

        payload = verify_token(token, expected_type="access")
        return payload["sub"]
    except Exception:
        return None


def _rate_limit_key(prefix: str, client_ip: str, window: float) -> tuple[str, dict]:
    now = time.time()
    key = f"{prefix}:{client_ip}"
    entry = _rate_limit_store.get(key, {"count": 0, "window_start": now})
    if now - entry["window_start"] > window:
        entry = {"count": 0, "window_start": now}
    entry["count"] += 1
    _rate_limit_store[key] = entry
    return key, entry


_AUTH_BRUTE_FORCE_PATHS = {
    "/auth/login": (60.0, "auth_login"),
    "/auth/login/form": (60.0, "auth_login"),
    "/auth/register": (3600.0, "auth_register"),
    "/auth/verify-otp": (60.0, "auth_otp"),
    "/auth/mfa/verify": (60.0, "auth_otp"),
    "/auth/forgot-password": (3600.0, "auth_register"),
}


@app.middleware("http")
async def rate_limit(request: Request, call_next) -> Response:
    client_ip = _resolve_client_ip(request)

    # Dedicated, stricter, IP-keyed brute-force protection on auth endpoints —
    # independent of RATE_LIMIT_RPM so it can't be disabled by turning off the
    # general API rate limit, and tight enough to actually deter guessing.
    auth_cfg = _AUTH_BRUTE_FORCE_PATHS.get(request.url.path)
    if auth_cfg:
        window, prefix = auth_cfg
        limit = (
            settings.AUTH_REGISTER_RATE_LIMIT_PER_HOUR
            if prefix == "auth_register"
            else settings.AUTH_LOGIN_RATE_LIMIT_PER_MIN
        )
        _, entry = _rate_limit_key(prefix, client_ip, window)
        if entry["count"] > limit:
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            logger.warning(
                event="auth_rate_limit_exceeded",
                path=request.url.path,
                client_ip=client_ip,
                count=entry["count"],
                limit=limit,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "message": "Too many attempts — please wait before retrying",
                    "request_id": request_id,
                },
            )

    if not settings.RATE_LIMIT_RPM:
        return await call_next(request)

    # Per-user when authenticated (Redis-backed, app/auth/rate_limit.py —
    # shared across all app instances, keyed on identity not network
    # location), falling back to the existing per-IP in-memory bucket for
    # anonymous/unauthenticated requests, which have no user identity to key
    # on. Before this, EVERY request — authenticated or not — shared one
    # bucket per client IP, so N users behind the same IP (a shared office/
    # university NAT, or a battery of load-test VUs from one host) collided
    # in one 60/min bucket regardless of how many distinct accounts were
    # involved. See perf/k6/README.md for how that surfaced.
    user_id = _rate_limit_user_id(request)

    if user_id:
        from app.auth.rate_limit import check_user_rate_limit

        try:
            check_user_rate_limit(user_id)
        except ValueError:
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            logger.warning(
                event="rate_limit_exceeded",
                user_id=user_id,
                limit=settings.RATE_LIMIT_RPM,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "message": "Rate limit exceeded — please slow down",
                    "request_id": request_id,
                },
            )
        return await call_next(request)

    _, entry = _rate_limit_key("ratelimit", client_ip, 60.0)

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
                "status": "error",
                "message": "Rate limit exceeded — please slow down",
                "request_id": request_id,
            },
        )

    return await call_next(request)


# AUDIT LOG WRITER — SECTION 5


def _write_audit_log(
    request_id: str,
    method: str,
    path: str,
    client_ip: str,
) -> None:
    try:
        import json

        entry = json.dumps(
            {
                "request_id": request_id,
                "method": method,
                "path": path,
                "client_ip": client_ip,
                "timestamp": time.time(),
            }
        )
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


# ROOT — see the STATIC SPA block at the end of this module.
#
# GET "/" is registered there instead of here, conditionally: the built UI
# (production) or this same JSON banner (API-only / local dev without a UI
# build). Two `@app.get("/")` handlers in one module is not two endpoints —
# FastAPI/Starlette matches routes in registration order, so whichever is
# registered FIRST wins and the second is permanently unreachable dead code.
# That exact mistake shipped once already: this handler here, registered
# before the static mount at the bottom of the file, silently shadowed the
# new UI route and kept serving the JSON banner on a build that had the UI
# fix merged in. Single source of truth now.


# HEALTH — SECTION 6


@app.get("/health", tags=["System"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.ENV,
    }


# VERSION / BUILD INTROSPECTION — Phase 29 §A. GIT_SHA/IMAGE_TAG are build-time
# env vars (set via Docker --build-arg in cd.yml, see Dockerfile's `runtime`
# target) — "unknown" outside a CD-built image (e.g. local dev) is expected,
# not a bug. Lets any running instance prove exactly what it's serving
# without needing shell/log access.
@app.get("/version", tags=["System"])
def version() -> dict[str, Any]:
    import json

    from app.prompt.prompt_builder import PROMPT_VERSION

    manifest_path = _Path(settings.HF_HOME) / "download_manifest.json"
    models: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
            models = [
                {
                    "model_id": e.get("model_id"),
                    "sha256": e.get("sha256"),
                    "revision": e.get("revision"),
                }
                for e in entries
            ]
        except Exception:
            pass

    return {
        "service": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "git_sha": _os.environ.get("GIT_SHA", "unknown"),
        "image_tag": _os.environ.get("IMAGE_TAG", "unknown"),
        "prompt_version": PROMPT_VERSION,
        "env": settings.ENV,
        "models": models,
    }


# READINESS — SECTION 6


@app.get("/ready", tags=["System"])
def readiness() -> dict[str, Any]:
    try:
        from app.core.infra_registry import infra
        from app.core.model_loader import model_loader

        models = model_loader.health_check()
        infra_status = infra.health_check()
        # embedder must be loaded AND the LLM must be actually reachable —
        # not just "constructed" (see model_loader.health_check()'s
        # llm_ready / GGUFModel.health_check()'s "ready" for why the two
        # differ). Without the llm_ready check, /ready reported "ready"
        # during the startup window while llama-server was still loading
        # its checkpoint shards, and a real /rag/query call in that window
        # hit connection-refused and tripped the LLM circuit breaker.
        all_ready = bool(models.get("embedder", False)) and bool(models.get("llm_ready", False))

        return {
            "status": "ready" if all_ready else "degraded",
            "models": models,
            "infra": infra_status,
        }

    except Exception as e:
        logger.error(event="readiness_failed", error=str(e))
        return {
            "status": "not_ready",
            "error": str(e),
        }


# STATUS — SECTION 6 — human/dashboard-facing JSON summary of model + infra
# health. Renamed from "/metrics" (Phase 31): a route named /metrics that
# returns a JSON dict, while the REAL Prometheus text-exposition metrics are
# served by prometheus_client.start_http_server on settings.PROMETHEUS_PORT
# (see _setup_prometheus() above), was two different things sharing one name.
# /metrics is now reserved for the real Prometheus scrape target; this JSON
# summary lives at /status. See monitoring/prometheus/prometheus.yml.


@app.get("/status", tags=["System"])
def status() -> dict[str, Any]:
    if not settings.PROMETHEUS_ENABLED:
        return {
            "status": "disabled",
            "message": "Set PROMETHEUS_ENABLED=true to enable metrics",
        }

    try:
        from app.core.infra_registry import infra
        from app.core.model_loader import model_loader

        return {
            "status": "ok",
            "models": model_loader.health_check(),
            "infra": infra.health_check(),
            "prometheus_port": settings.PROMETHEUS_PORT,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# INFRA HEALTH — SECTION 6


@app.get("/infra/health", tags=["System"])
def infra_health() -> dict[str, Any]:
    try:
        from app.core.infra_registry import infra

        return {
            "status": "ok",
            "infra": infra.health_check(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# TOOLS LIST — SECTION 4.9


@app.get("/tools", tags=["Agents"], response_model=None)
def list_tools() -> dict[str, Any] | JSONResponse:
    try:
        from app.agents.tool_registry import ToolRegistry

        registry = ToolRegistry()
        return {
            "status": "ok",
            "tools": registry.list_tools(),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# STATIC SPA — must stay LAST in this module
#
# The production image builds ui/ (Dockerfile stage `ui-builder`) into
# /app/ui_dist. When that directory is present this serves the React app at /;
# when it is absent — a plain source checkout, or `uvicorn app.main:app` during
# backend development — nothing below registers and the API behaves exactly as
# before. That conditional is deliberate: the deployed artifact must not require
# a separate `npm run dev` process to be reachable, and a backend-only checkout
# must not fail because a build output is missing.
#
# ui/src/api/client.js uses `const API = ''` (same-origin relative paths), so the
# bundle needs no build-time API URL and works under any hostname.

_UI_DIST = _Path(__file__).resolve().parents[1] / "ui_dist"

# Prefixes owned by the API. The catch-all below refuses them so a typo like
# /rag/quer returns a JSON 404 instead of silently handing back index.html —
# which would otherwise surface to a client as an unparseable HTML "response".
_API_PREFIXES = (
    "auth",
    "rag",
    "admin",
    "health",
    "metrics",
    "docs",
    "redoc",
    "openapi.json",
    "tools",
    "infra",
    "models",
)

if _UI_DIST.is_dir():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    _INDEX = _UI_DIST / "index.html"

    # Vite emits hashed filenames under assets/ — immutable, so cache hard.
    _assets = _UI_DIST / "assets"
    if _assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="ui-assets")

    @app.get("/", include_in_schema=False)
    def _ui_root() -> FileResponse:
        return FileResponse(_INDEX)

    @app.get("/{full_path:path}", include_in_schema=False, response_model=None)
    def _ui_catch_all(full_path: str) -> FileResponse | JSONResponse:
        """Serve real files; fall back to index.html for client-side routes.

        A SPA owns its own routing, so /login and /chat are not server routes —
        they must return index.html and let the router resolve them. Registered
        last, after every include_router call, so API routes always win.
        """
        top = full_path.split("/", 1)[0]
        if top in _API_PREFIXES:
            return JSONResponse(status_code=404, content={"detail": "Not Found"})

        if full_path:
            candidate = (_UI_DIST / full_path).resolve()
            # Containment check — blocks ../ traversal out of the bundle.
            if candidate.is_file() and _UI_DIST.resolve() in candidate.parents:
                return FileResponse(candidate)

        return FileResponse(_INDEX)

    logger.info(event="ui_static_mounted", path=str(_UI_DIST))
else:
    logger.info(event="ui_static_absent", detail="API-only mode; no ui_dist directory")

    @app.get("/", tags=["System"])
    def _root_api_only() -> dict[str, Any]:
        """API-only fallback: no ui_dist build present (e.g. `uvicorn app.main:app`
        run directly against a source checkout without `npm run build` first)."""
        return {
            "message": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "env": settings.ENV,
            "status": "running",
        }
