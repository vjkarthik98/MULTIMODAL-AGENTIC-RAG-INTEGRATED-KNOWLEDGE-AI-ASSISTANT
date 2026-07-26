import asyncio
import time
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Gauge, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.metrics import circuit_breaker_failures as _circuit_breaker_failures
from app.core.metrics import circuit_breaker_state as _circuit_breaker_state
from app.memory.mongo_memory import MongoMemory
from app.memory.redis_memory import RedisMemory
from app.retrieval.bm25_retriever import BM25Retriever
from app.vectorstore.qdrant_store import QdrantVectorStore

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_infra_init_duration = Histogram(
    "infra_init_duration_seconds",
    "Infrastructure service initialization duration",
    ["service"],
)
_infra_init_errors = Counter(
    "infra_init_errors_total",
    "Infrastructure initialization errors by service",
    ["service", "error_type"],
)

_infra_available = Gauge(
    "infra_service_available",
    "Whether an infra service is available (1=yes, 0=no)",
    ["service"],
)


# CIRCUIT BREAKER


class _CircuitBreaker:

    def __init__(
        self,
        name: str,
        fail_max: int = 5,
        reset_timeout: int = 60,
    ) -> None:
        self.name = name
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._opened_at = 0.0
        self._open = False

    def record_success(self) -> None:
        self._failures = 0
        self._open = False
        self._opened_at = 0.0
        _circuit_breaker_state.labels(service=self.name).set(0)
        _circuit_breaker_failures.labels(service=self.name).set(0)

    def record_failure(self) -> None:
        self._failures += 1
        _circuit_breaker_failures.labels(service=self.name).set(self._failures)

        if self._failures >= self.fail_max:
            self._open = True
            self._opened_at = time.time()
            _circuit_breaker_state.labels(service=self.name).set(1)
            logger.warning(
                "circuit_breaker_opened",
                service=self.name,
                failures=self._failures,
                fail_max=self.fail_max,
            )

    def is_open(self) -> bool:
        if self._open:
            # HALF-OPEN PROBE AFTER RESET TIMEOUT
            if time.time() - self._opened_at >= self.reset_timeout:
                self._open = False
                self._failures = 0
                self._opened_at = 0.0
                _circuit_breaker_state.labels(service=self.name).set(0)
                _circuit_breaker_failures.labels(service=self.name).set(0)
                logger.info("circuit_breaker_half_open", service=self.name)
                return False
            return True
        return False

    def check(self) -> None:
        if self.is_open():
            raise RuntimeError(
                f"CIRCUIT_OPEN_{self.name.upper()}: "
                f"service unavailable after {self._failures} failures"
            )


class InfraRegistry:

    def __init__(self) -> None:
        self._vector_store: QdrantVectorStore | None = None
        self._bm25: BM25Retriever | None = None
        self._memory: RedisMemory | None = None
        self._mongo: MongoMemory | None = None
        self._cache: Any | None = None
        self._cache_failed: bool = False

        self._initialized: bool = False

        # CIRCUIT BREAKERS PER SERVICE
        self._cb: dict[str, _CircuitBreaker] = {
            "qdrant": _CircuitBreaker(
                "qdrant",
                fail_max=getattr(settings, "QDRANT_CB_FAIL_MAX", 5),
                reset_timeout=getattr(settings, "QDRANT_CB_RESET_TIMEOUT", 60),
            ),
            "redis": _CircuitBreaker(
                "redis",
                fail_max=getattr(settings, "REDIS_CB_FAIL_MAX", 5),
                reset_timeout=getattr(settings, "REDIS_CB_RESET_TIMEOUT", 30),
            ),
            "mongo": _CircuitBreaker(
                "mongo",
                fail_max=getattr(settings, "MONGO_CB_FAIL_MAX", 5),
                reset_timeout=getattr(settings, "MONGO_CB_RESET_TIMEOUT", 60),
            ),
        }

        # INITIALIZE PROMETHEUS GAUGES
        for service in ("qdrant", "redis", "mongo", "bm25"):
            _circuit_breaker_state.labels(service=service).set(0)
            _infra_available.labels(service=service).set(0)

    # CHECK CIRCUIT BREAKER

    def _check_circuit(self, name: str) -> bool:
        cb = self._cb.get(name)
        if cb and cb.is_open():
            logger.warning("circuit_open_skipping_init", service=name)
            return False
        return True

    # RECORD FAILURE

    def _record_failure(self, name: str) -> None:
        cb = self._cb.get(name)
        if cb:
            cb.record_failure()
        _infra_available.labels(service=name).set(0)

    # RECORD SUCCESS

    def _record_success(self, name: str) -> None:
        cb = self._cb.get(name)
        if cb:
            cb.record_success()
        _infra_available.labels(service=name).set(1)

    # RETRY WRAPPER WITH EXPONENTIAL BACKOFF

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=8),
        reraise=True,
    )
    def _init_with_retry(self, fn, name: str) -> Any:
        start = time.time()
        with tracer.start_as_current_span(f"infra_init_{name}") as span:
            span.set_attribute("service", name)
            try:
                obj = fn()
                latency = round(time.time() - start, 2)
                _infra_init_duration.labels(service=name).observe(latency)
                span.set_attribute("latency", latency)
                span.set_status(Status(StatusCode.OK))
                logger.info("infra_initialized", service=name, latency=latency)
                return obj
            except Exception as exc:
                _infra_init_errors.labels(
                    service=name,
                    error_type=type(exc).__name__,
                ).inc()
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                raise

    # ASYNC WARMUP — INITIALIZE ALL SERVICES CONCURRENTLY

    async def warmup(self) -> None:
        if self._initialized:
            logger.info("infra_warmup_skipped")
            return

        logger.info("infra_warmup_started")

        from app.core.model_loader import model_loader as _loader

        def _warm_lang_detector():
            # Pre-build the lingua language detector (CPU, ~1s) so the first text
            # upload doesn't pay the cold build cost on the request path.
            try:
                from app.ingestion.txt_ingest import warm_language_detector

                warm_language_detector()
            except Exception:
                pass

        tasks = [
            # Infra services
            asyncio.to_thread(self.get_vector_store),
            asyncio.to_thread(self.get_bm25),
            asyncio.to_thread(self.get_memory),
            asyncio.to_thread(self.get_mongo),
            # Text-path essentials only — warm fast (embedder, reranker).
            asyncio.to_thread(_loader.get_embedder),
            asyncio.to_thread(_loader.get_reranker),
            # CPU NLP used by every text ingest — warm off the request path.
            asyncio.to_thread(_warm_lang_detector),
            # trocr/ner/whisper/qwen2_vl/diarizer/blip warm lazily on first use of
            # their modality via model_registry.ensure_for_modality() — keeps
            # startup fast and frees VRAM/RAM for text-only workloads.
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = [r for r in results if isinstance(r, Exception)]

        for f in failures:
            logger.warning("infra_warmup_partial_failure", error=str(f))

        self._initialized = True
        logger.info("infra_warmup_completed", failures=len(failures))

    # QDRANT VECTOR STORE

    def get_vector_store(self) -> QdrantVectorStore | None:
        if self._vector_store:
            return self._vector_store

        if not self._check_circuit("qdrant"):
            return None

        try:
            self._vector_store = self._init_with_retry(
                lambda: QdrantVectorStore(),
                "qdrant",
            )
            self._record_success("qdrant")
            return self._vector_store

        except Exception as exc:
            self._record_failure("qdrant")
            logger.error("qdrant_init_failed", error=str(exc))
            return None

    # LOCAL REDIS CACHE — hot-path ephemeral state (job status, embedding cache,
    # token blacklist, rate limits). Backed by LOCAL Redis (settings.LOCAL_CACHE_*),
    # NOT Upstash, so it never pays the ~200ms cloud round-trip. Returns None if the
    # local server is unavailable; callers must treat None as "no cache" and degrade
    # to in-process behavior.

    def get_cache(self) -> Any | None:
        if self._cache is not None:
            return self._cache
        if self._cache_failed:
            return None
        try:
            import redis as _redis_lib

            client = _redis_lib.Redis(
                host=settings.LOCAL_CACHE_HOST,
                port=settings.LOCAL_CACHE_PORT,
                db=settings.LOCAL_CACHE_DB,
                socket_timeout=getattr(settings, "REDIS_TIMEOUT", 5),
                socket_connect_timeout=2,
                decode_responses=True,
            )
            client.ping()
            self._cache = client
            logger.info(
                "local_cache_connected",
                host=settings.LOCAL_CACHE_HOST,
                port=settings.LOCAL_CACHE_PORT,
                db=settings.LOCAL_CACHE_DB,
            )
            return self._cache
        except Exception as exc:
            self._cache_failed = True
            logger.warning(
                "local_cache_unavailable",
                error=str(exc),
                hint="start local redis-server for fast hot-path state; degrading to in-process",
            )
            return None

    # BM25 RETRIEVER

    def get_bm25(self) -> BM25Retriever:
        if self._bm25:
            return self._bm25

        try:
            start = time.time()
            self._bm25 = BM25Retriever()
            latency = round(time.time() - start, 2)
            _infra_init_duration.labels(service="bm25").observe(latency)
            _infra_available.labels(service="bm25").set(1)
            logger.info("bm25_initialized", latency=latency)
        except Exception as exc:
            _infra_init_errors.labels(
                service="bm25",
                error_type=type(exc).__name__,
            ).inc()
            _infra_available.labels(service="bm25").set(0)
            logger.error("bm25_init_failed", error=str(exc))
            raise

        return self._bm25

    # REDIS MEMORY

    def get_memory(self) -> RedisMemory | None:
        if self._memory:
            return self._memory

        if not self._check_circuit("redis"):
            return None

        try:
            self._memory = self._init_with_retry(
                lambda: RedisMemory(),
                "redis",
            )
            self._record_success("redis")
            return self._memory

        except Exception as exc:
            self._record_failure("redis")
            logger.error("redis_init_failed", error=str(exc))
            return None

    # MONGO MEMORY

    def get_mongo(self) -> MongoMemory | None:
        if self._mongo:
            return self._mongo

        if not self._check_circuit("mongo"):
            return None

        try:
            self._mongo = self._init_with_retry(
                lambda: MongoMemory(),
                "mongo",
            )
            self._record_success("mongo")
            return self._mongo

        except Exception as exc:
            self._record_failure("mongo")
            logger.error("mongo_init_failed", error=str(exc))
            return None

    # HEALTH CHECK

    def health_check(self) -> dict[str, Any]:
        circuit_states = {name: cb.is_open() for name, cb in self._cb.items()}
        failure_counts = {name: cb._failures for name, cb in self._cb.items()}

        return {
            "qdrant": self._vector_store is not None,
            "redis": self._memory is not None,
            "mongo": self._mongo is not None,
            "bm25": self._bm25 is not None,
            "initialized": self._initialized,
            "circuits_open": circuit_states,
            "failure_counts": failure_counts,
        }

    # RESET ALL SERVICES AND CIRCUIT BREAKERS

    def reset(self) -> None:
        self._vector_store = None
        self._bm25 = None
        self._memory = None
        self._mongo = None
        self._cache = None
        self._cache_failed = False
        self._initialized = False

        for name, cb in self._cb.items():
            cb.record_success()
            _infra_available.labels(service=name).set(0)

        logger.warning("infra_reset")

    # GDPR PURGE — DELETE ALL DATA FOR A SESSION/USER ACROSS ALL SERVICES

    async def gdpr_purge(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, bool]:

        if not session_id and not user_id:
            raise ValueError("GDPR_PURGE_REQUIRES_SESSION_ID_OR_USER_ID")

        results: dict[str, bool] = {}

        # PURGE QDRANT
        try:
            if self._vector_store and session_id:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._vector_store.gdpr_purge,
                    user_id,
                    session_id,
                )
                results["qdrant"] = True
        except Exception as exc:
            logger.error("gdpr_purge_qdrant_failed", error=str(exc))
            results["qdrant"] = False

        # PURGE REDIS
        try:
            if self._memory and session_id:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._memory.clear_memory,
                    session_id,
                )
                results["redis"] = True
        except Exception as exc:
            logger.error("gdpr_purge_redis_failed", error=str(exc))
            results["redis"] = False

        # PURGE MONGO
        try:
            if self._mongo and session_id:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._mongo.clear_memory,
                    session_id,
                )
                results["mongo"] = True
        except Exception as exc:
            logger.error("gdpr_purge_mongo_failed", error=str(exc))
            results["mongo"] = False

        logger.info(
            "gdpr_purge_complete",
            session_id=session_id,
            user_id=user_id,
            results=results,
        )

        return results


# SINGLETON

infra = InfraRegistry()
