import asyncio
import time
from typing import Any, Dict, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.memory.mongo_memory import MongoMemory
from app.memory.redis_memory import RedisMemory
from app.retrieval.bm25_retriever import BM25Retriever
from app.utils.logger import get_logger
from app.vectorstore.qdrant_store import QdrantVectorStore

logger = get_logger(__name__)


class InfraRegistry:

    def __init__(self) -> None:
        self._vector_store: Optional[QdrantVectorStore] = None
        self._bm25: Optional[BM25Retriever]             = None
        self._memory: Optional[RedisMemory]             = None
        self._mongo: Optional[MongoMemory]              = None

        self._initialized: bool = False

        # CIRCUIT BREAKER STATE
        self._fail_counts: Dict[str, int] = {
            "qdrant": 0,
            "redis":  0,
            "mongo":  0,
        }
        self._max_failures: int = 3

    # CIRCUIT BREAKER

    def _check_circuit(self, name: str) -> bool:
        return self._fail_counts.get(name, 0) < self._max_failures

    def _record_failure(self, name: str) -> None:
        self._fail_counts[name] = self._fail_counts.get(name, 0) + 1
        logger.warning(
            event="circuit_failure_recorded",
            service=name,
            count=self._fail_counts[name],
            max=self._max_failures,
        )

    def _reset_failure(self, name: str) -> None:
        self._fail_counts[name] = 0

    def _is_circuit_open(self, name: str) -> bool:
        open_ = not self._check_circuit(name)
        if open_:
            logger.warning(event="circuit_open", service=name, failures=self._fail_counts[name])
        return open_

    # RETRY WRAPPER

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
    def _init_with_retry(self, fn, name: str):
        start = time.time()
        obj   = fn()
        logger.info(event="infra_initialized", service=name, latency=round(time.time() - start, 2))
        return obj

    # WARMUP

    async def warmup(self) -> None:
        if self._initialized:
            logger.info(event="infra_warmup_skipped")
            return

        logger.info(event="infra_warmup_started")

        tasks = [
            asyncio.to_thread(self.get_vector_store),
            asyncio.to_thread(self.get_bm25),
            asyncio.to_thread(self.get_mongo),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        failures = [r for r in results if isinstance(r, Exception)]

        for f in failures:
            logger.error(event="infra_warmup_error", error=str(f))

        if failures:
            raise RuntimeError(f"Infra warmup failed for {len(failures)} service(s)")

        self._initialized = True
        logger.info(event="infra_warmup_completed")

    # QDRANT

    def get_vector_store(self) -> Optional[QdrantVectorStore]:
        if self._vector_store:
            return self._vector_store

        if self._is_circuit_open("qdrant"):
            return None

        try:
            self._vector_store = self._init_with_retry(
                lambda: QdrantVectorStore(),
                "qdrant",
            )
            self._reset_failure("qdrant")
            return self._vector_store

        except Exception as e:
            self._record_failure("qdrant")
            logger.error(event="qdrant_init_failed", error=str(e))
            return None

    # BM25

    def get_bm25(self) -> BM25Retriever:
        if self._bm25:
            return self._bm25

        try:
            self._bm25 = BM25Retriever()
            logger.info(event="bm25_initialized")
        except Exception as e:
            logger.error(event="bm25_init_failed", error=str(e))
            raise

        return self._bm25

    # REDIS

    def get_memory(self) -> Optional[RedisMemory]:
        if not settings.USE_REDIS:
            logger.warning(event="redis_disabled")
            return None

        if self._memory:
            return self._memory

        if self._is_circuit_open("redis"):
            return None

        try:
            self._memory = self._init_with_retry(
                lambda: RedisMemory(),
                "redis",
            )
            self._reset_failure("redis")
            return self._memory

        except Exception as e:
            self._record_failure("redis")
            logger.error(event="redis_init_failed", error=str(e))
            return None

    # MONGO

    def get_mongo(self) -> Optional[MongoMemory]:
        if self._mongo:
            return self._mongo

        if self._is_circuit_open("mongo"):
            return None

        try:
            self._mongo = self._init_with_retry(
                lambda: MongoMemory(),
                "mongo",
            )
            self._reset_failure("mongo")
            return self._mongo

        except Exception as e:
            self._record_failure("mongo")
            logger.error(event="mongo_init_failed", error=str(e))
            return None

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        return {
            "qdrant":      self._vector_store is not None,
            "redis":       self._memory is not None,
            "mongo":       self._mongo is not None,
            "bm25":        self._bm25 is not None,
            "initialized": self._initialized,
            "fail_counts": self._fail_counts,
            "circuits_open": {
                name: not self._check_circuit(name)
                for name in self._fail_counts
            },
        }

    # RESET

    def reset(self) -> None:
        self._vector_store = None
        self._bm25         = None
        self._memory       = None
        self._mongo        = None
        self._initialized  = False
        self._fail_counts  = {"qdrant": 0, "redis": 0, "mongo": 0}
        logger.warning(event="infra_reset")


# SINGLETON

infra = InfraRegistry()