import asyncio
import time
from typing import Optional, Dict, Any

from tenacity import retry, stop_after_attempt, wait_exponential

from app.vectorstore.qdrant_store import QdrantVectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.memory.redis_memory import RedisMemory
from app.memory.mongo_memory import MongoMemory
from app.core.config import settings

from app.utils.logger import get_logger

logger = get_logger(__name__)


class InfraRegistry:
    def __init__(self) -> None:
        self._vector_store: Optional[QdrantVectorStore] = None
        self._bm25: Optional[BM25Retriever] = None
        self._memory: Optional[RedisMemory] = None
        self._mongo: Optional[MongoMemory] = None

        self._initialized: bool = False

        # circuit breaker state
        self._fail_counts = {
            "qdrant": 0,
            "redis": 0,
            "mongo": 0,
        }

        self._max_failures = 3

    #  CIRCUIT BREAKER 
    def _check_circuit(self, name: str) -> bool:
        return self._fail_counts[name] < self._max_failures

    def _record_failure(self, name: str):
        self._fail_counts[name] += 1

    def _reset_failure(self, name: str):
        self._fail_counts[name] = 0

    #  RETRY WRAPPER 
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
    def _init_with_retry(self, fn, name: str):
        start = time.time()
        obj = fn()
        latency = round(time.time() - start, 2)
        logger.info(event="infra_initialized", service=name, latency=latency)
        return obj

    #  WARMUP 
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

        for r in results:
            if isinstance(r, Exception):
                logger.error(event="infra_warmup_error", error=str(r))
                raise r

        self._initialized = True
        logger.info(event="infra_warmup_completed")

    #  QDRANT 
    def get_vector_store(self) -> Optional[QdrantVectorStore]:
        if self._vector_store:
            return self._vector_store

        if not self._check_circuit("qdrant"):
            logger.warning(event="qdrant_circuit_open")
            return None

        try:
            self._vector_store = self._init_with_retry(
                lambda: QdrantVectorStore(),
                "qdrant"
            )
            self._reset_failure("qdrant")
            return self._vector_store

        except Exception as e:
            self._record_failure("qdrant")
            logger.error(event="qdrant_init_failed", error=str(e))
            return None

    #  BM25 
    def get_bm25(self) -> BM25Retriever:
        if self._bm25:
            return self._bm25

        self._bm25 = BM25Retriever()
        logger.info(event="bm25_initialized")

        return self._bm25

    #  REDIS 
    def get_memory(self) -> Optional[RedisMemory]:

        if not getattr(settings,"USE_REDIS", True):
            logger.warning(event="redis_disabled")
            return None
        
        if self._memory:
            return self._memory


        try:
            self._memory = RedisMemory()
            logger.info(event="redis_initialized")
            return self._memory
        except Exception as e:
            logger.error(event="redis_skipped", error=str(e))
            return None
        

    #  MONGO 
    def get_mongo(self) -> Optional[MongoMemory]:
        if self._mongo:
            return self._mongo

        if not self._check_circuit("mongo"):
            logger.warning(event="mongo_circuit_open")
            return None

        try:
            self._mongo = self._init_with_retry(
                lambda: MongoMemory(),
                "mongo"
            )
            self._reset_failure("mongo")
            return self._mongo

        except Exception as e:
            self._record_failure("mongo")
            logger.error(event="mongo_init_failed", error=str(e))
            return None

    #  HEALTH CHECK 
    def health_check(self) -> Dict[str, Any]:
        return {
            "qdrant": self._vector_store is not None,
            "redis": self._memory is not None,
            "mongo": self._mongo is not None,
            "bm25": self._bm25 is not None,
            "fail_counts": self._fail_counts,
        }


# SINGLETON
infra = InfraRegistry()