from app.vectorstore.qdrant_store import QdrantVectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.memory.redis_memory import RedisMemory
from app.memory.mongo_memory import MongoMemory

from app.utils.logger import get_logger

logger = get_logger(__name__)


class InfraRegistry:
    def __init__(self):
        self._vector_store = None
        self._bm25 = None
        self._memory = None
        self._mongo = None
        self._initialized = False

    
    # WARMUP
    def warmup(self):
        if self._initialized:
            return

        logger.info("[Infra] Warmup started")

        try:
            self.get_vector_store()
            self.get_bm25()
            self.get_memory()
            self.get_mongo()

            self._initialized = True

            logger.info("[Infra] Warmup completed")

        except Exception as e:
            logger.error("[Infra] Warmup failed | %s", str(e))
            raise

    
    # QDRANT
    def get_vector_store(self):
        if self._vector_store:
            return self._vector_store

        logger.info("[Infra] Connecting Qdrant...")
        self._vector_store = QdrantVectorStore()

        return self._vector_store

    
    # BM25
    def get_bm25(self):
        if self._bm25:
            return self._bm25

        logger.info("[Infra] Initializing BM25...")
        self._bm25 = BM25Retriever()

        return self._bm25

    
    # REDIS
    def get_memory(self):
        if self._memory:
            return self._memory

        logger.info("[Infra] Connecting Redis...")
        self._memory = RedisMemory()

        return self._memory
    

    # MONGO
    def get_mongo(self):
        if self._mongo:
            return self._mongo

        logger.info("[Infra] Connecting MongoDB...")
        self._mongo = MongoMemory()
        return self._mongo


infra = InfraRegistry()