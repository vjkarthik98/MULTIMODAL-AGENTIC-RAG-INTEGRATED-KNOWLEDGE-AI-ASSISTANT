import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


def _get_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


def _get_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _get_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


class Settings:
    # CORE
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    APP_NAME: str = "Multimodal RAG Assistant"
    APP_VERSION: str = "0.21.0"
    APP_DESCRIPTION: str = "Production Multimodal RAG System"

    ENV: str = os.getenv("ENV", "production")
    DEBUG: bool = _get_bool("DEBUG", False)

    # PATHS
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", PROJECT_ROOT / "logs"))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # SERVER
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = _get_int("PORT", 8000)

    # LOGGING
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_JSON: bool = _get_bool("LOG_JSON", False)
    ENABLE_FILE_LOGGING: bool = _get_bool("ENABLE_FILE_LOGGING", True)

    LOG_FILE_NAME: str = os.getenv("LOG_FILE_NAME", "app.log")
    LOG_MAX_BYTES: int = _get_int("LOG_MAX_BYTES", 10 * 1024 * 1024)
    LOG_BACKUP_COUNT: int = _get_int("LOG_BACKUP_COUNT", 5)

    # PERFORMANCE
    THREAD_POOL_SIZE: int = _get_int("THREAD_POOL_SIZE", 4)
    MAX_PARALLEL_REQUESTS: int = _get_int("MAX_PARALLEL_REQUESTS", 50)

    REQUEST_TIMEOUT_SEC: int = _get_int("REQUEST_TIMEOUT_SEC", 30)
    RETRIEVAL_TIMEOUT: int = _get_int("RETRIEVAL_TIMEOUT", 10)

    EMBEDDING_TIMEOUT: int = _get_int("EMBEDDING_TIMEOUT", 15)
    VECTOR_DB_TIMEOUT: int = _get_int("VECTOR_DB_TIMEOUT", 10)

    MODEL_TIMEOUT_SEC: int = _get_int("MODEL_TIMEOUT_SEC", 120)
    SLOW_REQUEST_THRESHOLD: float = _get_float("SLOW_REQUEST_THRESHOLD", 2.0)

    # LLM
    LLM_MODEL_PATH: str = os.getenv(
        "LLM_MODEL_PATH",
        "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    )

    LLM_MAX_TOKENS: int = _get_int("LLM_MAX_TOKENS", 512)
    CONTEXT_MAX_TOKENS: int = _get_int("CONTEXT_MAX_TOKENS", 4096)

    LLM_TEMPERATURE: float = _get_float("LLM_TEMPERATURE", 0.2)
    LLM_TOP_P: float = _get_float("LLM_TOP_P", 0.9)

    MAX_PROMPT_CHARS: int = _get_int("MAX_PROMPT_CHARS", 8000)

    LLM_GPU_LAYERS: int = _get_int("LLM_GPU_LAYERS", 0)

    # INGESTION
    INGESTION_BATCH_SIZE: int = 32
    MAX_CHUNKS: int = 200

    # CHUNKING
    CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 512)
    CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 50)
    CHUNK_MIN_SIZE: int = _get_int("CHUNK_MIN_SIZE", 50)

    # EMBEDDINGS
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_BATCH_SIZE: int = _get_int("EMBEDDING_BATCH_SIZE", 32)
    EMBEDDING_CACHE_TTL: int = _get_int("EMBEDDING_CACHE_TTL", 86400)

    TEXT_EMBEDDING_DIM: int = _get_int("TEXT_EMBEDDING_DIM", 384)
    VISION_EMBEDDING_DIM: int = _get_int("VISION_EMBEDDING_DIM", 512)

    # RERANKER
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    

    # QDRANT
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")

    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = _get_int("QDRANT_PORT", 6333)
    QDRANT_TIMEOUT: int = _get_int("QDRANT_TIMEOUT", 10)

    TEXT_COLLECTION_NAME: str = os.getenv("TEXT_COLLECTION_NAME", "text_collection")
    VISION_COLLECTION_NAME: str = os.getenv("VISION_COLLECTION_NAME", "vision_collection")

    QDRANT_BATCH_SIZE: int = _get_int("QDRANT_BATCH_SIZE", 64)
    QDRANT_MAX_DOCS: int = _get_int("QDRANT_MAX_DOCS", 100000)

    QDRANT_ALLOW_RECREATE: bool = _get_bool("QDRANT_ALLOW_RECREATE", False)
    QDRANT_INIT_RETRIES: int = _get_int("QDRANT_INIT_RETRIES", 2)
    QDRANT_RETRY_DELAY: int = _get_int("QDRANT_RETRY_DELAY", 1)
    QDRANT_TEXT_MAX_CHARS: int = _get_int("QDRANT_TEXT_MAX_CHARS", 2000)

    # REDIS
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = _get_int("REDIS_PORT", 6379)
    REDIS_DB: int = _get_int("REDIS_DB", 0)

    REDIS_TIMEOUT: int = _get_int("REDIS_TIMEOUT", 5)
    REDIS_TTL_SECONDS: int = _get_int("REDIS_TTL_SECONDS", 86400)

   
    REDIS_KEY_PREFIX: str = os.getenv("REDIS_KEY_PREFIX", "chat")
    USE_REDIS: bool = _get_bool("USE_REDIS", False)

    # MONGO
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "rag_memory")

    DB_TIMEOUT_MS: int = _get_int("DB_TIMEOUT_MS", 5000)
    DB_MAX_POOL_SIZE: int = _get_int("DB_MAX_POOL_SIZE", 20)

    # BM25
    BM25_MAX_DOCS: int = 10000
    BM25_TOP_K: int = 10
    BM25_MIN_SCORE: float = 0.1
    BM25_MAX_TEXT_CHARS: int = _get_int("BM25_MAX_TEXT_CHARS", 1000)

    # HYBRID RETRIEVAL 
    HYBRID_WEIGHT_BM25: float = _get_float("HYBRID_WEIGHT_BM25", 0.4)
    HYBRID_WEIGHT_VECTOR: float = _get_float("HYBRID_WEIGHT_VECTOR", 0.6)
    HYBRID_WEIGHT_VISION: float = _get_float("HYBRID_WEIGHT_VISION", 0.2)
    HYBRID_CANDIDATES_MULTIPLIER: int = _get_int("HYBRID_CANDIDATES_MULTIPLIER", 3)
    HYBRID_CANDIDATES_MULTIPLIER: int = _get_int("HYBRID_CANDIDATES_MULTIPLIER", 3)
    HYBRID_MIN_SCORE: float = _get_float("HYBRID_MIN_SCORE", 0.05)

    # RERANK
    RERANK_TOP_K: int = _get_int("RERANK_TOP_K", 3)
    RERANK_MAX_INPUT: int = _get_int("RERANK_MAX_INPUT", 20)
    RERANK_CONTEXT_MAX_CHARS: int = _get_int("RERANK_CONTEXT_MAX_CHARS", 512)
    RERANK_MODEL_WEIGHT: float = _get_float("RERANK_MODEL_WEIGHT", 0.5)
    RERANK_FUSION_WEIGHT: float = _get_float("RERANK_FUSION_WEIGHT", 0.5)

    # FUSION
    FUSION_SIMILARITY_THRESHOLD: float = _get_float("FUSION_SIMILARITY_THRESHOLD", 0.7)
    FUSION_SCORE_WEIGHT: float = _get_float("FUSION_SCORE_WEIGHT", 0.6)
    FUSION_QUALITY_WEIGHT: float = _get_float("FUSION_QUALITY_WEIGHT", 0.2)
    
    FUSION_MODALITY_WEIGHT: float = _get_float("FUSION_MODALITY_WEIGHT", 0.2)
    
    # RETRIEVAL 
    VECTOR_TOP_K: int = _get_int("VECTOR_TOP_K", 10)
    BM25_TOP_K: int = _get_int("BM25_TOP_K", 10)

    MAX_CONTEXT_DOCS: int = _get_int("MAX_CONTEXT_DOCS", 5)
    MAX_CONTEXT_CHARS: int = _get_int("MAX_CONTEXT_CHARS", 4000)

    # QUERY DECOMPOSER 
    MAX_SUBQUERIES: int = _get_int("MAX_SUBQUERIES", 3)
    SUBQUERY_MAX_TOKENS: int = _get_int("SUBQUERY_MAX_TOKENS", 64)
    DECOMPOSITION_MIN_WORDS: int = _get_int("DECOMPOSITION_MIN_WORDS", 6)
    DECOMPOSITION_MAX_SUBQUERIES: int = _get_int("DECOMPOSITION_MAX_SUBQUERIES", 3)
    DECOMPOSITION_CONFIDENCE_THRESHOLD: float = _get_float("DECOMPOSITION_CONFIDENCE_THRESHOLD", 0.5)


    # WEB SEARCH
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    WEB_MAX_DOCS: int = 5
    WEB_MAX_RESULTS: int = 10
    WEB_DOC_MAX_CHARS: int = 1000
    WEB_CONTEXT_MAX_CHARS: int = 4000
    WEB_SEARCH_DEPTH: str = "advanced"

    # MEMORY
    MAX_HISTORY_MESSAGES: int = _get_int("MAX_HISTORY_MESSAGES", 40)
    MEMORY_TOP_K: int = _get_int("MEMORY_TOP_K", 5)
    MEMORY_MAX_CONTEXT_CHARS: int = _get_int("MEMORY_MAX_CONTEXT_CHARS", 2000)

    MEMORY_SIM_THRESHOLD: float = _get_float("MEMORY_SIM_THRESHOLD", 0.3)
    MEMORY_RECENCY_SCALE: float = _get_float("MEMORY_RECENCY_SCALE", 3600)

    MEMORY_ROLE_WEIGHTS: dict = {
        "user": 1.0,
        "assistant": 0.9,
        "system": 1.1
    }

    MEMORY_SUMMARY_MAX_CHARS: int = _get_int("MEMORY_SUMMARY_MAX_CHARS", 2000)
    MEMORY_SUMMARY_INPUT_CHARS: int = _get_int("MEMORY_SUMMARY_INPUT_CHARS", 4000)
    MIN_SUMMARY_LENGTH: int = _get_int("MIN_SUMMARY_LENGTH", 50)

    MAX_SYSTEM_MESSAGES: int = _get_int("MAX_SYSTEM_MESSAGES", 5)

    # RAG
    RAG_TOP_K: int = _get_int("RAG_TOP_K", 5)

    # AGENT
    AGENT_MAX_STEPS: int = _get_int("AGENT_MAX_STEPS", 10)
    AGENT_HIGH_CONFIDENCE: float = _get_float("AGENT_HIGH_CONFIDENCE", 0.7)
    AGENT_LOW_CONFIDENCE: float = _get_float("AGENT_LOW_CONFIDENCE", 0.4)
    AGENT_TOOL_TIMEOUT: int = _get_int("AGENT_TOOL_TIMEOUT", 10)
    AGENT_MAX_RETRIES: int = _get_int("AGENT_MAX_RETRIES", 2)
    AGENT_ENABLE_DECOMPOSITION: bool = _get_bool("AGENT_ENABLE_DECOMPOSITION", True)

    # UPLOAD
    MAX_FILE_SIZE_MB: int = _get_int("MAX_FILE_SIZE_MB", 50)
    UPLOAD_CHUNK_SIZE: int = _get_int("UPLOAD_CHUNK_SIZE", 1024 * 1024)

    # VALIDATION
    def validate(self) -> bool:
        errors = []

        if not self.LLM_MODEL_PATH:
            errors.append("LLM_MODEL_PATH missing")

        if not (self.QDRANT_URL or self.QDRANT_HOST):
            errors.append("QDRANT connection missing")

        if not self.MONGO_URI:
            errors.append("MONGO_URI required")

        if not self.REDIS_HOST:
            errors.append("REDIS_HOST required")

        if self.TEXT_EMBEDDING_DIM <= 0:
            errors.append("Invalid TEXT_EMBEDDING_DIM")

        if self.VISION_EMBEDDING_DIM <= 0:
            errors.append("Invalid VISION_EMBEDDING_DIM")

        if self.LLM_MAX_TOKENS > self.CONTEXT_MAX_TOKENS:
            errors.append("LLM_MAX_TOKENS cannot exceed CONTEXT_MAX_TOKENS")

        if not self.RERANKER_MODEL:
            errors.append("RERANKER_MODEL missing")

        if self.MAX_CHUNKS <= 0:
            errors.append("Invalid MAX_CHUNKS")

        if self.INGESTION_BATCH_SIZE <= 0:
            errors.append("Invalid INGESTION_BATCH_SIZE")

        if self.WEB_MAX_DOCS <= 0:
            errors.append("Invalid WEB_MAX_DOCS")

        if self.WEB_MAX_RESULTS <= 0:
            errors.append("Invalid WEB_MAX_RESULTS")

        if self.MEMORY_TOP_K <= 0:
            errors.append("Invalid MEMORY_TOP_K")

        if self.RAG_TOP_K <= 0:
            errors.append("Invalid RAG_TOP_K")

        if self.AGENT_MAX_STEPS <= 0:
            errors.append("Invalid AGENT_MAX_STEPS")

        if errors:
            raise ValueError("Config validation failed:\n" + "\n".join(errors))

        return True


settings = Settings()
settings.validate()