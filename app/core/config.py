import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(override=False)


# HELPERS

def _get_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("true", "1", "yes")


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _get_str(key: str, default: str = "") -> str:
    return os.getenv(key, default) or default


def _get_optional_str(key: str) -> Optional[str]:
    val = os.getenv(key, "").strip()
    return val if val else None


def _get_list(key: str, default: Optional[List[str]] = None) -> List[str]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default or []
    return [item.strip() for item in raw.split(",") if item.strip()]


# SETTINGS

class Settings:

    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    # CORE APPLICATION
    APP_NAME: str        = _get_str("APP_NAME", "Multimodal RAG Agentic Knowledge AI Assistant")
    APP_VERSION: str     = _get_str("APP_VERSION", "0.20.0")
    APP_DESCRIPTION: str = _get_str("APP_DESCRIPTION", "Production Multimodal RAG + Agentic AI System")
    ENV: str             = _get_str("ENV", "development")
    DEBUG: bool          = _get_bool("DEBUG", False)

    # PATHS
    DATA_DIR: Path           = Path(_get_str("DATA_DIR",           str(PROJECT_ROOT / "data")))
    LOG_DIR: Path            = Path(_get_str("LOG_DIR",            str(PROJECT_ROOT / "logs")))
    UPLOAD_STAGING_DIR: Path = Path(_get_str("UPLOAD_STAGING_DIR", str(PROJECT_ROOT / "data" / "staging")))
    VIDEO_FRAMES_DIR: Path   = Path(_get_str("VIDEO_FRAMES_DIR",   str(PROJECT_ROOT / "data" / "temp_frames")))
    PDF_IMAGE_DIR: Path      = Path(_get_str("PDF_IMAGE_DIR",      str(PROJECT_ROOT / "data" / "images")))
    TEST_FIXTURES_DIR: Path  = Path(_get_str("TEST_FIXTURES_DIR",  str(PROJECT_ROOT / "tests" / "fixtures" / "phase24")))

    # SERVER
    HOST: str = _get_str("HOST", "127.0.0.1")
    PORT: int = _get_int("PORT", 8000)

    # LOGGING
    LOG_LEVEL: str            = _get_str("LOG_LEVEL", "INFO")
    LOG_JSON: bool            = _get_bool("LOG_JSON", False)
    LOG_SHOW_TIMESTAMP: bool  = _get_bool("LOG_SHOW_TIMESTAMP", False)
    ENABLE_FILE_LOGGING: bool = _get_bool("ENABLE_FILE_LOGGING", True)
    LOG_FILE_NAME: str        = _get_str("LOG_FILE_NAME", "app.log")
    LOG_MAX_BYTES: int        = _get_int("LOG_MAX_BYTES", 10_485_760)
    LOG_BACKUP_COUNT: int     = _get_int("LOG_BACKUP_COUNT", 5)

    # PERFORMANCE
    THREAD_POOL_SIZE: int         = _get_int("THREAD_POOL_SIZE", 4)
    MAX_PARALLEL_REQUESTS: int    = _get_int("MAX_PARALLEL_REQUESTS", 20)
    REQUEST_TIMEOUT_SEC: int      = _get_int("REQUEST_TIMEOUT_SEC", 60)
    RETRIEVAL_TIMEOUT: int        = _get_int("RETRIEVAL_TIMEOUT", 10)
    EMBEDDING_TIMEOUT: int        = _get_int("EMBEDDING_TIMEOUT", 15)
    VECTOR_DB_TIMEOUT: int        = _get_int("VECTOR_DB_TIMEOUT", 10)
    MODEL_TIMEOUT_SEC: int        = _get_int("MODEL_TIMEOUT_SEC", 120)
    SLOW_REQUEST_THRESHOLD: float = _get_float("SLOW_REQUEST_THRESHOLD", 3.0)

    # LLM
    LLM_MODEL_PATH: str     = _get_str("LLM_MODEL_PATH", "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    LLM_MAX_TOKENS: int     = _get_int("LLM_MAX_TOKENS", 512)
    CONTEXT_MAX_TOKENS: int = _get_int("CONTEXT_MAX_TOKENS", 4096)
    LLM_TEMPERATURE: float  = _get_float("LLM_TEMPERATURE", 0.2)
    LLM_TOP_P: float        = _get_float("LLM_TOP_P", 0.9)
    MAX_PROMPT_CHARS: int   = _get_int("MAX_PROMPT_CHARS", 8000)
    LLM_GPU_LAYERS: int     = _get_int("LLM_GPU_LAYERS", 0)
    LLM_THREADS: int        = _get_int("LLM_THREADS", 8)
    LLM_N_BATCH: int        = _get_int("LLM_N_BATCH", 512)

    # EMBEDDINGS
    EMBEDDING_MODEL: str      = _get_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_BATCH_SIZE: int = _get_int("EMBEDDING_BATCH_SIZE", 32)
    EMBEDDING_CACHE_TTL: int  = _get_int("EMBEDDING_CACHE_TTL", 86400)
    TEXT_EMBEDDING_DIM: int   = _get_int("TEXT_EMBEDDING_DIM", 384)
    VISION_EMBEDDING_DIM: int = _get_int("VISION_EMBEDDING_DIM", 512)

    # VISION MODELS
    CLIP_MODEL: str      = _get_str("CLIP_MODEL", "openai/clip-vit-base-patch32")
    BLIP_MODEL: str      = _get_str("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
    BLIP_MAX_TOKENS: int = _get_int("BLIP_MAX_TOKENS", 64)
    BLIP_NUM_BEAMS: int  = _get_int("BLIP_NUM_BEAMS", 3)
    MAX_IMAGE_DIM: int   = _get_int("MAX_IMAGE_DIM", 1024)

    # ASR WHISPER
    WHISPER_MODEL: str            = _get_str("WHISPER_MODEL", "base")
    AUDIO_SAMPLE_RATE: int        = _get_int("AUDIO_SAMPLE_RATE", 16000)
    MAX_AUDIO_SEGMENTS: int       = _get_int("MAX_AUDIO_SEGMENTS", 500)
    AUDIO_SNR_THRESHOLD_DB: float = _get_float("AUDIO_SNR_THRESHOLD_DB", 10.0)
    AUDIO_SILENCE_GAP_MS: int     = _get_int("AUDIO_SILENCE_GAP_MS", 200)
    WHISPER_DOMAIN_VOCAB: List[str] = _get_list(
        "WHISPER_DOMAIN_VOCAB",
        [
            "RAG", "FAISS", "Qdrant", "LlamaIndex", "LangChain", "RAGAS",
            "Pinecone", "ChromaDB", "Mistral", "OpenAI", "Anthropic",
            "embedding", "retrieval", "chunking", "reranker", "BM25",
        ]
    )

    # RERANKER
    RERANKER_MODEL: str           = _get_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANK_TOP_K: int             = _get_int("RERANK_TOP_K", 5)
    RERANK_MAX_INPUT: int         = _get_int("RERANK_MAX_INPUT", 20)
    RERANK_CONTEXT_MAX_CHARS: int = _get_int("RERANK_CONTEXT_MAX_CHARS", 512)
    RERANK_MODEL_WEIGHT: float    = _get_float("RERANK_MODEL_WEIGHT", 0.5)
    RERANK_FUSION_WEIGHT: float   = _get_float("RERANK_FUSION_WEIGHT", 0.5)
    RERANK_POSITION_WEIGHT: float = _get_float("RERANK_POSITION_WEIGHT", 0.1)

    RERANK_MODALITY_WEIGHTS: Dict[str, float] = {
        "text":  1.0,
        "image": 0.95,
        "audio": 1.05,
        "video": 1.1,
    }

    # QDRANT
    QDRANT_URL: Optional[str]     = _get_optional_str("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = _get_optional_str("QDRANT_API_KEY")
    QDRANT_HOST: str              = _get_str("QDRANT_HOST", "localhost")
    QDRANT_PORT: int              = _get_int("QDRANT_PORT", 6333)
    QDRANT_TIMEOUT: int           = _get_int("QDRANT_TIMEOUT", 10)
    TEXT_COLLECTION_NAME: str     = _get_str("TEXT_COLLECTION_NAME", "text_collection")
    VISION_COLLECTION_NAME: str   = _get_str("VISION_COLLECTION_NAME", "vision_collection")
    COLLECTION_NAME: str          = _get_str("COLLECTION_NAME", "rag_collection")
    QDRANT_BATCH_SIZE: int        = _get_int("QDRANT_BATCH_SIZE", 64)
    QDRANT_MAX_DOCS: int          = _get_int("QDRANT_MAX_DOCS", 100_000)
    QDRANT_ALLOW_RECREATE: bool   = _get_bool("QDRANT_ALLOW_RECREATE", False)
    QDRANT_INIT_RETRIES: int      = _get_int("QDRANT_INIT_RETRIES", 2)
    QDRANT_RETRY_DELAY: int       = _get_int("QDRANT_RETRY_DELAY", 1)
    QDRANT_TEXT_MAX_CHARS: int    = _get_int("QDRANT_TEXT_MAX_CHARS", 2000)

    # REDIS
    REDIS_URL: Optional[str]       = _get_optional_str("REDIS_URL")
    REDIS_TOKEN: Optional[str]     = _get_optional_str("REDIS_TOKEN")
    REDIS_HOST: str                = _get_str("REDIS_HOST", "localhost")
    REDIS_PORT: int                = _get_int("REDIS_PORT", 6379)
    REDIS_DB: int                  = _get_int("REDIS_DB", 0)
    REDIS_TIMEOUT: int             = _get_int("REDIS_TIMEOUT", 5)
    REDIS_TTL_SECONDS: int         = _get_int("REDIS_TTL_SECONDS", 86400)
    REDIS_QUERY_CACHE_TTL: int     = _get_int("REDIS_QUERY_CACHE_TTL", 3600)
    REDIS_EMBEDDING_CACHE_TTL: int = _get_int("REDIS_EMBEDDING_CACHE_TTL", 86400)
    REDIS_KEY_PREFIX: str          = _get_str("REDIS_KEY_PREFIX", "rag")
    USE_REDIS: bool                = _get_bool("USE_REDIS", False)

    # MONGODB
    MONGO_URI: str        = _get_str("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str    = _get_str("MONGO_DB_NAME", "rag_memory")
    DB_TIMEOUT_MS: int    = _get_int("DB_TIMEOUT_MS", 5000)
    DB_MAX_POOL_SIZE: int = _get_int("DB_MAX_POOL_SIZE", 20)

    # INGESTION
    INGESTION_BATCH_SIZE: int = _get_int("INGESTION_BATCH_SIZE", 32)
    MAX_CHUNKS: int           = _get_int("MAX_CHUNKS", 200)

    # CHUNKING
    CHUNK_SIZE: int     = _get_int("CHUNK_SIZE", 512)
    CHUNK_OVERLAP: int  = _get_int("CHUNK_OVERLAP", 50)
    CHUNK_MIN_SIZE: int = _get_int("CHUNK_MIN_SIZE", 50)

    # BM25
    BM25_MAX_DOCS: int       = _get_int("BM25_MAX_DOCS", 10_000)
    BM25_TOP_K: int          = _get_int("BM25_TOP_K", 10)
    BM25_MIN_SCORE: float    = _get_float("BM25_MIN_SCORE", 0.1)
    BM25_MAX_TEXT_CHARS: int = _get_int("BM25_MAX_TEXT_CHARS", 1000)
    BM25_MAX_TOKENS: int     = _get_int("BM25_MAX_TOKENS", 256)

    BM25_MODALITY_WEIGHTS: Dict[str, float] = {
        "text":  1.0,
        "image": 0.9,
        "audio": 1.0,
        "video": 1.1,
    }

    # HYBRID RETRIEVAL
    HYBRID_WEIGHT_BM25: float         = _get_float("HYBRID_WEIGHT_BM25", 0.4)
    HYBRID_WEIGHT_VECTOR: float       = _get_float("HYBRID_WEIGHT_VECTOR", 0.6)
    HYBRID_WEIGHT_VISION: float       = _get_float("HYBRID_WEIGHT_VISION", 0.2)
    HYBRID_CANDIDATES_MULTIPLIER: int = _get_int("HYBRID_CANDIDATES_MULTIPLIER", 3)
    HYBRID_MIN_SCORE: float           = _get_float("HYBRID_MIN_SCORE", 0.05)

    # FUSION
    FUSION_SIMILARITY_THRESHOLD: float = _get_float("FUSION_SIMILARITY_THRESHOLD", 0.7)
    FUSION_SCORE_WEIGHT: float         = _get_float("FUSION_SCORE_WEIGHT", 0.6)
    FUSION_QUALITY_WEIGHT: float       = _get_float("FUSION_QUALITY_WEIGHT", 0.2)
    FUSION_MODALITY_WEIGHT: float      = _get_float("FUSION_MODALITY_WEIGHT", 0.2)
    FUSION_MAX_INPUT: int              = _get_int("FUSION_MAX_INPUT", 30)
    FUSION_MAX_TEXT_CHARS: int         = _get_int("FUSION_MAX_TEXT_CHARS", 1200)
    FUSION_MIN_SCORE: float            = _get_float("FUSION_MIN_SCORE", 0.05)

    FUSION_MODALITY_WEIGHTS: Dict[str, float] = {
        "text":  1.0,
        "image": 0.9,
        "audio": 1.1,
        "video": 1.15,
    }

    # RETRIEVAL
    DEFAULT_TOP_K: int     = _get_int("DEFAULT_TOP_K", 5)
    VECTOR_TOP_K: int      = _get_int("VECTOR_TOP_K", 10)
    MAX_CONTEXT_DOCS: int  = _get_int("MAX_CONTEXT_DOCS", 5)
    MAX_CONTEXT_CHARS: int = _get_int("MAX_CONTEXT_CHARS", 4000)
    RAG_TOP_K: int         = _get_int("RAG_TOP_K", 5)
    RAG_DOC_MAX_CHARS: int = _get_int("RAG_DOC_MAX_CHARS", 1200)

    # QUERY DECOMPOSITION
    MAX_SUBQUERIES: int                       = _get_int("MAX_SUBQUERIES", 3)
    SUBQUERY_MAX_TOKENS: int                  = _get_int("SUBQUERY_MAX_TOKENS", 64)
    DECOMPOSITION_MIN_WORDS: int              = _get_int("DECOMPOSITION_MIN_WORDS", 6)
    DECOMPOSITION_MAX_SUBQUERIES: int         = _get_int("DECOMPOSITION_MAX_SUBQUERIES", 3)
    DECOMPOSITION_CONFIDENCE_THRESHOLD: float = _get_float("DECOMPOSITION_CONFIDENCE_THRESHOLD", 0.5)
    DECOMPOSITION_KEYWORDS: List[str]         = [
        "compare", "difference", "vs", "process", "steps", "multiple",
    ]

    # WEB SEARCH
    TAVILY_API_KEY: str        = _get_str("TAVILY_API_KEY", "")
    WEB_MAX_DOCS: int          = _get_int("WEB_MAX_DOCS", 5)
    WEB_MAX_RESULTS: int       = _get_int("WEB_MAX_RESULTS", 10)
    WEB_DOC_MAX_CHARS: int     = _get_int("WEB_DOC_MAX_CHARS", 1000)
    WEB_CONTEXT_MAX_CHARS: int = _get_int("WEB_CONTEXT_MAX_CHARS", 4000)
    WEB_SEARCH_DEPTH: str      = _get_str("WEB_SEARCH_DEPTH", "advanced")

    # MEMORY
    MAX_HISTORY_MESSAGES: int      = _get_int("MAX_HISTORY_MESSAGES", 40)
    MAX_SYSTEM_MESSAGES: int       = _get_int("MAX_SYSTEM_MESSAGES", 5)
    MEMORY_TOP_K: int              = _get_int("MEMORY_TOP_K", 5)
    MEMORY_MAX_CONTEXT_CHARS: int  = _get_int("MEMORY_MAX_CONTEXT_CHARS", 2000)
    MEMORY_SIM_THRESHOLD: float    = _get_float("MEMORY_SIM_THRESHOLD", 0.3)
    MEMORY_RECENCY_SCALE: float    = _get_float("MEMORY_RECENCY_SCALE", 3600.0)
    MEMORY_SUMMARY_MAX_CHARS: int  = _get_int("MEMORY_SUMMARY_MAX_CHARS", 2000)
    MEMORY_SUMMARY_INPUT_CHARS: int = _get_int("MEMORY_SUMMARY_INPUT_CHARS", 4000)
    MIN_SUMMARY_LENGTH: int        = _get_int("MIN_SUMMARY_LENGTH", 50)

    MEMORY_ROLE_WEIGHTS: Dict[str, float] = {
        "user":      1.0,
        "assistant": 0.9,
        "system":    1.1,
    }

    # AGENT
    AGENT_MAX_STEPS: int                  = _get_int("AGENT_MAX_STEPS", 10)
    AGENT_TIMEOUT_SEC: int                = _get_int("AGENT_TIMEOUT_SEC", 10)
    AGENT_STEP_TIMEOUT_SEC: int           = _get_int("AGENT_STEP_TIMEOUT_SEC", 5)
    AGENT_HIGH_CONFIDENCE: float          = _get_float("AGENT_HIGH_CONFIDENCE", 0.7)
    AGENT_LOW_CONFIDENCE: float           = _get_float("AGENT_LOW_CONFIDENCE", 0.4)
    AGENT_TOOL_TIMEOUT: int               = _get_int("AGENT_TOOL_TIMEOUT", 10)
    AGENT_MAX_RETRIES: int                = _get_int("AGENT_MAX_RETRIES", 2)
    AGENT_ENABLE_DECOMPOSITION: bool      = _get_bool("AGENT_ENABLE_DECOMPOSITION", True)
    AGENT_QUERY_EXPANSION_ENABLED: bool   = _get_bool("AGENT_QUERY_EXPANSION_ENABLED", True)

    # VIDEO
    VIDEO_FRAME_INTERVAL_SEC: int = _get_int("VIDEO_FRAME_INTERVAL_SEC", 2)
    MAX_VIDEO_FRAMES: int         = _get_int("MAX_VIDEO_FRAMES", 20)
    MAX_VIDEO_DURATION_SEC: int   = _get_int("MAX_VIDEO_DURATION_SEC", 300)
    FFMPEG_PATH: str              = _get_str("FFMPEG_PATH", "ffmpeg")
    FFMPEG_TIMEOUT_SEC: int       = _get_int("FFMPEG_TIMEOUT_SEC", 120)
    SCENE_CHANGE_THRESHOLD: float = _get_float("SCENE_CHANGE_THRESHOLD", 25.0)

    # FILE SIZE LIMITS
    MAX_FILE_SIZE_TEXT: int  = _get_int("MAX_FILE_SIZE_TEXT",  10 * 1024 * 1024)
    MAX_FILE_SIZE_PDF: int   = _get_int("MAX_FILE_SIZE_PDF",  100 * 1024 * 1024)
    MAX_FILE_SIZE_DOCX: int  = _get_int("MAX_FILE_SIZE_DOCX",  50 * 1024 * 1024)
    MAX_FILE_SIZE_XLSX: int  = _get_int("MAX_FILE_SIZE_XLSX",  50 * 1024 * 1024)
    MAX_FILE_SIZE_IMAGE: int = _get_int("MAX_FILE_SIZE_IMAGE", 20 * 1024 * 1024)
    MAX_FILE_SIZE_AUDIO: int = _get_int("MAX_FILE_SIZE_AUDIO", 200 * 1024 * 1024)
    MAX_FILE_SIZE_VIDEO: int = _get_int("MAX_FILE_SIZE_VIDEO", 2 * 1024 * 1024 * 1024)
    MAX_FILE_SIZE_MB: int    = _get_int("MAX_FILE_SIZE_MB", 50)
    UPLOAD_CHUNK_SIZE: int   = _get_int("UPLOAD_CHUNK_SIZE", 1_048_576)

    @property
    def FILE_SIZE_LIMITS(self) -> Dict[str, int]:
        return {
            "text":  self.MAX_FILE_SIZE_TEXT,
            "pdf":   self.MAX_FILE_SIZE_PDF,
            "docx":  self.MAX_FILE_SIZE_DOCX,
            "xlsx":  self.MAX_FILE_SIZE_XLSX,
            "image": self.MAX_FILE_SIZE_IMAGE,
            "audio": self.MAX_FILE_SIZE_AUDIO,
            "video": self.MAX_FILE_SIZE_VIDEO,
        }

    # INPUT VALIDATION
    PREFLIGHT_MAX_MS: int             = _get_int("PREFLIGHT_MAX_MS", 50)
    WAV_HEADER_VALIDATION_MAX_MS: int = _get_int("WAV_HEADER_VALIDATION_MAX_MS", 5)
    ALLOWED_MIME_TYPES: List[str]     = _get_list("ALLOWED_MIME_TYPES", [])
    ALLOWED_FILE_TYPES: List[str]     = _get_list("ALLOWED_FILE_TYPES", [])

    # CROSS MODAL FUSION
    CROSS_MODAL_FUSION_TIMEOUT: float        = _get_float("CROSS_MODAL_FUSION_TIMEOUT", 5.0)
    CROSS_MODAL_MIN_CONFIDENCE: float        = _get_float("CROSS_MODAL_MIN_CONFIDENCE", 0.1)
    CROSS_MODAL_MAX_CHUNKS_PER_MODALITY: int = _get_int("CROSS_MODAL_MAX_CHUNKS_PER_MODALITY", 5)

    # LATENCY TARGETS
    LATENCY_TARGET_IMAGE_MS: int       = _get_int("LATENCY_TARGET_IMAGE_MS", 3000)
    LATENCY_TARGET_AUDIO_RTF: float    = _get_float("LATENCY_TARGET_AUDIO_RTF", 0.5)
    LATENCY_TARGET_PDF_MS: int         = _get_int("LATENCY_TARGET_PDF_MS", 10_000)
    LATENCY_TARGET_CROSS_MODAL_MS: int = _get_int("LATENCY_TARGET_CROSS_MODAL_MS", 5000)
    LATENCY_TARGET_PREFLIGHT_MS: int   = _get_int("LATENCY_TARGET_PREFLIGHT_MS", 50)
    LATENCY_TARGET_CACHE_HIT_MS: int   = _get_int("LATENCY_TARGET_CACHE_HIT_MS", 100)
    LATENCY_TARGET_EMBED_BATCH_MS: int = _get_int("LATENCY_TARGET_EMBED_BATCH_MS", 2000)

    # OBSERVABILITY
    LANGFUSE_PUBLIC_KEY: Optional[str] = _get_optional_str("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY: Optional[str] = _get_optional_str("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST: str                 = _get_str("LANGFUSE_HOST", "https://cloud.langfuse.com")
    LANGFUSE_ENABLED: bool             = _get_bool("LANGFUSE_ENABLED", False)
    PROMETHEUS_ENABLED: bool           = _get_bool("PROMETHEUS_ENABLED", False)
    PROMETHEUS_PORT: int               = _get_int("PROMETHEUS_PORT", 9090)

    # SECURITY
    SECRET_KEY: str                       = _get_str("SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
    CORS_ORIGINS: List[str]               = _get_list("CORS_ORIGINS", ["http://localhost:7860", "http://localhost:8000"])
    RATE_LIMIT_RPM: int                   = _get_int("RATE_LIMIT_RPM", 60)
    PII_DETECTION_ENABLED: bool           = _get_bool("PII_DETECTION_ENABLED", False)
    PROMPT_INJECTION_FILTER_ENABLED: bool = _get_bool("PROMPT_INJECTION_FILTER_ENABLED", False)

    # DSA LAYER
    LRU_CACHE_MAXSIZE: int         = _get_int("LRU_CACHE_MAXSIZE", 1000)
    PRIORITY_QUEUE_TOP_K: int      = _get_int("PRIORITY_QUEUE_TOP_K", 5)
    SLIDING_WINDOW_MAX_TOKENS: int = _get_int("SLIDING_WINDOW_MAX_TOKENS", 4096)
    INVERTED_INDEX_MAX_VOCAB: int  = _get_int("INVERTED_INDEX_MAX_VOCAB", 50_000)

    # TEST FIXTURES
    TEST_AUDIO_SAMPLE_RATE: int  = _get_int("TEST_AUDIO_SAMPLE_RATE", 44100)
    TEST_AUDIO_DURATION_SEC: int = _get_int("TEST_AUDIO_DURATION_SEC", 10)
    TEST_IMAGE_WIDTH: int        = _get_int("TEST_IMAGE_WIDTH", 800)
    TEST_IMAGE_HEIGHT: int       = _get_int("TEST_IMAGE_HEIGHT", 600)

    # DIRECTORIES
    def _create_directories(self) -> None:
        dirs = [
            self.DATA_DIR,
            self.LOG_DIR,
            self.UPLOAD_STAGING_DIR,
            self.VIDEO_FRAMES_DIR,
            self.PDF_IMAGE_DIR,
            self.TEST_FIXTURES_DIR,
        ]
        for d in dirs:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except PermissionError as exc:
                import sys
                print(f"[config] WARNING: Cannot create directory {d}: {exc}", file=sys.stderr)

    # VALIDATION
    def validate(self) -> bool:

        errors: List[str] = []

        if not self.LLM_MODEL_PATH:
            errors.append("LLM_MODEL_PATH is required")

        if self.LLM_MAX_TOKENS > self.CONTEXT_MAX_TOKENS:
            errors.append(
                f"LLM_MAX_TOKENS ({self.LLM_MAX_TOKENS}) cannot exceed "
                f"CONTEXT_MAX_TOKENS ({self.CONTEXT_MAX_TOKENS})"
            )

        if self.LLM_THREADS < 1:
            errors.append("LLM_THREADS must be >= 1")

        if self.TEXT_EMBEDDING_DIM <= 0:
            errors.append(f"TEXT_EMBEDDING_DIM must be > 0, got {self.TEXT_EMBEDDING_DIM}")

        if self.VISION_EMBEDDING_DIM <= 0:
            errors.append(f"VISION_EMBEDDING_DIM must be > 0, got {self.VISION_EMBEDDING_DIM}")

        if self.EMBEDDING_BATCH_SIZE <= 0:
            errors.append(f"EMBEDDING_BATCH_SIZE must be > 0, got {self.EMBEDDING_BATCH_SIZE}")

        if not self.RERANKER_MODEL:
            errors.append("RERANKER_MODEL is required")

        if not self.QDRANT_URL and not self.QDRANT_HOST:
            errors.append("Either QDRANT_URL or QDRANT_HOST must be set")

        if not self.MONGO_URI:
            errors.append("MONGO_URI is required")

        if not self.REDIS_URL and not self.REDIS_HOST:
            errors.append("Either REDIS_URL or REDIS_HOST must be set")

        if self.MAX_CHUNKS <= 0:
            errors.append(f"MAX_CHUNKS must be > 0, got {self.MAX_CHUNKS}")

        if self.INGESTION_BATCH_SIZE <= 0:
            errors.append(f"INGESTION_BATCH_SIZE must be > 0, got {self.INGESTION_BATCH_SIZE}")

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            errors.append(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be < CHUNK_SIZE ({self.CHUNK_SIZE})"
            )

        if self.WEB_MAX_DOCS <= 0:
            errors.append(f"WEB_MAX_DOCS must be > 0, got {self.WEB_MAX_DOCS}")

        if self.WEB_MAX_RESULTS <= 0:
            errors.append(f"WEB_MAX_RESULTS must be > 0, got {self.WEB_MAX_RESULTS}")

        if self.MEMORY_TOP_K <= 0:
            errors.append(f"MEMORY_TOP_K must be > 0, got {self.MEMORY_TOP_K}")

        if self.RAG_TOP_K <= 0:
            errors.append(f"RAG_TOP_K must be > 0, got {self.RAG_TOP_K}")

        if self.AGENT_MAX_STEPS <= 0:
            errors.append(f"AGENT_MAX_STEPS must be > 0, got {self.AGENT_MAX_STEPS}")

        fusion_sum = (
            self.FUSION_SCORE_WEIGHT
            + self.FUSION_QUALITY_WEIGHT
            + self.FUSION_MODALITY_WEIGHT
        )
        if not (0.95 <= fusion_sum <= 1.05):
            errors.append(f"FUSION weights must sum to ~1.0, got {fusion_sum:.3f}")

        if self.ENV != "development" and self.SECRET_KEY == "CHANGE_ME_IN_PRODUCTION":
            errors.append("SECRET_KEY must be changed from default in non-development environments")

        if self.CROSS_MODAL_FUSION_TIMEOUT <= 0:
            errors.append("CROSS_MODAL_FUSION_TIMEOUT must be > 0")

        if errors:
            formatted = "\n  - ".join(errors)
            raise ValueError(
                f"Config validation failed with {len(errors)} error(s):\n  - {formatted}"
            )

        return True


# SINGLETON

def _build_settings() -> Settings:
    s = Settings()
    s._create_directories()
    s.validate()
    return s


settings: Settings = _build_settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return settings