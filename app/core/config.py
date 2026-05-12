from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(override=False)


# HELPERS

def _bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("true", "1", "yes")

def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default

def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default

def _str(key: str, default: str = "") -> str:
    return os.getenv(key, default) or default

def _opt(key: str) -> Optional[str]:
    val = os.getenv(key, "").strip()
    return val if val else None

def _list(key: str, default: Optional[List[str]] = None) -> List[str]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default or []
    return [item.strip() for item in raw.split(",") if item.strip()]


# SETTINGS CLASS

class Settings:

    # PROJECT ROOT
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    # CORE APPLICATION
    APP_NAME: str        = _str("APP_NAME", "Multimodal RAG Agentic Knowledge AI Assistant")
    APP_VERSION: str     = _str("APP_VERSION", "0.25.0")
    APP_DESCRIPTION: str = _str("APP_DESCRIPTION", "Production Multimodal RAG + Agentic AI System — Phase 24")
    ENV: str             = _str("ENV", "development")
    DEBUG: bool          = _bool("DEBUG", False)

    # SERVER
    HOST: str = _str("HOST", "127.0.0.1")
    PORT: int = _int("PORT", 8000)

    # PATHS
    DATA_DIR: Path           = Path(_str("DATA_DIR",           str(PROJECT_ROOT / "data")))
    LOG_DIR: Path            = Path(_str("LOG_DIR",            str(PROJECT_ROOT / "logs")))
    UPLOAD_STAGING_DIR: Path = Path(_str("UPLOAD_STAGING_DIR", str(PROJECT_ROOT / "data" / "staging")))
    VIDEO_FRAMES_DIR: Path   = Path(_str("VIDEO_FRAMES_DIR",   str(PROJECT_ROOT / "data" / "temp_frames")))
    PDF_IMAGE_DIR: Path      = Path(_str("PDF_IMAGE_DIR",      str(PROJECT_ROOT / "data" / "images")))
    TEST_FIXTURES_DIR: Path  = Path(_str("TEST_FIXTURES_DIR",  str(PROJECT_ROOT / "tests" / "fixtures" / "phase24")))
    TEMP_DIR: Path           = Path(_str("TEMP_DIR",           str(PROJECT_ROOT / "data" / "temp")))
    AUDIT_LOG_PATH: Path     = Path(_str("AUDIT_LOG_PATH",     str(PROJECT_ROOT / "logs" / "audit.log")))
    CHROOT_BASE: Path        = Path(_str("CHROOT_BASE",        str(PROJECT_ROOT / "data")))

    # LOGGING
    LOG_LEVEL: str             = _str("LOG_LEVEL", "INFO")
    LOG_JSON: bool             = _bool("LOG_JSON", False)
    LOG_SHOW_TIMESTAMP: bool   = _bool("LOG_SHOW_TIMESTAMP", True)
    ENABLE_FILE_LOGGING: bool  = _bool("ENABLE_FILE_LOGGING", True)
    LOG_FILE_NAME: str         = _str("LOG_FILE_NAME", "app.log")
    LOG_MAX_BYTES: int         = _int("LOG_MAX_BYTES", 10_485_760)
    LOG_BACKUP_COUNT: int      = _int("LOG_BACKUP_COUNT", 5)
    CORRELATION_ID_HEADER: str = _str("CORRELATION_ID_HEADER", "X-Correlation-ID")

    # OPENTELEMETRY
    OTEL_ENABLED: bool                  = _bool("OTEL_ENABLED", False)
    OTEL_EXPORTER_OTLP_ENDPOINT: str    = _str("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    OTEL_SERVICE_NAME: str              = _str("OTEL_SERVICE_NAME", "multimodal-rag-assistant")
    OTEL_SAMPLING_RATIO: float          = _float("OTEL_SAMPLING_RATIO", 1.0)

    # PROMETHEUS
    PROMETHEUS_ENABLED: bool      = _bool("PROMETHEUS_ENABLED", False)
    PROMETHEUS_PORT: int          = _int("PROMETHEUS_PORT", 9090)
    PROMETHEUS_METRICS_PATH: str  = _str("PROMETHEUS_METRICS_PATH", "/metrics")

    # PERFORMANCE
    THREAD_POOL_SIZE: int           = _int("THREAD_POOL_SIZE", 4)
    ASYNC_SEMAPHORE_WORKERS: int    = _int("ASYNC_SEMAPHORE_WORKERS", 5)
    MAX_PARALLEL_REQUESTS: int      = _int("MAX_PARALLEL_REQUESTS", 20)
    REQUEST_TIMEOUT_SEC: int        = _int("REQUEST_TIMEOUT_SEC", 120)
    FILE_PROCESSING_TIMEOUT_SEC: int = _int("FILE_PROCESSING_TIMEOUT_SEC", 120)
    LLM_CALL_TIMEOUT_SEC: int       = _int("LLM_CALL_TIMEOUT_SEC", 30)
    RETRIEVAL_TIMEOUT: int          = _int("RETRIEVAL_TIMEOUT", 10)
    EMBEDDING_TIMEOUT: int          = _int("EMBEDDING_TIMEOUT", 15)
    VECTOR_DB_TIMEOUT: int          = _int("VECTOR_DB_TIMEOUT", 10)
    MODEL_TIMEOUT_SEC: int          = _int("MODEL_TIMEOUT_SEC", 120)
    SLOW_REQUEST_THRESHOLD: float   = _float("SLOW_REQUEST_THRESHOLD", 3.0)
    INGESTION_WORKER_COUNT: int     = _int("INGESTION_WORKER_COUNT", 3)

    # LLM
    LLM_MODEL_PATH: str      = _str("LLM_MODEL_PATH", "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    LLM_MAX_TOKENS: int      = _int("LLM_MAX_TOKENS", 512)
    CONTEXT_MAX_TOKENS: int  = _int("CONTEXT_MAX_TOKENS", 4096)
    LLM_TEMPERATURE: float   = _float("LLM_TEMPERATURE", 0.2)
    LLM_TOP_P: float         = _float("LLM_TOP_P", 0.9)
    MAX_PROMPT_CHARS: int    = _int("MAX_PROMPT_CHARS", 8000)
    LLM_GPU_LAYERS: int      = _int("LLM_GPU_LAYERS", 0)
    LLM_THREADS: int         = _int("LLM_THREADS", 8)
    LLM_N_BATCH: int         = _int("LLM_N_BATCH", 512)
    LLM_MAX_RETRIES: int     = _int("LLM_MAX_RETRIES", 3)
    LLM_RETRY_WAIT_MIN: int  = _int("LLM_RETRY_WAIT_MIN", 1)
    LLM_RETRY_WAIT_MAX: int  = _int("LLM_RETRY_WAIT_MAX", 10)

    # EMBEDDINGS
    EMBEDDING_MODEL: str                = _str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MULTILINGUAL_EMBEDDING_MODEL: str   = _str("MULTILINGUAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    EMBEDDING_BATCH_SIZE: int           = _int("EMBEDDING_BATCH_SIZE", 32)
    EMBEDDING_MAX_BATCH_SIZE: int       = _int("EMBEDDING_MAX_BATCH_SIZE", 100)
    EMBEDDING_CACHE_TTL: int            = _int("EMBEDDING_CACHE_TTL", 2_592_000)
    TEXT_EMBEDDING_DIM: int             = _int("TEXT_EMBEDDING_DIM", 384)
    MATRYOSHKA_SHORT_DIM: int           = _int("MATRYOSHKA_SHORT_DIM", 256)
    MATRYOSHKA_LONG_DIM: int            = _int("MATRYOSHKA_LONG_DIM", 1536)

    # VISION MODELS
    CLIP_MODEL: str                  = _str("CLIP_MODEL", "openai/clip-vit-base-patch32")
    VISION_EMBEDDING_DIM: int        = _int("VISION_EMBEDDING_DIM", 512)
    BLIP_MODEL: str                  = _str("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
    BLIP_MAX_TOKENS: int             = _int("BLIP_MAX_TOKENS", 64)
    BLIP_NUM_BEAMS: int              = _int("BLIP_NUM_BEAMS", 3)
    MAX_IMAGE_DIM: int               = _int("MAX_IMAGE_DIM", 1024)
    MAX_IMAGE_SIZE_MP: int           = _int("MAX_IMAGE_SIZE_MP", 50)
    THUMBNAIL_WIDTH: int             = _int("THUMBNAIL_WIDTH", 256)
    THUMBNAIL_HEIGHT: int            = _int("THUMBNAIL_HEIGHT", 256)
    IMAGE_PRIVACY_MODE: bool         = _bool("IMAGE_PRIVACY_MODE", False)
    SOLID_COLOR_THRESHOLD: float     = _float("SOLID_COLOR_THRESHOLD", 0.95)
    CLIP_MAX_LENGTH: int             = _int("CLIP_MAX_LENGTH", 77)

    # ASR WHISPER
    WHISPER_MODEL: str              = _str("WHISPER_MODEL", "base")
    AUDIO_SAMPLE_RATE: int          = _int("AUDIO_SAMPLE_RATE", 16000)
    MAX_AUDIO_DURATION_SEC: int     = _int("MAX_AUDIO_DURATION_SEC", 10800)
    MAX_AUDIO_SEGMENTS: int         = _int("MAX_AUDIO_SEGMENTS", 500)
    AUDIO_SNR_THRESHOLD_DB: float   = _float("AUDIO_SNR_THRESHOLD_DB", 10.0)
    AUDIO_SILENCE_GAP_MS: int       = _int("AUDIO_SILENCE_GAP_MS", 200)
    AUDIO_CHUNK_DURATION_SEC: int   = _int("AUDIO_CHUNK_DURATION_SEC", 1800)
    DIARIZATION_ENABLED: bool       = _bool("DIARIZATION_ENABLED", False)
    HF_TOKEN: Optional[str]         = _opt("HF_TOKEN")
    WHISPER_DOMAIN_VOCAB: List[str] = _list("WHISPER_DOMAIN_VOCAB", [
        "RAG", "FAISS", "Qdrant", "LlamaIndex", "LangChain", "RAGAS",
        "Pinecone", "ChromaDB", "Mistral", "OpenAI", "Anthropic",
        "embedding", "retrieval", "chunking", "reranker", "BM25",
    ])
    WHISPER_FILLER_WORDS: List[str] = _list("WHISPER_FILLER_WORDS", ["uh", "um", "er", "hmm"])

    # RERANKER
    RERANKER_MODEL: str            = _str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANK_TOP_K: int              = _int("RERANK_TOP_K", 5)
    RERANK_MAX_INPUT: int          = _int("RERANK_MAX_INPUT", 20)
    RERANK_CONTEXT_MAX_CHARS: int  = _int("RERANK_CONTEXT_MAX_CHARS", 512)
    RERANK_MODEL_WEIGHT: float     = _float("RERANK_MODEL_WEIGHT", 0.5)
    RERANK_FUSION_WEIGHT: float    = _float("RERANK_FUSION_WEIGHT", 0.5)
    RERANK_POSITION_WEIGHT: float  = _float("RERANK_POSITION_WEIGHT", 0.1)
    COHERE_API_KEY: Optional[str]  = _opt("COHERE_API_KEY")

    RERANK_MODALITY_WEIGHTS: Dict[str, float] = {
        "text":  1.0,
        "image": 0.95,
        "audio": 1.05,
        "video": 1.1,
    }

    # QDRANT
    QDRANT_URL: Optional[str]       = _opt("QDRANT_URL")
    QDRANT_API_KEY: Optional[str]   = _opt("QDRANT_API_KEY")
    QDRANT_HOST: str                = _str("QDRANT_HOST", "localhost")
    QDRANT_PORT: int                = _int("QDRANT_PORT", 6333)
    QDRANT_TIMEOUT: int             = _int("QDRANT_TIMEOUT", 10)
    TEXT_COLLECTION_NAME: str       = _str("TEXT_COLLECTION_NAME", "text_collection")
    VISION_COLLECTION_NAME: str     = _str("VISION_COLLECTION_NAME", "vision_collection")
    COLLECTION_NAME: str            = _str("COLLECTION_NAME", "rag_collection")
    QDRANT_BATCH_SIZE: int          = _int("QDRANT_BATCH_SIZE", 100)
    QDRANT_MAX_DOCS: int            = _int("QDRANT_MAX_DOCS", 100_000)
    QDRANT_ALLOW_RECREATE: bool     = _bool("QDRANT_ALLOW_RECREATE", False)
    QDRANT_INIT_RETRIES: int        = _int("QDRANT_INIT_RETRIES", 3)
    QDRANT_RETRY_DELAY: int         = _int("QDRANT_RETRY_DELAY", 2)
    QDRANT_TEXT_MAX_CHARS: int      = _int("QDRANT_TEXT_MAX_CHARS", 2000)
    QDRANT_SOFT_DELETE: bool        = _bool("QDRANT_SOFT_DELETE", True)
    QDRANT_CB_FAIL_MAX: int         = _int("QDRANT_CB_FAIL_MAX", 5)
    QDRANT_CB_RESET_TIMEOUT: int    = _int("QDRANT_CB_RESET_TIMEOUT", 60)

    # REDIS
    REDIS_URL: Optional[str]         = _opt("REDIS_URL")
    REDIS_TOKEN: Optional[str]       = _opt("REDIS_TOKEN")
    REDIS_HOST: str                  = _str("REDIS_HOST", "localhost")
    REDIS_PORT: int                  = _int("REDIS_PORT", 6379)
    REDIS_DB: int                    = _int("REDIS_DB", 0)
    REDIS_TIMEOUT: int               = _int("REDIS_TIMEOUT", 5)
    REDIS_TTL_SECONDS: int           = _int("REDIS_TTL_SECONDS", 86400)
    REDIS_QUERY_CACHE_TTL: int       = _int("REDIS_QUERY_CACHE_TTL", 3600)
    REDIS_EMBEDDING_CACHE_TTL: int   = _int("REDIS_EMBEDDING_CACHE_TTL", 2_592_000)
    REDIS_KEY_PREFIX: str            = _str("REDIS_KEY_PREFIX", "rag")
    USE_REDIS: bool                  = _bool("USE_REDIS", False)
    REDIS_CB_FAIL_MAX: int           = _int("REDIS_CB_FAIL_MAX", 5)
    REDIS_CB_RESET_TIMEOUT: int      = _int("REDIS_CB_RESET_TIMEOUT", 30)
    REDIS_MAX_WINDOW_TURNS: int      = _int("REDIS_MAX_WINDOW_TURNS", 40)

    # MONGODB
    MONGO_URI: str                       = _str("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str                   = _str("MONGO_DB_NAME", "rag_memory")
    DB_TIMEOUT_MS: int                   = _int("DB_TIMEOUT_MS", 5000)
    DB_MAX_POOL_SIZE: int                = _int("DB_MAX_POOL_SIZE", 20)
    MONGO_CB_FAIL_MAX: int               = _int("MONGO_CB_FAIL_MAX", 5)
    MONGO_CB_RESET_TIMEOUT: int          = _int("MONGO_CB_RESET_TIMEOUT", 60)
    MONGO_MESSAGES_COLLECTION: str       = _str("MONGO_MESSAGES_COLLECTION", "messages")
    MONGO_SUMMARIES_COLLECTION: str      = _str("MONGO_SUMMARIES_COLLECTION", "summaries")
    MONGO_AUDIT_COLLECTION: str          = _str("MONGO_AUDIT_COLLECTION", "audit_log")

    # CHUNKING
    CHUNK_SIZE: int                  = _int("CHUNK_SIZE", 512)
    CHUNK_OVERLAP: int               = _int("CHUNK_OVERLAP", 50)
    CHUNK_OVERLAP_RATIO: float       = _float("CHUNK_OVERLAP_RATIO", 0.15)
    CHUNK_MIN_SIZE: int              = _int("CHUNK_MIN_SIZE", 50)
    MAX_CHUNKS: int                  = _int("MAX_CHUNKS", 200)
    INGESTION_BATCH_SIZE: int        = _int("INGESTION_BATCH_SIZE", 32)
    CHUNK_MAX_TOKENS: int            = _int("CHUNK_MAX_TOKENS", 512)
    CHUNK_MINHASH_THRESHOLD: float   = _float("CHUNK_MINHASH_THRESHOLD", 0.85)
    CHUNK_MINHASH_PERMUTATIONS: int  = _int("CHUNK_MINHASH_PERMUTATIONS", 128)
    CHUNK_ADAPTIVE_ENABLED: bool     = _bool("CHUNK_ADAPTIVE_ENABLED", True)
    LATE_CHUNKING_ENABLED: bool      = _bool("LATE_CHUNKING_ENABLED", False)

    # BM25
    BM25_MAX_DOCS: int        = _int("BM25_MAX_DOCS", 10_000)
    BM25_TOP_K: int           = _int("BM25_TOP_K", 10)
    BM25_MIN_SCORE: float     = _float("BM25_MIN_SCORE", 0.1)
    BM25_MAX_TEXT_CHARS: int  = _int("BM25_MAX_TEXT_CHARS", 1000)
    BM25_MAX_TOKENS: int      = _int("BM25_MAX_TOKENS", 256)

    BM25_MODALITY_WEIGHTS: Dict[str, float] = {
        "text":  1.0,
        "image": 0.9,
        "audio": 1.0,
        "video": 1.1,
    }

    # HYBRID RETRIEVAL
    HYBRID_WEIGHT_BM25: float          = _float("HYBRID_WEIGHT_BM25", 0.4)
    HYBRID_WEIGHT_VECTOR: float        = _float("HYBRID_WEIGHT_VECTOR", 0.6)
    HYBRID_WEIGHT_VISION: float        = _float("HYBRID_WEIGHT_VISION", 0.2)
    HYBRID_CANDIDATES_MULTIPLIER: int  = _int("HYBRID_CANDIDATES_MULTIPLIER", 3)
    HYBRID_MIN_SCORE: float            = _float("HYBRID_MIN_SCORE", 0.05)
    HYBRID_RRF_K: int                  = _int("HYBRID_RRF_K", 60)
    HYBRID_MMR_LAMBDA: float           = _float("HYBRID_MMR_LAMBDA", 0.7)
    HYBRID_SCORE_THRESHOLD: float      = _float("HYBRID_SCORE_THRESHOLD", 0.1)

    # FUSION
    FUSION_SIMILARITY_THRESHOLD: float  = _float("FUSION_SIMILARITY_THRESHOLD", 0.7)
    FUSION_SCORE_WEIGHT: float          = _float("FUSION_SCORE_WEIGHT", 0.6)
    FUSION_QUALITY_WEIGHT: float        = _float("FUSION_QUALITY_WEIGHT", 0.2)
    FUSION_MODALITY_WEIGHT: float       = _float("FUSION_MODALITY_WEIGHT", 0.2)
    FUSION_MAX_INPUT: int               = _int("FUSION_MAX_INPUT", 30)
    FUSION_MAX_TEXT_CHARS: int          = _int("FUSION_MAX_TEXT_CHARS", 1200)
    FUSION_MIN_SCORE: float             = _float("FUSION_MIN_SCORE", 0.05)

    FUSION_MODALITY_WEIGHTS: Dict[str, float] = {
        "text":  1.0,
        "image": 0.9,
        "audio": 1.1,
        "video": 1.15,
    }

    # RETRIEVAL
    DEFAULT_TOP_K: int      = _int("DEFAULT_TOP_K", 5)
    VECTOR_TOP_K: int       = _int("VECTOR_TOP_K", 10)
    MAX_CONTEXT_DOCS: int   = _int("MAX_CONTEXT_DOCS", 5)
    MAX_CONTEXT_CHARS: int  = _int("MAX_CONTEXT_CHARS", 4000)
    RAG_TOP_K: int          = _int("RAG_TOP_K", 5)
    RAG_DOC_MAX_CHARS: int  = _int("RAG_DOC_MAX_CHARS", 1200)

    # QUERY DECOMPOSITION
    MAX_SUBQUERIES: int                        = _int("MAX_SUBQUERIES", 3)
    SUBQUERY_MAX_TOKENS: int                   = _int("SUBQUERY_MAX_TOKENS", 64)
    DECOMPOSITION_MIN_WORDS: int               = _int("DECOMPOSITION_MIN_WORDS", 6)
    DECOMPOSITION_MAX_SUBQUERIES: int          = _int("DECOMPOSITION_MAX_SUBQUERIES", 3)
    DECOMPOSITION_CONFIDENCE_THRESHOLD: float  = _float("DECOMPOSITION_CONFIDENCE_THRESHOLD", 0.5)
    DECOMPOSITION_KEYWORDS: List[str]          = [
        "compare", "difference", "vs", "process", "steps", "multiple",
    ]

    # WEB SEARCH
    TAVILY_API_KEY: str                    = _str("TAVILY_API_KEY", "")
    SERPAPI_KEY: Optional[str]             = _opt("SERPAPI_KEY")
    WEB_MAX_DOCS: int                      = _int("WEB_MAX_DOCS", 5)
    WEB_MAX_RESULTS: int                   = _int("WEB_MAX_RESULTS", 10)
    WEB_DOC_MAX_CHARS: int                 = _int("WEB_DOC_MAX_CHARS", 1000)
    WEB_CONTEXT_MAX_CHARS: int             = _int("WEB_CONTEXT_MAX_CHARS", 4000)
    WEB_SEARCH_DEPTH: str                  = _str("WEB_SEARCH_DEPTH", "advanced")
    WEB_SEARCH_BLOCK_PRIVATE_IPS: bool     = _bool("WEB_SEARCH_BLOCK_PRIVATE_IPS", True)
    WEB_SEARCH_DOMAIN_ALLOWLIST: List[str] = _list("WEB_SEARCH_DOMAIN_ALLOWLIST", [])

    # MEMORY
    MAX_HISTORY_MESSAGES: int        = _int("MAX_HISTORY_MESSAGES", 40)
    MAX_SYSTEM_MESSAGES: int         = _int("MAX_SYSTEM_MESSAGES", 5)
    MEMORY_TOP_K: int                = _int("MEMORY_TOP_K", 5)
    MEMORY_MAX_CONTEXT_CHARS: int    = _int("MEMORY_MAX_CONTEXT_CHARS", 2000)
    MEMORY_SIM_THRESHOLD: float      = _float("MEMORY_SIM_THRESHOLD", 0.3)
    MEMORY_RECENCY_SCALE: float      = _float("MEMORY_RECENCY_SCALE", 3600.0)
    MEMORY_SUMMARY_MAX_CHARS: int    = _int("MEMORY_SUMMARY_MAX_CHARS", 2000)
    MEMORY_SUMMARY_INPUT_CHARS: int  = _int("MEMORY_SUMMARY_INPUT_CHARS", 4000)
    MIN_SUMMARY_LENGTH: int          = _int("MIN_SUMMARY_LENGTH", 50)
    SLIDING_WINDOW_MAX_TOKENS: int   = _int("SLIDING_WINDOW_MAX_TOKENS", 4096)

    MEMORY_ROLE_WEIGHTS: Dict[str, float] = {
        "user":      1.0,
        "assistant": 0.9,
        "system":    1.1,
    }

    # AGENT
    AGENT_MAX_STEPS: int                 = _int("AGENT_MAX_STEPS", 10)
    AGENT_TIMEOUT_SEC: int               = _int("AGENT_TIMEOUT_SEC", 60)
    AGENT_STEP_TIMEOUT_SEC: int          = _int("AGENT_STEP_TIMEOUT_SEC", 30)
    AGENT_HIGH_CONFIDENCE: float         = _float("AGENT_HIGH_CONFIDENCE", 0.7)
    AGENT_LOW_CONFIDENCE: float          = _float("AGENT_LOW_CONFIDENCE", 0.4)
    AGENT_TOOL_TIMEOUT: int              = _int("AGENT_TOOL_TIMEOUT", 30)
    AGENT_MAX_RETRIES: int               = _int("AGENT_MAX_RETRIES", 3)
    AGENT_ENABLE_DECOMPOSITION: bool     = _bool("AGENT_ENABLE_DECOMPOSITION", True)
    AGENT_QUERY_EXPANSION_ENABLED: bool  = _bool("AGENT_QUERY_EXPANSION_ENABLED", True)
    AGENT_MAX_ITERATIONS: int            = _int("AGENT_MAX_ITERATIONS", 20)
    AGENT_PARALLEL_TOOLS: bool           = _bool("AGENT_PARALLEL_TOOLS", True)

    # VIDEO
    VIDEO_FRAME_INTERVAL_SEC: int            = _int("VIDEO_FRAME_INTERVAL_SEC", 2)
    MAX_VIDEO_FRAMES: int                    = _int("MAX_VIDEO_FRAMES", 20)
    MAX_VIDEO_DURATION_SEC: int              = _int("MAX_VIDEO_DURATION_SEC", 7200)
    FFMPEG_PATH: str                         = _str("FFMPEG_PATH", "ffmpeg")
    FFMPEG_TIMEOUT_SEC: int                  = _int("FFMPEG_TIMEOUT_SEC", 120)
    SCENE_CHANGE_THRESHOLD: float            = _float("SCENE_CHANGE_THRESHOLD", 25.0)
    VIDEO_DUPLICATE_FRAME_THRESHOLD: float   = _float("VIDEO_DUPLICATE_FRAME_THRESHOLD", 0.98)
    VIDEO_SUBTITLE_EXTRACTION: bool          = _bool("VIDEO_SUBTITLE_EXTRACTION", True)
    VIDEO_HDR_TONEMAPPING: bool              = _bool("VIDEO_HDR_TONEMAPPING", True)
    VIDEO_DEINTERLACE: bool                  = _bool("VIDEO_DEINTERLACE", False)
    VIDEO_THUMBNAIL_GRID_ROWS: int           = _int("VIDEO_THUMBNAIL_GRID_ROWS", 3)
    VIDEO_THUMBNAIL_GRID_COLS: int           = _int("VIDEO_THUMBNAIL_GRID_COLS", 3)
    VIDEO_STREAM_THRESHOLD_BYTES: int        = _int("VIDEO_STREAM_THRESHOLD_BYTES", 10_737_418_240)

    # FILE SIZE LIMITS
    MAX_FILE_SIZE_TEXT: int   = _int("MAX_FILE_SIZE_TEXT",  10 * 1024 * 1024)
    MAX_FILE_SIZE_PDF: int    = _int("MAX_FILE_SIZE_PDF",  100 * 1024 * 1024)
    MAX_FILE_SIZE_DOCX: int   = _int("MAX_FILE_SIZE_DOCX",  50 * 1024 * 1024)
    MAX_FILE_SIZE_XLSX: int   = _int("MAX_FILE_SIZE_XLSX",  50 * 1024 * 1024)
    MAX_FILE_SIZE_IMAGE: int  = _int("MAX_FILE_SIZE_IMAGE", 20 * 1024 * 1024)
    MAX_FILE_SIZE_AUDIO: int  = _int("MAX_FILE_SIZE_AUDIO", 200 * 1024 * 1024)
    MAX_FILE_SIZE_VIDEO: int  = _int("MAX_FILE_SIZE_VIDEO", 2 * 1024 * 1024 * 1024)
    MAX_FILE_SIZE_MB: int     = _int("MAX_FILE_SIZE_MB", 500)
    UPLOAD_CHUNK_SIZE: int    = _int("UPLOAD_CHUNK_SIZE", 1_048_576)

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
    PREFLIGHT_MAX_MS: int              = _int("PREFLIGHT_MAX_MS", 50)
    WAV_HEADER_VALIDATION_MAX_MS: int  = _int("WAV_HEADER_VALIDATION_MAX_MS", 5)
    ALLOWED_MIME_TYPES: List[str]      = _list("ALLOWED_MIME_TYPES", [])
    ALLOWED_FILE_TYPES: List[str]      = _list("ALLOWED_FILE_TYPES", [])
    MIN_FREE_DISK_MB: int              = _int("MIN_FREE_DISK_MB", 500)
    STRIP_NULL_BYTES: bool             = _bool("STRIP_NULL_BYTES", True)
    STRIP_BOM: bool                    = _bool("STRIP_BOM", True)
    MAGIC_BYTE_MIME_DETECTION: bool    = _bool("MAGIC_BYTE_MIME_DETECTION", True)

    # DEDUP AND SECURITY
    DEDUP_ENABLED: bool                   = _bool("DEDUP_ENABLED", True)
    CLAMAV_ENABLED: bool                  = _bool("CLAMAV_ENABLED", False)
    CLAMAV_HOST: str                      = _str("CLAMAV_HOST", "localhost")
    CLAMAV_PORT: int                      = _int("CLAMAV_PORT", 3310)
    TEMP_FILE_ENCRYPTION: bool            = _bool("TEMP_FILE_ENCRYPTION", False)

    # CROSS-MODAL FUSION
    CROSS_MODAL_FUSION_TIMEOUT: float         = _float("CROSS_MODAL_FUSION_TIMEOUT", 5.0)
    CROSS_MODAL_MIN_CONFIDENCE: float         = _float("CROSS_MODAL_MIN_CONFIDENCE", 0.1)
    CROSS_MODAL_MAX_CHUNKS_PER_MODALITY: int  = _int("CROSS_MODAL_MAX_CHUNKS_PER_MODALITY", 5)

    # LATENCY TARGETS
    LATENCY_TARGET_IMAGE_MS: int        = _int("LATENCY_TARGET_IMAGE_MS", 3000)
    LATENCY_TARGET_AUDIO_RTF: float     = _float("LATENCY_TARGET_AUDIO_RTF", 0.5)
    LATENCY_TARGET_PDF_MS: int          = _int("LATENCY_TARGET_PDF_MS", 10_000)
    LATENCY_TARGET_CROSS_MODAL_MS: int  = _int("LATENCY_TARGET_CROSS_MODAL_MS", 5000)
    LATENCY_TARGET_PREFLIGHT_MS: int    = _int("LATENCY_TARGET_PREFLIGHT_MS", 50)
    LATENCY_TARGET_CACHE_HIT_MS: int    = _int("LATENCY_TARGET_CACHE_HIT_MS", 100)
    LATENCY_TARGET_EMBED_BATCH_MS: int  = _int("LATENCY_TARGET_EMBED_BATCH_MS", 2000)

    # SLO TARGETS
    SLO_TEXT_P95_MS: int    = _int("SLO_TEXT_P95_MS", 2000)
    SLO_PDF_P95_MS: int     = _int("SLO_PDF_P95_MS", 30000)
    SLO_WORD_P95_MS: int    = _int("SLO_WORD_P95_MS", 15000)
    SLO_EXCEL_P95_MS: int   = _int("SLO_EXCEL_P95_MS", 45000)
    SLO_IMAGE_P95_MS: int   = _int("SLO_IMAGE_P95_MS", 10000)
    SLO_AUDIO_P95_MS: int   = _int("SLO_AUDIO_P95_MS", 300000)
    SLO_VIDEO_P95_MS: int   = _int("SLO_VIDEO_P95_MS", 900000)

    # OBSERVABILITY
    LANGFUSE_PUBLIC_KEY: Optional[str]  = _opt("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY: Optional[str]  = _opt("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST: str                  = _str("LANGFUSE_HOST", "https://cloud.langfuse.com")
    LANGFUSE_ENABLED: bool              = _bool("LANGFUSE_ENABLED", False)

    # SECURITY
    SECRET_KEY: str                        = _str("SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
    CORS_ORIGINS: List[str]                = _list("CORS_ORIGINS", ["http://localhost:7860", "http://localhost:8000"])
    RATE_LIMIT_RPM: int                    = _int("RATE_LIMIT_RPM", 60)
    PII_DETECTION_ENABLED: bool            = _bool("PII_DETECTION_ENABLED", False)
    PII_ENTITIES: List[str]                = _list("PII_ENTITIES", ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"])
    PII_REDACTION_OPERATOR: str            = _str("PII_REDACTION_OPERATOR", "replace")
    PROMPT_INJECTION_FILTER_ENABLED: bool  = _bool("PROMPT_INJECTION_FILTER_ENABLED", False)
    AUDIT_LOG_ENABLED: bool                = _bool("AUDIT_LOG_ENABLED", True)
    SSRF_BLOCK_PRIVATE_IPS: bool           = _bool("SSRF_BLOCK_PRIVATE_IPS", True)

    # DOCUMENT PROCESSING
    PDF_OCR_TEXT_DENSITY_THRESHOLD: float  = _float("PDF_OCR_TEXT_DENSITY_THRESHOLD", 0.1)
    LIBREOFFICE_ENABLED: bool              = _bool("LIBREOFFICE_ENABLED", True)
    LIBREOFFICE_PATH: str                  = _str("LIBREOFFICE_PATH", "libreoffice")
    LIBREOFFICE_TIMEOUT_SEC: int           = _int("LIBREOFFICE_TIMEOUT_SEC", 60)
    PDF_SIGNATURE_DETECTION: bool          = _bool("PDF_SIGNATURE_DETECTION", True)
    PDF_STRIP_JAVASCRIPT: bool             = _bool("PDF_STRIP_JAVASCRIPT", True)
    PDF_FORM_EXTRACTION: bool              = _bool("PDF_FORM_EXTRACTION", True)
    WORD_TRACK_CHANGES: bool               = _bool("WORD_TRACK_CHANGES", True)
    WORD_MACRO_DETECTION: bool             = _bool("WORD_MACRO_DETECTION", True)

    # EXCEL PROCESSING
    EXCEL_STREAM_READ_ONLY: bool  = _bool("EXCEL_STREAM_READ_ONLY", True)
    EXCEL_MAX_ROWS: int           = _int("EXCEL_MAX_ROWS", 1_000_000)

    # KEYWORD EXTRACTION
    KEYWORD_EXTRACTOR: str       = _str("KEYWORD_EXTRACTOR", "yake")
    KEYWORD_MAX_PER_CHUNK: int   = _int("KEYWORD_MAX_PER_CHUNK", 10)
    KEYBERT_MODEL: str           = _str("KEYBERT_MODEL", "all-MiniLM-L6-v2")

    # DSA LAYER
    LRU_CACHE_MAXSIZE: int         = _int("LRU_CACHE_MAXSIZE", 1000)
    PRIORITY_QUEUE_TOP_K: int      = _int("PRIORITY_QUEUE_TOP_K", 5)
    INVERTED_INDEX_MAX_VOCAB: int  = _int("INVERTED_INDEX_MAX_VOCAB", 50_000)

    # TEST FIXTURES
    TEST_AUDIO_SAMPLE_RATE: int   = _int("TEST_AUDIO_SAMPLE_RATE", 44100)
    TEST_AUDIO_DURATION_SEC: int  = _int("TEST_AUDIO_DURATION_SEC", 10)
    TEST_IMAGE_WIDTH: int         = _int("TEST_IMAGE_WIDTH", 800)
    TEST_IMAGE_HEIGHT: int        = _int("TEST_IMAGE_HEIGHT", 600)

    # DIRECTORY CREATION
    def _create_directories(self) -> None:
        dirs = [
            self.DATA_DIR,
            self.LOG_DIR,
            self.UPLOAD_STAGING_DIR,
            self.VIDEO_FRAMES_DIR,
            self.PDF_IMAGE_DIR,
            self.TEST_FIXTURES_DIR,
            self.TEMP_DIR,
            self.AUDIT_LOG_PATH.parent,
        ]
        for d in dirs:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except PermissionError as exc:
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

        if self.CHUNK_MINHASH_THRESHOLD < 0.0 or self.CHUNK_MINHASH_THRESHOLD > 1.0:
            errors.append(f"CHUNK_MINHASH_THRESHOLD must be 0.0–1.0, got {self.CHUNK_MINHASH_THRESHOLD}")

        if self.OTEL_SAMPLING_RATIO < 0.0 or self.OTEL_SAMPLING_RATIO > 1.0:
            errors.append(f"OTEL_SAMPLING_RATIO must be 0.0–1.0, got {self.OTEL_SAMPLING_RATIO}")

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


# TESTS

# ============================================================
# TESTS — Phase 24 Upgrade
# Run: pytest app/core/config.py -v
# ============================================================

import pytest


class TestSettings:

    def test_defaults_are_valid(self):
        s = Settings()
        assert s.APP_VERSION == "0.25.0"
        assert s.TEXT_EMBEDDING_DIM == 384
        assert s.VISION_EMBEDDING_DIM == 512
        assert s.CHUNK_OVERLAP < s.CHUNK_SIZE
        assert s.AGENT_MAX_STEPS > 0
        assert s.RAG_TOP_K > 0
        assert s.MEMORY_TOP_K > 0

    def test_fusion_weights_sum_to_one(self):
        s = Settings()
        total = s.FUSION_SCORE_WEIGHT + s.FUSION_QUALITY_WEIGHT + s.FUSION_MODALITY_WEIGHT
        assert 0.95 <= total <= 1.05

    def test_file_size_limits_property(self):
        s = Settings()
        limits = s.FILE_SIZE_LIMITS
        assert "text" in limits
        assert "pdf" in limits
        assert "video" in limits
        assert limits["video"] > limits["audio"]
        assert limits["pdf"] > limits["text"]

    def test_chunk_overlap_less_than_chunk_size(self):
        s = Settings()
        assert s.CHUNK_OVERLAP < s.CHUNK_SIZE

    def test_llm_max_tokens_within_context(self):
        s = Settings()
        assert s.LLM_MAX_TOKENS <= s.CONTEXT_MAX_TOKENS

    def test_new_phase24_fields_present(self):
        s = Settings()
        assert hasattr(s, "ASYNC_SEMAPHORE_WORKERS")
        assert hasattr(s, "OTEL_ENABLED")
        assert hasattr(s, "DEDUP_ENABLED")
        assert hasattr(s, "MAGIC_BYTE_MIME_DETECTION")
        assert hasattr(s, "CHUNK_MINHASH_THRESHOLD")
        assert hasattr(s, "HYBRID_RRF_K")
        assert hasattr(s, "QDRANT_SOFT_DELETE")
        assert hasattr(s, "REDIS_EMBEDDING_CACHE_TTL")
        assert hasattr(s, "MONGO_CB_FAIL_MAX")
        assert hasattr(s, "PII_ENTITIES")
        assert hasattr(s, "SSRF_BLOCK_PRIVATE_IPS")
        assert hasattr(s, "LIBREOFFICE_ENABLED")
        assert hasattr(s, "VIDEO_DUPLICATE_FRAME_THRESHOLD")

    def test_circuit_breaker_defaults(self):
        s = Settings()
        assert s.QDRANT_CB_FAIL_MAX == 5
        assert s.REDIS_CB_FAIL_MAX == 5
        assert s.MONGO_CB_FAIL_MAX == 5
        assert s.QDRANT_CB_RESET_TIMEOUT > 0
        assert s.REDIS_CB_RESET_TIMEOUT > 0

    def test_otel_sampling_ratio_range(self):
        s = Settings()
        assert 0.0 <= s.OTEL_SAMPLING_RATIO <= 1.0

    def test_minhash_threshold_range(self):
        s = Settings()
        assert 0.0 <= s.CHUNK_MINHASH_THRESHOLD <= 1.0

    def test_env_development_secret_key_allowed(self):
        s = Settings()
        assert s.ENV == "development"

    def test_paths_are_path_objects(self):
        s = Settings()
        assert isinstance(s.DATA_DIR, Path)
        assert isinstance(s.LOG_DIR, Path)
        assert isinstance(s.TEMP_DIR, Path)
        assert isinstance(s.AUDIT_LOG_PATH, Path)
        assert isinstance(s.CHROOT_BASE, Path)

    def test_whisper_filler_words_list(self):
        s = Settings()
        assert isinstance(s.WHISPER_FILLER_WORDS, list)
        assert "uh" in s.WHISPER_FILLER_WORDS

    def test_pii_entities_list(self):
        s = Settings()
        assert isinstance(s.PII_ENTITIES, list)
        assert "EMAIL_ADDRESS" in s.PII_ENTITIES

    def test_slo_targets_positive(self):
        s = Settings()
        assert s.SLO_TEXT_P95_MS > 0
        assert s.SLO_PDF_P95_MS > 0
        assert s.SLO_VIDEO_P95_MS > s.SLO_TEXT_P95_MS

    def test_validate_raises_on_invalid_chunk_overlap(self):
        s = Settings()
        original = s.CHUNK_OVERLAP
        s.CHUNK_OVERLAP = s.CHUNK_SIZE + 10
        with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
            s.validate()
        s.CHUNK_OVERLAP = original

    def test_validate_raises_on_bad_fusion_weights(self):
        s = Settings()
        original = s.FUSION_SCORE_WEIGHT
        s.FUSION_SCORE_WEIGHT = 0.9
        with pytest.raises(ValueError, match="FUSION weights"):
            s.validate()
        s.FUSION_SCORE_WEIGHT = original

    def test_get_settings_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])