from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(override=True)


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

    # MODEL CACHE — permanent .hf_cache (survives instance stop/start; mount as EBS volume on EC2)
    HF_HOME: str = _str("HF_HOME", str(Path(__file__).resolve().parents[2] / ".hf_cache"))
    MODEL_CACHE_REQUIRE_MANIFEST: bool = _bool("MODEL_CACHE_REQUIRE_MANIFEST", False)

    # CORE APPLICATION
    APP_NAME: str        = _str("APP_NAME", "Multimodal Agentic RAG Integrated Knowledge AI Assistant")
    APP_VERSION: str     = _str("APP_VERSION", "1.0.0")
    APP_DESCRIPTION: str = _str("APP_DESCRIPTION", "Production Multimodal Agentic RAG Integrated AI System")
    ENV: str             = _str("ENV", "development")
    DEBUG: bool          = _bool("DEBUG", False)

    # SERVER
    HOST: str      = _str("HOST",      "0.0.0.0")
    PORT: int      = _int("PORT",      8000)
    ROOT_PATH: str = _str("ROOT_PATH", "")

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
    LOG_SHOW_TIMESTAMP: bool   = _bool("LOG_SHOW_TIMESTAMP", False)
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

    # PERFORMANCE — defaults tuned for Tesla T4 GPU deployment
    THREAD_POOL_SIZE: int           = _int("THREAD_POOL_SIZE", 8)
    ASYNC_SEMAPHORE_WORKERS: int    = _int("ASYNC_SEMAPHORE_WORKERS", 10)
    MAX_PARALLEL_REQUESTS: int      = _int("MAX_PARALLEL_REQUESTS", 32)
    REQUEST_TIMEOUT_SEC: int        = _int("REQUEST_TIMEOUT_SEC", 120)
    FILE_PROCESSING_TIMEOUT_SEC: int = _int("FILE_PROCESSING_TIMEOUT_SEC", 300)
    LLM_CALL_TIMEOUT_SEC: int       = _int("LLM_CALL_TIMEOUT_SEC", 60)
    RETRIEVAL_TIMEOUT: int          = _int("RETRIEVAL_TIMEOUT", 10)
    EMBEDDING_TIMEOUT: int          = _int("EMBEDDING_TIMEOUT", 5)
    VECTOR_DB_TIMEOUT: int          = _int("VECTOR_DB_TIMEOUT", 10)
    MODEL_TIMEOUT_SEC: int          = _int("MODEL_TIMEOUT_SEC", 120)
    SLOW_REQUEST_THRESHOLD: float   = _float("SLOW_REQUEST_THRESHOLD", 2.0)
    INGESTION_WORKER_COUNT: int     = _int("INGESTION_WORKER_COUNT", 6)

    # LLM
    LLM_MODEL_PATH: str      = _str("LLM_MODEL_PATH", "./.hf_cache/gguf/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    LLM_MAX_TOKENS: int      = _int("LLM_MAX_TOKENS", 768)
    CONTEXT_MAX_TOKENS: int  = _int("CONTEXT_MAX_TOKENS", 8192)
    LLM_TEMPERATURE: float          = _float("LLM_TEMPERATURE", 0.2)
    LLM_TEMPERATURE_FACTUAL: float  = _float("LLM_TEMPERATURE_FACTUAL", 0.1)
    LLM_TEMPERATURE_GENERATIVE: float = _float("LLM_TEMPERATURE_GENERATIVE", 0.35)
    LLM_TOP_P: float         = _float("LLM_TOP_P", 0.9)
    LLM_TOP_K_SAMPLING: int  = _int("LLM_TOP_K_SAMPLING", 40)
    LLM_MIN_P: float         = _float("LLM_MIN_P", 0.05)
    LLM_REPEAT_PENALTY: float = _float("LLM_REPEAT_PENALTY", 1.1)
    MAX_PROMPT_CHARS: int    = _int("MAX_PROMPT_CHARS", 8000)
    LLM_GPU_LAYERS: int      = _int("LLM_GPU_LAYERS", -1)
    LLM_THREADS: int         = _int("LLM_THREADS", 4)
    LLM_N_BATCH: int         = _int("LLM_N_BATCH", 1024)
    LLM_USE_MLOCK: bool      = _bool("LLM_USE_MLOCK", True)
    LLM_MAX_RETRIES: int     = _int("LLM_MAX_RETRIES", 3)
    LLM_RETRY_WAIT_MIN: int  = _int("LLM_RETRY_WAIT_MIN", 1)
    LLM_RETRY_WAIT_MAX: int  = _int("LLM_RETRY_WAIT_MAX", 10)

    # EMBEDDINGS
    EMBEDDING_MODEL: str                = _str("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    EMBEDDING_BATCH_SIZE: int           = _int("EMBEDDING_BATCH_SIZE", 128)
    EMBEDDING_MAX_BATCH_SIZE: int       = _int("EMBEDDING_MAX_BATCH_SIZE", 128)
    EMBEDDING_MAX_SEQ_LEN: int          = _int("EMBEDDING_MAX_SEQ_LEN", 512)
    INGESTION_MICRO_BATCH: int          = _int("INGESTION_MICRO_BATCH", 32)
    # Clear CUDA cache every N micro-batches during ingestion (OOM guard
    # without the per-chunk empty_cache+gc tax that dominated embed latency).
    INGESTION_CACHE_CLEAR_EVERY: int    = _int("INGESTION_CACHE_CLEAR_EVERY", 8)
    # Parallelism caps for heavy ML operations — keep within A10G VRAM budget
    VIDEO_CAPTION_CONCURRENCY: int      = _int("VIDEO_CAPTION_CONCURRENCY", 3)
    # Shared GPU semaphore — max concurrent ingestion jobs using GPU (embed/Whisper/LLaVA).
    # 3 jobs × peak ~3GB each = ~9GB working memory + ~14GB resident models = ~23GB on A10G.
    MAX_CONCURRENT_GPU_JOBS: int        = _int("MAX_CONCURRENT_GPU_JOBS", 3)
    AUDIO_TRANSCRIPTION_WORKERS: int    = _int("AUDIO_TRANSCRIPTION_WORKERS", 2)
    PDF_OCR_WORKERS: int                = _int("PDF_OCR_WORKERS", 8)
    EMBEDDING_CACHE_TTL: int            = _int("EMBEDDING_CACHE_TTL", 2_592_000)
    TEXT_EMBEDDING_DIM: int             = _int("TEXT_EMBEDDING_DIM", 1024)
    # BGE-large requires instruction prefix on queries only (not documents)
    BGE_QUERY_INSTRUCTION: str          = _str("BGE_QUERY_INSTRUCTION",
                                               "Represent this sentence for searching relevant passages: ")

    # MODALITY FEATURE FLAGS — set to false to skip model load and all related processing
    ENABLE_VISION: bool              = _bool("ENABLE_VISION", True)
    ENABLE_AUDIO: bool               = _bool("ENABLE_AUDIO", True)

    # FINANCE-GRADE PROCESSING FEATURE FLAGS
    PDF_USE_PDFPLUMBER: bool         = _bool("PDF_USE_PDFPLUMBER", True)
    VIDEO_SCENE_DETECTION: bool      = _bool("VIDEO_SCENE_DETECTION", True)
    EXCEL_SEMANTIC_GROUP: bool       = _bool("EXCEL_SEMANTIC_GROUP", True)
    AUDIO_SPEAKER_SUBINDEX_ENABLED: bool = _bool("AUDIO_SPEAKER_SUBINDEX_ENABLED", False)

    # MODEL DEVICE / WARMUP — Tesla T4 / A10G all_gpu profile
    # Profiles: "auto" (CUDA → all_gpu, else cpu), "hybrid", "all_gpu", "all_cpu"
    MODELS_DEVICE_PROFILE: str       = _str("MODELS_DEVICE_PROFILE", "all_gpu")
    VRAM_BUDGET_GB: float            = _float("VRAM_BUDGET_GB", 13.0)
    WARMUP_AT_STARTUP: bool          = _bool("WARMUP_AT_STARTUP", True)
    WARMUP_MODELS: List[str]         = _list("WARMUP_MODELS", ["text_embedder", "llm", "reranker", "siglip", "blip", "whisper"])
    MODEL_PARALLEL_LOAD: bool        = _bool("MODEL_PARALLEL_LOAD", True)
    LLM_GPU_LAYERS_AUTO: bool        = _bool("LLM_GPU_LAYERS_AUTO", True)
    LLM_GPU_LAYERS_ALL: int          = _int("LLM_GPU_LAYERS_ALL", -1)
    LLM_N_CTX: int                   = _int("LLM_N_CTX", 8192)
    EMBEDDER_HALF_PRECISION: bool    = _bool("EMBEDDER_HALF_PRECISION", True)
    VISION_HALF_PRECISION: bool      = _bool("VISION_HALF_PRECISION", True)
    WHISPER_COMPUTE_TYPE: str        = _str("WHISPER_COMPUTE_TYPE", "int8_float16")
    EMBEDDER_DEVICE: str             = _str("EMBEDDER_DEVICE", "cuda")
    RERANKER_DEVICE: str             = _str("RERANKER_DEVICE", "cuda")
    SIGLIP_DEVICE: str               = _str("SIGLIP_DEVICE", "cuda")
    BLIP_DEVICE: str                 = _str("BLIP_DEVICE", "cuda")
    BLIP2_DEVICE: str                = _str("BLIP2_DEVICE", "cuda")
    LLAVA_DEVICE: str                = _str("LLAVA_DEVICE", "cuda")
    TROCR_DEVICE: str                = _str("TROCR_DEVICE", "cuda")
    DIARIZER_DEVICE: str             = _str("DIARIZER_DEVICE", "cuda")
    NER_DEVICE: str                  = _str("NER_DEVICE", "cuda")
    WHISPER_DEVICE: str              = _str("WHISPER_DEVICE", "cuda")
    LLM_DEVICE_HINT: str             = _str("LLM_DEVICE_HINT", "cuda")

    # VISION MODELS
    SIGLIP_MODEL: str                = _str("SIGLIP_MODEL", "google/siglip-so400m-patch14-384")
    VISION_EMBEDDING_DIM: int        = _int("VISION_EMBEDDING_DIM", 1152)
    # BLIP-1 kept for backward-compat during transition; new ingestion uses BLIP2
    BLIP_MODEL: str                  = _str("BLIP_MODEL", "Salesforce/blip-image-captioning-large")
    BLIP_MAX_TOKENS: int             = _int("BLIP_MAX_TOKENS", 100)
    BLIP_NUM_BEAMS: int              = _int("BLIP_NUM_BEAMS", 2)
    # BLIP2 — replaces BLIP-1 for image captioning (Phase 2+)
    BLIP2_MODEL: str                 = _str("BLIP2_MODEL", "Salesforce/blip2-opt-2.7b")
    BLIP2_LOAD_IN_8BIT: bool         = _bool("BLIP2_LOAD_IN_8BIT", True)
    BLIP2_MAX_TOKENS: int            = _int("BLIP2_MAX_TOKENS", 200)
    # LLaVA — video frame captioning
    LLAVA_MODEL: str                 = _str("LLAVA_MODEL", "llava-hf/llava-1.5-7b-hf")
    LLAVA_LOAD_IN_8BIT: bool         = _bool("LLAVA_LOAD_IN_8BIT", True)
    LLAVA_MAX_TOKENS: int            = _int("LLAVA_MAX_TOKENS", 300)
    # TrOCR — printed OCR for financial documents
    TROCR_MODEL: str                 = _str("TROCR_MODEL", "microsoft/trocr-large-printed")
    MAX_IMAGE_DIM: int               = _int("MAX_IMAGE_DIM", 1024)
    MAX_IMAGE_SIZE_MP: int           = _int("MAX_IMAGE_SIZE_MP", 50)
    THUMBNAIL_WIDTH: int             = _int("THUMBNAIL_WIDTH", 256)
    THUMBNAIL_HEIGHT: int            = _int("THUMBNAIL_HEIGHT", 256)
    IMAGE_PRIVACY_MODE: bool         = _bool("IMAGE_PRIVACY_MODE", False)
    SOLID_COLOR_THRESHOLD: float     = _float("SOLID_COLOR_THRESHOLD", 0.95)

    # ASR WHISPER
    WHISPER_MODEL: str              = _str("WHISPER_MODEL", "large-v3")
    AUDIO_SAMPLE_RATE: int          = _int("AUDIO_SAMPLE_RATE", 16000)
    MAX_AUDIO_DURATION_SEC: int     = _int("MAX_AUDIO_DURATION_SEC", 10800)
    MAX_AUDIO_SEGMENTS: int         = _int("MAX_AUDIO_SEGMENTS", 500)
    AUDIO_SNR_THRESHOLD_DB: float   = _float("AUDIO_SNR_THRESHOLD_DB", 10.0)
    AUDIO_SILENCE_GAP_MS: int       = _int("AUDIO_SILENCE_GAP_MS", 200)
    AUDIO_CHUNK_DURATION_SEC: int   = _int("AUDIO_CHUNK_DURATION_SEC", 1800)
    DIARIZATION_ENABLED: bool       = _bool("DIARIZATION_ENABLED", False)
    AUDIO_DIARIZATION_ENABLED: bool = _bool("AUDIO_DIARIZATION_ENABLED", False)
    # Pyannote speaker diarization (requires HF_TOKEN + model access approval)
    DIARIZATION_MODEL: str          = _str("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    # Finance NER
    NER_MODEL: str                  = _str("NER_MODEL", "dslim/bert-base-NER")
    # FinBERT — finance-domain tone/sentiment classifier (yiyanghkust/finbert-tone)
    FINBERT_MODEL: str              = _str("FINBERT_MODEL", "yiyanghkust/finbert-tone")
    FINBERT_ENABLED: bool           = _bool("FINBERT_ENABLED", True)
    FINBERT_BATCH_SIZE: int         = _int("FINBERT_BATCH_SIZE", 32)
    HF_TOKEN: Optional[str]         = _opt("HF_TOKEN")
    WHISPER_DOMAIN_VOCAB: List[str] = _list("WHISPER_DOMAIN_VOCAB", [
        "RAG", "FAISS", "Qdrant", "LlamaIndex", "LangChain", "RAGAS",
        "Pinecone", "ChromaDB", "Mistral", "OpenAI", "Anthropic",
        "embedding", "retrieval", "chunking", "reranker", "BM25",
    ])
    WHISPER_FILLER_WORDS: List[str] = _list("WHISPER_FILLER_WORDS", ["uh", "um", "er", "hmm"])

    # RERANKER
    RERANKER_MODEL: str             = _str("RERANKER_MODEL", "BAAI/bge-reranker-large")
    RERANK_TOP_K: int               = _int("RERANK_TOP_K", 8)
    RERANK_MAX_INPUT: int           = _int("RERANK_MAX_INPUT", 20)
    RERANK_CONTEXT_MAX_CHARS: int   = _int("RERANK_CONTEXT_MAX_CHARS", 1024)
    RERANK_MODEL_WEIGHT: float      = _float("RERANK_MODEL_WEIGHT", 0.7)
    RERANK_FUSION_WEIGHT: float     = _float("RERANK_FUSION_WEIGHT", 0.3)
    RERANK_POSITION_WEIGHT: float   = _float("RERANK_POSITION_WEIGHT", 0.1)
    RERANK_SCORE_THRESHOLD: float   = _float("RERANK_SCORE_THRESHOLD", 0.1)
    RERANK_BATCH_SIZE: int          = _int("RERANK_BATCH_SIZE", 32)
    RERANK_SOURCE_MAX_CHUNKS: int   = _int("RERANK_SOURCE_MAX_CHUNKS", 3)
    MMR_ENABLED: bool               = _bool("MMR_ENABLED", True)
    MMR_LAMBDA: float               = _float("MMR_LAMBDA", 0.7)
    COHERE_API_KEY: Optional[str]   = _opt("COHERE_API_KEY")

    RERANK_MODALITY_WEIGHTS: Dict[str, float] = {
        "text":  1.0,
        "table": 1.15,
        "image": 0.75,
        "audio": 0.85,
        "video": 0.75,
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

    # CIRCUIT BREAKER — SHARED DEFAULTS
    CIRCUIT_BREAKER_MAX_FAILURES: int    = _int("CIRCUIT_BREAKER_MAX_FAILURES", 5)
    CIRCUIT_BREAKER_RESET_TIMEOUT: int   = _int("CIRCUIT_BREAKER_RESET_TIMEOUT", 60)

    # RETRY — SHARED DEFAULTS
    RETRY_MAX_ATTEMPTS: int   = _int("RETRY_MAX_ATTEMPTS", 3)
    RETRY_WAIT_MIN_SEC: float = _float("RETRY_WAIT_MIN_SEC", 1.0)
    RETRY_WAIT_MAX_SEC: float = _float("RETRY_WAIT_MAX_SEC", 10.0)

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
    USE_MONGO: bool                  = _bool("USE_MONGO", True)
    REDIS_CB_FAIL_MAX: int           = _int("REDIS_CB_FAIL_MAX", 5)
    REDIS_CB_RESET_TIMEOUT: int      = _int("REDIS_CB_RESET_TIMEOUT", 30)
    REDIS_MAX_WINDOW_TURNS: int      = _int("REDIS_MAX_WINDOW_TURNS", 40)
    REDIS_MAX_CONNECTIONS: int       = _int("REDIS_MAX_CONNECTIONS", 10)

    # MONGODB
    MONGO_URI: str                       = _str("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str                   = _str("MONGO_DB_NAME", "rag_memory")
    DB_TIMEOUT_MS: int                   = _int("DB_TIMEOUT_MS", 5000)
    DB_MAX_POOL_SIZE: int                = _int("DB_MAX_POOL_SIZE", 20)
    MONGO_CB_FAIL_MAX: int               = _int("MONGO_CB_FAIL_MAX", 5)
    MONGO_CB_RESET_TIMEOUT: int          = _int("MONGO_CB_RESET_TIMEOUT", 60)
    MONGO_MESSAGES_COLLECTION: str       = _str("MONGO_MESSAGES_COLLECTION", "messages")
    MONGO_SUMMARIES_COLLECTION: str      = _str("MONGO_SUMMARIES_COLLECTION", "summaries")
    MONGO_MEMORY_COLLECTION: str         = _str("MONGO_MEMORY_COLLECTION", "messages")
    MONGO_SUMMARY_COLLECTION: str        = _str("MONGO_SUMMARY_COLLECTION", "summaries")
    MONGO_AUDIT_COLLECTION: str          = _str("MONGO_AUDIT_COLLECTION", "audit_log")
    MONGO_SESSIONS_COLLECTION: str       = _str("MONGO_SESSIONS_COLLECTION", "chat_sessions")
    MONGO_FEEDBACK_COLLECTION: str       = _str("MONGO_FEEDBACK_COLLECTION", "feedback")

    # TEXT REPAIR — per-pass toggles for the broken-corpus hardening layer
    # (mojibake fix, noise-line strip, whitespace recovery, OCR normalization,
    # footnote strip, placeholder skip, title-mismatch flag, version tagging).
    # All default ON; each pass is a no-op on clean text so the cost is low.
    TEXT_REPAIR_ENABLED: bool        = _bool("TEXT_REPAIR_ENABLED",     True)
    TEXT_REPAIR_MOJIBAKE: bool       = _bool("TEXT_REPAIR_MOJIBAKE",    True)
    TEXT_REPAIR_NOISE_LINES: bool    = _bool("TEXT_REPAIR_NOISE_LINES", True)
    TEXT_REPAIR_WHITESPACE: bool     = _bool("TEXT_REPAIR_WHITESPACE",  True)
    TEXT_REPAIR_OCR: bool            = _bool("TEXT_REPAIR_OCR",         True)
    TEXT_REPAIR_FOOTNOTES: bool      = _bool("TEXT_REPAIR_FOOTNOTES",   True)
    TEXT_REPAIR_PLACEHOLDERS: bool   = _bool("TEXT_REPAIR_PLACEHOLDERS", True)
    TEXT_REPAIR_TITLE_MISMATCH: bool = _bool("TEXT_REPAIR_TITLE_MISMATCH", True)
    TEXT_REPAIR_VERSION_TAG: bool    = _bool("TEXT_REPAIR_VERSION_TAG", True)

    # CHUNKING
    CHUNK_SIZE: int                  = _int("CHUNK_SIZE", 1024)
    CHUNK_OVERLAP: int               = _int("CHUNK_OVERLAP", 128)
    CHUNK_HASH_ID: bool              = _bool("CHUNK_HASH_ID", True)
    FINANCE_NUMBER_PROTECT: bool     = _bool("FINANCE_NUMBER_PROTECT", True)
    CHUNK_TARGET_TOKENS: int         = _int("CHUNK_TARGET_TOKENS", 400)
    CHUNK_OVERLAP_SENTENCES: int     = _int("CHUNK_OVERLAP_SENTENCES", 1)
    CHUNK_OVERLAP_RATIO: float       = _float("CHUNK_OVERLAP_RATIO", 0.15)
    CHUNK_MIN_SIZE: int              = _int("CHUNK_MIN_SIZE", 50)
    MAX_CHUNKS: int                  = _int("MAX_CHUNKS", 200)
    INGESTION_BATCH_SIZE: int        = _int("INGESTION_BATCH_SIZE", 16)
    CHUNK_MAX_TOKENS: int            = _int("CHUNK_MAX_TOKENS", 512)
    CHUNK_MINHASH_THRESHOLD: float   = _float("CHUNK_MINHASH_THRESHOLD", 0.85)
    CHUNK_MINHASH_PERMUTATIONS: int  = _int("CHUNK_MINHASH_PERMUTATIONS", 128)
    CHUNK_ADAPTIVE_ENABLED: bool     = _bool("CHUNK_ADAPTIVE_ENABLED", True)
    LATE_CHUNKING_ENABLED: bool      = _bool("LATE_CHUNKING_ENABLED", False)
    EXCEL_ROWS_PER_CHUNK: int        = _int("EXCEL_ROWS_PER_CHUNK", 25)

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
    HYBRID_WEIGHT_BM25: float          = _float("HYBRID_WEIGHT_BM25", 0.40)
    HYBRID_WEIGHT_VECTOR: float        = _float("HYBRID_WEIGHT_VECTOR", 0.60)
    HYBRID_WEIGHT_VISION: float        = _float("HYBRID_WEIGHT_VISION", 0.2)
    HYBRID_CANDIDATES_MULTIPLIER: int  = _int("HYBRID_CANDIDATES_MULTIPLIER", 3)
    HYBRID_MIN_SCORE: float            = _float("HYBRID_MIN_SCORE", 0.20)
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
    DEFAULT_TOP_K: int      = _int("DEFAULT_TOP_K", 8)
    VECTOR_TOP_K: int       = _int("VECTOR_TOP_K", 15)
    MAX_CONTEXT_DOCS: int   = _int("MAX_CONTEXT_DOCS", 5)
    MAX_CONTEXT_CHARS: int  = _int("MAX_CONTEXT_CHARS", 5500)
    RAG_TOP_K: int          = _int("RAG_TOP_K", 8)
    RAG_DOC_MAX_CHARS: int  = _int("RAG_DOC_MAX_CHARS", 1200)

    # STREAMING — token flush holdback (see RAGPipeline.stream)
    STREAM_PREFIX_GATE_CHARS: int = _int("STREAM_PREFIX_GATE_CHARS", 160)
    STREAM_HOLDBACK_CHARS: int    = _int("STREAM_HOLDBACK_CHARS", 48)

    # QUERY DECOMPOSITION
    MAX_SUBQUERIES: int                        = _int("MAX_SUBQUERIES", 3)
    SUBQUERY_MAX_TOKENS: int                   = _int("SUBQUERY_MAX_TOKENS", 64)
    DECOMPOSITION_MIN_WORDS: int               = _int("DECOMPOSITION_MIN_WORDS", 6)
    # Heuristic gate: only call the LLM decomposer for structurally multi-part
    # queries (≥2 question marks or an explicit compare/multi-part marker).
    # Measured logs show decompose yields 1 subquery ~100% of the time on
    # simple questions — a pure LLM-latency tax on every query otherwise.
    DECOMPOSITION_HEURISTIC_GATE: bool         = _bool("DECOMPOSITION_HEURISTIC_GATE", True)
    DECOMPOSITION_MAX_SUBQUERIES: int          = _int("DECOMPOSITION_MAX_SUBQUERIES", 3)
    DECOMPOSITION_CONFIDENCE_THRESHOLD: float  = _float("DECOMPOSITION_CONFIDENCE_THRESHOLD", 0.5)
    DECOMPOSITION_KEYWORDS: List[str]          = _list("DECOMPOSITION_KEYWORDS", [
        "and", "also", "then", "but",
    ])

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
    MEMORY_SUMMARY_EVERY_N_TURNS: int = _int("MEMORY_SUMMARY_EVERY_N_TURNS", 10)
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
    AGENT_TOKEN_BUDGET: int              = _int("AGENT_TOKEN_BUDGET", 32000)
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
    FRAME_DARKNESS_THRESHOLD: float          = _float("FRAME_DARKNESS_THRESHOLD", 10.0)
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

    # HALLUCINATION
    HALLUCINATION_THRESHOLD: float = _float("HALLUCINATION_THRESHOLD", 0.5)

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

    # MULTI-USER
    DEFAULT_DEV_USER_ID: str = _str("DEFAULT_DEV_USER_ID", "a3f9c821-4b2e-11ef-9454-0242ac120002")
    TEST_USER_2_ID: str      = _str("TEST_USER_2_ID",      "b7d2e109-4b2e-11ef-9454-0242ac120002")

    # GUEST MODE — trial session limits
    GUEST_QUERY_LIMIT:        int = _int("GUEST_QUERY_LIMIT",        5)
    GUEST_FILE_LIMIT:         int = _int("GUEST_FILE_LIMIT",         2)
    GUEST_SESSION_TTL_HOURS:  int = _int("GUEST_SESSION_TTL_HOURS", 24)

    # JWT AUTHENTICATION — Phase 27
    JWT_SECRET_KEY: str                  = _str("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
    JWT_ALGORITHM: str                   = _str("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int     = _int("ACCESS_TOKEN_EXPIRE_MINUTES", 30)
    REFRESH_TOKEN_EXPIRE_DAYS: int       = _int("REFRESH_TOKEN_EXPIRE_DAYS", 7)
    AUTH_COLLECTION: str                 = _str("AUTH_COLLECTION", "users")
    PASSWORD_MIN_ZXCVBN_SCORE: int       = _int("PASSWORD_MIN_ZXCVBN_SCORE", 2)
    PASSWORD_MIN_LENGTH: int             = _int("PASSWORD_MIN_LENGTH", 8)
    AUTH_ENABLED: bool                   = _bool("AUTH_ENABLED", True)

    # EMAIL (Gmail SMTP) — OTP + password-reset
    SMTP_HOST: str                       = _str("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int                       = _int("SMTP_PORT", 587)
    SMTP_USER: Optional[str]             = _opt("SMTP_USER")
    SMTP_PASSWORD: Optional[str]         = _opt("SMTP_PASSWORD")
    EMAIL_FROM: str                      = _str("EMAIL_FROM", "magikaiassistant@gmail.com")
    EMAIL_FROM_NAME: str                 = _str("EMAIL_FROM_NAME", "MAGIK AI")
    FRONTEND_URL: str                    = _str("FRONTEND_URL", "http://localhost:5173")

    # OTP / password-reset tuning
    OTP_TTL_SECONDS: int                 = _int("OTP_TTL_SECONDS", 600)        # 10 min
    OTP_MAX_ATTEMPTS: int                = _int("OTP_MAX_ATTEMPTS", 3)
    OTP_LOCKOUT_SECONDS: int             = _int("OTP_LOCKOUT_SECONDS", 900)    # 15 min
    RESET_TOKEN_TTL_SECONDS: int         = _int("RESET_TOKEN_TTL_SECONDS", 3600)  # 1 hr
    # Dev-only: log OTP/reset links to console instead of sending email
    DEV_OTP_LOG: bool                    = _bool("DEV_OTP_LOG", False)

    # SECURITY
    SECRET_KEY: str                        = _str("SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
    DATABASE_URL: str                      = _str("DATABASE_URL", "sqlite:///./data/rag_users.db")
    CORS_ORIGINS: List[str]                = _list("CORS_ORIGINS", ["*"])
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
    EXCEL_MAX_ROWS: int           = _int("EXCEL_MAX_ROWS", 50_000)

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
            self.TEST_FIXTURES_DIR,
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

        if self.USE_MONGO and not self.MONGO_URI:
            errors.append("MONGO_URI is required when USE_MONGO=true")

        if self.USE_REDIS and not self.REDIS_URL and not self.REDIS_HOST:
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

        if self.ENV != "development" and self.JWT_SECRET_KEY == "CHANGE_ME_IN_PRODUCTION":
            errors.append("JWT_SECRET_KEY must be changed from default in non-development environments")

        if self.AUTH_ENABLED and len(self.JWT_SECRET_KEY) < 32:
            errors.append("JWT_SECRET_KEY must be at least 32 characters long")

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

    def get_service_status(self) -> Dict[str, bool]:
        return {
            "redis":  self.USE_REDIS,
            "mongo":  self.USE_MONGO,
            "qdrant": bool(self.QDRANT_URL or self.QDRANT_HOST),
        }


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
        assert s.APP_VERSION == "1.0.0"
        assert s.TEXT_EMBEDDING_DIM == 1024
        assert s.VISION_EMBEDDING_DIM > 0
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