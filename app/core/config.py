import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:

    
    # CORE
    
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    APP_NAME: str        = "Multimodal RAG Assistant"
    APP_VERSION: str     = "0.20.0"
    APP_DESCRIPTION: str = "Production Multimodal RAG System"

    ENV: str   = "production"
    DEBUG: bool = False

    
    # PATHS
    
    DATA_DIR: Path = (PROJECT_ROOT / "data").resolve()
    LOG_DIR: Path  = (PROJECT_ROOT / "logs").resolve()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    
    # SERVER
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    
    # LOGGING
    
    LOG_LEVEL: str          = "INFO"
    LOG_JSON: bool          = True       # Structured JSON logs for production
    ENABLE_FILE_LOGGING: bool = True
    LOG_FILE_NAME: str      = "app.log"
    LOG_MAX_BYTES: int      = 10 * 1024 * 1024   # 10 MB per file
    LOG_BACKUP_COUNT: int   = 5

    
    # MULTIMODAL MODELS
    
    CLIP_MODEL: str      = "openai/clip-vit-base-patch32"
    WHISPER_MODEL: str   = "base"
    BLIP_MODEL: str      = "Salesforce/blip-image-captioning-large"
    RERANKER_MODEL: str  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # BLIP inference controls
    BLIP_MAX_TOKENS: int = 128
    BLIP_NUM_BEAMS: int  = 4

    
    # PERFORMANCE
    
    THREAD_POOL_SIZE: int       = 4
    MAX_PARALLEL_REQUESTS: int  = 20
    REQUEST_TIMEOUT_SEC: int    = 30
    RETRIEVAL_TIMEOUT: int      = 10

    
    # LLM / MODEL
    
    MODEL_TIMEOUT_SEC: int  = 120
    LLM_MODEL_PATH: str     = r"./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    LLM_MAX_TOKENS: int     = 512
    CONTEXT_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float  = 0.2
    LLM_TOP_P: float        = 0.9

    MAX_PROMPT_CHARS: int   = 8_000

    
    # EMBEDDINGS
    
    EMBEDDING_MODEL: str      = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32

    TEXT_EMBEDDING_DIM: int  = 384
    VISION_EMBEDDING_DIM: int = 512
    CLIP_MAX_LENGTH: int     = 77

    
    # CHUNKING
    
    CHUNK_SIZE: int    = 300
    CHUNK_OVERLAP: int = 80
    MAX_CHUNKS: int    = 1_000

    
    # INGESTION
    
    INGESTION_BATCH_SIZE: int = 64
    MAX_INGESTED_DOCS: int    = 5_000
    MAX_FILE_SIZE_MB: int     = 50
    UPLOAD_CHUNK_SIZE: int    = 1024 * 1024   # 1 MB streaming upload chunks

    
    # RETRIEVAL
    
    DEFAULT_TOP_K: int = 5
    RAG_TOP_K: int     = 5

    RAG_DOC_MAX_CHARS: int          = 1_000
    HYBRID_CANDIDATES_MULTIPLIER: int = 3

    HYBRID_WEIGHT_BM25: float   = 0.5
    HYBRID_WEIGHT_VECTOR: float = 0.4
    HYBRID_WEIGHT_VISION: float = 0.1

    
    # BM25
    
    BM25_TOP_K: int          = 5
    BM25_MAX_DOCS: int       = 10_000
    BM25_MAX_TEXT_CHARS: int = 1_000
    BM25_MAX_TOKENS: int     = 256

    
    # RERANK
    
    RERANK_TOP_K: int            = 5
    RERANK_MAX_INPUT: int        = 50
    RERANK_CONTEXT_MAX_CHARS: int = 512

    RERANK_MODEL_WEIGHT: float    = 0.7
    RERANK_FUSION_WEIGHT: float   = 0.3
    RERANK_POSITION_WEIGHT: float = 0.1

    RERANK_MODALITY_WEIGHTS: dict = {
        "text":  1.00,
        "image": 1.10,
        "audio": 1.05,
        "video": 1.15,
    }

    
    # RESULT FUSION
    
    FUSION_SIMILARITY_THRESHOLD: float = 0.85
    FUSION_MAX_INPUT: int               = 50
    FUSION_MAX_TEXT_CHARS: int          = 1_000
    FUSION_SCORE_WEIGHT: float          = 0.6
    FUSION_QUALITY_WEIGHT: float        = 0.3
    FUSION_MODALITY_WEIGHT: float       = 0.1

    
    # VECTOR STORE — QDRANT
    
    QDRANT_URL: str | None     = None        # Set to remote URL in cloud deployments
    QDRANT_API_KEY: str | None = None        # Set when using Qdrant Cloud

    QDRANT_HOST: str    = "localhost"
    QDRANT_PORT: int    = 6333
    QDRANT_TIMEOUT: int = 10

    TEXT_COLLECTION_NAME: str   = "text_collection"
    VISION_COLLECTION_NAME: str = "vision_collection"
    COLLECTION_NAME: str        = "text_collection"   # Legacy alias (integration tests)

    QDRANT_BATCH_SIZE: int      = 64
    QDRANT_MAX_DOCS: int        = 10_000
    QDRANT_TEXT_MAX_CHARS: int  = 1_000

    
    # CACHE — REDIS
    
    REDIS_HOST: str         = "localhost"
    REDIS_PORT: int         = 6379
    REDIS_DB: int           = 0
    REDIS_TIMEOUT: int      = 5      # socket connect + read timeout (seconds)
    REDIS_TTL_SECONDS: int  = 86_400 # 24 h default key TTL

    
    # MEMORY STORE — MONGODB
    
    MONGO_URI: str      = "mongodb://localhost:27017"
    MONGO_DB_NAME: str  = "rag_memory"
    DB_TIMEOUT_MS: int  = 5_000    # serverSelectionTimeoutMS
    DB_MAX_POOL_SIZE: int = 20

    
    # MEMORY / CONVERSATION HISTORY
    
    MAX_HISTORY_MESSAGES: int       = 40
    MAX_SYSTEM_MESSAGES: int        = 5
    MEMORY_TOP_K: int               = 5
    MEMORY_SIM_THRESHOLD: float     = 0.75
    MEMORY_RECENCY_SCALE: float     = 3_600.0    # seconds; controls recency decay
    MEMORY_SUMMARY_THRESHOLD: int   = 20         # messages before summarisation
    MEMORY_MAX_CONTEXT_CHARS: int   = 4_000
    MEMORY_SUMMARY_MAX_CHARS: int   = 2_000
    MEMORY_SUMMARY_INPUT_CHARS: int = 6_000
    MIN_SUMMARY_LENGTH: int         = 50

    
    # AGENT
    
    AGENT_MAX_STEPS: int        = 10
    AGENT_TIMEOUT_SEC: int      = 30
    AGENT_HIGH_CONFIDENCE: float = 0.7
    AGENT_LOW_CONFIDENCE: float  = 0.4

    
    # REASONING / QUERY DECOMPOSITION
    
    MAX_SUBQUERIES: int           = 4
    DECOMPOSITION_MIN_WORDS: int  = 8

    
    # AUDIO / VIDEO INGESTION
    
    MAX_AUDIO_DURATION_SEC: int  = 600     # 10 minutes
    MAX_AUDIO_SEGMENTS: int      = 50
    AUDIO_SAMPLE_RATE: int       = 16_000  # Hz; Whisper native rate

    MAX_VIDEO_DURATION_SEC: int  = 1_800   # 30 minutes
    MAX_VIDEO_FRAMES: int        = 60
    VIDEO_FRAME_INTERVAL_SEC: int = 30

    FFMPEG_PATH: str         = r"C:\Users\karth\Desktop\Production 1\ffmpeg.exe"
    FFMPEG_TIMEOUT_SEC: int  = 120

    
    # WEB SEARCH (Tavily)
    
    TAVILY_API_KEY: str        = os.getenv("TAVILY_API_KEY", "")

    WEB_MAX_RESULTS: int         = 5
    WEB_MAX_DOCS: int            = 5
    WEB_DOC_MAX_CHARS: int       = 500
    WEB_CONTEXT_MAX_CHARS: int   = 2_000
    WEB_SEARCH_DEPTH: str        = "advanced"

    
    # VALIDATION
    
    def validate(self) -> bool:
        errors: list[str] = []

        # --- LLM ---
        if not self.LLM_MODEL_PATH:
            errors.append("LLM_MODEL_PATH is required")

        # --- Chunking ---
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            errors.append(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be < CHUNK_SIZE ({self.CHUNK_SIZE})"
            )

        # --- Hybrid weights ---
        for name, val in (
            ("HYBRID_WEIGHT_BM25",    self.HYBRID_WEIGHT_BM25),
            ("HYBRID_WEIGHT_VECTOR",  self.HYBRID_WEIGHT_VECTOR),
            ("HYBRID_WEIGHT_VISION",  self.HYBRID_WEIGHT_VISION),
        ):
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} must be in [0, 1], got {val}")

        # --- Hybrid weights sum ---
        hybrid_sum = (
            self.HYBRID_WEIGHT_BM25
            + self.HYBRID_WEIGHT_VECTOR
            + self.HYBRID_WEIGHT_VISION
        )
        if not (0.99 <= hybrid_sum <= 1.01):
            errors.append(
                f"Hybrid weights must sum to 1.0, got {hybrid_sum:.3f}"
            )

        # --- Fusion weights sum ---
        fusion_sum = (
            self.FUSION_SCORE_WEIGHT
            + self.FUSION_QUALITY_WEIGHT
            + self.FUSION_MODALITY_WEIGHT
        )
        if not (0.99 <= fusion_sum <= 1.01):
            errors.append(
                f"Fusion weights must sum to 1.0, got {fusion_sum:.3f}"
            )

        # --- Rerank weights ---
        for name, val in (
            ("RERANK_MODEL_WEIGHT",    self.RERANK_MODEL_WEIGHT),
            ("RERANK_FUSION_WEIGHT",   self.RERANK_FUSION_WEIGHT),
            ("RERANK_POSITION_WEIGHT", self.RERANK_POSITION_WEIGHT),
        ):
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} must be in [0, 1], got {val}")

        # --- Confidence thresholds ---
        if not (0.0 <= self.AGENT_LOW_CONFIDENCE < self.AGENT_HIGH_CONFIDENCE <= 1.0):
            errors.append(
                "AGENT_LOW_CONFIDENCE must be < AGENT_HIGH_CONFIDENCE and both in [0, 1]"
            )

        # --- Memory ---
        if self.MEMORY_SUMMARY_THRESHOLD > self.MAX_HISTORY_MESSAGES:
            errors.append(
                "MEMORY_SUMMARY_THRESHOLD should not exceed MAX_HISTORY_MESSAGES"
            )

        if errors:
            raise ValueError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )

        return True


settings = Settings()
settings.validate()
