from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    # CORE 
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    APP_NAME = os.getenv("APP_NAME", "Multimodal RAG Assistant")
    APP_VERSION = os.getenv("APP_VERSION", "0.20.0")
    APP_DESCRIPTION = os.getenv(
        "APP_DESCRIPTION",
        "Production Multimodal RAG System"
    )

    ENV = os.getenv("ENV", "development")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # PATHS 
    DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data")).resolve()
    LOG_DIR = Path(os.getenv("LOG_DIR", PROJECT_ROOT / "logs")).resolve()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # LOGGING 
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_JSON = os.getenv("LOG_JSON", "false").lower() == "true"

    ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
    LOG_FILE_NAME = os.getenv("LOG_FILE_NAME", "app.log")
    LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 5 * 1024 * 1024))
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 3))

    # FILE UPLOAD 
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 10))
    UPLOAD_CHUNK_SIZE = int(os.getenv("UPLOAD_CHUNK_SIZE", 1024 * 1024))

    # LLM 
    LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 512))
    CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", 4096))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))
    LLM_TOP_P = float(os.getenv("LLM_TOP_P", 0.9))

    MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", 8000))

    # EMBEDDING 
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    TEXT_EMBEDDING_DIM = int(os.getenv("TEXT_EMBEDDING_DIM", 384))
    VISION_EMBEDDING_DIM = int(os.getenv("VISION_EMBEDDING_DIM", 512))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))

    # MODELS 
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    BLIP_MODEL = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-large")
    CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # CHUNKING 
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
    MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", 1000))

    # INGESTION 
    INGESTION_BATCH_SIZE = int(os.getenv("INGESTION_BATCH_SIZE", 64))
    MAX_INGESTED_DOCS = int(os.getenv("MAX_INGESTED_DOCS", 5000))

    # VIDEO 
    VIDEO_FRAME_INTERVAL_SEC = int(os.getenv("VIDEO_FRAME_INTERVAL_SEC", 2))
    MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", 200))
    MAX_VIDEO_DURATION_SEC = int(os.getenv("MAX_VIDEO_DURATION_SEC", 600))

    # AUDIO 
    AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))
    MAX_AUDIO_SEGMENTS = int(os.getenv("MAX_AUDIO_SEGMENTS", 200))
    MAX_AUDIO_DURATION_SEC = int(os.getenv("MAX_AUDIO_DURATION_SEC", 600))

    # IMAGE 
    MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", 1024))
    BLIP_MAX_TOKENS = 64
    BLIP_NUM_BEAMS = 3

    # FFMPEG 
    FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
    FFMPEG_TIMEOUT_SEC = int(os.getenv("FFMPEG_TIMEOUT_SEC", 60))

    # RETRIEVAL 
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", 5))
    RAG_DOC_MAX_CHARS = int(os.getenv("RAG_DOC_MAX_CHARS", 500))
    RAG_MAX_TOTAL_CHARS = int(os.getenv("RAG_MAX_TOTAL_CHARS", 3000))

    # HYBRID 
    HYBRID_CANDIDATES_MULTIPLIER = int(
        os.getenv("HYBRID_CANDIDATES_MULTIPLIER", 3)
    )

    # BM25 
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", 5))
    BM25_MAX_DOCS = int(os.getenv("BM25_MAX_DOCS", 10000))
    BM25_MAX_TEXT_CHARS = int(os.getenv("BM25_MAX_TEXT_CHARS", 1000))
    BM25_MAX_TOKENS = int(os.getenv("BM25_MAX_TOKENS", 256))
    BM25_CANDIDATE_MULTIPLIER = int(os.getenv("BM25_CANDIDATE_MULTIPLIER", 3))

    BM25_MODALITY_WEIGHTS = {
        "text": 1.0,
        "image": 1.05,
        "audio": 1.1,
        "video": 1.15,
    }

    # RERANK 
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 5))
    RERANK_MAX_INPUT = int(os.getenv("RERANK_MAX_INPUT", 50))
    RERANK_CONTEXT_MAX_CHARS = int(os.getenv("RERANK_CONTEXT_MAX_CHARS", 512))

    RERANK_MODEL_WEIGHT = float(os.getenv("RERANK_MODEL_WEIGHT", 0.7))
    RERANK_FUSION_WEIGHT = float(os.getenv("RERANK_FUSION_WEIGHT", 0.3))
    RERANK_POSITION_WEIGHT = float(os.getenv("RERANK_POSITION_WEIGHT", 0.1))

    RERANK_MODALITY_WEIGHTS = {
        "text": 1.0,
        "image": 1.1,
        "audio": 1.05,
        "video": 1.15,
    }

    # FUSION 
    FUSION_TOP_K = int(os.getenv("FUSION_TOP_K", 5))
    FUSION_MAX_INPUT = int(os.getenv("FUSION_MAX_INPUT", 100))
    FUSION_SIMILARITY_THRESHOLD = float(os.getenv("FUSION_SIMILARITY_THRESHOLD", 0.85))

    FUSION_SCORE_WEIGHT = float(os.getenv("FUSION_SCORE_WEIGHT", 0.6))
    FUSION_LENGTH_WEIGHT = float(os.getenv("FUSION_LENGTH_WEIGHT", 0.2))
    FUSION_MODALITY_WEIGHT = float(os.getenv("FUSION_MODALITY_WEIGHT", 0.2))

    FUSION_MAX_TEXT_CHARS = int(os.getenv("FUSION_MAX_TEXT_CHARS", 500))
    FUSION_HASH_CHARS = int(os.getenv("FUSION_HASH_CHARS", 200))
    FUSION_NORMALIZATION_EPS = float(os.getenv("FUSION_NORMALIZATION_EPS", 1e-6))

    FUSION_MODALITY_WEIGHTS = {
        "text": 1.0,
        "image": 0.9,
        "audio": 1.1,
        "video": 1.15,
    }

    # MEMORY 
    MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", 20))
    MAX_SYSTEM_MESSAGES = int(os.getenv("MAX_SYSTEM_MESSAGES", 5))

    MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", 5))
    MEMORY_SIM_THRESHOLD = float(os.getenv("MEMORY_SIM_THRESHOLD", 0.35))
    MEMORY_RECENCY_SCALE = int(os.getenv("MEMORY_RECENCY_SCALE", 3600))

    MEMORY_MAX_CONTEXT_CHARS = int(os.getenv("MEMORY_MAX_CONTEXT_CHARS", 4000))
    MEMORY_HISTORY_MAX_CHARS = int(os.getenv("MEMORY_HISTORY_MAX_CHARS", 2000))
    MEMORY_SUMMARY_MAX_CHARS = int(os.getenv("MEMORY_SUMMARY_MAX_CHARS", 1500))
    MEMORY_SUMMARY_INPUT_CHARS = int(os.getenv("MEMORY_SUMMARY_INPUT_CHARS", 4000))
    MEMORY_SUMMARY_THRESHOLD = int(os.getenv("MEMORY_SUMMARY_THRESHOLD", 10))
    MEMORY_MIN_MESSAGES_FOR_SUMMARY = int(os.getenv("MEMORY_MIN_MESSAGES_FOR_SUMMARY", 5))

    MEMORY_ROLE_WEIGHTS = {
        "user": 1.3,
        "assistant": 1.0,
        "system": 1.2,
    }

    # Compatibility aliases
    MEMORY_MAX_HISTORY = MAX_HISTORY_MESSAGES

    # REDIS 
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_TIMEOUT = int(os.getenv("REDIS_TIMEOUT", 5))
    REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", 86400))
    REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "chat")

    # MONGO 
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "rag_memory")
    DB_TIMEOUT_MS = int(os.getenv("DB_TIMEOUT_MS", 5000))
    DB_MAX_POOL_SIZE = int(os.getenv("DB_MAX_POOL_SIZE", 50))

    # QDRANT 
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", 10))

    QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", 64))
    QDRANT_MAX_DOCS = int(os.getenv("QDRANT_MAX_DOCS", 10000))
    QDRANT_TEXT_MAX_CHARS = int(os.getenv("QDRANT_TEXT_MAX_CHARS", 1000))

    TEXT_COLLECTION_NAME = os.getenv("TEXT_COLLECTION_NAME", "text_collection")
    VISION_COLLECTION_NAME = os.getenv("VISION_COLLECTION_NAME", "vision_collection")

    # WEB SEARCH 
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", 7))
    WEB_MAX_DOCS = int(os.getenv("WEB_MAX_DOCS", 5))
    WEB_DOC_MAX_CHARS = int(os.getenv("WEB_DOC_MAX_CHARS", 400))
    WEB_CONTEXT_MAX_CHARS = int(os.getenv("WEB_CONTEXT_MAX_CHARS", 2000))
    WEB_SEARCH_DEPTH = os.getenv("WEB_SEARCH_DEPTH", "advanced")

    # QUERY DECOMPOSITION 
    MAX_SUBQUERIES = int(os.getenv("MAX_SUBQUERIES", 4))
    DECOMPOSITION_MIN_WORDS = int(os.getenv("DECOMPOSITION_MIN_WORDS", 12))

    # AGENT 
    AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", 3))
    AGENT_TIMEOUT_SEC = int(os.getenv("AGENT_TIMEOUT_SEC", 30))

    # STREAM 
    STREAM_CHUNK_SIZE = int(os.getenv("STREAM_CHUNK_SIZE", 50))

    # PROMPT 
    CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", 2000))
    MEMORY_MAX_CONTEXT = MEMORY_MAX_CONTEXT_CHARS

    # VALIDATION 
    def validate(self):

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be < CHUNK_SIZE")

        if self.TEXT_EMBEDDING_DIM <= 0:
            raise ValueError("Invalid TEXT_EMBEDDING_DIM")

        if self.MAX_PROMPT_CHARS <= 0:
            raise ValueError("Invalid MAX_PROMPT_CHARS")

        if self.RAG_TOP_K <= 0:
            raise ValueError("Invalid RAG_TOP_K")
        

    # GLOBAL COMPATIBILITY 
    # Retrieval / Fusion
    SIMILARITY_THRESHOLD = FUSION_SIMILARITY_THRESHOLD

    # Top-k alignment
    DEFAULT_TOP_K = RAG_TOP_K

    # Context alignment
    CONTEXT_MAX_CHARS = CONTEXT_MAX_CHARS
    MEMORY_MAX_CONTEXT = MEMORY_MAX_CONTEXT_CHARS

    # Memory Compatibility
    MEMORY_MAX_HISTORY = MAX_HISTORY_MESSAGES


settings = Settings()
settings.validate()