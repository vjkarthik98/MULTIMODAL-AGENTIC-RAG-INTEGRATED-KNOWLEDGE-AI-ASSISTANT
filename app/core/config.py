from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:

    # CORE PATHS
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data")).resolve()
    LOG_DIR = Path(os.getenv("LOG_DIR", PROJECT_ROOT / "logs")).resolve()

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ENVIRONMENT
    ENV = os.getenv("ENV", "development")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"


    # QDRANT CONFIG
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    TEXT_COLLECTION_NAME = os.getenv("QDRANT_TEXT_COLLECTION", "text_collection")
    VISION_COLLECTION_NAME = os.getenv("QDRANT_VISION_COLLECTION", "vision_collection")


    # MODEL CONGIF
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

    BLIP_MODEL = os.getenv(
        "BLIP_MODEL",
        "Salesforce/blip-image-captioning-large"
    )

    CLIP_MODEL = os.getenv(
        "CLIP_MODEL",
        "openai/clip-vit-base-patch32"
    )

    RERANKER_MODEL = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # MULTIMODAL PROCESSING
    FFMPEG_PATH = os.getenv(
        "FFMPEG_PATH",
        str(PROJECT_ROOT / "ffmpeg.exe")
    )

    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

    FRAME_INTERVAL_SECONDS = int(
        os.getenv("FRAME_INTERVAL_SECONDS", 2)
    )

    # RETRIEVAL + PIPELINE TUNING
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
    MAX_TOP_K = int(os.getenv("MAX_TOP_K", 10))

    FUSION_TOP_K = int(os.getenv("FUSION_TOP_K", 6))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 5))

    MEMORY_MAX_HISTORY = int(os.getenv("MEMORY_MAX_HISTORY", 20))
    MEMORY_SUMMARY_THRESHOLD = int(os.getenv("MEMORY_SUMMARY_THRESHOLD", 6))

    CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKEN", 2000))


    # LLM GENERATION CONTROL
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERTURE", 0.2))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 512))


    # LOGGING
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    # VALIDATION
    def validate(self):
        errors = []

        if self.QDRANT_URL is None and not self.QDRANT_HOST:
            errors.append("QDRANT config missing (URL or HOST required)")

        if not self.EMBEDDING_MODEL:
            errors.append("EMBEDDING_MODEL not set")

        if self.FRAME_INTERVAL_SECONDS <=0:
            errors.append("FRAME_INTERVAL_SECONDS must be > 0")

        if errors:
            raise ValueError(f"[CONFIG VALIDATION FAILED]: {errors}")
        
# singleton
settings = Settings()

# Run Validation at import 
settings.validate()

