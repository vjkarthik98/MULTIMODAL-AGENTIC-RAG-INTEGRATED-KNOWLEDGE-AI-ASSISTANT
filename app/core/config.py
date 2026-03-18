import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:

    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

    COLLECTION_NAME = os.getenv("COLLETION_NAME", "multimodal_rag")

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

settings = Settings()