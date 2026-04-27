from app.vectorstore.qdrant_store import QdrantVectorStore
from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


def initialize_qdrant():
    store = QdrantVectorStore()

    try:
        # Text collection
        store._ensure_collection(
            settings.TEXT_COLLECTION_NAME,
            settings.TEXT_EMBEDDING_DIM
        )

        # Vision collection
        store._ensure_collection(
            settings.VISION_COLLECTION_NAME,
            settings.VISION_EMBEDDING_DIM
        )

        logger.info(
            "[QdrantInit] collections ready | text=%s vision=%s",
            settings.TEXT_COLLECTION_NAME,
            settings.VISION_COLLECTION_NAME
        )

    except Exception as e:
        logger.error("[QdrantInit] failed | %s", str(e))
        raise


if __name__ == "__main__":
    initialize_qdrant()