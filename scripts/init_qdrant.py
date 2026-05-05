import time

from app.core.config import settings
from app.utils.logger import get_logger
from app.core.infra_registry import infra

logger = get_logger(__name__)


class QdrantInitializer:

    def __init__(self):
        self.store = infra.get_vector_store()

        self.recreate_allowed = getattr(settings, "QDRANT_ALLOW_RECREATE", False)
        self.max_retries = getattr(settings, "QDRANT_INIT_RETRIES", 2)
        self.retry_delay = getattr(settings, "QDRANT_RETRY_DELAY", 1)

    # CONNECTION
    def _ping(self):
        try:
            self.store.client.get_collections()
        except Exception as e:
            logger.error(event="qdrant_connection_failed", error=str(e))
            raise

    # MAIN
    def initialize(self):

        start = time.time()

        logger.info(event="qdrant_init_start")

        self._ping()

        for attempt in range(self.max_retries + 1):

            try:
                self._ensure(
                    settings.TEXT_COLLECTION_NAME,
                    settings.TEXT_EMBEDDING_DIM,
                )

                self._ensure(
                    settings.VISION_COLLECTION_NAME,
                    settings.VISION_EMBEDDING_DIM,
                )

                logger.info(
                    event="qdrant_init_success",
                    latency=round(time.time() - start, 2)
                )

                return True

            except Exception as e:

                logger.warning(
                    event="qdrant_init_retry",
                    attempt=attempt,
                    error=str(e)
                )

                if attempt >= self.max_retries:
                    logger.error(event="qdrant_init_failed")
                    raise

                time.sleep(self.retry_delay)

    # ENSURE
    def _ensure(self, name: str, dim: int):

        try:
            info = self.store.client.get_collection(name)
            existing_dim = info.config.params.vectors.size

            if existing_dim != dim:

                logger.warning(
                    event="qdrant_dim_mismatch",
                    collection=name,
                    existing=existing_dim,
                    expected=dim
                )

                if not self.recreate_allowed:
                    raise ValueError(f"DIMENSION_MISMATCH_{name}")

                self._recreate(name, dim)

            else:
                logger.info(
                    event="qdrant_collection_ok",
                    collection=name,
                    dim=dim
                )

        except Exception:
            logger.info(
                event="qdrant_create_collection",
                collection=name,
                dim=dim
            )

            self.store._ensure_collection(name, dim)

    # RECREATE
    def _recreate(self, name: str, dim: int):

        logger.warning(
            event="qdrant_recreate_collection",
            collection=name
        )

        self.store.client.delete_collection(name)
        self.store._ensure_collection(name, dim)


# ENTRY
def initialize_qdrant():
    return QdrantInitializer().initialize()


if __name__ == "__main__":
    initialize_qdrant()