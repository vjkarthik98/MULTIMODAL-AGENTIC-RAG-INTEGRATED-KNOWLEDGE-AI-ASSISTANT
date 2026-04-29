import time

from app.vectorstore.qdrant_store import QdrantVectorStore
from app.core.config import settings
from app.utils.logger import get_logger
from app.core.infra_registry import infra


logger = get_logger(__name__)


class QdrantInitializer:

    def __init__(self):
        self.store = infra.get_vector_store()

        self.recreate_allowed = getattr(
            settings,
            "QDRANT_ALLOW_RECREATE",
            False
        )

        self.max_retries = getattr(
            settings,
            "QDRANT_INIT_RETRIES",
            2
        )

    #  MAIN 
    def initialize(self):

        start = time.time()

        logger.info("[QdrantInit] starting initialization")

        for attempt in range(self.max_retries + 1):

            try:
                self._ensure_collection(
                    settings.TEXT_COLLECTION_NAME,
                    settings.TEXT_EMBEDDING_DIM,
                    "text"
                )

                self._ensure_collection(
                    settings.VISION_COLLECTION_NAME,
                    settings.VISION_EMBEDDING_DIM,
                    "vision"
                )

                logger.info(
                    "[QdrantInit] success | latency=%.2fs",
                    time.time() - start
                )

                return True

            except Exception as e:

                logger.warning(
                    "[QdrantInit] attempt %s failed | %s",
                    attempt,
                    str(e)
                )

                if attempt >= self.max_retries:
                    logger.error("[QdrantInit] failed permanently")
                    raise

                time.sleep(1)

    #  ENSURE COLLECTION 
    def _ensure_collection(self, name: str, dim: int, label: str):

        logger.info("[QdrantInit] checking collection=%s", name)

        try:
            info = self.store.client.get_collection(name)
            existing_dim = (
                info.config.params.vectors.size
            )

            if existing_dim != dim:

                logger.warning(
                    "[QdrantInit] dimension mismatch | %s existing=%s expected=%s",
                    name,
                    existing_dim,
                    dim
                )

                if self.recreate_allowed:

                    logger.warning(
                        "[QdrantInit] recreating collection=%s",
                        name
                    )

                    self.store.client.delete_collection(name)

                    self.store._ensure_collection(name, dim)

                else:
                    raise ValueError(
                        f"Dimension mismatch for {name}"
                    )

            else:
                logger.info(
                    "[QdrantInit] collection OK | %s dim=%s",
                    name,
                    dim
                )

        except Exception:

            # COLLECTION DOES NOT EXIST → CREATE
            logger.info(
                "[QdrantInit] creating collection=%s dim=%s",
                name,
                dim
            )

            self.store._ensure_collection(name, dim)


#  ENTRYPOINT 
def initialize_qdrant():
    initializer = QdrantInitializer()
    return initializer.initialize()


if __name__ == "__main__":
    initialize_qdrant()