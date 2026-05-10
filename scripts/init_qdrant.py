import time
from typing import Optional

from app.core.config import settings
from app.core.infra_registry import infra
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantInitializer:

    def __init__(self) -> None:
        self.store = infra.get_vector_store()

        if not self.store:
            raise RuntimeError(
                "QDRANT_UNAVAILABLE: Vector store could not be initialized. "
                "Check QDRANT_URL or QDRANT_HOST in your .env file."
            )

        self.recreate_allowed = settings.QDRANT_ALLOW_RECREATE
        self.max_retries      = settings.QDRANT_INIT_RETRIES
        self.retry_delay      = settings.QDRANT_RETRY_DELAY

    # CONNECTION

    def _ping(self) -> None:
        try:
            self.store.client.get_collections()
            logger.info(event="qdrant_ping_ok")
        except Exception as e:
            logger.error(event="qdrant_ping_failed", error=str(e))
            raise

    # MAIN

    def initialize(self) -> bool:
        start = time.time()

        logger.info(event="qdrant_init_start")

        self._ping()

        for attempt in range(self.max_retries + 1):
            try:
                t_text = time.time()
                self._ensure(settings.TEXT_COLLECTION_NAME, settings.TEXT_EMBEDDING_DIM)
                text_latency = round(time.time() - t_text, 2)

                t_vis = time.time()
                self._ensure(settings.VISION_COLLECTION_NAME, settings.VISION_EMBEDDING_DIM)
                vis_latency = round(time.time() - t_vis, 2)

                logger.info(
                    event="qdrant_init_success",
                    text_collection=settings.TEXT_COLLECTION_NAME,
                    vision_collection=settings.VISION_COLLECTION_NAME,
                    text_latency=text_latency,
                    vision_latency=vis_latency,
                    total_latency=round(time.time() - start, 2),
                )

                return True

            except Exception as e:
                logger.warning(
                    event="qdrant_init_retry",
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error=str(e),
                )

                if attempt >= self.max_retries:
                    logger.error(event="qdrant_init_failed", error=str(e))
                    raise

                time.sleep(self.retry_delay * (attempt + 1))

        return False

    # ENSURE COLLECTION

    def _ensure(self, name: str, dim: int) -> None:
        try:
            info         = self.store.client.get_collection(name)
            existing_dim = info.config.params.vectors.size

            if existing_dim != dim:
                logger.warning(
                    event="qdrant_dim_mismatch",
                    collection=name,
                    existing_dim=existing_dim,
                    expected_dim=dim,
                    hint="Set QDRANT_ALLOW_RECREATE=true to auto-recreate (DATA LOSS)",
                )

                if not self.recreate_allowed:
                    raise ValueError(
                        f"DIMENSION_MISMATCH: {name} has dim={existing_dim}, "
                        f"expected {dim}. Set QDRANT_ALLOW_RECREATE=true to fix."
                    )

                self._recreate(name, dim)

            else:
                point_count = getattr(info, "points_count", "unknown")
                logger.info(
                    event="qdrant_collection_ok",
                    collection=name,
                    dim=dim,
                    points=point_count,
                )

        except ValueError:
            raise

        except Exception:
            # COLLECTION DOES NOT EXIST — CREATE IT
            logger.info(
                event="qdrant_collection_creating",
                collection=name,
                dim=dim,
            )
            self.store._ensure_collection(name, dim)

            logger.info(
                event="qdrant_collection_created",
                collection=name,
                dim=dim,
            )

    # RECREATE COLLECTION

    def _recreate(self, name: str, dim: int) -> None:
        logger.warning(
            event="qdrant_recreating_collection",
            collection=name,
            reason="dimension_mismatch",
        )

        try:
            self.store.client.delete_collection(name)
            logger.info(event="qdrant_collection_deleted", collection=name)
        except Exception as e:
            logger.warning(
                event="qdrant_delete_failed",
                collection=name,
                error=str(e),
            )

        self.store._ensure_collection(name, dim)

        logger.info(
            event="qdrant_collection_recreated",
            collection=name,
            dim=dim,
        )


# ENTRY POINT

def initialize_qdrant() -> bool:
    return QdrantInitializer().initialize()


if __name__ == "__main__":
    initialize_qdrant()