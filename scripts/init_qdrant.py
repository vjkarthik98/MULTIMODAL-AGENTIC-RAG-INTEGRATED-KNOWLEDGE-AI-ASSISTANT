from __future__ import annotations

import time
from typing import Any, Dict, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# QDRANT CLIENT IMPORTS

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        HnswConfigDiff,
        OptimizersConfigDiff,
        PayloadSchemaType,
        VectorParams,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning(event="qdrant_client_not_installed", hint="pip install qdrant-client")


# COLLECTION SCHEMA — PHASE 24 UNIVERSAL METADATA PAYLOAD FIELDS

PAYLOAD_INDEXES: Dict[str, Any] = {
    "session_id":        "keyword",
    "user_id":           "keyword",
    "modality":          "keyword",
    "subtype":           "keyword",
    "source_type":       "keyword",
    "source":            "keyword",
    "doc_id":            "keyword",
    "content_type":      "keyword",
    "embedding_space":   "keyword",
    "language":          "keyword",
    "checksum_sha256":   "keyword",
    "chunk_id":          "integer",
    "page":              "integer",
    "ingestion_time":    "float",
    "deleted":           "bool",
    "section_number":    "integer",
    "is_forward_looking": "bool",
}

PAYLOAD_SCHEMA_TYPE_MAP: Dict[str, Any] = {}

if QDRANT_AVAILABLE:
    PAYLOAD_SCHEMA_TYPE_MAP = {
        "keyword": PayloadSchemaType.KEYWORD,
        "integer": PayloadSchemaType.INTEGER,
        "float":   PayloadSchemaType.FLOAT,
        "bool":    PayloadSchemaType.BOOL,
    }


# COLLECTION DEFINITION

class CollectionSpec:
    """Specification for a single Qdrant collection."""

    def __init__(
        self,
        name: str,
        dim: int,
        description: str,
    ) -> None:
        self.name        = name
        self.dim         = dim
        self.description = description


# QDRANT INITIALIZER

class QdrantInitializer:
    """
    Initializes and validates Qdrant collections on application startup.

    Responsibilities:
      - Ping Qdrant to confirm connectivity.
      - Create collections if they do not exist.
      - Detect and handle dimension mismatches.
      - Create filterable payload indexes for all Phase 24 metadata fields.
      - Retry with exponential back-off on transient failures.
      - Graceful degradation: log warnings, never crash the app.
    """

    def __init__(self) -> None:
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "QDRANT_CLIENT_NOT_INSTALLED: pip install qdrant-client"
            )

        self.store = self._resolve_vector_store()

        if not self.store:
            raise RuntimeError(
                "QDRANT_UNAVAILABLE: Vector store could not be initialized. "
                "Check QDRANT_URL or QDRANT_HOST in your .env file."
            )

        self.client: QdrantClient = self.store.client

        self.recreate_allowed = settings.QDRANT_ALLOW_RECREATE
        self.max_retries      = settings.QDRANT_INIT_RETRIES
        self.retry_delay      = settings.QDRANT_RETRY_DELAY

        # COLLECTIONS TO INITIALIZE
        self.collections: list[CollectionSpec] = [
            CollectionSpec(
                name        = settings.TEXT_COLLECTION_NAME,
                dim         = settings.TEXT_EMBEDDING_DIM,
                description = "Dense text + audio + table embeddings (sentence-transformers)",
            ),
            CollectionSpec(
                name        = settings.VISION_COLLECTION_NAME,
                dim         = settings.VISION_EMBEDDING_DIM,
                description = "SigLIP visual embeddings (images + video frames)",
            ),
        ]

    # RESOLVE VECTOR STORE FROM INFRA REGISTRY

    def _resolve_vector_store(self) -> Optional[Any]:
        try:
            from app.core.infra_registry import infra
            return infra.get_vector_store()
        except Exception as exc:
            logger.error(event="qdrant_store_resolution_failed", error=str(exc))
            return None

    # PING

    def _ping(self) -> None:
        try:
            self.client.get_collections()
            logger.info(event="qdrant_ping_ok")
        except Exception as exc:
            logger.error(event="qdrant_ping_failed", error=str(exc))
            raise

    # COLLECTION EXISTS

    def _collection_exists(self, name: str) -> bool:
        try:
            return any(
                c.name == name
                for c in self.client.get_collections().collections
            )
        except Exception:
            return False

    # GET COLLECTION DIM

    def _get_existing_dim(self, name: str) -> Optional[int]:
        try:
            info = self.client.get_collection(name)
            return info.config.params.vectors.size
        except Exception:
            return None

    # GET COLLECTION POINT COUNT

    def _get_point_count(self, name: str) -> int:
        try:
            info = self.client.get_collection(name)
            return getattr(info, "points_count", 0) or 0
        except Exception:
            return 0

    # CREATE COLLECTION

    def _create_collection(self, spec: CollectionSpec) -> None:
        logger.info(
            event="qdrant_collection_creating",
            collection=spec.name,
            dim=spec.dim,
            description=spec.description,
        )

        self.client.create_collection(
            collection_name=spec.name,
            vectors_config=VectorParams(
                size=spec.dim,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=settings.QDRANT_HNSW_M if hasattr(settings, "QDRANT_HNSW_M") else 16,
                ef_construct=settings.QDRANT_HNSW_EF if hasattr(settings, "QDRANT_HNSW_EF") else 128,
                full_scan_threshold=10000,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000,
            ),
        )

        logger.info(
            event="qdrant_collection_created",
            collection=spec.name,
            dim=spec.dim,
        )

    # RECREATE COLLECTION

    def _recreate_collection(self, spec: CollectionSpec) -> None:
        logger.warning(
            event="qdrant_recreating_collection",
            collection=spec.name,
            reason="dimension_mismatch",
        )

        try:
            self.client.delete_collection(spec.name)
            logger.info(event="qdrant_collection_deleted", collection=spec.name)
        except Exception as exc:
            logger.warning(
                event="qdrant_delete_failed",
                collection=spec.name,
                error=str(exc),
            )

        self._create_collection(spec)

    # CREATE PAYLOAD INDEXES

    def _create_payload_indexes(self, collection_name: str) -> None:
        """
        Create filterable payload indexes for all Phase 24 metadata fields.
        Idempotent — skips fields that already have an index.
        """
        created  = 0
        skipped  = 0
        failed   = 0

        for field_name, field_type_str in PAYLOAD_INDEXES.items():
            schema_type = PAYLOAD_SCHEMA_TYPE_MAP.get(field_type_str)
            if schema_type is None:
                logger.warning(
                    event="qdrant_unknown_payload_type",
                    field=field_name,
                    type=field_type_str,
                )
                skipped += 1
                continue

            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                created += 1

            except Exception as exc:
                err_str = str(exc).lower()
                if "already exists" in err_str or "conflict" in err_str:
                    skipped += 1
                else:
                    failed += 1
                    logger.warning(
                        event="qdrant_payload_index_failed",
                        collection=collection_name,
                        field=field_name,
                        error=str(exc),
                    )

        logger.info(
            event="qdrant_payload_indexes_done",
            collection=collection_name,
            indexes_created=created,
            indexes_skipped=skipped,
            indexes_failed=failed,
        )

    # ENSURE SINGLE COLLECTION

    def _ensure(self, spec: CollectionSpec) -> None:
        """
        Ensure a single collection exists with the correct dimension.

        Flow:
          1. Check if collection exists.
          2. If not → create.
          3. If yes → validate dimension.
          4. Dimension mismatch + recreate_allowed → recreate.
          5. Dimension mismatch + recreate_allowed=False → raise ValueError.
          6. After create/validate → create payload indexes.
        """
        if not self._collection_exists(spec.name):
            self._create_collection(spec)
            self._create_payload_indexes(spec.name)
            return

        # VALIDATE EXISTING DIMENSION
        existing_dim = self._get_existing_dim(spec.name)
        point_count  = self._get_point_count(spec.name)

        if existing_dim is None:
            logger.warning(
                event="qdrant_dim_unreadable",
                collection=spec.name,
                hint="Proceeding without dimension validation",
            )
            self._create_payload_indexes(spec.name)
            return

        if existing_dim != spec.dim:
            logger.warning(
                event="qdrant_dim_mismatch",
                collection=spec.name,
                existing_dim=existing_dim,
                expected_dim=spec.dim,
                point_count=point_count,
                hint="Set QDRANT_ALLOW_RECREATE=true to auto-recreate (DATA LOSS)",
            )

            if not self.recreate_allowed:
                raise ValueError(
                    f"DIMENSION_MISMATCH: collection '{spec.name}' "
                    f"has dim={existing_dim}, expected {spec.dim}. "
                    f"Set QDRANT_ALLOW_RECREATE=true to auto-recreate (DATA LOSS)."
                )

            self._recreate_collection(spec)
            self._create_payload_indexes(spec.name)
            return

        # COLLECTION OK — ENSURE PAYLOAD INDEXES
        logger.info(
            event="qdrant_collection_ok",
            collection=spec.name,
            dim=existing_dim,
            points=point_count,
        )
        self._create_payload_indexes(spec.name)

    # MAIN INITIALIZE

    def initialize(self) -> bool:
        """
        Initialize all Qdrant collections with retries.

        Returns:
            True on success.

        Raises:
            RuntimeError: if all retries are exhausted.
        """
        start = time.time()
        logger.info(event="qdrant_init_start")

        # CONNECTIVITY CHECK
        self._ping()

        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                for spec in self.collections:
                    t_col = time.time()
                    self._ensure(spec)
                    logger.info(
                        event="qdrant_collection_initialized",
                        collection=spec.name,
                        dim=spec.dim,
                        latency=round(time.time() - t_col, 2),
                    )

                # WARM COLLECTION CACHE IN VECTOR STORE
                self._warm_collection_cache()

                logger.info(
                    event="qdrant_init_success",
                    collections=[c.name for c in self.collections],
                    total_latency=round(time.time() - start, 2),
                )
                return True

            except ValueError:
                # DIMENSION MISMATCH WITHOUT RECREATE — DO NOT RETRY
                raise

            except Exception as exc:
                last_exc = exc
                backoff  = self.retry_delay * (attempt + 1)

                logger.warning(
                    event="qdrant_init_retry",
                    attempt=attempt,
                    max_retries=self.max_retries,
                    backoff_sec=backoff,
                    error=str(exc),
                )

                if attempt >= self.max_retries:
                    logger.error(
                        event="qdrant_init_failed",
                        error=str(last_exc),
                        total_latency=round(time.time() - start, 2),
                    )
                    raise RuntimeError(
                        f"QDRANT_INIT_FAILED after {self.max_retries + 1} attempts: {last_exc}"
                    )

                time.sleep(backoff)

        return False

    # WARM COLLECTION CACHE IN VECTOR STORE SINGLETON

    def _warm_collection_cache(self) -> None:
        """
        Populate the QdrantVectorStore._collection_cache set
        so search calls skip the exists() check on hot path.
        """
        try:
            from app.core.infra_registry import infra
            store = infra.get_vector_store()
            if store and hasattr(store, "_collection_cache"):
                for spec in self.collections:
                    store._collection_cache.add(spec.name)
                logger.info(
                    event="qdrant_collection_cache_warmed",
                    collections=[c.name for c in self.collections],
                )
        except Exception as exc:
            logger.warning(event="qdrant_cache_warm_failed", error=str(exc))

    # COLLECTION STATS

    def stats(self) -> Dict[str, Any]:
        """Return point counts and status for all managed collections."""
        result: Dict[str, Any] = {}

        for spec in self.collections:
            try:
                info = self.client.get_collection(spec.name)
                result[spec.name] = {
                    "points_count":  getattr(info, "points_count", 0),
                    "vectors_count": getattr(info, "vectors_count", 0),
                    "status":        str(getattr(info, "status", "unknown")),
                    "dim":           spec.dim,
                }
            except Exception as exc:
                result[spec.name] = {"error": str(exc)}

        return result

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        """Quick connectivity + collection existence check."""
        ok       = True
        details: Dict[str, Any] = {}

        try:
            self._ping()
            details["ping"] = "ok"
        except Exception as exc:
            ok = False
            details["ping"] = str(exc)

        for spec in self.collections:
            exists = self._collection_exists(spec.name)
            details[spec.name] = {
                "exists": exists,
                "dim":    spec.dim,
            }
            if not exists:
                ok = False

        return {
            "ok":         ok,
            "details":    details,
            "collections": [c.name for c in self.collections],
        }


# ENTRY POINT — CALLED FROM main.py LIFESPAN AND scripts/

def initialize_qdrant() -> bool:
    """
    Public entry point for Qdrant initialization.

    Called during:
      - Application startup lifespan (main.py)
      - CLI: python scripts/init_qdrant.py

    Returns True on success, raises RuntimeError on failure.
    """
    return QdrantInitializer().initialize()


if __name__ == "__main__":
    initialize_qdrant()