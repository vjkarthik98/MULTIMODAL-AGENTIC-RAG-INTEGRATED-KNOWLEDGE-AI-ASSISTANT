import math
import time
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantVectorStore:

    def __init__(self) -> None:

        self.client = (
            QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=settings.QDRANT_TIMEOUT,
            )
            if settings.QDRANT_URL
            else QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=settings.QDRANT_TIMEOUT,
            )
        )

        self.text_collection   = settings.TEXT_COLLECTION_NAME
        self.vision_collection = settings.VISION_COLLECTION_NAME

        self.batch_size = settings.QDRANT_BATCH_SIZE
        self.max_docs   = settings.QDRANT_MAX_DOCS
        self.text_dim   = settings.TEXT_EMBEDDING_DIM
        self.vision_dim = settings.VISION_EMBEDDING_DIM

        self._collection_cache: set        = set()
        self.modality_filter: Optional[str] = None

        logger.info(event="qdrant_initialized")

    # RETRY

    def _retry(self, fn, retries: int = 3):
        for i in range(retries):
            try:
                return fn()
            except Exception as e:
                if i == retries - 1:
                    raise
                logger.warning(event="qdrant_retry", attempt=i + 1, error=str(e))
                time.sleep(0.5 * (i + 1))

    # COLLECTION

    def _collection_exists(self, name: str) -> bool:
        try:
            return any(
                c.name == name
                for c in self.client.get_collections().collections
            )
        except Exception:
            return False

    def _ensure_collection(self, name: str, dim: int) -> None:
        if name in self._collection_cache:
            return

        if not self._collection_exists(name):
            logger.info(event="qdrant_create_collection", name=name, dim=dim)
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        self._collection_cache.add(name)

    # EMBEDDING VALIDATION

    def _valid_vector(self, emb: List[float], expected_dim: int) -> bool:
        if not isinstance(emb, list):
            return False
        if len(emb) != expected_dim:
            return False
        if any(math.isnan(v) or math.isinf(v) for v in emb):
            return False
        return True

    # PAYLOAD

    def _payload(self, d) -> Dict[str, Any]:
        s = dict(d.structure or {})

        return {
            "text":            str(d.text or "")[:settings.QDRANT_TEXT_MAX_CHARS],
            "doc_id":          s.get("doc_id"),
            "chunk_id":        d.chunk_id,
            "modality":        d.modality,
            "subtype":         getattr(d, "subtype", None),
            "content_type":    s.get("content_type"),
            "session_id":      s.get("session_id"),
            "embedding_space": s.get("embedding_space", "text"),
            "source":          str(d.source or "")[:200],
            "source_type":     getattr(d, "source_type", None),
            "page":            getattr(d, "page", None),
        }

    # INSERT

    def insert_documents(self, documents: List, session_id: str = "") -> None:

        if not documents:
            return

        start     = time.time()
        documents = documents[:self.max_docs]

        text_points:   List[PointStruct] = []
        vision_points: List[PointStruct] = []
        skipped = 0

        for d in documents:
            emb   = getattr(d, "embedding", None)
            space = (d.structure or {}).get("embedding_space", "text")

            if space == "vision":
                if not self._valid_vector(emb, self.vision_dim):
                    skipped += 1
                    continue
                vision_points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload=self._payload(d),
                    )
                )
            else:
                if not self._valid_vector(emb, self.text_dim):
                    skipped += 1
                    continue
                text_points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload=self._payload(d),
                    )
                )

        def _insert(collection_name: str, points: List[PointStruct]) -> None:
            for i in range(0, len(points), self.batch_size):
                batch = points[i:i + self.batch_size]
                self._retry(
                    lambda b=batch: self.client.upsert(
                        collection_name=collection_name,
                        points=b,
                    )
                )

        if text_points:
            self._ensure_collection(self.text_collection, self.text_dim)
            _insert(self.text_collection, text_points)

        if vision_points:
            self._ensure_collection(self.vision_collection, self.vision_dim)
            _insert(self.vision_collection, vision_points)

        total     = len(text_points) + len(vision_points)
        latency   = round(time.time() - start, 2)
        throughput = round(total / max(latency, 1e-6), 1)

        logger.info(
            event="qdrant_insert_success",
            text=len(text_points),
            vision=len(vision_points),
            skipped=skipped,
            throughput_per_sec=throughput,
            latency=latency,
            session_id=session_id,
        )

    # FILTER

    def _build_filter(self, session_id: Optional[str] = None) -> Optional[Filter]:
        conditions = []

        if session_id:
            conditions.append(
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            )

        if self.modality_filter:
            conditions.append(
                FieldCondition(key="modality", match=MatchValue(value=self.modality_filter))
            )

        return Filter(must=conditions) if conditions else None

    # SEARCH

    def _search(
        self,
        collection: str,
        vector: List[float],
        limit: int,
        session_id: Optional[str],
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:

        if collection not in self._collection_cache:
            logger.warning(
                event="qdrant_search_collection_not_ready",
                collection=collection,
                session_id=session_id,
            )
            return []

        start = time.time()

        try:
            res    = self._retry(
                lambda: self.client.query_points(
                    collection_name=collection,
                    query=vector,
                    limit=limit,
                    query_filter=self._build_filter(session_id),
                    score_threshold=score_threshold if score_threshold > 0 else None,
                )
            )
            points = getattr(res, "points", [])

            results = [
                {
                    "text":     p.payload.get("text"),
                    "score":    float(p.score),
                    "metadata": p.payload,
                }
                for p in points
                if p.payload.get("text")
            ]

            logger.debug(
                event="qdrant_search_success",
                collection=collection,
                results=len(results),
                latency=round(time.time() - start, 3),
                session_id=session_id,
            )

            return results

        except Exception as e:
            logger.error(
                event="qdrant_search_failed",
                collection=collection,
                session_id=session_id,
                error=str(e),
            )
            return []

    # PUBLIC SEARCH

    def search_text(
        self,
        query_vector: List[float],
        limit: int = None,
        session_id: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        return self._search(
            self.text_collection,
            query_vector,
            limit or settings.RAG_TOP_K,
            session_id,
            score_threshold,
        )

    def search_vision(
        self,
        query_vector: List[float],
        limit: int = None,
        session_id: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        return self._search(
            self.vision_collection,
            query_vector,
            limit or settings.RAG_TOP_K,
            session_id,
            score_threshold,
        )

    # MODALITY FILTER

    def set_modality_filter(self, modality: Optional[str]) -> None:
        self.modality_filter = modality

    # DELETE BY SESSION

    def delete_by_session(self, session_id: str) -> None:
        if not session_id:
            return

        for collection in (self.text_collection, self.vision_collection):
            if collection not in self._collection_cache:
                continue
            try:
                self._retry(
                    lambda c=collection: self.client.delete(
                        collection_name=c,
                        points_selector=Filter(
                            must=[
                                FieldCondition(
                                    key="session_id",
                                    match=MatchValue(value=session_id),
                                )
                            ]
                        ),
                    )
                )
                logger.info(
                    event="qdrant_session_deleted",
                    collection=collection,
                    session_id=session_id,
                )
            except Exception as e:
                logger.error(
                    event="qdrant_delete_failed",
                    collection=collection,
                    session_id=session_id,
                    error=str(e),
                )

    # COLLECTION STATS

    def collection_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}

        for name in (self.text_collection, self.vision_collection):
            try:
                info         = self.client.get_collection(name)
                stats[name]  = {
                    "points_count":  info.points_count,
                    "vectors_count": info.vectors_count,
                    "status":        str(info.status),
                }
            except Exception as e:
                stats[name] = {"error": str(e)}

        return stats