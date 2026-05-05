import uuid
import time
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.utils.logger import get_logger

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = get_logger(__name__)


class QdrantVectorStore:

    def __init__(self):

        self.client = (
            QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=settings.QDRANT_TIMEOUT
            )
            if settings.QDRANT_URL
            else QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=settings.QDRANT_TIMEOUT
            )
        )

        self.text_collection = settings.TEXT_COLLECTION_NAME
        self.vision_collection = settings.VISION_COLLECTION_NAME

        self.batch_size = settings.QDRANT_BATCH_SIZE
        self.max_docs = settings.QDRANT_MAX_DOCS

        self.text_dim = settings.TEXT_EMBEDDING_DIM
        self.vision_dim = settings.VISION_EMBEDDING_DIM

        self._collection_cache = set()
        self.modality_filter: Optional[str] = None

        logger.info(event="qdrant_initialized")

    #  RETRY 
    def _retry(self, fn, retries=3):
        for i in range(retries):
            try:
                return fn()
            except Exception as e:
                if i == retries - 1:
                    raise
                logger.warning(event="qdrant_retry", error=str(e))
                time.sleep(0.5 * (i + 1))

    #  COLLECTION 
    def _collection_exists(self, name: str) -> bool:
        try:
            return any(c.name == name for c in self.client.get_collections().collections)
        except Exception:
            return False

    def _ensure_collection(self, name: str, dim: int):

        if name in self._collection_cache:
            return

        if not self._collection_exists(name):
            logger.warning(event="create_collection", name=name)

            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        self._collection_cache.add(name)

    #  PAYLOAD 
    def _payload(self, d) -> Dict[str, Any]:

        s = dict(d.structure or {})

        return {
            "text": str(d.text or "")[:settings.QDRANT_TEXT_MAX_CHARS],
            "doc_id": s.get("doc_id"),
            "chunk_id": d.chunk_id,
            "modality": d.modality,
            "content_type": s.get("content_type"),
            "session_id": s.get("session_id"),
            "embedding_space": s.get("embedding_space", "text"),
            "source": d.source,
        }

    #  INSERT 
    def insert_documents(self, documents: List):

        if not documents:
            return

        start = time.time()
        documents = documents[:self.max_docs]

        text_points, vision_points = [], []

        for d in documents:
            emb = getattr(d, "embedding", None)
            if not isinstance(emb, list):
                continue

            space = (d.structure or {}).get("embedding_space", "text")

            if space == "vision":
                if len(emb) != self.vision_dim:
                    continue
                collection = "vision"
            else:
                if len(emb) != self.text_dim:
                    continue
                collection = "text"

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload=self._payload(d),
            )

            if collection == "vision":
                vision_points.append(point)
            else:
                text_points.append(point)

        def _insert(collection_name, points):
            for i in range(0, len(points), self.batch_size):
                batch = points[i:i + self.batch_size]

                self._retry(lambda: self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                ))

        if text_points:
            self._ensure_collection(self.text_collection, self.text_dim)
            _insert(self.text_collection, text_points)

        if vision_points:
            self._ensure_collection(self.vision_collection, self.vision_dim)
            _insert(self.vision_collection, vision_points)

        logger.info(
            event="qdrant_insert",
            text=len(text_points),
            vision=len(vision_points),
            latency=round(time.time() - start, 2)
        )

    #  FILTER 
    def _filter(self, session_id=None):

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

    #  SEARCH 
    def _search(self, collection, vector, limit, session_id):

        if collection not in self._collection_cache:
            return []

        try:
            res = self._retry(lambda: self.client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                query_filter=self._filter(session_id)
            ))

            points = getattr(res, "points", [])

            return [
                {
                    "text": p.payload.get("text"),
                    "score": float(p.score),
                    "metadata": p.payload,
                }
                for p in points if p.payload.get("text")
            ]

        except Exception as e:
            logger.error(event="qdrant_search_failed", error=str(e))
            return []

    #  PUBLIC 
    def search_text(self, query_vector, limit=None, session_id=None):
        return self._search(
            self.text_collection,
            query_vector,
            limit or settings.RAG_TOP_K,
            session_id
        )

    def search_vision(self, query_vector, limit=None, session_id=None):
        return self._search(
            self.vision_collection,
            query_vector,
            limit or settings.RAG_TOP_K,
            session_id
        )

    def set_modality_filter(self, modality: str):
        self.modality_filter = modality