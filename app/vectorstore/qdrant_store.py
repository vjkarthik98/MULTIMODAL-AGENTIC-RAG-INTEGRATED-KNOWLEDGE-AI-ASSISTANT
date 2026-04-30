import uuid
import time
from typing import List

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

    # INIT
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

        self._collection_cache = set()
        self.modality_filter = None

        logger.info("[QdrantStore] initialized")

    # RETRY
    def _retry(self, fn, retries=2):
        for i in range(retries):
            try:
                return fn()
            except Exception as e:
                if i == retries - 1:
                    raise
                logger.warning("[QdrantStore][RETRY] %s", str(e))
                time.sleep(0.5)

    # CHECK COLLECTION EXISTS
    def _collection_exists(self, name: str) -> bool:
        try:
            collections = self.client.get_collections().collections
            return any(c.name == name for c in collections)
        except Exception as e:
            logger.error("[QdrantStore] collection check failed | %s", str(e))
            return False

    # ENSURE COLLECTION
    def _ensure_collection(self, name: str, dim: int):

        if name in self._collection_cache:
            return

        try:
            exists = self._collection_exists(name)

            if not exists:
                logger.warning("[QdrantStore] creating collection=%s", name)

                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=dim,
                        distance=Distance.COSINE
                    ),
                )

            self._collection_cache.add(name)

        except Exception as e:
            logger.error("[QdrantStore] collection ensure failed | %s", str(e))
            raise

    # BUILD PAYLOAD
    def _build_payload(self, d):

        structure = dict(d.structure or {})
        text = str(d.text or "")[:settings.QDRANT_TEXT_MAX_CHARS]

        return {
            "text": text,
            "doc_id": structure.get("doc_id"),
            "chunk_id": d.chunk_id,
            "modality": d.modality,
            "content_type": structure.get("content_type"),
            "session_id": structure.get("session_id"),
            "embedding_space": structure.get("embedding_space", "text"),
            "source": d.source,
        }

    # INSERT DOCUMENTS
    def insert_documents(self, documents: List):

        if not documents:
            return

        start = time.time()
        documents = documents[:self.max_docs]

        dim = len(documents[0].embedding) if documents[0].embedding else None

        if not dim:
            logger.warning("[QdrantStore] no embeddings")
            return

        text_points, vision_points = [], []

        for d in documents:
            emb = getattr(d, "embedding", None)

            if not isinstance(emb, list) or len(emb) != dim:
                continue

            payload = self._build_payload(d)

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload=payload,
            )

            if payload.get("embedding_space") == "vision":
                vision_points.append(point)
            else:
                text_points.append(point)

        def _insert(collection, points):
            for i in range(0, len(points), self.batch_size):
                batch = points[i:i + self.batch_size]

                self._retry(lambda: self.client.upsert(
                    collection_name=collection,
                    points=batch
                ))

        if text_points:
            self._ensure_collection(self.text_collection, dim)
            _insert(self.text_collection, text_points)

        if vision_points:
            self._ensure_collection(self.vision_collection, dim)
            _insert(self.vision_collection, vision_points)

        logger.info(
            "[QdrantStore] insert | text=%s vision=%s latency=%ss",
            len(text_points),
            len(vision_points),
            round(time.time() - start, 2)
        )

    # FILTER
    def _build_filter(self, session_id=None):

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

    # SAFE SEARCH CORE
    def _search(self, collection, query_vector, limit, session_id):

        start = time.time()

        # use cache instead of API call
        if collection not in self._collection_cache:
            logger.warning("[QdrantStore] collection not initialized=%s", collection)
            return []

        try:
            results = self._retry(lambda: self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                query_filter=self._build_filter(session_id)
            ))

            points = getattr(results, "points", [])

            if not points:
                return []

            output = []

            for r in points:
                text = r.payload.get("text")
                if not text:
                    continue

                output.append({
                    "text": text,
                    "score": float(r.score),
                    "metadata": r.payload
                })

            return output

        except Exception as e:
            logger.error("[QdrantStore] search failed | %s", str(e))
            return []

    # SEARCH TEXT
    def search_text(self, query_vector, limit=None, session_id=None):
        return self._search(
            self.text_collection,
            query_vector,
            limit or settings.RAG_TOP_K,
            session_id
        )

    # SEARCH VISION
    def search_vision(self, query_vector, limit=None, session_id=None):
        return self._search(
            self.vision_collection,
            query_vector,
            limit or settings.RAG_TOP_K,
            session_id
        )

    # MODALITY FILTER
    def set_modality_filter(self, modality: str):
        self.modality_filter = modality