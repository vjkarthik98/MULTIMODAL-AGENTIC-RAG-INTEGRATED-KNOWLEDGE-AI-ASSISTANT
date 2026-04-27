import uuid
import time
from typing import List

from app.core.config import settings
from app.utils.logger import get_logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )
except ImportError:
    QdrantClient = None


logger = get_logger(__name__)


class QdrantVectorStore:

    def __init__(self):
        if QdrantClient is None:
            raise ImportError("qdrant-client is required")

        if settings.QDRANT_URL:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=settings.QDRANT_TIMEOUT
            )
        else:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=settings.QDRANT_TIMEOUT
            )

        self.text_collection = settings.TEXT_COLLECTION_NAME
        self.vision_collection = settings.VISION_COLLECTION_NAME

        self.batch_size = settings.QDRANT_BATCH_SIZE
        self.max_docs = settings.QDRANT_MAX_DOCS

        self.modality_filter = None

        logger.info("[QdrantStore] initialized")

    # COLLECTION 
    def _collection_exists(self, name: str) -> bool:
        try:
            collections = self.client.get_collections().collections
            return any(c.name == name for c in collections)
        except Exception as e:
            logger.warning("[QdrantStore] collection check failed | %s", str(e))
            return False

    def _ensure_collection(self, name: str, dim: int):
        if self._collection_exists(name):
            return

        logger.warning(
            "[QdrantStore] creating collection=%s dim=%s",
            name,
            dim
        )

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE
            ),
        )

    # VECTOR SIZE 
    def _get_vector_size(self, documents):
        for d in documents:
            emb = getattr(d, "embedding", None)
            if isinstance(emb, list):
                return len(emb)
        return None

    # PAYLOAD 
    def _build_payload(self, document):
        structure = dict(document.structure or {})

        text = str(document.text or "")[:settings.QDRANT_TEXT_MAX_CHARS]

        payload = {
            "text": text,
            "text_preview": text[:200],
            "doc_id": structure.get("doc_id"),
            "chunk_id": document.chunk_id,
            "modality": document.modality,
            "content_type": structure.get("content_type"),
            "session_id": structure.get("session_id"),
            "embedding_space": structure.get("embedding_space", "text"),  # FIX
            "structure": structure,
            "source": document.source,
        }

        if document.extra_metadata:
            payload.update(document.extra_metadata)

        return payload

    # INSERT 
    def insert_documents(self, documents: List):

        if not documents:
            return

        start = time.time()

        documents = documents[:self.max_docs]

        dim = self._get_vector_size(documents)
        if not dim:
            logger.warning("[QdrantStore] no embeddings")
            return

        text_points = []
        vision_points = []

        for d in documents:
            emb = getattr(d, "embedding", None)

            if not isinstance(emb, list) or len(emb) != dim:
                continue

            payload = self._build_payload(d)
            embedding_space = payload.get("embedding_space", "text")

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload=payload,
            )

            if embedding_space == "vision":
                vision_points.append(point)
            else:
                text_points.append(point)

        # TEXT
        if text_points:
            self._ensure_collection(self.text_collection, dim)

            for i in range(0, len(text_points), self.batch_size):
                try:
                    self.client.upsert(
                        collection_name=self.text_collection,
                        points=text_points[i:i + self.batch_size]
                    )
                except Exception as e:
                    logger.error("[QdrantStore] text insert failed | %s", str(e))

        # VISION
        if vision_points:
            self._ensure_collection(self.vision_collection, dim)

            for i in range(0, len(vision_points), self.batch_size):
                try:
                    self.client.upsert(
                        collection_name=self.vision_collection,
                        points=vision_points[i:i + self.batch_size]
                    )
                except Exception as e:
                    logger.error("[QdrantStore] vision insert failed | %s", str(e))

        logger.info(
            "[QdrantStore] insert complete | text=%s vision=%s latency=%ss",
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

    # SEARCH TEXT 
    def search_text(self, query_vector, limit=None, session_id=None):

        if not self._collection_exists(self.text_collection):
            logger.warning("[QdrantStore] text collection missing → skipping")
            return []

        try:
            response = self.client.query_points(
                collection_name=self.text_collection,
                query=query_vector,
                limit=limit or settings.RAG_TOP_K,
                query_filter=self._build_filter(session_id)
            )

            results = getattr(response, "points", [])

            return [
                {
                    "text": r.payload.get("text"),
                    "score": float(r.score),
                    "metadata": r.payload
                }
                for r in results
            ]

        except Exception as e:
            logger.error("[QdrantStore] text search failed | %s", str(e))
            return []

    # SEARCH VISION 
    def search_vision(self, query_vector, limit=None, session_id=None):

        if not self._collection_exists(self.vision_collection):
            logger.warning("[QdrantStore] vision collection missing → skipping")
            return []

        try:
            response = self.client.query_points(
                collection_name=self.vision_collection,
                query=query_vector,
                limit=limit or settings.RAG_TOP_K,
                query_filter=self._build_filter(session_id)
            )

            results = getattr(response, "points", [])

            return [
                {
                    "text": r.payload.get("text"),
                    "score": float(r.score),
                    "metadata": r.payload
                }
                for r in results
            ]

        except Exception as e:
            logger.error("[QdrantStore] vision search failed | %s", str(e))
            return []

    # MODALITY FILTER 
    def set_modality_filter(self, modality: str):
        self.modality_filter = modality

    def get_client(self):
        return self.client