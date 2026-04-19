import uuid

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
except ImportError:  # pragma: no cover - handled at runtime
    QdrantClient = None
    Distance = FieldCondition = Filter = MatchValue = PointStruct = VectorParams = None


logger = get_logger(__name__)


class QdrantVectorStore:
    def __init__(self):
        if QdrantClient is None:
            raise ImportError("qdrant-client is required to use QdrantVectorStore")

        if settings.QDRANT_URL:
            self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        else:
            self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

        self.TEXT_COLLECTION = settings.TEXT_COLLECTION_NAME
        self.VISION_COLLECTION = settings.VISION_COLLECTION_NAME
        self.modality_filter = None

        logger.info("[QdrantStore] Initialized")

    def create_collection(self, collection_name: str, vector_size: int):
        logger.info("[QdrantStore] Creating collection=%s | dim=%s", collection_name, vector_size)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def _collection_exists(self, collection_name: str) -> bool:
        collections = self.client.get_collections().collections
        return collection_name in {collection.name for collection in collections}

    def _ensure_collection(self, collection_name: str, vector_size: int):
        if not self._collection_exists(collection_name):
            logger.warning("[QdrantStore] Creating missing collection=%s", collection_name)
            self.create_collection(collection_name, vector_size)

    def _get_vector_size(self, documents):
        for document in documents:
            embedding = getattr(document, "embedding", None)
            if embedding:
                return len(embedding)
        return None

    def _base_payload(self, document):
        structure = dict(document.structure or {})
        payload = {
            "text": document.text,
            "text_preview": document.text[:200] if document.text else "",
            "doc_id": structure.get("doc_id"),
            "chunk_id": document.chunk_id,
            "total_chunks": structure.get("total_chunks", 1),
            "modality": document.modality,
            "subtype": document.subtype,
            "source": document.source,
            "source_type": document.source_type,
            "page": document.page,
            "session_id": structure.get("session_id"),
            "structure": structure,
            "embedding_space": structure.get("embedding_space"),
            "content_type": structure.get("content_type"),
            "start_time": structure.get("start_time"),
            "end_time": structure.get("end_time"),
            "duration": structure.get("duration"),
            "segment_index": structure.get("segment_index"),
            "frame_index": structure.get("frame_index"),
            "timestamp": structure.get("timestamp"),
            "linked_segment_index": structure.get("linked_segment_index"),
        }
        if document.extra_metadata:
            payload.update(document.extra_metadata)
        return payload

    def set_modality_filter(self, modality: str):
        self.modality_filter = modality

    def insert_documents(self, documents):
        if not documents:
            logger.warning("[QdrantStore] No documents to insert")
            return

        vector_size = self._get_vector_size(documents)
        if vector_size is None:
            logger.warning("[QdrantStore] No text embeddings available for insert")
            return

        try:
            self._ensure_collection(self.TEXT_COLLECTION, vector_size)
            points = []

            for document in documents:
                if not document.embedding or len(document.embedding) != vector_size:
                    continue

                payload = self._base_payload(document)
                payload["embedding_space"] = "text"
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=document.embedding,
                        payload=payload,
                    )
                )

            if not points:
                logger.warning("[QdrantStore] No valid text points prepared")
                return

            logger.info("[QdrantStore] Inserting TEXT points=%s", len(points))
            self.client.upsert(collection_name=self.TEXT_COLLECTION, points=points)
            logger.info("[QdrantStore] TEXT insert successful")

        except Exception as exc:
            logger.error("[QdrantStore] TEXT insert failed | error=%s", exc)
            raise

    def insert_vision_documents(self, documents):
        if not documents:
            logger.warning("[QdrantStore] No VISION documents to insert")
            return

        vector_size = self._get_vector_size(documents)
        if vector_size is None:
            logger.warning("[QdrantStore] No vision embeddings available for insert")
            return

        try:
            self._ensure_collection(self.VISION_COLLECTION, vector_size)
            points = []

            for document in documents:
                if not document.embedding or len(document.embedding) != vector_size:
                    continue

                payload = self._base_payload(document)
                payload["embedding_space"] = "vision"
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=document.embedding,
                        payload=payload,
                    )
                )

            if not points:
                logger.warning("[QdrantStore] No valid vision points prepared")
                return

            logger.info("[QdrantStore] Inserting VISION points=%s", len(points))
            self.client.upsert(collection_name=self.VISION_COLLECTION, points=points)
            logger.info("[QdrantStore] VISION insert successful")

        except Exception as exc:
            logger.error("[QdrantStore] VISION insert failed | error=%s", exc)
            raise

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

    def _query_collection(self, collection_name, query_vector, limit, query_filter):
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
            )
            return response.points

        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
        )

    def search_text(self, query_vector, limit=5, session_id=None):
        try:
            if not self._collection_exists(self.TEXT_COLLECTION):
                logger.warning("[QdrantStore] TEXT collection does not exist")
                return []

            results = self._query_collection(
                self.TEXT_COLLECTION,
                query_vector,
                limit,
                self._build_filter(session_id=session_id),
            )
            logger.info("[QdrantStore] TEXT results=%s | session_id=%s", len(results), session_id)
            return [
                {"text": point.payload.get("text"), "score": point.score, "metadata": point.payload}
                for point in results
            ]

        except Exception as exc:
            logger.error("[QdrantStore] TEXT search failed | error=%s", exc)
            return []

    def search_vision(self, query_vector, limit=5, session_id=None):
        try:
            if not self._collection_exists(self.VISION_COLLECTION):
                logger.warning("[QdrantStore] VISION collection does not exist")
                return []

            results = self._query_collection(
                self.VISION_COLLECTION,
                query_vector,
                limit,
                self._build_filter(session_id=session_id),
            )
            logger.info("[QdrantStore] VISION results=%s | session_id=%s", len(results), session_id)
            return [
                {"text": point.payload.get("text"), "score": point.score, "metadata": point.payload}
                for point in results
            ]

        except Exception as exc:
            logger.error("[QdrantStore] VISION search failed | error=%s", exc)
            return []

    def get_client(self):
        return self.client
