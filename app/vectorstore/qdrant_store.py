import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.core.config import settings

# Logger
logger = logging.getLogger(__name__)


class QdrantVectorStore:

    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )

        self.TEXT_COLLECTION = "text_collection"

        logger.info("[QdrantStore] Initialized client")

    # -----------------------
    # COLLECTION CREATION
    # -----------------------
    def create_collection(self, collection_name: str, vector_size: int):
        logger.info(
            f"[QdrantStore] Creating collection={collection_name} | dim={vector_size}"
        )

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    def _ensure_collection(self, collection_name: str, vector_size: int):
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            logger.warning(
                f"[QdrantStore] Collection missing → creating {collection_name}"
            )
            self.create_collection(collection_name, vector_size)

    # -----------------------
    # INSERT
    # -----------------------
    def insert_documents(self, documents):

        if not documents:
            logger.warning("[QdrantStore] No documents to insert")
            return

        try:
            document_id = str(uuid.uuid4())
            points = []

            for doc in documents:
                metadata = doc.get("metadata", {})

                collection_name = self.TEXT_COLLECTION
                vector_size = len(doc["embedding"])

                self._ensure_collection(collection_name, vector_size)

                payload = {
                    "text": doc["text"],
                    "document_id": document_id,
                    **metadata
                }

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=doc["embedding"],
                    payload=payload
                )

                points.append(point)

            logger.info(
                f"[QdrantStore] Inserting points | count={len(points)}"
            )

            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            logger.info("[QdrantStore] Insert successful")

        except Exception as e:
            logger.error(f"[QdrantStore] Insert failed | error={str(e)}")
            raise

    # -----------------------
    # TEXT SEARCH
    # -----------------------
    def search_text(self, query_vector, limit=5, source_filter=None, session_id=None):

        try:
            logger.debug(
                f"[QdrantStore] Text search | limit={limit} | session_id={session_id}"
            )

            conditions = []

            if source_filter:
                conditions.append(
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_filter)
                    )
                )

            if session_id:
                conditions.append(
                    FieldCondition(
                        key="session_id",
                        match=MatchValue(value=session_id)
                    )
                )

            query_filter = Filter(must=conditions) if conditions else None

            results = self.client.query_points(
                collection_name=self.TEXT_COLLECTION,
                query=query_vector,
                limit=limit,
                query_filter=query_filter
            )

            logger.debug(
                f"[QdrantStore] Retrieved results | count={len(results.points)}"
            )

            return [
                {
                    "text": point.payload["text"],
                    "score": point.score,
                    "metadata": point.payload
                }
                for point in results.points
            ]

        except Exception as e:
            logger.error(f"[QdrantStore] Search failed | error={str(e)}")
            return []

    # -----------------------
    # IMAGE SEARCH
    # -----------------------
    def search_image(self, query_vector, limit=5):

        try:
            logger.debug(f"[QdrantStore] Image search | limit={limit}")

            results = self.client.query_points(
                collection_name=self.TEXT_COLLECTION,
                query=query_vector,
                limit=limit
            )

            return [
                {
                    "text": point.payload.get("text", ""),
                    "score": point.score,
                    "metadata": point.payload
                }
                for point in results.points
            ]

        except Exception as e:
            logger.error(f"[QdrantStore] Image search failed | error={str(e)}")
            return []

    def get_client(self):
        return self.client