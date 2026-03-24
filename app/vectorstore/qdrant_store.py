import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import settings

class QdrantVectorStore:

    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )

    def create_collection(self, collection_name: str, vector_size: int = 384):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    def insert_documents(self, documents):
        # Step 1 : ensure collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if settings.COLLECTION_NAME not in collection_names:
            self.create_collection(settings.COLLECTION_NAME)
        
        # Step 2 : insert points
        from qdrant_client.models import PointStruct
        points = []

        for idx, doc in enumerate(documents):
            points.append(
                PointStruct(
                    id= str(uuid.uuid4()),
                    vector=doc["embedding"],
                    payload={
                        "text": doc["text"],
                        "source": doc.get("metadata", {}).get("source", "unknown"),
                        "chunk_id": idx,
                        "modality": doc.get("metadata", {}).get("modality", "text"),
                        **doc.get("metadata", {})
                    }
                )
            )
        
        self.client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points
        )

    def search(self, query_vector, limit=5):
        results = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=query_vector,
            limit=limit
        )

        return [
            {
                "text": point.payload["text"],   # payload
                "score": point.score,          # score
                "metadata": {
                    k: v for k, v in point.payload.items() if k != "text"
                }
            }
            for point in results.points 
        ]
    
    def get_client(self):
        return self.client