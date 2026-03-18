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
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    def insert_vector(self, collection_name: str, vector, payload: dict, point_id: int):

        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload= {
                        "text": chunk,
                        "source": file_name,
                        "modality": text
                    }
                )
            ]
        )
        
    def search_vector(self, collection_name: str, query_vector, limit: int = 3):

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit
        )

        return results.points
    
    def get_client(self):
        return self.client