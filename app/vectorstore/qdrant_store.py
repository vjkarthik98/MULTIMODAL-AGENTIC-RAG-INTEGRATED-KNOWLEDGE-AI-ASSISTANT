import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.core.config import settings

class QdrantVectorStore:

    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )

    # Create collection with given size
    def create_collection(self, collection_name: str, vector_size: int = int):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
    # Ensure collection exists
    def _ensure_collection(self, collection_name: str, vector_size: int):
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
    
        if collection_name not in collection_names:
            self.create_collection(collection_name, vector_size)
    
    # Insert Multimodal
    def insert_documents(self, documents):

        document_id = str(uuid.uuid4())

        for doc in documents:
            metadata = doc.get("metadata", {})
            modality = metadata.get("modality", "text")

            # Decide collection + dimension
            if modality == "image":
                collection_name = "image_collection"
                vector_size = 768
            else:
                collection_name = "text_collection"
                vector_size = 384

            # Ensure correct collection
            self._ensure_collection(collection_name, vector_size)

            payload = {
                # core metadata (clean + controlled)
                "text": doc["text"],
                "document_id": document_id,
                **metadata 
            }

            point = PointStruct(
                id=str(uuid.uuid4()), # unique per chunk
                vector=doc["embedding"],
                payload=payload
                )
        
            # Insert oer collection
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
    # TEXT SEARCH
    def search_text(self, query_vector, limit=5, source_filter = None):

        query_filter = None

        if source_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_filter)
                    )
                ]
            )
        
        results = self.client.query_points(
            collection_name="text_collection",
            query=query_vector,
            limit=limit,
            query_filter=query_filter
        )

        return [
            { 
                "text": point.payload["text"],
                "score": point.score,
                "metadata": point.payload
            }
            for point in results.points
        ]
    
    # IMAGE SEARCH
    def search_image(self, query_vector, limit=5):
 
        results = self.client.query_points(
            collection_name="image_collection",
            query=query_vector,
            limit=limit
        )

        return [
            {
                "text": point.payload["text"],   # payload
                "score": point.score,          # score
                "metadata": point.payload
            }
            for point in results.points
        ]
    
    def get_client(self):
        return self.client