from app.vectorstore.qdrant_store import QdrantVectorStore

# Initialize
qdrant = QdrantVectorStore()

# create collection
qdrant.create_collection("multimodal_rag")

# sample vector (dummy embedding)
vector = [0.1] * 384

# insert data
qdrant.client.upsert(
    collection_name="multimodal_rag",
    points=[
        {
            "id": 1,
            "vector": vector,
            "payload":{
                "text": "Artificial intelligence is the simulation of human intelligence.",
                "source": "test_doc"

            }
        }
    ]
)

print("sample data inserted!")