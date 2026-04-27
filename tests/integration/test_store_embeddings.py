from app.vectorstore.qdrant_store import QdrantVectorStore
from app.embeddings.text_embedder import TextEmbedder
from app.core.config import settings


def test_store_embedding():

    # intialize components
    vector_store = QdrantVectorStore()
    embedder = TextEmbedder()

    # ensure collection exists
    vector_store.create_collection(settings.COLLECTION_NAME)

    text = "Artificial intelligence is transforming industries."

    # generate embedding 
    vector = embedder.embed_text(text)

    # store in qdrant
    vector_store.insert_vector(
        collection_name=settings.COLLECTION_NAME,
        vector=vector,
        payload={"text": text},
        point_id=1
    )
    
    print("Embedding stored successfully!")


if __name__ == "__main__":

    test_store_embedding()