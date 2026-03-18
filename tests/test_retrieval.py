from app.vectorstore.qdrant_store import QdrantVectorStore
from app.embeddings.text_embedder import TextEmbedder
from app.core.config import settings


def test_semantic_search():

    vector_store = QdrantVectorStore()
    embedder = TextEmbedder()

    print("Creating collection...")
    vector_store.create_collection(settings.COLLECTION_NAME)

    text = "Artificial intelligence is transforming industries."

    print("Generating embedding...")
    vector = embedder.embed_text(text)

    print("Inserting vector...")
    vector_store.insert_vector(
        collection_name=settings.COLLECTION_NAME,
        vector=vector,
        payload={"text": text},
        point_id=1
    )

    query = "How is AI changing business?"

    print("Generating query embedding...")
    query_vector = embedder.embed_text(query)

    print("Searching...")
    results = vector_store.search_vector(
        collection_name=settings.COLLECTION_NAME,
        query_vector=query_vector
    )

    print("Results:", results)

    for result in results:
        print("Score:", result.score)
        print("Retrieved Text:", result.payload["text"])


if __name__ == "__main__":
    test_semantic_search()