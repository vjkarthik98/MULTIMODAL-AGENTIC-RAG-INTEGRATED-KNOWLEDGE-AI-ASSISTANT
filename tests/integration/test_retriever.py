from app.vectorstore.qdrant_store import QdrantVectorStore
from app.embeddings.text_embedder import TextEmbedder
from app.retrieval.retriever import Retriever
from app.core.config import settings


def test_retriever():

    vector_store = QdrantVectorStore()
    embedder = TextEmbedder()

    # recreate collection
    vector_store.create_collection(settings.COLLECTION_NAME)

    text = "Artificial intelligence is transforming industries."

    vector = embedder.embed_text(text)

    # store vector
    vector_store.insert_vector(
        collection_name=settings.COLLECTION_NAME,
        vector=vector,
        payload={"text": text},
        point_id=1
    )

    retriever = Retriever()

    query = "How is AI changing Business?" 

    results = retriever.retrieval(query)

    print("Retrieved Documents:")
    for doc in results:
        print(doc)


if __name__ == "__main__":
    test_retriever()
