from app.embeddings.text_embedder import TextEmbedder

def test_text_embedding():
    embedder = TextEmbedder()

    text = "Artificial intelligence is transforming industries."

    vector = embedder.embed_text(text)

    print("Embedding length:", len(vector))
    print("First 5 values:", vector[:5])

if __name__ == "__main__":

    test_text_embedding()