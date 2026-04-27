from app.memory.memory_filter import filter_relevant_history
from app.embeddings.text_embedder import TextEmbedder

embedder = TextEmbedder()

history = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assitant", "content": "ML is a subset of AI"},
    {"role": "user", "content": "Explain cricket"},
]

query = "Explain machine Learning"

filtered = filter_relevant_history(query, history, embedder)

print(filtered)
