import requests

url = "http://127.0.0.1:8000/rag/query/stream"

response = requests.post(
    url,
    json={"query": "Explain RAG in simple terms"},
    stream=True
)

for chunk in response.iter_content(chunk_size=50):
    if chunk:
        print(chunk.decode("utf-8"), end="", flush=True)