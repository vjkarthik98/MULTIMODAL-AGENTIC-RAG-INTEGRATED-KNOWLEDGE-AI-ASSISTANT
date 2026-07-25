from app.reasoning.query_decomposer import QueryDecomposer
from app.llm.gguf_model import GGUFModel 

llm = GGUFModel()

decomposer = QueryDecomposer(llm)

query = "Explain AI, machine learning, and their differences in real-world applications"

result = decomposer.decompose(query)

print("\nSUB-QUERIES:\n", result)