from src.rag_system.generation.ollama_generator import OllamaGenerator

llm = OllamaGenerator()

prompt = "Explain Retrieval Augumented Generation in 3 lines."

response = llm.generate(prompt)

print("\nLLm Response:\n")
print(response)