from app.llm.gguf_model import GGUFModel

model = GGUFModel()

prompt = """[INST] 
Explain RAG in simple terms 
[/INST]
"""

response = model.generate(prompt)
print("\nResponse:\n", response)