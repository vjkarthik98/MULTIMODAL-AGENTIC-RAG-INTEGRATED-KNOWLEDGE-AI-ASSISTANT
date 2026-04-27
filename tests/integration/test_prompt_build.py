from src.rag_system.prompt.prompt_builder import PromptBuilder

builder = PromptBuilder()

contexts = [
    "RAG stands for Retrieval Augmented Generation.",
    "It combines document retrieval with language models."
]

query = "What is RAG?"

prompt = builder.build_prompt(query, contexts)

print(prompt)