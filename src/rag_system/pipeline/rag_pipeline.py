from app.retrieval.retriever import Retriever
from src.rag_system.prompt.prompt_builder import PromptBuilder
from src.rag_system.generation.ollama_generator import OllamaGenerator

class RAGPipeline:

    def __init__(self):

        self.retriever = Retriever()
        self.prompt_builder = PromptBuilder()
        self.llm = OllamaGenerator()

    def run(self, query):

        docs = self.retriever.retrieval(query)

        print("\nRetrieved Docs:\n")
        print(docs)

        contexts = [doc["text"] for doc in docs]
        sources = [doc["source"] for doc in docs]

        prompt = self.prompt_builder.build_prompt(query, contexts)

        answer = self.llm.generate(prompt)

        return {
            "answer" : answer,
            "sources": sources
        }