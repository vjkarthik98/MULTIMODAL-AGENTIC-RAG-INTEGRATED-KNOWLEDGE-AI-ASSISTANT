from app.retrieval.retriever import Retriever
from app.prompt.prompt_builder import PromptBuilder
from app.llm.gguf_model import GGUFModel 

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.prompt_builder = PromptBuilder()
        self.llm = GGUFModel()

    def run(self, query: str):
        # Step 1: Retrieve relevant documents
        docs = self.retriever.retrieval(query)

        # Step 2: Extract context + sources
        contexts = [doc["text"] for doc in docs]
        sources = [doc.get("source", "") for doc in docs]

        # Step 3: Build prompt
        prompt = self.prompt_builder.build_prompt(query, contexts)

        # Step 4: Generate answer
        answer = self.llm.generate(prompt)

        # Step 5: Return Structured output
        return {
            "answer": answer,
            "sources": sources
        }

    def stream(self, query: str):
        docs = self.retriever.retrieval(query)

        contexts = [doc.get("text", "")[:500] for doc in docs][:3]

        prompt = self.prompt_builder.build_prompt(query, contexts)

        return self.llm.stream(prompt)