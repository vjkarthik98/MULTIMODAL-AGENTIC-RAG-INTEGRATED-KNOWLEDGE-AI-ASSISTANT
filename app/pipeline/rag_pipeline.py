from app.retrieval.retriever import Retriever
from app.prompt.prompt_builder import PromptBuilder
from app.llm.gguf_model import GGUFModel 


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.prompt_builder = PromptBuilder()
        self.llm = GGUFModel()

    def run(self, query: str):
        # Step 1: Retrieve relevant documents (TOP-K CONTROL)
        docs = self.retriever.retrieval(query, top_k=2)
        
        # Remove duplicate texts
        unique_docs = []
        seen_texts = set()

        for doc in docs:
            text = doc["text"]
            if text not in seen_texts:
                unique_docs.append(doc)
                seen_texts.add(text)

        docs = unique_docs

        if not docs:
            return{
                "answer": "I don't Know",
                "sources": []
            }

        # Step 2: Extract context with Audio TimeStamps
        context_parts = []

        for i, doc in enumerate(docs):
            metadata = doc.get("metadata", {})
            modality = metadata.get("modality", "text")

            if modality == "audio":
                start = metadata.get("start_time", 0)
                end = metadata.get("end_time", 0)

                context_parts.append(
                    f"[Source {i + 1} | Audio {start:.2f}s]\n{doc['text'][:150]}"
                )
            else:
                context_parts.append(
                    f"[Source {i + 1}]\n{doc['text'][:150]}"
                )
        
        context = "\n\n".join(context_parts)

        # Step 3: Context size control 
        context = context[:800]

        # Step 4: Extract sources correctly
        sources = [
            doc.get("metadata", {}).get("source", "unknown")
            for doc in docs                     
        ]
    
        # Step 5: Build prompt
        prompt = self.prompt_builder.build_prompt(query, context)
        if not prompt:
            raise ValueError("Prompt is empty or None")
        
        print("\n--- PROMPT PREVIEW ---")
        print(prompt[:300])
        print("--------------------\n")

        # Step 6: Generate answer
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": sources
        }

    def stream(self, query: str):
        docs = self.retriever.retrieval(query, top_k = 3)

        context_parts = []

        for i, doc in enumerate(docs):
            metadata = doc.get("metadata", {})
            modality = metadata.get("modality", "text")

            if modality == "audio":
                start = metadata.get("start_time", 0)
                end = metadata.get("end_time", 0)

                context_parts.append(
                    f"[Source {i + 1} | Audio {start:.2f}s]\n{doc['text'][:150]}"
                )
            else:
                context_parts.append(
                    f"[Source {i + 1}]\n{doc['text'][:150]}"
                )
        
        context = "\n\n".join(context_parts)

        # Context size control
        context = context[:2000]

        prompt = self.prompt_builder.build_prompt(query, context)
        
        return self.llm.stream(prompt)
