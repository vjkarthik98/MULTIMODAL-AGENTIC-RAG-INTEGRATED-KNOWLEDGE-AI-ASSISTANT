from app.retrieval.retriever import Retriever
from app.prompt.prompt_builder import PromptBuilder
from app.llm.gguf_model import GGUFModel 
from app.memory.redis_memory import RedisMemory
from app.memory.mongo_memory import MongoMemory
from app.memory.memory_manager import MemoryManager



class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.prompt_builder = PromptBuilder()
        self.llm = GGUFModel()
        self.memory_manager = MemoryManager(self.llm)
        self.mongo_memory = MongoMemory()

    def run(self, query: str, session_id: str = "default"):

        # Step 0: Load Memory
        history = self.memory_manager.get_history(session_id)

        # Convert history -> text
        history_text = ""
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            history_text += f"{role.upper()}: {content}\n"

        # Step 1: Retrieve relevant documents (TOP-K CONTROL)
        docs = self.retriever.retrieval(query, top_k=5)
        print("\n RETRIEVED DOCS:\n", docs)

        # Normalize docs(handle tuple issue)
        normalized_docs = []

        for doc in docs:
            if isinstance(doc, dict):
                normalized_docs.append(doc)
            
            elif isinstance(doc, tuple):
                # assume structure: (text, score, metadata)
                normalized_docs.append({
                    "text": doc[0],
                    "metadata": doc[2] if len(doc) > 2 else {}
                })
        docs = normalized_docs
        
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

        # Step 2: Smart Context Aggregation
        audio_texts = []
        frame_texts = []
        other_texts = []

        for doc in docs:
            metadata = doc.get("metadata", {})
            modality = metadata.get("modality", "text")

            if modality in  ["audio", "video_audio"]:
                audio_texts.append(doc["text"])

            elif modality == "video_frame":
                frame_texts.append(doc["text"])
            
            else: 
                other_texts.append(doc["text"])

        # Combine texts
        combined_audio = " ".join(audio_texts)
        combined_frames = " ".join(frame_texts)
        combined_other = " ".join(other_texts)

        # Step 3: Build Strong Context

        context = f"""
    AUDIO CONTENT (main meaning):
    {combined_audio}

    VISUAL CONTENT (supporting details):
    {combined_frames}

    OTHER CONTENT:
    {combined_other}
    """
        # Limit size
        context = context[:1200]

        # Step 4: Sources
        sources = list(set([
            doc.get("metadata", {}).get("source", "unknown")
            for doc in docs
        ]))

        # Step 5: Dynamic Role Deteciton
        modality_types = set([
            doc.get("metadata", {}).get("modality", "text")
            for doc in docs
        ])
        
        if "image" in modality_types:
            system_role = "You are an AI assistant analyzing images."
        elif "audio" in modality_types or "video_audio" in modality_types:
            system_role = "You are an AI assistant analyzing audio content."
        elif "video_frame" in modality_types:
            system_role = "You are an AI assistant analyzing video content."
        else:
            system_role = "You are an AI assistant answering general knowledge questions."

        # Step 6: Dynamic Prompt  
        prompt = f"""
        {system_role}

        Conversation History:
        {history_text}

        Use the following information:

        {context}

        Question: {query}

        TASK:
        - Use the provided context strictly
        - Do not hallucinate
        - Answer clearly and concisely
        - If unsure, say you don't know

        FINAL ANSWER:
        """
 
        print("\n--- PROMPT PREVIEW ---")
        print(prompt[:500])
        print("--------------------\n")

        # Step 6: Generate answer
        print("\n[PIPELINE] Generating answer...")

        answer = self.llm.generate(prompt)

        print("\n[PIPELINE] Answer generated:")
        print(answer)

        # Step 7: Store Memory (Summarized)
        memory_result = self.memory_manager.add_interaction(
            session_id,
            query,
            answer
        )
        if memory_result.get("summarized"):
            print("\n MEMORY SUMMARIZED:")
            print(memory_result["summary"])

        # Step 8: Store in MongoDB
        self.mongo_memory.store_message(session_id, "user", query)
        self.mongo_memory.store_message(session_id, "assistant", answer)

        # Step 9: Return
        return {
            "answer": answer,
            "sources": sources
        }

    def stream(self, query: str):
        docs = self.retriever.retrieval(query, top_k = 3)
        print("\n=== RETRIEVED DOCS===")
        for d in docs:
            print(d["text"], "|", d["metadata"])
        print("=================\n")

        context_parts = []

        for i, doc in enumerate(docs):
            metadata = doc.get("metadata", {})
            modality = metadata.get("modality", "text")

            if modality in ["audio", "video_audio"]:
                start = metadata.get("start_time", 0)
                context_parts.append(
                    f"[Source {i + 1} | Audio {start:.2f}s]\n{doc['text'][:150]}"
                )
            
            elif modality == "video_frame":
                timestamp = metadata.get("timestamp", 0)
                context_parts.append(
                    f"[Source {i + 1} | Frame {timestamp:.2f}s]\n{doc['text'][:150]}"
                )
            
            else:
                context_parts.append(
                    f"[Source {i + 1}]\n{doc['text'][:150]}"
                )
        
        context = "\n\n".join(context_parts)


        # Context size control
        context = context[:2000]

        prompt = self.prompt_builder.build_prompt(query, context)
        
        # Wrap LLM stream properly
        def generator():
            for token in self.llm.stream(prompt):
                yield token + "\n"
        
        return generator()
