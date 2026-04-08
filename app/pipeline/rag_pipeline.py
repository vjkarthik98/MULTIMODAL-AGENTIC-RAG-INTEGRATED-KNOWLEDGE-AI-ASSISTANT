from app.retrieval.retriever import Retriever
from app.prompt.prompt_builder import PromptBuilder
from app.core.model_loader import model_loader
from app.memory.redis_memory import RedisMemory
from app.memory.mongo_memory import MongoMemory
from app.memory.memory_manager import MemoryManager
import logging

# ✅ Logger
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.prompt_builder = PromptBuilder()
        self.llm = model_loader.get_llm()
        self.memory_manager = MemoryManager(self.llm)
        self.mongo_memory = MongoMemory()

    def run(self, query: str, session_id: str = "default"):

        logger.info(f"[RAGPipeline] session_id={session_id} | Run started")

        # Step 0: Load Memory
        history = self.memory_manager.get_history(session_id)

        history_text = ""
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            history_text += f"{role.upper()}: {content}\n"

        # Step 1: Retrieval
        docs = self.retriever.retrieval(query, top_k=5)

        logger.debug(f"[RAGPipeline] session_id={session_id} | Retrieved docs count={len(docs)}")

        # Normalize docs
        normalized_docs = []
        for doc in docs:
            if isinstance(doc, dict):
                normalized_docs.append(doc)
            elif isinstance(doc, tuple):
                normalized_docs.append({
                    "text": doc[0],
                    "metadata": doc[2] if len(doc) > 2 else {}
                })
        docs = normalized_docs

        # Remove duplicates
        unique_docs = []
        seen_texts = set()

        for doc in docs:
            text = doc["text"]
            if text not in seen_texts:
                unique_docs.append(doc)
                seen_texts.add(text)

        docs = unique_docs

        if not docs:
            logger.warning(f"[RAGPipeline] session_id={session_id} | No docs retrieved")
            return {
                "answer": "I don't Know",
                "sources": []
            }

        # Step 2: Context Aggregation
        audio_texts, frame_texts, other_texts = [], [], []

        for doc in docs:
            metadata = doc.get("metadata", {})
            modality = metadata.get("modality", "text")

            if modality in ["audio", "video_audio"]:
                audio_texts.append(doc["text"])
            elif modality == "video_frame":
                frame_texts.append(doc["text"])
            else:
                other_texts.append(doc["text"])

        context = f"""
AUDIO CONTENT:
{' '.join(audio_texts)}

VISUAL CONTENT:
{' '.join(frame_texts)}

OTHER CONTENT:
{' '.join(other_texts)}
"""
        context = context[:1200]

        # Step 3: Sources
        sources = list(set([
            doc.get("metadata", {}).get("source", "unknown")
            for doc in docs
        ]))

        # Step 4: Role Detection
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

        # Step 5: Prompt
        prompt = f"""
{system_role}

Conversation History:
{history_text}

Context:
{context}

Question: {query}

TASK:
- Use the provided context strictly
- Do not hallucinate
- Answer clearly and concisely
- If unsure, say you don't know

FINAL ANSWER:
"""

        logger.debug(f"[RAGPipeline] session_id={session_id} | Prompt built")

        # Step 6: LLM
        logger.info(f"[RAGPipeline] session_id={session_id} | Generating answer")
        answer = self.llm.generate(prompt)

        logger.info(f"[RAGPipeline] session_id={session_id} | Answer generated")

        # Step 7: Memory
        memory_result = self.memory_manager.add_interaction(
            session_id,
            query,
            answer
        )

        if memory_result.get("summarized"):
            logger.info(f"[RAGPipeline] session_id={session_id} | Memory summarized")

        # Step 8: Mongo
        self.mongo_memory.store_message(session_id, "user", query)
        self.mongo_memory.store_message(session_id, "assistant", answer)

        logger.info(f"[RAGPipeline] session_id={session_id} | Run completed")

        return {
            "answer": answer,
            "sources": sources
        }

    def stream(self, query: str, session_id: str = "default"):

        logger.info(f"[RAGPipeline] session_id={session_id} | Stream started")

        docs = self.retriever.retrieval(query, top_k=3)

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
        context = context[:2000]

        prompt = self.prompt_builder.build_prompt(query, context)

        def generator():
            for token in self.llm.stream(prompt):
                yield token + "\n"

        return generator()