from typing import Dict, Any, List
import time

from app.core.config import settings
from app.utils.logger import get_logger
from app.core.model_loader import model_loader

from app.retrieval.retriever import Retriever
from app.prompt.prompt_builder import PromptBuilder

from app.memory.mongo_memory import MongoMemory
from app.memory.memory_manager import MemoryManager


logger = get_logger(__name__)


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.prompt_builder = PromptBuilder()

        
        try:
            self.llm = model_loader.get_llm()
        except Exception as e:
            logger.warning("[RAGPipeline] LLM unavailable | %s", str(e))
            self.llm = None
            
        self.embedder = model_loader.get_embedder()

        self.memory_manager = MemoryManager(self.llm, self.embedder)
        self.mongo_memory = MongoMemory()

    def run(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        start_time = time.time()

        if not query or not query.strip():
            return {
                "answer": "Query cannot be empty.",
                "error": "invalid_query"
            }

        logger.info(f"[RAGPipeline][START] session_id={session_id}")

        try:
            history = self.memory_manager.get_history(session_id)
            history_text = self._format_history(history)

            docs = self.retriever.retrieval(
                query=query,
                session_id=session_id,
                top_k=settings.DEFAULT_TOP_K
            )

            if not docs:
                return self._empty_response(start_time)

            docs = self._normalize_docs(docs)
            docs = self._deduplicate_docs(docs)

            context = self._build_multimodal_context(docs)
            sources = self._extract_sources(docs)

            system_role = self._detect_role(docs)

            full_context = self._compose_context(
                system_role,
                history_text,
                context
            )

            prompt = self.prompt_builder.build_prompt(
                query=query,
                context=full_context
            )

            answer = self.llm.generate(prompt)

            self.memory_manager.add_interaction(
                session_id=session_id,
                query=query,
                response=answer
            )

            self.mongo_memory.store_message(session_id, "user", query)
            self.mongo_memory.store_message(session_id, "assistant", answer)

            latency = round(time.time() - start_time, 2)

            return {
                "answer": answer,
                "sources": sources,
                "metadata": {
                    "retrieved_docs": len(docs),
                    "latency": latency
                }
            }

        except Exception as e:
            logger.error(f"[RAGPipeline][ERROR] {str(e)}")

            return {
                "answer": "Something went wrong.",
                "error": str(e)
            }

    def stream(self, query: str, session_id: str = "default"):
        logger.info(f"[RAGPipeline][STREAM] session_id={session_id}")

        docs = self.retriever.retrieval(
            query=query,
            session_id=session_id,
            top_k=min(3, settings.DEFAULT_TOP_K)
        )

        context = self._build_stream_context(docs)

        prompt = self.prompt_builder.build_prompt(query, context)

        def generator():
            for token in self.llm.stream(prompt):
                yield token + "\n"

        return generator()

    def _format_history(self, history: List[Dict]) -> str:
        max_chars = settings.MEMORY_MAX_CONTEXT_CHARS

        formatted = []
        total_len = 0

        for msg in reversed(history):
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")

            line = f"{role}: {content}"
            if total_len + len(line) > max_chars:
                break

            formatted.append(line)
            total_len += len(line)

        return "\n".join(reversed(formatted))

    def _normalize_docs(self, docs: List[Any]) -> List[Dict]:
        normalized = []

        for doc in docs:
            if isinstance(doc, dict):
                normalized.append(doc)
            elif isinstance(doc, tuple):
                normalized.append({
                    "text": doc[0],
                    "metadata": doc[2] if len(doc) > 2 else {}
                })

        return normalized

    def _deduplicate_docs(self, docs: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []

        for d in docs:
            key = (
                str(d.get("metadata", {}).get("doc_id")),
                str(d.get("metadata", {}).get("chunk_id")),
                d.get("text", "")[:100]
            )

            if key not in seen:
                seen.add(key)
                unique.append(d)

        return unique

    def _build_multimodal_context(self, docs: List[Dict]) -> str:
        audio, visual, text = [], [], []

        for d in docs:
            modality = d.get("metadata", {}).get("modality", "text")

            if modality in ["audio", "video_audio"]:
                audio.append(d["text"])
            elif modality in ["image", "video_frame"]:
                visual.append(d["text"])
            else:
                text.append(d["text"])

        return f"""
AUDIO:
{' '.join(audio)[:600]}

VISUAL:
{' '.join(visual)[:600]}

TEXT:
{' '.join(text)[:800]}
""".strip()

    def _detect_role(self, docs: List[Dict]) -> str:
        modalities = {d.get("metadata", {}).get("modality", "text") for d in docs}

        if "image" in modalities:
            return "You are an expert in visual understanding."
        if "audio" in modalities:
            return "You are an expert in audio understanding."
        if "video" in modalities or "video_frame" in modalities:
            return "You are an expert in video analysis."

        return "You are a highly accurate knowledge assistant."

    def _compose_context(self, role: str, history: str, context: str) -> str:
        return f"""
{role}

CONVERSATION HISTORY:
{history}

CONTEXT:
{context}

INSTRUCTIONS:
- Use ONLY provided context
- No hallucination
- Be precise and concise
"""

    def _extract_sources(self, docs: List[Dict]) -> List[str]:
        sources = set()

        for d in docs:
            src = d.get("metadata", {}).get("source")
            if src:
                sources.add(src)

        return list(sources)

    def _build_stream_context(self, docs: List[Dict]) -> str:
        parts = []

        for d in docs:
            metadata = d.get("metadata", {})
            modality = metadata.get("modality", "text")

            if modality in ["audio", "video_audio"]:
                t = metadata.get("start_time", 0)
                parts.append(f"[Audio {t:.2f}s]\n{d['text'][:150]}")

            elif modality == "video_frame":
                ts = metadata.get("timestamp", 0)
                parts.append(f"[Frame {ts:.2f}s]\n{d['text'][:150]}")
            else:
                parts.append(d["text"][:150])

        return "\n\n".join(parts)[:settings.MAX_PROMPT_CHARS]

    def _empty_response(self, start_time):
        return {
            "answer": "I don't know based on available data.",
            "sources": [],
            "latency": round(time.time() - start_time, 2)
        }