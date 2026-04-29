from typing import Dict, Any, List
import time

from app.core.config import settings
from app.utils.logger import get_logger
from app.core.model_loader import model_loader

from app.retrieval.retriever import Retriever
from app.prompt.prompt_builder import PromptBuilder

from app.memory.memory_manager import MemoryManager
from app.core.infra_registry import infra


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

        # MEMORY INIT (FIXED)
        self.memory_manager = MemoryManager()   
        self.mongo_memory = infra.get_mongo()

    # NORMALIZE QUERY
    def _normalize_query(self, query: str) -> str:
        return " ".join(query.strip().split())

    def run(self, query: str, session_id: str = "default") -> Dict[str, Any]:

        start_time = time.time()

        if not query or not query.strip():
            return {"answer": "Query cannot be empty.", "error": "invalid_query"}

        query = self._normalize_query(query)

        logger.info("[RAGPipeline][START] session_id=%s", session_id)

        try:
            # MEMORY
            history = self.memory_manager.get_history(session_id)
            history_text = self._format_history(history)

            # RETRIEVAL
            try:
                docs = self.retriever.retrieval(
                    query=query,
                    session_id=session_id,
                    top_k=settings.DEFAULT_TOP_K
                )
            except Exception as e:
                logger.error("[RAGPipeline][RETRIEVAL_FAIL] %s", str(e))
                docs = []

            if not docs:
                return self._empty_response(start_time)

            docs = self._normalize_docs(docs)
            docs = self._deduplicate_docs(docs)

            docs = docs[:settings.MAX_CHUNKS]

            # CONTEXT
            context = self._build_multimodal_context(docs)
            sources = self._extract_sources(docs)

            system_role = self._detect_role(docs)

            full_context = self._compose_context(
                system_role,
                history_text,
                context
            )

            full_context = full_context[:settings.MAX_PROMPT_CHARS]

            prompt = self.prompt_builder.build_prompt(
                query=query,
                context=full_context
            )

            # LLM
            try:
                answer = self.llm.generate(prompt) if self.llm else ""
            except Exception as e:
                logger.error("[RAGPipeline][LLM_FAIL] %s", str(e))
                return self._empty_response(start_time)

            answer = (answer or "").strip()

            if not answer:
                answer = "I could not generate a reliable answer."

            # MEMORY WRITE (SAFE)
            try:
                self.memory_manager.add_interaction(
                    session_id=session_id,
                    query=query,
                    response=answer
                )

                self.mongo_memory.store_message(session_id, "user", query)
                self.mongo_memory.store_message(session_id, "assistant", answer)

            except Exception as e:
                logger.warning("[RAGPipeline][MEMORY_FAIL] %s", str(e))

            latency = round(time.time() - start_time, 2)

            logger.info("[RAGPipeline][SUCCESS] latency=%ss", latency)

            return {
                "answer": answer,
                "sources": sources,
                "metadata": {
                    "retrieved_docs": len(docs),
                    "latency": latency
                }
            }

        except Exception as e:
            logger.error("[RAGPipeline][ERROR] %s", str(e))
            return {"answer": "Something went wrong.", "error": str(e)}

    # STREAMING (SAFE)
    def stream(self, query: str, session_id: str = "default"):

        logger.info("[RAGPipeline][STREAM] session_id=%s", session_id)

        try:
            docs = self.retriever.retrieval(
                query=query,
                session_id=session_id,
                top_k=min(3, settings.DEFAULT_TOP_K)
            )
        except Exception:
            docs = []

        context = self._build_stream_context(docs)

        prompt = self.prompt_builder.build_prompt(query, context)

        def generator():
            try:
                for token in self.llm.stream(prompt):
                    yield token + "\n"
            except Exception:
                yield "Streaming failed.\n"

        return generator()

    # HELPERS

    def _format_history(self, history: List[Dict]) -> str:
        max_chars = settings.MEMORY_MAX_CONTEXT_CHARS

        formatted, total_len = [], 0

        for msg in reversed(history):
            line = f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}"

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
        seen, unique = set(), []

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
        parts = []
        current_len = 0
        max_chars = settings.MAX_PROMPT_CHARS // 2

        for d in docs:
            text = d.get("text", "").strip()
            if not text:
                continue

            chunk = text[:300]

            if current_len + len(chunk) > max_chars:
                break

            parts.append(chunk)
            current_len += len(chunk)

        return "\n\n".join(parts)

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
        return f"{role}\n\n{history}\n\n{context}"

    def _extract_sources(self, docs: List[Dict]) -> List[str]:
        return list({
            d.get("metadata", {}).get("source")
            for d in docs if d.get("metadata", {}).get("source")
        })

    def _build_stream_context(self, docs: List[Dict]) -> str:
        return "\n\n".join(
            d.get("text", "")[:150] for d in docs
        )[:settings.MAX_PROMPT_CHARS]

    def _empty_response(self, start_time):
        return {
            "answer": "I don't know based on available data.",
            "sources": [],
            "latency": round(time.time() - start_time, 2)
        }