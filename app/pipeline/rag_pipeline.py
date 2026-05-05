from typing import Dict, Any, List
import time
import hashlib

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
            logger.warning(event="llm_unavailable", error=str(e))
            self.llm = None

        self.memory_manager = MemoryManager()
        self.mongo_memory = infra.get_mongo()

    #  NORMALIZE 
    def _normalize(self, query: str) -> str:
        return " ".join(query.strip().split())

    #  HASH 
    def _hash(self, text: str, meta: Dict):
        base = f"{text[:100]}|{meta.get('doc_id')}|{meta.get('chunk_id')}"
        return hashlib.sha256(base.encode()).hexdigest()

    #  MAIN 
    def run(self, query: str, session_id: str = "default") -> Dict[str, Any]:

        start = time.time()

        if not query or not query.strip():
            return {"answer": "Query cannot be empty."}

        query = self._normalize(query)

        try:
            #  MEMORY 
            history = self.memory_manager.get_history(session_id)
            history_text = self._format_history(history)

            #  RETRIEVAL 
            docs = self.retriever.retrieval(
                query=query,
                session_id=session_id,
                top_k=settings.DEFAULT_TOP_K
            )

            if not docs:
                return self._empty(start)

            docs = self._normalize_docs(docs)
            docs = self._dedup(docs)
            docs = docs[:settings.MAX_CHUNKS]

            #  CONTEXT 
            context = self._build_context(docs)
            sources = self._sources(docs)
            role = self._role(docs)

            full_context = self._compose(role, history_text, context)
            full_context = full_context[:settings.MAX_PROMPT_CHARS]

            prompt = self.prompt_builder.build_prompt(query, full_context)

            #  LLM 
            answer = ""
            try:
                if self.llm:
                    answer = self.llm.generate(prompt)
            except Exception as e:
                logger.error(event="llm_failed", error=str(e))

            answer = (answer or "").strip() or "I don't know based on available data."

            #  MEMORY WRITE 
            self._store_memory(session_id, query, answer)

            latency = round(time.time() - start, 2)

            return {
                "answer": answer,
                "sources": sources,
                "latency": latency,
                "metadata": {
                    "docs": len(docs)
                }
            }

        except Exception as e:
            logger.error(event="rag_failed", error=str(e))
            return {"answer": "Something went wrong."}

    #  STREAM 
    def stream(self, query: str, session_id: str = "default"):

        docs = self.retriever.retrieval(
            query=query,
            session_id=session_id,
            top_k=min(3, settings.DEFAULT_TOP_K)
        )

        context = self._build_context(docs)
        prompt = self.prompt_builder.build_prompt(query, context)

        def generator():
            try:
                for token in self.llm.stream(prompt):
                    yield token
            except Exception:
                yield "Streaming failed."

        return generator()

    #  HELPERS 
    def _format_history(self, history: List[Dict]) -> str:

        max_chars = settings.MEMORY_MAX_CONTEXT_CHARS
        out, total = [], 0

        for msg in reversed(history):
            line = f"{msg.get('role','user').upper()}: {msg.get('content','')}"
            if total + len(line) > max_chars:
                break
            out.append(line)
            total += len(line)

        return "\n".join(reversed(out))

    def _normalize_docs(self, docs: List[Any]) -> List[Dict]:

        out = []

        for d in docs:
            if isinstance(d, dict):
                out.append(d)
            elif isinstance(d, tuple):
                out.append({
                    "text": d[0],
                    "metadata": d[2] if len(d) > 2 else {}
                })

        return out

    def _dedup(self, docs: List[Dict]):

        seen = set()
        unique = []

        for d in docs:
            h = self._hash(d.get("text", ""), d.get("metadata", {}))
            if h in seen:
                continue
            seen.add(h)
            unique.append(d)

        return unique

    def _build_context(self, docs: List[Dict]) -> str:

        parts = []
        total = 0
        max_chars = settings.MAX_PROMPT_CHARS // 2

        for d in docs:
            text = d.get("text", "").strip()
            if not text:
                continue

            chunk = text[:300]

            if total + len(chunk) > max_chars:
                break

            parts.append(chunk)
            total += len(chunk)

        return "\n\n".join(parts)

    def _role(self, docs: List[Dict]):

        modalities = {d.get("metadata", {}).get("modality", "text") for d in docs}

        if "image" in modalities:
            return "You are an expert in visual reasoning."
        if "audio" in modalities:
            return "You are an expert in audio reasoning."
        if "video" in modalities:
            return "You are an expert in video reasoning."

        return "You are a factual assistant."

    def _compose(self, role: str, history: str, context: str):

        return f"{role}\n\n{history}\n\n{context}"

    def _sources(self, docs: List[Dict]):

        return list({
            d.get("metadata", {}).get("source")
            for d in docs if d.get("metadata", {}).get("source")
        })

    def _store_memory(self, session_id, query, answer):

        try:
            self.memory_manager.add_interaction(session_id, query, answer)
            self.mongo_memory.store_message(session_id, "user", query)
            self.mongo_memory.store_message(session_id, "assistant", answer)
        except Exception:
            pass

    def _empty(self, start):

        return {
            "answer": "I don't know based on available data.",
            "sources": [],
            "latency": round(time.time() - start, 2)
        }