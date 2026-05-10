import hashlib
import time
import unicodedata
from typing import Any, Dict, Iterator, List

from app.core.config import settings
from app.core.infra_registry import infra
from app.core.model_loader import model_loader
from app.memory.memory_manager import MemoryManager
from app.prompt.prompt_builder import PromptBuilder
from app.retrieval.retriever import Retriever
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:

    def __init__(self) -> None:
        self.retriever      = Retriever()
        self.prompt_builder = PromptBuilder()

        try:
            self.llm = model_loader.get_llm()
        except Exception as e:
            logger.warning(event="llm_unavailable", error=str(e))
            self.llm = None

        self.memory_manager = MemoryManager()
        self.mongo_memory   = infra.get_mongo()

    # NORMALIZE

    def _normalize(self, query: str) -> str:
        query = unicodedata.normalize("NFC", str(query or ""))
        return " ".join(query.strip().split())

    # HASH

    def _hash(self, text: str, meta: Dict) -> str:
        base = f"{text[:100]}|{meta.get('doc_id')}|{meta.get('chunk_id')}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    # MAIN

    def run(self, query: str, session_id: str = "default") -> Dict[str, Any]:

        start = time.time()

        if not query or not query.strip():
            return {"answer": "Query cannot be empty."}

        query = self._normalize(query)[:settings.MAX_PROMPT_CHARS]

        try:
            # MEMORY
            history      = self.memory_manager.get_history(session_id)
            history_text = self._format_history(history)

            # RETRIEVAL
            t_ret = time.time()
            docs  = self.retriever.retrieval(
                query=query,
                session_id=session_id,
                top_k=settings.DEFAULT_TOP_K,
            )
            retrieval_latency = round(time.time() - t_ret, 2)

            if not docs:
                return self._empty(start)

            docs = self._normalize_docs(docs)
            docs = self._dedup(docs)
            docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)
            docs = docs[:settings.RAG_TOP_K]

            # CONTEXT
            context      = self._build_context(docs)
            sources      = self._sources(docs)
            full_context = self._compose(history_text, context)
            full_context = full_context[:settings.MAX_PROMPT_CHARS]

            prompt = self.prompt_builder.build_prompt(
                query=query,
                context=full_context,
                session_id=session_id,
            )

            # LLM GENERATE
            answer = ""
            t_llm  = time.time()

            try:
                if self.llm:
                    answer = self.llm.generate(
                        prompt,
                        session_id=session_id,
                    )
            except Exception as e:
                logger.error(
                    event="rag_llm_failed",
                    error=str(e),
                    session_id=session_id,
                )

            llm_latency = round(time.time() - t_llm, 2)
            answer      = (answer or "").strip() or "I don't know based on available data."

            # MEMORY WRITE
            if answer and len(answer.strip()) > 5:
                self._store_memory(session_id, query, answer)

            latency = round(time.time() - start, 2)

            logger.info(
                event="rag_pipeline_success",
                docs=len(docs),
                retrieval_latency=retrieval_latency,
                llm_latency=llm_latency,
                latency=latency,
                session_id=session_id,
            )

            return {
                "answer":   answer,
                "sources":  sources,
                "latency":  latency,
                "metadata": {
                    "docs":              len(docs),
                    "retrieval_latency": retrieval_latency,
                    "llm_latency":       llm_latency,
                },
            }

        except Exception as e:
            logger.error(
                event="rag_pipeline_failed",
                error=str(e),
                session_id=session_id,
            )
            return {"answer": "Something went wrong. Please try again."}

    # STREAM

    def stream(
        self,
        query: str,
        session_id: str = "default",
    ) -> Iterator[str]:

        query = self._normalize(query)[:settings.MAX_PROMPT_CHARS]

        docs = self.retriever.retrieval(
            query=query,
            session_id=session_id,
            top_k=min(3, settings.DEFAULT_TOP_K),
        )

        docs    = self._normalize_docs(docs)
        context = self._build_context(docs)
        prompt  = self.prompt_builder.build_prompt(
            query=query,
            context=context,
            session_id=session_id,
        )

        def generator() -> Iterator[str]:
            try:
                if not self.llm:
                    yield "LLM unavailable."
                    return
                for token in self.llm.stream(prompt, session_id=session_id):
                    yield token
            except Exception as e:
                logger.error(
                    event="rag_stream_failed",
                    error=str(e),
                    session_id=session_id,
                )
                yield "Streaming failed."

        return generator()

    # FORMAT HISTORY

    def _format_history(self, history: List[Dict]) -> str:
        max_chars  = settings.MEMORY_MAX_CONTEXT_CHARS
        out: List  = []
        total: int = 0

        for msg in reversed(history):
            role    = msg.get("role", "user").upper()
            content = msg.get("content", "")
            line    = f"{role}: {content}"

            if total + len(line) > max_chars:
                break

            out.append(line)
            total += len(line)

        return "\n".join(reversed(out))

    # NORMALIZE DOCS

    def _normalize_docs(self, docs: List[Any]) -> List[Dict]:
        out: List[Dict] = []

        for d in docs:
            if isinstance(d, dict):
                out.append(d)
            elif isinstance(d, tuple):
                out.append({
                    "text":     d[0] if len(d) > 0 else "",
                    "score":    d[1] if len(d) > 1 else 0.0,
                    "metadata": d[2] if len(d) > 2 else {},
                })

        return out

    # DEDUP

    def _dedup(self, docs: List[Dict]) -> List[Dict]:
        seen:   set       = set()
        unique: List[Dict] = []

        for d in docs:
            h = self._hash(d.get("text", ""), d.get("metadata", {}))
            if h in seen:
                continue
            seen.add(h)
            unique.append(d)

        return unique

    # BUILD CONTEXT

    def _build_context(self, docs: List[Dict]) -> str:
        parts:     List[str] = []
        total:     int       = 0
        max_chars: int       = settings.MAX_CONTEXT_CHARS

        for d in docs:
            text     = d.get("text", "").strip()
            meta     = d.get("metadata", {}) or {}
            modality = meta.get("modality", "text")
            source   = meta.get("source", "")
            subtype  = meta.get("subtype", "")

            if not text:
                continue

            label = f"[{modality.upper()}"
            if subtype:
                label += f"/{subtype}"
            if source:
                label += f" | {source}"
            label += "]"

            chunk = f"{label} {text}"[:settings.RAG_DOC_MAX_CHARS]

            if total + len(chunk) > max_chars:
                break

            parts.append(chunk)
            total += len(chunk)

        return "\n\n".join(parts)

    # COMPOSE CONTEXT

    def _compose(self, history: str, context: str) -> str:
        parts: List[str] = []
        if history:
            parts.append(history)
        if context:
            parts.append(context)
        return "\n\n".join(parts)

    # SOURCES

    def _sources(self, docs: List[Dict]) -> List[str]:
        return list({
            d.get("metadata", {}).get("source")
            for d in docs
            if d.get("metadata", {}).get("source")
        })

    # STORE MEMORY

    def _store_memory(self, session_id: str, query: str, answer: str) -> None:
        try:
            self.memory_manager.add_interaction(session_id, query, answer)

            if self.mongo_memory:
                self.mongo_memory.store_message(session_id, "user",      query)
                self.mongo_memory.store_message(session_id, "assistant", answer)

        except Exception as e:
            logger.warning(
                event="rag_memory_store_failed",
                error=str(e),
                session_id=session_id,
            )

    # EMPTY RESPONSE

    def _empty(self, start: float) -> Dict[str, Any]:
        return {
            "answer":   "I don't know based on available data.",
            "sources":  [],
            "latency":  round(time.time() - start, 2),
            "metadata": {"docs": 0},
        }