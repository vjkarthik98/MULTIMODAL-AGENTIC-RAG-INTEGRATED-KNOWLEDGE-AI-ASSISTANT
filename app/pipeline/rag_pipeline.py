from __future__ import annotations

import asyncio
import hashlib
import time
import unicodedata
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# PROMETHEUS METRICS — SECTION 6

def _get_metrics():
    try:
        from prometheus_client import Counter, Histogram
        rag_duration = Histogram(
            "rag_pipeline_duration_seconds",
            "RAG pipeline total duration",
            ["stage"],
        )
        llm_latency = Histogram(
            "llm_call_latency_seconds",
            "LLM call latency by model",
            ["model"],
        )
        rag_errors = Counter(
            "rag_pipeline_errors_total",
            "RAG pipeline errors by stage",
            ["stage"],
        )
        retrieval_latency = Histogram(
            "retrieval_latency_seconds",
            "Retrieval latency",
            ["retriever_type"],
        )
        return {
            "rag_duration":       rag_duration,
            "llm_latency":        llm_latency,
            "rag_errors":         rag_errors,
            "retrieval_latency":  retrieval_latency,
        }
    except Exception:
        return {}


_METRICS: Dict[str, Any] = {}

if settings.PROMETHEUS_ENABLED:
    try:
        _METRICS = _get_metrics()
    except Exception:
        pass


def _record_stage(stage: str, latency: float) -> None:
    try:
        if "rag_duration" in _METRICS:
            _METRICS["rag_duration"].labels(stage=stage).observe(latency)
    except Exception:
        pass


def _record_llm(model: str, latency: float) -> None:
    try:
        if "llm_latency" in _METRICS:
            _METRICS["llm_latency"].labels(model=model).observe(latency)
    except Exception:
        pass


def _record_error(stage: str) -> None:
    try:
        if "rag_errors" in _METRICS:
            _METRICS["rag_errors"].labels(stage=stage).inc()
    except Exception:
        pass


def _record_retrieval(retriever_type: str, latency: float) -> None:
    try:
        if "retrieval_latency" in _METRICS:
            _METRICS["retrieval_latency"].labels(retriever_type=retriever_type).observe(latency)
    except Exception:
        pass


# NORMALIZE — SECTION 2.3

def _normalize(query: str) -> str:
    query = unicodedata.normalize("NFC", str(query or ""))
    query = query.replace("\x00", "")
    query = query.lstrip("\ufeff\ufffe")
    return " ".join(query.strip().split())


# PROMPT INJECTION SANITIZATION — SECTION 5

_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard the above",
    "forget everything",
    "you are now",
    "act as",
    "jailbreak",
]


def _sanitize(query: str) -> str:
    lower = query.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lower:
            idx   = lower.find(pattern)
            query = query[:idx].strip()
            logger.warning(event="rag_injection_stripped", pattern=pattern)
            break
    return query


# HASH FOR DEDUP

def _hash(text: str, meta: Dict[str, Any]) -> str:
    base = f"{text[:100]}|{meta.get('doc_id')}|{meta.get('chunk_id')}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# DOCUMENT NORMALIZATION

def _normalize_docs(docs: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
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


# DEDUP DOCS

def _dedup_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen:   set                = set()
    unique: List[Dict[str, Any]] = []
    for d in docs:
        h = _hash(d.get("text", ""), d.get("metadata", {}))
        if h in seen:
            continue
        seen.add(h)
        unique.append(d)
    return unique


# BUILD CONTEXT STRING — SECTION 4.6

def _build_context(
    docs: List[Dict[str, Any]],
    max_chars: int,
) -> str:
    parts: List[str] = []
    total: int       = 0

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


# COMPOSE CONTEXT + HISTORY

def _compose(history: str, context: str) -> str:
    parts: List[str] = []
    if history:
        parts.append(history)
    if context:
        parts.append(context)
    return "\n\n".join(parts)


# FORMAT HISTORY — SECTION 4.7

def _format_history(
    history: List[Dict[str, Any]],
    max_chars: int,
) -> str:
    out:   List[str] = []
    total: int       = 0

    for msg in reversed(history):
        role    = msg.get("role", "user").upper()
        content = msg.get("content", "").strip()
        line    = f"{role}: {content}"
        if total + len(line) > max_chars:
            break
        out.append(line)
        total += len(line)

    return "\n".join(reversed(out))


# SOURCES EXTRACTOR

def _extract_sources(docs: List[Dict[str, Any]]) -> List[str]:
    return list({
        d.get("metadata", {}).get("source")
        for d in docs
        if d.get("metadata", {}).get("source")
    })


# RAG PIPELINE CLASS

class RAGPipeline:

    def __init__(self) -> None:
        self._retriever     = None
        self._prompt_builder = None
        self._llm           = None
        self._memory_mgr    = None
        self._mongo         = None

    # LAZY INIT — AVOID CIRCULAR IMPORTS

    def _get_retriever(self):
        if self._retriever is None:
            from app.retrieval.retriever import Retriever
            self._retriever = Retriever()
        return self._retriever

    def _get_prompt_builder(self):
        if self._prompt_builder is None:
            from app.prompt.prompt_builder import PromptBuilder
            self._prompt_builder = PromptBuilder()
        return self._prompt_builder

    def _get_llm(self):
        if self._llm is None:
            try:
                from app.core.model_loader import model_loader
                self._llm = model_loader.get_llm()
            except Exception as e:
                logger.warning(event="llm_unavailable", error=str(e))
                self._llm = None
        return self._llm

    def _get_memory_manager(self):
        if self._memory_mgr is None:
            from app.memory.memory_manager import MemoryManager
            self._memory_mgr = MemoryManager()
        return self._memory_mgr

    def _get_mongo(self):
        if self._mongo is None:
            try:
                from app.core.infra_registry import infra
                self._mongo = infra.get_mongo()
            except Exception:
                self._mongo = None
        return self._mongo

    # STORE MEMORY — SECTION 4.7

    def _store_memory(
        self,
        session_id: str,
        query: str,
        answer: str,
    ) -> None:
        if not answer or len(answer.strip()) < 5:
            return
        try:
            mgr = self._get_memory_manager()
            mgr.add_interaction(session_id, query, answer)

            mongo = self._get_mongo()
            if mongo:
                mongo.store_message(session_id, "user",      query)
                mongo.store_message(session_id, "assistant", answer)

        except Exception as e:
            logger.warning(
                event="rag_memory_store_failed",
                error=str(e),
                session_id=session_id,
            )

    # FALLBACK LLM RESPONSE — SECTION 4.6

    def _fallback_response(
        self,
        query: str,
        session_id: str,
    ) -> str:
        try:
            llm = self._get_llm()
            if not llm:
                return "I don't know based on available data."
            prompt = f"Answer clearly and concisely:\n{query}"
            return llm.generate(
                prompt,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=0.2,
                session_id=session_id,
            ) or "I don't know based on available data."
        except Exception as e:
            logger.error(
                event="rag_fallback_failed",
                error=str(e),
                session_id=session_id,
            )
            return "I don't know based on available data."

    # EMPTY RESPONSE — no docs retrieved, do NOT call LLM

    def _empty(self, start: float) -> Dict[str, Any]:
        return {
            "answer":     "No relevant documents found. Please ingest documents first.",
            "confidence": 0.0,
            "sources":    [],
            "latency":    round(time.time() - start, 2),
            "metadata":   {"docs": 0},
        }

    # MAIN RUN — SECTION 4.6

    def run(
        self,
        query: str,
        session_id: str = "default",
    ) -> Dict[str, Any]:

        start    = time.time()
        trace_id = str(uuid.uuid4())

        if not query or not query.strip():
            return {"answer": "Query cannot be empty.", "trace_id": trace_id}

        # NORMALIZE + SANITIZE — SECTION 2.3 / 5
        query = _normalize(query)
        query = _sanitize(query)
        query = query[:settings.MAX_PROMPT_CHARS]

        try:
            # MEMORY HISTORY — SECTION 4.7
            t_mem = time.time()
            try:
                mgr     = self._get_memory_manager()
                history = mgr.get_history(session_id)
            except Exception as e:
                logger.warning(event="rag_memory_fetch_failed", error=str(e))
                history = []

            history_text = _format_history(history, settings.MEMORY_MAX_CONTEXT_CHARS)
            _record_stage("memory", round(time.time() - t_mem, 3))

            # RETRIEVAL — SECTION 4.5
            t_ret = time.time()
            try:
                retriever = self._get_retriever()
                raw_docs  = retriever.retrieval(
                    query=query,
                    session_id=session_id,
                    top_k=settings.DEFAULT_TOP_K,
                )
            except Exception as e:
                logger.error(
                    event="rag_retrieval_failed",
                    error=str(e),
                    session_id=session_id,
                )
                _record_error("retrieval")
                raw_docs = []

            retrieval_latency = round(time.time() - t_ret, 3)
            _record_retrieval("retriever", retrieval_latency)
            _record_stage("retrieval", retrieval_latency)

            if not raw_docs:
                return self._empty(start)

            # NORMALIZE + DEDUP DOCS
            docs = _normalize_docs(raw_docs)
            docs = _dedup_docs(docs)
            docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)
            docs = docs[:settings.RAG_TOP_K]

            # SCORE THRESHOLD GUARD — treat low-score results as empty retrieval
            docs = [d for d in docs if d.get("score", 0.0) >= settings.FUSION_MIN_SCORE]
            if not docs:
                logger.warning(
                    event="rag_all_chunks_below_min_score",
                    threshold=settings.FUSION_MIN_SCORE,
                    session_id=session_id,
                )
                return self._empty(start)

            # CONTEXT ASSEMBLY
            context      = _build_context(docs, settings.MAX_CONTEXT_CHARS)
            sources      = _extract_sources(docs)
            full_context = _compose(history_text, context)
            full_context = full_context[:settings.MAX_PROMPT_CHARS]

            # PROMPT BUILD — SECTION 4.9
            t_prompt = time.time()
            try:
                builder = self._get_prompt_builder()
                prompt  = builder.build_prompt(
                    query=query,
                    context=full_context,
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(event="rag_prompt_build_failed", error=str(e))
                prompt = (
                    f"Answer from context only.\n\n"
                    f"CONTEXT:\n{full_context}\n\n"
                    f"QUERY:\n{query}\n\nAnswer:"
                )
            _record_stage("prompt_build", round(time.time() - t_prompt, 3))

            # LLM GENERATE — SECTION 4.6 FALLBACK CHAIN
            t_llm  = time.time()
            answer = ""

            try:
                llm = self._get_llm()
                if llm:
                    answer = llm.generate(
                        prompt,
                        max_tokens=settings.LLM_MAX_TOKENS,
                        temperature=settings.LLM_TEMPERATURE,
                        top_p=settings.LLM_TOP_P,
                        session_id=session_id,
                    )
                else:
                    raise RuntimeError("LLM_UNAVAILABLE")

            except Exception as e:
                logger.error(
                    event="rag_llm_failed",
                    error=str(e),
                    session_id=session_id,
                )
                _record_error("llm")
                # FALLBACK CHAIN — GGUF LOCAL — SECTION 4.6
                answer = self._fallback_response(query, session_id)

            llm_latency = round(time.time() - t_llm, 3)
            _record_llm(settings.LLM_MODEL_PATH, llm_latency)
            _record_stage("llm", llm_latency)

            answer = (answer or "").strip() or "I don't know based on available data."

            # MEMORY WRITE — SECTION 4.7
            self._store_memory(session_id, query, answer)

            total_latency = round(time.time() - start, 2)

            logger.info(
                event="rag_pipeline_success",
                docs=len(docs),
                retrieval_latency=retrieval_latency,
                llm_latency=llm_latency,
                latency=total_latency,
                session_id=session_id,
                trace_id=trace_id,
            )

            return {
                "answer":   answer,
                "sources":  sources,
                "latency":  total_latency,
                "trace_id": trace_id,
                "metadata": {
                    "docs":              len(docs),
                    "retrieval_latency": retrieval_latency,
                    "llm_latency":       llm_latency,
                    "memory_turns":      len(history),
                },
            }

        except Exception as e:
            _record_error("pipeline")
            logger.error(
                event="rag_pipeline_failed",
                error=str(e),
                session_id=session_id,
                trace_id=trace_id,
            )
            return {
                "answer":   "Something went wrong. Please try again.",
                "sources":  [],
                "latency":  round(time.time() - start, 2),
                "trace_id": trace_id,
                "error":    str(e),
            }

    # STREAM — SECTION 4.6 SSE / WEBSOCKET TOKEN STREAMING

    def stream(
        self,
        query: str,
        session_id: str = "default",
    ) -> Iterator[str]:

        query = _normalize(query)
        query = _sanitize(query)
        query = query[:settings.MAX_PROMPT_CHARS]

        def _generator() -> Iterator[str]:
            try:
                retriever = self._get_retriever()
                raw_docs  = retriever.retrieval(
                    query=query,
                    session_id=session_id,
                    top_k=min(3, settings.DEFAULT_TOP_K),
                )

                docs    = _normalize_docs(raw_docs)
                docs    = _dedup_docs(docs)
                context = _build_context(docs, settings.MAX_CONTEXT_CHARS)

                builder = self._get_prompt_builder()
                prompt  = builder.build_prompt(
                    query=query,
                    context=context,
                    session_id=session_id,
                )

                llm = self._get_llm()
                if not llm:
                    yield "LLM unavailable."
                    return

                collected_tokens: List[str] = []

                for token in llm.stream(
                    prompt,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=settings.LLM_TEMPERATURE,
                    top_p=settings.LLM_TOP_P,
                    session_id=session_id,
                ):
                    collected_tokens.append(token)
                    yield token

                # MEMORY WRITE AFTER STREAM COMPLETES
                answer = "".join(collected_tokens).strip()
                if answer:
                    self._store_memory(session_id, query, answer)

            except Exception as e:
                logger.error(
                    event="rag_stream_failed",
                    error=str(e),
                    session_id=session_id,
                )
                _record_error("stream")
                yield "Streaming failed."

        return _generator()

    # ASYNC STREAM — SECTION 4.6

    async def stream_async(
        self,
        query: str,
        session_id: str = "default",
    ) -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        gen  = await loop.run_in_executor(None, self.stream, query, session_id)

        for token in gen:
            yield token

    # ASYNC RUN — SECTION 4.6

    async def run_async(
        self,
        query: str,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, self.run, query, session_id),
            timeout=settings.REQUEST_TIMEOUT_SEC,
        )


