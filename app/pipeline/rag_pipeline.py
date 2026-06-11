from __future__ import annotations

import asyncio
import hashlib
import re
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


# STREAM RERANKER SINGLETON — loaded once, shared across all streaming requests
import threading as _threading
_stream_reranker      = None
_stream_reranker_lock = _threading.Lock()


def _get_stream_reranker():
    """Return the pre-warmed Reranker singleton used by the streaming path.
    Uses the same underlying cross-encoder model as query_pipeline."""
    global _stream_reranker
    if _stream_reranker is not None:
        return _stream_reranker
    with _stream_reranker_lock:
        if _stream_reranker is not None:
            return _stream_reranker
        try:
            from app.retrieval.reranker import Reranker
            _stream_reranker = Reranker()
        except Exception as _e:
            logger.warning(event="rag_stream_reranker_init_failed", error=str(_e))
    return _stream_reranker


# NORMALIZE — SECTION 2.3

def _normalize(query: str) -> str:
    query = unicodedata.normalize("NFC", str(query or ""))
    query = query.replace("\x00", "")
    query = query.lstrip("\ufeff\ufffe")
    return " ".join(query.strip().split())


# PROMPT INJECTION SANITIZATION — delegates to unified guardrail (Phase 26)

def _sanitize(query: str) -> str:
    from app.guardrails.input_guard import sanitize as _guard_sanitize
    return _guard_sanitize(query, surface="rag_pipeline")


# FINANCIAL FIGURE NORMALIZER — post-processing layer that replaces rounded
# billion figures in LLM output with the exact million figures that appear
# verbatim in the retrieved context chunks. This is deterministic and does not
# depend on the LLM following prompt instructions about rounding.

_ROUNDED_BILLIONS_RE = re.compile(
    r'\$\s*(\d{1,4}(?:\.\d+)?)\s*billion', re.IGNORECASE
)
_EXACT_MILLIONS_RE = re.compile(r'\b(\d{2,3},\d{3})\b')
_CHUNK_TEXT_CITATION_RE = re.compile(
    r'\[\s*\d+\s*,\s*[\'"].*?[\'"]\s*\]',  # [2, 'chunk text'] → strip
    re.DOTALL,
)
_SOURCE_LABEL_RE = re.compile(
    r'\(\s*[\w\-\.]+\.(?:txt|pdf|docx|xlsx)\s*[^)]*\)\s*[A-Z][A-Z\s]{10,}',
    re.IGNORECASE,
)


def _fix_financial_figures(answer: str, context_texts: List[str]) -> str:
    """
    Replace '$X.X billion' rounded figures with '$XXX,XXX million' exact figures
    from the retrieved context. Matching tolerance: within 0.5%.
    Also strips citation-leakage patterns like [2, 'chunk text...'].
    """
    # Strip [n, 'chunk text'] citation leakage
    answer = _CHUNK_TEXT_CITATION_RE.sub(
        lambda m: "[" + m.group(0).split(",")[0].strip("[ ") + "]",
        answer,
    )

    # Find all exact million figures across all context chunks
    combined = " ".join(str(t) for t in context_texts)
    exact_figs = _EXACT_MILLIONS_RE.findall(combined)
    if not exact_figs:
        return answer

    def _replace(m: re.Match) -> str:
        rounded_val = float(m.group(1))
        best: Optional[str] = None
        best_diff = float("inf")
        for fig in exact_figs:
            fig_val = float(fig.replace(",", "")) / 1000.0  # millions → billions
            diff = abs(fig_val - rounded_val) / max(rounded_val, 0.001)
            if diff < 0.005 and diff < best_diff:  # within 0.5%
                best_diff = diff
                best = fig
        return f"${best} million" if best else m.group(0)

    return _ROUNDED_BILLIONS_RE.sub(_replace, answer)


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


# PHASE 24.8 — STANDARDISED SOURCES ARRAY

def _build_p248_sources(docs: List[Dict[str, Any]], max_items: int = 3) -> List[Dict[str, Any]]:
    import os as _os
    out: List[Dict[str, Any]] = []
    for doc in docs[:max_items]:
        meta  = doc.get("metadata") or {}
        text  = doc.get("text") or ""
        score = doc.get("final_score") if doc.get("final_score") is not None else doc.get("score")
        try:
            score = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0

        src_raw = meta.get("source") or meta.get("filename") or meta.get("file_path") or "unknown"
        source_name = _os.path.basename(str(src_raw)) if src_raw != "unknown" else "unknown"
        modality = str(meta.get("modality") or "text")

        page_number: Optional[int] = None
        raw_page = meta.get("page_number") if meta.get("page_number") is not None else meta.get("page")
        if isinstance(raw_page, int):
            page_number = raw_page
        elif raw_page is not None:
            try:
                page_number = int(raw_page)
            except (TypeError, ValueError):
                pass

        start_time: Optional[float] = None
        end_time:   Optional[float] = None
        for sk, tk in (("start_time", "timestamp_start"), ("end_time", "timestamp_end")):
            raw = meta.get(sk) if meta.get(sk) is not None else meta.get(tk)
            if raw is not None:
                try:
                    val = float(raw)
                    if sk == "start_time":
                        start_time = val
                    else:
                        end_time = val
                except (TypeError, ValueError):
                    pass

        # DOCX/Word: nearest heading is the locator (no page numbers at render time).
        # Excel/table: sheet name extracted from the chunk text prefix "[Sheet: X, Rows Y-Z]"
        # so existing indexed data works without re-upload.
        raw_section = meta.get("section_title")
        section_title = str(raw_section).strip() if raw_section else None

        if not section_title and modality in ("table", "excel"):
            import re as _re
            _m = _re.match(r'\[Sheet:\s*([^,\]\n]+)', str(text))
            if _m:
                section_title = _m.group(1).strip()

        # Image: prefer chart title extracted from OCR text over BLIP caption,
        # because BLIP often produces generic descriptions ("Graph showing...").
        # The OCR text contains the actual chart title as a readable line.
        if modality == "image" and not section_title:
            _title_m = re.search(
                r'(?:U\.S\.?|US)\s+(?:GDP|GNP|CPI|unemployment|employment)[^\n\r]{5,60}',
                str(text), re.IGNORECASE,
            )
            if _title_m:
                section_title = _title_m.group(0).strip()[:80]
            else:
                caption = meta.get("caption")
                if caption:
                    section_title = str(caption).strip()

        out.append({
            "text":          str(text)[:200],
            "score":         round(score, 6),
            "source":        source_name,
            "page_number":   page_number,
            "section_title": section_title,
            "start_time":    start_time,
            "end_time":      end_time,
            "modality":      modality,
            "doc_id":        str(meta.get("doc_id") or meta.get("chunk_id") or ""),
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


# AUTO FILE-SCOPE & SOURCE-COHERENCE HELPERS (duplicated from query_pipeline to
# avoid a circular import — rag_pipeline must not import query_pipeline at module level)

_AUTO_SCOPE_RE_STREAM = re.compile(
    r'\b[\w\-]{2,60}\.(?:pdf|txt|docx|xlsx|xls|doc|mp3|mp4|wav|jpg|jpeg|png)\b',
    re.IGNORECASE,
)

_COHERENCE_MAX_SOURCES_STREAM = 3
_COHERENCE_GAP_ABS_STREAM     = 0.45
_COHERENCE_ABS_FLOOR_STREAM   = 0.04


def _detect_filename_scope_stream(query: str) -> Optional[List[str]]:
    matches = _AUTO_SCOPE_RE_STREAM.findall(query)
    return [m.lower() for m in matches] if matches else None


def _source_coherence_filter_stream(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(docs) <= 1:
        return docs
    top_raw = (docs[0].get("metadata") or {}).get("_reranker_raw")
    seen_sources: Dict[str, Any] = {}
    kept: List[Dict[str, Any]] = []
    for doc in docs:
        meta = doc.get("metadata") or {}
        src  = meta.get("source", "")
        raw  = meta.get("_reranker_raw")
        if not kept:
            seen_sources[src] = raw
            kept.append(doc)
            continue
        if raw is not None and raw < _COHERENCE_ABS_FLOOR_STREAM:
            continue
        if src not in seen_sources:
            if len(seen_sources) >= _COHERENCE_MAX_SOURCES_STREAM:
                continue
            if top_raw is not None and raw is not None and raw < top_raw - _COHERENCE_GAP_ABS_STREAM:
                continue
            seen_sources[src] = raw
        kept.append(doc)
    return kept


# BUILD CONTEXT STRING — SECTION 4.6
#
# Renders retrieved chunks in Claude-style numbered references:
#
#   [1] (rag_test_corpus.txt — DOC-001 — Transformer Architecture) <chunk text>
#   [2] (rag_test_corpus.txt — DOC-002) <chunk text>
#
# The number aligns 1-to-1 with the position in the canonical `sources`
# list returned to the API, so the UI can render
#   Sources:
#   [1] rag_test_corpus.txt — DOC-001 (Transformer Architecture)
#   [2] rag_test_corpus.txt — DOC-002

def _build_context(
    docs: List[Dict[str, Any]],
    max_chars: int,
) -> str:
    parts: List[str] = []
    total: int       = 0

    for idx, d in enumerate(docs, start=1):
        text          = d.get("text", "").strip()
        meta          = d.get("metadata", {}) or {}
        source        = meta.get("source") or ""
        section_id    = meta.get("section_id")
        section_title = meta.get("section_title")
        page          = meta.get("page")
        error_markers = meta.get("error_markers") or []
        doc_version   = meta.get("doc_version")

        if not text:
            continue

        # LABEL — readable for the LLM and stable for citation parsing
        label_parts: List[str] = []
        if source:
            label_parts.append(str(source))
        if section_id:
            label_parts.append(str(section_id))
        elif page is not None:
            label_parts.append(f"p.{page}")
        if section_title:
            label_parts.append(str(section_title))
        if doc_version:
            label_parts.append(f"version={doc_version}")

        provenance = " — ".join(label_parts) if label_parts else "unknown"
        label      = f"[{idx}] ({provenance})"

        # When the chunk carries in-corpus self-flags (e.g. "intentional
        # error", "does not exist", "WRONG LABEL"), surface them on a
        # separate header line so the LLM can treat the claim as suspect.
        # The prompt builder's general branch explains this exact format.
        if error_markers:
            joined = "; ".join(str(m) for m in error_markers[:4])
            label = f"{label}\n⚠ ERROR_MARKERS={joined}"

        chunk = f"{label} {text}"[:settings.RAG_DOC_MAX_CHARS]

        if total + len(chunk) > max_chars:
            break

        parts.append(chunk)
        total += len(chunk)

    return "\n\n".join(parts)


# KEY-FACT EXTRACTOR — for queries that ask about specific events (acquisitions,
# mergers, dates) the LLM often misses sentences buried mid-chunk because the
# prompt is flattened to a single line. Prepending a "KEY FACT" line surfaces
# the most relevant sentence right after "CONTEXT:" where the LLM reads first.

_MA_QUERY_KEYWORDS  = frozenset(["acquisition", "merger", "acquired", "deal", "takeover", "purchased"])
_MA_CHUNK_KEYWORDS  = frozenset(["acquired", "acquisition", "merger", "assumed", "fdic", "purchase"])

# Phrases that mark an LLM refusal (model declined to answer despite context).
# Used by the streaming path: a refusal here is suppressed and the accurate
# meta-path answer is streamed instead, so the user never sees the flash.
_LLM_REFUSAL_PHRASES = (
    "could not find this in the provided sources",
    "could not find", "cannot find", "couldn't find",
    "no relevant information", "not in the provided", "not found in",
    "not mentioned in", "not provided in", "is not available",
    "i don't know", "i do not know", "no information about",
)

def _is_llm_refusal(text: str) -> bool:
    if not text:
        return True
    t = text.lower()
    return any(p in t for p in _LLM_REFUSAL_PHRASES)


# Streaming holdback tuning — see RAGPipeline.stream(). The prefix gate must be
# long enough to contain every _LLM_REFUSAL_PHRASES opener; the holdback tail
# must exceed the longest PII entity (emails/SSNs/phones contain no spaces, so
# an in-progress entity always sits inside the unflushed tail).
_STREAM_PREFIX_GATE = settings.STREAM_PREFIX_GATE_CHARS
_STREAM_HOLDBACK    = settings.STREAM_HOLDBACK_CHARS


# SANDWICH REORDER — Liu et al. "Lost in the Middle" (2023):
# LLMs attend best to the beginning and end of long prompts; middle positions
# are attended to least.  Placing the best-ranked chunk first and the
# second-best last keeps the two most relevant passages in the high-attention
# zones.  Middle chunks are sorted by descending relevance so any accidental
# attention still lands on the most relevant remaining content.
#
# Input:  docs sorted descending by reranker score (docs[0] is best).
# Output: [best, 3rd, 4th, …, 2nd-best]  (sandwich around the middle filler).

def _sandwich_reorder(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(docs) <= 2:
        return docs
    best   = docs[0]
    second = docs[1]
    middle = docs[2:]   # already sorted descending — middle filler
    return [best] + middle + [second]


def _adaptive_temperature(query: str) -> float:
    """Derive the generation temperature from the query type (factual vs generative)."""
    try:
        from app.prompt.prompt_builder import get_generation_temperature, _detect_query_type  # noqa: PLC0415
        return get_generation_temperature(_detect_query_type(query))
    except Exception:
        return settings.LLM_TEMPERATURE

def _prepend_key_facts(docs: List[Dict[str, Any]], query: str, context: str) -> str:
    """If the query is about an M&A event, extract the most relevant sentences
    from the top chunks and prepend them so they land first in the flat prompt."""
    if not query or not docs:
        return context
    ql = query.lower()
    if not any(kw in ql for kw in _MA_QUERY_KEYWORDS):
        return context
    facts = []
    for doc in docs[:3]:
        text = doc.get("text", "") or ""
        # Split on sentence boundaries and collect M&A sentences
        for sent in text.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|"):
            sl = sent.lower()
            if any(kw in sl for kw in _MA_CHUNK_KEYWORDS) and len(sent.strip()) > 30:
                facts.append(sent.strip())
        if len(facts) >= 3:
            break
    if not facts:
        return context
    # Cap prefix at 300 chars to avoid pushing total prompt past llama.cpp's
    # token context window (4096 tokens) on dense documents like PDFs.
    prefix_body = " | ".join(facts[:3])[:300]
    prefix = "KEY FACTS (answer the query from these): " + prefix_body + " "
    return prefix + context


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
    #
    # We use HybridRetriever directly (the live `query_pipeline` already
    # does the same). The older `app.retrieval.retriever.Retriever` class
    # is kept on disk for legacy integration tests but is no longer on the
    # production path — it had its own RRF/MMR/query-expansion that
    # diverged from HybridRetriever's, causing identical queries to return
    # different results depending on entry point.

    def _get_retriever(self):
        if self._retriever is None:
            from app.core.infra_registry import infra
            from app.core.model_loader import model_loader
            from app.retrieval.hybrid_retriever import HybridRetriever

            bm25         = infra.get_bm25()
            vector_store = infra.get_vector_store()
            embedder     = model_loader.get_embedder()
            clip_embed   = None
            if settings.ENABLE_VISION:
                try:
                    clip_embed = model_loader.get_siglip_text_embedder()
                except Exception as exc:
                    logger.warning(event="siglip_text_embedder_unavailable", error=str(exc))

            self._retriever = HybridRetriever(
                bm25=bm25,
                vector_store=vector_store,
                embedder=embedder,
                clip_text_embedder=clip_embed,
            )
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
            # add_interaction writes user + assistant to both Redis and MongoDB
            # via MemoryManager internally — do NOT call mongo.store_message()
            # separately or every turn gets written twice to the messages collection.
            mgr = self._get_memory_manager()
            mgr.add_interaction(session_id, query, answer)

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
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        start    = time.time()
        trace_id = str(uuid.uuid4())

        if not query or not query.strip():
            return {"answer": "Query cannot be empty.", "trace_id": trace_id}

        # NORMALIZE + SANITIZE — SECTION 2.3 / 5
        query = _normalize(query)
        _pre_sanitize = query
        query = _sanitize(query)

        if not query and _pre_sanitize:
            return {
                "answer": (
                    "⚠️ Your message was blocked by the security guardrail. "
                    "It matched a restricted pattern (prompt injection or policy violation). "
                    "Please rephrase your query."
                ),
                "decision": "blocked",
                "source":   "input_guard",
                "trace_id": trace_id,
            }

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

            # RETRIEVAL — SECTION 4.5 — HybridRetriever.search() returns
            # the same Dict[text/metadata/score/embedding] shape Retriever.retrieval()
            # did, but routed through the single canonical RRF + MMR + heuristic
            # query-expansion path used by the live query_pipeline.
            t_ret = time.time()
            try:
                retriever = self._get_retriever()
                raw_docs  = retriever.search(
                    query=query,
                    session_id=session_id,
                    top_k=settings.DEFAULT_TOP_K,
                    user_id=user_id,
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
            docs = _sandwich_reorder(docs)

            # CONTEXT ASSEMBLY
            from app.core.response import build_sources
            canonical_sources = build_sources(docs)
            for i, s in enumerate(canonical_sources, start=1):
                s["index"] = i
            context = _build_context(docs, settings.MAX_CONTEXT_CHARS)

            # Phase 24.8 — standardised sources array with page_number/start_time/end_time
            p248_sources = _build_p248_sources(docs)
            sources = p248_sources
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

            # PII PROMPT STRIP — Phase 26 P1: scrub PII from prompt before LLM sees it
            try:
                from app.guardrails.pii import strip_pii_from_prompt
                prompt, _pii_stripped = strip_pii_from_prompt(prompt)
                if _pii_stripped:
                    logger.info(event="rag_pipeline_pii_stripped_from_prompt", session_id=session_id)
            except Exception as _pii_err:
                logger.warning(event="rag_pipeline_pii_prompt_strip_failed", error=str(_pii_err))

            # LLM GENERATE — SECTION 4.6 FALLBACK CHAIN
            t_llm  = time.time()
            answer = ""

            try:
                llm = self._get_llm()
                if llm:
                    answer = llm.generate(
                        prompt,
                        max_tokens=settings.LLM_MAX_TOKENS,
                        temperature=_adaptive_temperature(query),
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

            # OUTPUT GUARD — Phase 26: scrub artifacts, citations, PII before memory write
            ctx_texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
            try:
                from app.guardrails.output_guard import check as _output_guard_check
                _og = _output_guard_check(
                    answer,
                    context_chunks=ctx_texts,
                    sources=sources,
                    session_id=session_id,
                    correlation_id=trace_id,
                )
                answer = _og.text
                if _og.hallucination_warning:
                    logger.warning(event="rag_pipeline_hallucination_flagged", session_id=session_id)
                if _og.fabricated_citations:
                    logger.warning(
                        event="rag_pipeline_fabricated_citations_removed",
                        citations=_og.fabricated_citations[:5],
                        session_id=session_id,
                    )
            except Exception as _og_err:
                logger.warning(event="rag_pipeline_output_guard_failed", error=str(_og_err), session_id=session_id)

            # CITATION TRACKING — filter sources to cited [n] indices, then strip them.
            try:
                from app.core.response import extract_cited_indices, strip_inline_citations
                _cited_idx = extract_cited_indices(answer)
                if _cited_idx:
                    _filtered = [s for i, s in enumerate(sources, 1) if i in _cited_idx]
                    if _filtered:
                        sources = _filtered[:5]
                else:
                    sources = sources[:3]
                answer = strip_inline_citations(answer)
            except Exception as _cit_err:
                logger.warning(event="rag_pipeline_citation_tracking_failed", error=str(_cit_err), session_id=session_id)

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

            # Phase 24.8 — confidence + hallucination_warning
            _scores = [s["score"] for s in sources[:3] if isinstance(s.get("score"), (int, float))]
            confidence = round(max(0.0, min(sum(_scores) / len(_scores) if _scores else 0.0, 1.0)), 6)
            hallucination_warning = confidence < settings.AGENT_LOW_CONFIDENCE

            return {
                "answer":               answer,
                "sources":              sources,
                "confidence":           confidence,
                "hallucination_warning": hallucination_warning,
                "latency":              total_latency,
                "trace_id":             trace_id,
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
        user_id: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> Iterator[str]:

        query = _normalize(query)
        query = _sanitize(query)
        query = query[:settings.MAX_PROMPT_CHARS]

        def _generator() -> Iterator[str]:
            try:
                retriever = self._get_retriever()
                _auto_scope    = _detect_filename_scope_stream(query) if not sources else None
                _stream_filters = (
                    {"sources": sources}    if sources
                    else {"sources": _auto_scope} if _auto_scope
                    else None
                )
                raw_docs  = retriever.search(
                    query=query,
                    session_id=session_id,
                    top_k=settings.DEFAULT_TOP_K,
                    user_id=user_id,
                    filters=_stream_filters,
                )

                docs    = _normalize_docs(raw_docs)
                docs    = _dedup_docs(docs)

                # RETRIEVAL GATE — the genuine "no documents" case is decided
                # HERE, deterministically. The retriever already applies
                # HYBRID_MIN_SCORE / BM25_MIN_SCORE, so an empty result means
                # nothing relevant exists: emit the canonical message and skip
                # the LLM. When docs DO exist and the model still refuses, that
                # refusal is caught below and the accurate meta answer is used
                # instead — so the user never sees a spurious refusal flash.
                if not docs:
                    logger.info(
                        event="rag_stream_no_relevant_docs",
                        session_id=session_id,
                    )
                    yield (
                        "I could not find any relevant documents in your "
                        "knowledge base to answer this question."
                    )
                    return

                # RERANK — use the module-level singleton so the cross-encoder model
                # is loaded once and shared across all streaming requests. Fixes
                # "lost in the middle" failures where BM25/Qdrant cosine rank 1
                # differs from semantic rank 1 (e.g. quantum-computing chunk beats
                # Business-Segments for acquisition queries without reranking).
                try:
                    _rer = _get_stream_reranker()
                    if _rer is not None:
                        _reranked = _rer.rerank(query, docs, top_k=settings.RAG_TOP_K, session_id=session_id)
                        if _reranked:
                            docs = _source_coherence_filter_stream(_reranked)
                        else:
                            docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)[:settings.RAG_TOP_K]
                    else:
                        docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)[:settings.RAG_TOP_K]
                except Exception as _re_err:
                    logger.warning(event="rag_stream_rerank_failed", error=str(_re_err))
                    docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)[:settings.RAG_TOP_K]
                docs = _sandwich_reorder(docs)

                context = _build_context(docs, settings.MAX_CONTEXT_CHARS)
                context = _prepend_key_facts(docs, query, context)

                builder = self._get_prompt_builder()
                prompt  = builder.build_prompt(
                    query=query,
                    context=context,
                    session_id=session_id,
                )

                # PII PROMPT STRIP — same as non-streaming path (Phase 26 P1)
                try:
                    from app.guardrails.pii import strip_pii_from_prompt as _spfp
                    prompt, _pii_stripped = _spfp(prompt)
                    if _pii_stripped:
                        logger.info(event="rag_stream_pii_stripped_from_prompt",
                                    session_id=session_id)
                except Exception as _pii_err:
                    logger.warning(event="rag_stream_pii_prompt_strip_failed",
                                   error=str(_pii_err))

                llm = self._get_llm()
                if not llm:
                    yield "LLM unavailable."
                    return

                # TRUE TOKEN STREAMING with a holdback buffer.
                # - The first _STREAM_PREFIX_GATE chars are held back so a refusal
                #   ("I could not find…") is detected BEFORE anything reaches the
                #   client — preserving the refusal-sentinel UX below.
                # - After the gate, segments are flushed at whitespace boundaries
                #   keeping a _STREAM_HOLDBACK-char tail, so a PII entity that
                #   spans tokens is never split across a flush (it sits in the
                #   tail until complete, then gets scrubbed).
                # - The FULL output guard still runs on the complete answer; its
                #   canonical text is sent as a \x00REPLACE\x00 sentinel so the
                #   client and persistence always end on the guarded version.
                from app.guardrails.pii import scrub_pii as _scrub_pii_seg

                collected_tokens: List[str] = []
                hold = ""
                refusal_mode = False
                prefix_checked = False

                def _flush(seg: str) -> str:
                    try:
                        scrubbed, _ = _scrub_pii_seg(seg)
                        return scrubbed
                    except Exception:
                        return seg

                for token in llm.stream(
                    prompt,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=_adaptive_temperature(query),
                    top_p=settings.LLM_TOP_P,
                    session_id=session_id,
                ):
                    collected_tokens.append(token)
                    if refusal_mode:
                        continue
                    hold += token
                    if not prefix_checked:
                        if len(hold) < _STREAM_PREFIX_GATE:
                            continue
                        if _is_llm_refusal(hold):
                            refusal_mode = True
                            continue
                        prefix_checked = True
                    cut = len(hold) - _STREAM_HOLDBACK
                    if cut <= 0:
                        continue
                    ws = hold.rfind(" ", 0, cut)
                    if ws <= 0:
                        continue
                    out = _flush(hold[:ws])
                    hold = hold[ws:]
                    if out:
                        yield out

                # OUTPUT GUARD — Phase 26 P1b: guard the COMPLETE answer.
                # Whole-answer checks (groundedness, template artifacts, toxicity,
                # cross-segment PII) cannot run mid-stream; their canonical result
                # is delivered via the REPLACE sentinel after the token stream.
                answer = "".join(collected_tokens).strip()
                _ctx: List[str] = [
                    d.get("text", "") if isinstance(d, dict)
                    else (d.page_content if hasattr(d, "page_content") else "")
                    for d in docs
                ]
                try:
                    from app.guardrails.output_guard import check as _og_check
                    _sources = [{"filename": d.get("metadata", {}).get("source", "") if isinstance(d, dict) else ""} for d in docs]
                    _og = _og_check(answer, context_chunks=_ctx, sources=_sources, session_id=session_id)
                    answer = _og.text
                except Exception as _og_err:
                    logger.warning(event="rag_stream_output_guard_failed", error=str(_og_err))

                # FINANCIAL FIGURE NORMALIZER — deterministic post-processor:
                # replace any '$X.X billion' the LLM wrote with the exact
                # '$XXX,XXX million' figure from context (if within 0.5%).
                # This is context-grounded and does not depend on prompt rules.
                try:
                    answer = _fix_financial_figures(answer, _ctx)
                except Exception as _fin_err:
                    logger.warning(event="rag_fin_normalizer_failed", error=str(_fin_err))

                # REFUSAL HANDLING — docs WERE retrieved (we passed the retrieval
                # gate above), so a refusal here means the model declined despite
                # relevant context. Do NOT stream the refusal text: it would flash
                # letter-by-letter and then be replaced, which is the exact UX bug
                # we are fixing. Instead emit a sentinel so the client fetches the
                # accurate meta-path answer (its lazy /rag/query fallback). The
                # genuine "no documents" case never reaches here — it is handled
                # by the deterministic retrieval gate above.
                if not answer or _is_llm_refusal(answer):
                    logger.info(event="rag_stream_llm_refused_using_meta", session_id=session_id)
                    yield "\x00REFUSAL\x00"
                    return

                # CITATION TRACKING — parse [n] indices, filter source chips, then strip.
                try:
                    from app.core.response import extract_cited_indices, strip_inline_citations
                    _cited_idx = extract_cited_indices(answer)
                    if _cited_idx:
                        _cited_docs = [d for i, d in enumerate(docs, 1) if i in _cited_idx]
                        _source_docs = _cited_docs[:5] if _cited_docs else docs[:3]
                    else:
                        _source_docs = docs[:3]
                    answer = strip_inline_citations(answer)
                except Exception:
                    _source_docs = docs[:3]

                # REPLACE sentinel — canonical guarded answer. The client swaps
                # its streamed buffer for this text; the API stream layer is the
                # single persistence point (memory + Mongo + cache) — writing
                # here too would duplicate every exchange in MongoDB.
                yield "\x00REPLACE\x00" + answer

                # Emit sources so the client can display them immediately.
                try:
                    import json as _json
                    _p248 = _build_p248_sources(_source_docs)
                    yield "\x00SOURCES\x00" + _json.dumps(_p248)
                except Exception:
                    pass

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


