"""base_embedder.py

Shared base for all per-modality embedders (Phase 3).

Every modality embedder extends BaseEmbedder and overrides _build_embed_text()
to return the modality-specific enriched string for encoding. All encoding,
caching, batching, and validation logic lives here once.

text_embedder.py (legacy pipeline path) is kept untouched alongside this.
"""

from __future__ import annotations

import asyncio as _asyncio
import hashlib
import json
import math
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path as _Path
from typing import Any

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# FINANCE NUMBER NORMALIZATION
# Identical to the copy in text_embedder.py — kept here so base_embedder has
# no import dependency on text_embedder (avoids circular risk).
# ══════════════════════════════════════════════════════════════════════════════

_SCALE_EXPAND: dict[str, tuple[float, str, str]] = {
    "b": (1e9, "billion", "million"),
    "bn": (1e9, "billion", "million"),
    "t": (1e12, "trillion", "billion"),
    "tn": (1e12, "trillion", "billion"),
    "m": (1e6, "million", "billion"),
    "mn": (1e6, "million", "billion"),
    "k": (1e3, "thousand", "million"),
}

_FIN_NUM_RE = re.compile(r'([$€£¥₹]?)' r'([\d,]+\.?\d*)' r'\s*([BMKTbmkt]n?)\b')
_PCT_RE = re.compile(r'([\d.]+)\s*%')
_BPS_RE = re.compile(r'([\d.]+)\s*bps\b', re.IGNORECASE)
_QTR_RE = re.compile(r'\bQ([1-4])\s*(?:FY)?\s*(\d{2,4})\b', re.IGNORECASE)
_HY_RE = re.compile(r'\bH([12])\s+(\d{4})\b', re.IGNORECASE)

_QTR_WORDS = {"1": "first", "2": "second", "3": "third", "4": "fourth"}
_HY_WORDS = {"1": "first", "2": "second"}


def normalize_finance_numbers(text: str) -> str:
    """Append expanded forms of finance numbers so scale variants match at query time."""
    extras: list[str] = []
    for m in _FIN_NUM_RE.finditer(text):
        raw_num = m.group(2).replace(",", "")
        suffix = m.group(3).lower().rstrip("n")
        if suffix not in _SCALE_EXPAND:
            continue
        try:
            num_val = float(raw_num)
        except ValueError:
            continue
        mult, full_word, cross_word = _SCALE_EXPAND[suffix]
        cross_divisor = 1e9 if cross_word == "billion" else 1e6 if cross_word == "million" else 1e3
        cross_val = num_val * mult / cross_divisor
        extras.append(f"{num_val} {full_word}")
        extras.append(f"{num_val} {full_word} dollars")
        cross_str = f"{cross_val:.1f}".rstrip("0").rstrip(".")
        extras.append(f"{cross_str} {cross_word}")
    for m in _PCT_RE.finditer(text):
        extras.append(f"{m.group(1)} percent")
    for m in _BPS_RE.finditer(text):
        extras.append(f"{m.group(1)} basis points")
    for m in _QTR_RE.finditer(text):
        q, yr = m.group(1), m.group(2)
        if len(yr) == 2:
            yr = "20" + yr
        extras.append(f"Q{q} fiscal year {yr}")
        extras.append(f"{_QTR_WORDS.get(q, 'Q'+q)} quarter {yr}")
    for m in _HY_RE.finditer(text):
        h, yr = m.group(1), m.group(2)
        extras.append(f"{_HY_WORDS.get(h, 'H'+h)} half {yr}")
    return (text + " " + " ".join(extras)) if extras else text


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def valid_embedding(emb: list[float], expected_dim: int) -> bool:
    if not isinstance(emb, list) or len(emb) != expected_dim:
        return False
    return not any(math.isnan(v) or math.isinf(v) for v in emb)


def sanitize_text(text: str) -> str | None:
    if not text:
        return None
    text = unicodedata.normalize("NFC", text.strip())
    if settings.STRIP_NULL_BYTES and "\x00" in text:
        text = text.replace("\x00", "")
    if settings.STRIP_BOM:
        text = text.lstrip("﻿￾")
    if not text:
        return None
    try:
        from app.guardrails.input_guard import sanitize as _guard

        text = _guard(text, surface="embedder") or text
    except Exception:
        pass
    if len(text) > settings.MAX_PROMPT_CHARS:
        text = text[: settings.MAX_PROMPT_CHARS]
    return text or None


# ══════════════════════════════════════════════════════════════════════════════
# REDIS + LRU EMBEDDING CACHE
# ══════════════════════════════════════════════════════════════════════════════


class _EmbeddingCache:
    """Two-tier embedding cache: in-process dict + LOCAL Redis (~0.5ms/op).

    Backed by infra.get_cache() (local Redis), NOT Upstash — a per-chunk cloud
    write at ~200ms each was adding ~16s to every ingest. Local writes are ~0.5ms
    so the per-doc set stays inline without blocking the GPU encode loop. The
    in-process dict serves immediate re-reads within the same process.
    """

    def __init__(self) -> None:
        self._local: dict[str, list[float]] = {}
        self._cache = None
        self._cache_resolved = False

    def _get_cache(self):
        if self._cache_resolved:
            return self._cache
        try:
            from app.core.infra_registry import infra

            self._cache = infra.get_cache()
        except Exception:
            self._cache = None
        self._cache_resolved = True
        return self._cache

    def _key(self, text: str, model: str, dim: int) -> str:
        raw = f"{model}:{dim}:{text[:500]}"
        return "emb:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, model: str, dim: int) -> list[float] | None:
        key = self._key(text, model, dim)
        if key in self._local:
            return self._local[key]
        cache = self._get_cache()
        if cache is not None:
            try:
                raw = cache.get(key)
                if raw:
                    cached = json.loads(raw)
                    if isinstance(cached, list):
                        self._local[key] = cached
                        return cached
            except Exception:
                pass
        return None

    def set(self, text: str, model: str, dim: int, embedding: list[float]) -> None:
        key = self._key(text, model, dim)
        self._local[key] = embedding
        cache = self._get_cache()
        if cache is not None:
            try:
                cache.set(key, json.dumps(embedding), ex=settings.EMBEDDING_CACHE_TTL)
            except Exception:
                pass


_shared_cache = _EmbeddingCache()


# ══════════════════════════════════════════════════════════════════════════════
# BASE EMBEDDER
# ══════════════════════════════════════════════════════════════════════════════


class BaseEmbedder(ABC):
    """Abstract base for all per-modality embedders.

    Subclasses override _build_embed_text(doc, cleaned_text) to return the
    modality-specific enriched string. All encoding, caching, and batching
    is handled here.
    """

    def _get_model(self):
        """Lazy accessor for the shared TextEmbedder singleton."""
        from app.core.model_loader import model_loader

        return model_loader.get_embedder()

    # ── Abstract interface ──────────────────────────────────────────────────

    @abstractmethod
    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        """Build the full enriched string to encode for this document.

        Args:
            doc: IngestedDocument instance.
            cleaned_text: Already sanitized + finance-normalized text.

        Returns:
            Final string to pass to the encoder (max MAX_PROMPT_CHARS).
        """
        ...

    # ── Public API ──────────────────────────────────────────────────────────

    def embed_documents(
        self,
        docs: list[Any],
        session_id: str = "default",
    ) -> list[Any]:
        """Embed a list of IngestedDocument objects.

        Sets doc.embedding on each successful doc. Returns list of docs that
        received a valid embedding.
        """
        if not docs:
            return []
        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        embedder = self._get_model()
        start = time.time()

        texts: list[str] = []
        valid_docs: list[Any] = []
        seen: dict[str, bool] = {}

        for doc in docs:
            try:
                raw = getattr(doc, "text", "") or ""
                clean = sanitize_text(raw)
                if not clean:
                    continue
                clean = normalize_finance_numbers(clean)
                enriched = self._build_embed_text(doc, clean)
                enriched = enriched[: settings.MAX_PROMPT_CHARS]
                h = hashlib.sha256(enriched.encode("utf-8")).hexdigest()
                if h in seen:
                    continue
                seen[h] = True
                texts.append(enriched)
                valid_docs.append(doc)
            except Exception as exc:
                logger.warning(event="embed_doc_prep_skip", error=str(exc), session_id=session_id)

        if not valid_docs:
            return []

        cap = settings.INGESTION_BATCH_SIZE * 10
        texts = texts[:cap]
        valid_docs = valid_docs[:cap]

        results: list[Any] = []

        for i in range(0, len(texts), embedder.batch_size):
            batch_texts = texts[i : i + embedder.batch_size]
            batch_docs = valid_docs[i : i + embedder.batch_size]
            try:
                embs = embedder._encode_with_retry(embedder.model, batch_texts)
                for doc, emb, txt in zip(batch_docs, embs, batch_texts, strict=False):
                    if not valid_embedding(emb, embedder.expected_dim):
                        continue
                    doc.embedding = emb
                    struct = dict(getattr(doc, "structure", {}) or {})
                    struct["embedding_space"] = "text"
                    struct["embedding_model"] = embedder.model_name
                    doc.structure = struct
                    _shared_cache.set(txt, embedder.model_name, embedder.expected_dim, emb)
                    results.append(doc)
            except Exception as exc:
                logger.error(
                    event="embed_batch_failed",
                    batch_start=i,
                    error=str(exc),
                    session_id=session_id,
                )

        # ── FinBERT finance tone annotation ──────────────────────────────────
        # Augment chunk metadata with finance sentiment/tone from FinBERT
        # (yiyanghkust/finbert-tone) per Phase 2 Global Rules requirement.
        # This adds a 'finance_tone' field to structure without changing vectors.
        if getattr(settings, "FINBERT_ENABLED", False) and results:
            self._annotate_finbert_tone(results)

        latency = round(time.time() - start, 3)
        throughput = round(len(results) / max(latency, 1e-6), 1)
        logger.info(
            event="embed_documents_done",
            modality=self.__class__.__name__,
            embedded=len(results),
            total=len(valid_docs),
            throughput_per_sec=throughput,
            latency=latency,
            session_id=session_id,
        )
        return results

    def _annotate_finbert_tone(self, docs: list[Any]) -> None:
        """Run FinBERT tone classification on embedded docs and store result in structure.

        Labels: 'positive' | 'negative' | 'neutral'
        Score stored in structure['finance_tone'] and structure['finance_tone_score'].
        Runs in batches; silently skips on any error so it never blocks embedding.
        """
        try:
            from app.core.model_loader import model_loader

            finbert = model_loader.get_finbert()
            if finbert is None:
                return
        except Exception:
            return

        batch_size = getattr(settings, "FINBERT_BATCH_SIZE", 32)
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts = []
            for doc in batch:
                t = getattr(doc, "text", "") or ""
                texts.append(t[:512])  # FinBERT max 512 tokens
            try:
                preds = finbert(texts, truncation=True, max_length=512)
                for doc, pred in zip(batch, preds, strict=False):
                    struct = dict(getattr(doc, "structure", {}) or {})
                    struct["finance_tone"] = pred["label"].lower()
                    struct["finance_tone_score"] = round(float(pred["score"]), 4)
                    doc.structure = struct
            except Exception as exc:
                logger.debug(event="finbert_batch_skip", error=str(exc))

    def embed_query(self, query: str, session_id: str = "default") -> list[float]:
        """Embed a query string.

        Applies BGE_QUERY_INSTRUCTION from config as explicit prefix when the
        underlying model has no registered query prompt (ensures finance-domain
        instruction is always active regardless of SentenceTransformers version).
        """
        embedder = self._get_model()
        # Only prepend if the model hasn't already registered a query prompt;
        # prevents double-prefixing for instruction-tuned models (Qwen3/E5).
        if not (
            getattr(embedder, "_query_prompt_name", None)
            or getattr(embedder, "_query_prompt_text", None)
        ):
            instruction = getattr(settings, "BGE_QUERY_INSTRUCTION", "")
            if instruction:
                query = instruction + query
        return embedder.embed_text(query, session_id)


# ══════════════════════════════════════════════════════════════════════════════
# MODALITY-SPECIFIC CONTEXT ENRICHMENT
# Used by TextEmbedder.embed_documents() (legacy pipeline path) to produce a
# context-rich string before encoding. Per-modality embedders use
# _build_embed_text() instead.
# ══════════════════════════════════════════════════════════════════════════════


def _modality_prefix(doc: Any) -> str:
    m = (getattr(doc, "modality", "") or "").lower()
    st = (getattr(doc, "subtype", "") or "").lower()
    s = getattr(doc, "structure", {}) or {}
    if m == "table":
        return "Table: "
    if m == "image":
        return "OCR: " if st == "ocr" else "Image: "
    if m == "audio":
        ts = s.get("timestamp_start")
        return f"Audio {ts:.1f}s: " if ts is not None else "Audio: "
    if m == "video":
        ts = s.get("timestamp_start")
        fi = s.get("frame_index")
        if st == "speech":
            return f"Video speech {ts:.1f}s: " if ts is not None else "Video speech: "
        if st == "frame":
            if fi is not None and ts is not None:
                return f"Video frame {fi} @{ts:.1f}s: "
            if fi is not None:
                return f"Video frame {fi}: "
            return f"Video frame @{ts:.1f}s: " if ts is not None else "Video frame: "
        if st == "ocr":
            return "Video OCR: "
    if m == "text" and st == "heading":
        return "Heading: "
    return ""


def _enrich_doc(doc: Any, text: str) -> str:
    """Build context-enriched embedding input for a document (legacy shared path)."""
    context: list[str] = []
    src_type = (getattr(doc, "source_type", "") or "").lower()
    modality = (getattr(doc, "modality", "") or "").lower()
    s = getattr(doc, "structure", {}) or {}

    if getattr(doc, "source_type", None):
        context.append(f"[{doc.source_type.upper()}]")
    if modality:
        context.append(f"[{modality.upper()}]")
    if getattr(doc, "page", None):
        context.append(f"[Page {doc.page}]")
    if getattr(doc, "source", None):
        context.append(f"[{doc.source}]")

    section_title = s.get("section_title") or ""
    if section_title and len(section_title) <= 120:
        context.append(f"[Section: {section_title}]")

    sub_idx = s.get("sub_chunk_index")
    sub_total = s.get("total_sub_chunks")
    if sub_idx is not None and sub_total and int(sub_total) > 1:
        context.append(f"[Part {int(sub_idx)+1}/{int(sub_total)}]")

    row_start, row_end = s.get("row_start"), s.get("row_end")
    if row_start is not None and row_end is not None:
        context.append(f"[Rows {row_start}-{row_end}]")

    ts_start, ts_end = s.get("timestamp_start"), s.get("timestamp_end")
    if modality in ("audio", "video") and ts_start is not None:
        if ts_end is not None:
            context.append(f"[{float(ts_start):.1f}s-{float(ts_end):.1f}s]")
        else:
            context.append(f"[{float(ts_start):.1f}s]")

    if modality == "video" and s.get("frame_index") is not None:
        context.append(f"[Frame {s['frame_index']}]")

    context_prefix = " ".join(context) + " " + _modality_prefix(doc)

    if src_type == "word" or (s.get("content_type") or "").lower().startswith("docx_"):
        hierarchy = s.get("section_hierarchy") or []
        if hierarchy:
            text = f"Section: {' > '.join(str(h) for h in hierarchy)} | {text}"

    if src_type == "excel" or (s.get("content_type") or "").lower().startswith("excel_"):
        sheet = (s.get("sheet") or s.get("sheet_name") or "").strip()
        unit_scale = (s.get("unit_scale") or "").strip()
        parts = []
        if sheet:
            parts.append(f"Sheet: {sheet}")
        if unit_scale:
            parts.append(f"({unit_scale})")
        if parts:
            text = " | ".join(parts) + " | " + text

    if modality in ("audio", "video") and (s.get("content_type") or "").endswith(
        ("speech", "audio_speech_segment", "video_speech")
    ):
        speaker = (s.get("speaker") or s.get("speaker_role") or "").strip()
        ts_s, ts_e = s.get("timestamp_start"), s.get("timestamp_end")
        header = f"[SPEAKER: {speaker}]" if speaker else ""
        if ts_s is not None and ts_e is not None:
            header += f" [{float(ts_s):.0f}s-{float(ts_e):.0f}s]"
        entities = s.get("finance_entities") or {}
        entity_tokens: list[str] = []
        if isinstance(entities, dict):
            for v in entities.values():
                if isinstance(v, list):
                    entity_tokens.extend(str(x) for x in v[:3])
        suffix = f" [ENTITIES: {', '.join(entity_tokens)}]" if entity_tokens else ""
        if header:
            text = f"{header} {text}{suffix}"
        elif entity_tokens:
            text = f"{text}{suffix}"

    if modality == "video":
        combined = (s.get("combined_text") or "").strip()
        if combined and len(combined) > len(text):
            text = combined

    if modality == "image" or src_type == "image":
        image_type = (s.get("image_type") or "").strip()
        if image_type:
            text = f"{image_type.replace('_', ' ')}: {text}"

    return (context_prefix + text).strip()[: settings.MAX_PROMPT_CHARS]


# ══════════════════════════════════════════════════════════════════════════════
# TEXT EMBEDDER — BGE SentenceTransformer model wrapper
# Used by BaseEmbedder._get_model() and MultimodalEmbedder.
# Moved from text_embedder.py (now deleted).
# ══════════════════════════════════════════════════════════════════════════════


class TextEmbedder:
    """BGE-large-en-v1.5 (or configured model) SentenceTransformer wrapper.

    Provides: embed_text, embed_texts, embed_documents, embed_query,
    embed_matryoshka, embed_sparse.  Used as the shared encoding kernel by
    all per-modality embedders via model_loader.get_embedder().
    """

    def __init__(self, model_name: str, batch_size: int, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.expected_dim = settings.TEXT_EMBEDDING_DIM
        self.max_text_len = settings.MAX_PROMPT_CHARS
        self.model = SentenceTransformer(model_name, device=device)

        self._query_prompt_name: str | None = None
        self._query_prompt_text: str | None = None
        _prompts = getattr(self.model, "prompts", {}) or {}
        if "query" in _prompts:
            self._query_prompt_name = "query"
        elif any(k in model_name.lower() for k in ("qwen3", "e5-instruct", "gte-qwen")):
            self._query_prompt_text = (
                "Instruct: Given a financial document query, retrieve the most "
                "relevant passage that answers the question\nQuery: "
            )

        logger.info(
            event="text_embedder_initialized",
            model=model_name,
            device=device,
            dim=self.expected_dim,
            query_prompt_name=self._query_prompt_name,
            query_prompt_text=bool(self._query_prompt_text),
        )

    def _encode_with_retry(
        self, model, texts: list[str], max_retries: int = 3
    ) -> list[list[float]]:
        import time as _time

        wait = 1.0
        for attempt in range(max_retries):
            try:
                t_start = _time.time()
                embs = model.encode(
                    texts,
                    batch_size=len(texts),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                elapsed_ms = round((_time.time() - t_start) * 1000, 1)
                if elapsed_ms > settings.LATENCY_TARGET_EMBED_BATCH_MS:
                    logger.warning(
                        event="embed_batch_latency_exceeded",
                        latency_ms=elapsed_ms,
                        target_ms=settings.LATENCY_TARGET_EMBED_BATCH_MS,
                        batch_size=len(texts),
                    )
                return [e.tolist() for e in embs]
            except Exception as e:
                logger.warning(event="embed_encode_retry", attempt=attempt, error=str(e))
                if attempt >= max_retries - 1:
                    raise
                _time.sleep(wait)
                wait = min(wait * 2, 10.0)
        return []

    def embed_text(self, text: str, session_id: str = "default") -> list[float]:
        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")
        clean = sanitize_text(text)
        if not clean:
            raise ValueError("EMPTY_TEXT")
        clean = normalize_finance_numbers(clean)
        _cache_model_key = (
            self.model_name + ":q"
            if (self._query_prompt_name or self._query_prompt_text)
            else self.model_name
        )
        cached = _shared_cache.get(clean, _cache_model_key, self.expected_dim)
        if cached is not None:
            logger.debug(event="embed_cache_hit", session_id=session_id)
            return cached
        import time as _time

        t_start = _time.time()
        _kwargs: dict = dict(convert_to_numpy=True, normalize_embeddings=True)
        if self._query_prompt_name:
            _kwargs["prompt_name"] = self._query_prompt_name
            _text_to_encode = clean
        elif self._query_prompt_text:
            _text_to_encode = self._query_prompt_text + clean
        else:
            _text_to_encode = clean
        emb = self.model.encode(_text_to_encode, **_kwargs).tolist()
        if not valid_embedding(emb, self.expected_dim):
            raise ValueError("INVALID_EMBEDDING_SINGLE")
        _shared_cache.set(clean, _cache_model_key, self.expected_dim, emb)
        logger.debug(
            event="embed_single_success",
            latency_ms=round((_time.time() - t_start) * 1000, 1),
            session_id=session_id,
        )
        return emb

    def embed_texts(self, texts: list[str], session_id: str = "default") -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            raw_batch = texts[i : i + self.batch_size]
            clean_batch: list[str] = []
            cached_results: list[list[float] | None] = []
            for t in raw_batch:
                c = sanitize_text(t)
                if not c:
                    cached_results.append(None)
                    clean_batch.append("")
                    continue
                hit = _shared_cache.get(c, self.model_name, self.expected_dim)
                if hit is not None:
                    cached_results.append(hit)
                    clean_batch.append("")
                else:
                    cached_results.append(None)
                    clean_batch.append(c)
            to_encode = [(idx, t) for idx, t in enumerate(clean_batch) if t]
            encoded: dict[str, list[float]] = {}
            if to_encode:
                indices, batch_texts = zip(*to_encode, strict=False)
                try:
                    embs = self._encode_with_retry(self.model, list(batch_texts))
                    for idx, emb in zip(indices, embs, strict=False):
                        if valid_embedding(emb, self.expected_dim):
                            encoded[idx] = emb
                            _shared_cache.set(
                                clean_batch[idx], self.model_name, self.expected_dim, emb
                            )
                except Exception as e:
                    logger.error(
                        event="embed_texts_batch_failed", error=str(e), session_id=session_id
                    )
            for idx, cached_emb in enumerate(cached_results):
                if cached_emb is not None:
                    results.append(cached_emb)
                elif idx in encoded:
                    results.append(encoded[idx])
        return results

    def embed_documents(self, documents: list[Any], session_id: str = "default") -> list[Any]:
        import time as _time

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")
        if not documents:
            return []
        start = _time.time()
        texts: list[str] = []
        valid_docs: list[Any] = []
        seen: dict[str, bool] = {}
        for doc in documents:
            try:
                raw = getattr(doc, "text", "") or ""
                clean = sanitize_text(raw)
                if not clean:
                    continue
                clean = normalize_finance_numbers(clean)
                enriched = _enrich_doc(doc, clean)
                h = hashlib.sha256(enriched.encode("utf-8")).hexdigest()
                if h in seen:
                    continue
                seen[h] = True
                texts.append(enriched)
                valid_docs.append(doc)
            except Exception as e:
                logger.warning(event="embed_doc_skip", error=str(e), session_id=session_id)
        if not valid_docs:
            return []
        cap = settings.INGESTION_BATCH_SIZE * 10
        texts, valid_docs = texts[:cap], valid_docs[:cap]
        results: list[Any] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_docs = valid_docs[i : i + self.batch_size]
            try:
                embs = self._encode_with_retry(self.model, batch_texts)
                for doc, emb in zip(batch_docs, embs, strict=False):
                    if not valid_embedding(emb, self.expected_dim):
                        continue
                    doc.embedding = emb
                    struct = dict(doc.structure or {})
                    struct["embedding_space"] = "text"
                    struct["embedding_model"] = self.model_name
                    doc.structure = struct
                    text_for_cache = sanitize_text(getattr(doc, "text", "") or "")
                    if text_for_cache:
                        _shared_cache.set(text_for_cache, self.model_name, self.expected_dim, emb)
                    results.append(doc)
            except Exception as e:
                logger.error(
                    event="embed_batch_failed", batch_start=i, error=str(e), session_id=session_id
                )
        total_latency = round(_time.time() - start, 3)
        logger.info(
            event="embed_documents_success",
            embedded=len(results),
            total=len(valid_docs),
            throughput_per_sec=round(len(results) / max(total_latency, 1e-6), 1),
            latency=total_latency,
            session_id=session_id,
        )
        return results

    def embed_query(self, query: str, session_id: str = "default") -> list[float]:
        return self.embed_text(query, session_id)

    def embed_matryoshka(
        self, text: str, session_id: str = "default"
    ) -> tuple[list[float], list[float]]:
        full = self.embed_text(text, session_id)
        short = full[: settings.MATRYOSHKA_SHORT_DIM]
        norm = math.sqrt(sum(v * v for v in short)) + 1e-10
        return [v / norm for v in short], full

    def embed_sparse(self, text: str, vocab_size: int = 30000) -> dict[str, float]:
        import re as _re

        clean = sanitize_text(text)
        if not clean:
            return {}
        tokens = _re.findall(r"\b[a-z0-9]+\b", clean.lower())
        tf: dict[int, float] = {}
        for tok in tokens:
            idx = hash(tok) % vocab_size
            tf[idx] = tf.get(idx, 0.0) + 1.0
        total = sum(tf.values()) + 1e-10
        return {k: round(v / total, 6) for k, v in tf.items()}

    def health_check(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "device": self.device,
            "dim": self.expected_dim,
            "batch_size": self.batch_size,
            "cache_local": len(_shared_cache._local),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MULTIMODAL EMBEDDER — text + vision orchestration
# Moved from multimodal_embedder.py (now deleted).
# ══════════════════════════════════════════════════════════════════════════════


class MultimodalEmbedderError(Exception):
    pass


class NoValidDocumentsError(MultimodalEmbedderError):
    pass


class EmbeddingDimensionError(MultimodalEmbedderError):
    pass


class SessionIdRequiredError(MultimodalEmbedderError):
    pass


def _mm_valid_embedding(emb: Any, expected_dim: int) -> bool:
    if not isinstance(emb, list) or len(emb) != expected_dim:
        return False
    return not any(math.isnan(v) or math.isinf(v) for v in emb)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _modality_counts(docs: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in docs:
        m = getattr(d, "modality", "unknown")
        counts[m] = counts.get(m, 0) + 1
    return counts


def _resolve_asset_path(doc: Any) -> str | None:
    structure = getattr(doc, "structure", {}) or {}
    for key in ("asset_path", "frame_path", "source_path"):
        path = structure.get(key)
        if not path:
            continue
        p = _Path(str(path))
        if p.exists():
            return str(p)
        logger.warning(event="multimodal_asset_path_not_found", key=key, path=str(path))
    return None


def _route_documents(docs: list[Any]) -> tuple[list[Any], list[Any]]:
    """Route documents to the text (BGE) or vision (SigLIP) embedder.

    Routing trusts the explicit ``structure["embedding_space"]`` set by the
    chunkers — a "vision" doc goes to SigLIP/vision_collection, everything else to
    BGE/text_collection. A single doc is NEVER routed to both: doing so aliased the
    same object through both embedders, where the second embed overwrote the first
    (e.g. video frames collapsing to one point, images losing their text vector).
    Chunkers that want both representations emit two separate docs (one per space).

    Legacy docs without an explicit embedding_space fall back to modality/subtype.
    """
    text_docs: list[Any] = []
    vision_docs: list[Any] = []
    for doc in docs:
        try:
            space = (getattr(doc, "structure", {}) or {}).get("embedding_space")
            if space == "vision":
                vision_docs.append(doc)
                continue
            if space == "text":
                text_docs.append(doc)
                continue
            # ── Legacy fallback: no explicit embedding_space on the doc ──────────
            modality = getattr(doc, "modality", "")
            subtype = getattr(doc, "subtype", "") or ""
            if (
                modality == "image"
                and subtype in ("caption", "image_frame")
                and _resolve_asset_path(doc)
            ):
                vision_docs.append(doc)
            elif modality == "video" and subtype == "frame":
                vision_docs.append(doc)
            else:
                text_docs.append(doc)
        except Exception as exc:
            logger.warning(event="multimodal_route_failed", error=str(exc))
    return text_docs, vision_docs


class MultimodalEmbedder:
    """Routes IngestedDocuments to TextEmbedder (BGE) or ImageEmbedder (SigLIP).

    Used by ingestion_pipeline.py for the vision embedding path.
    Moved from multimodal_embedder.py.
    """

    def __init__(self, text_embedder, image_embedder) -> None:
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.max_docs = settings.INGESTION_BATCH_SIZE * 10
        self._text_model_name = settings.EMBEDDING_MODEL
        self._vision_model_name = settings.SIGLIP_MODEL
        logger.info(
            event="multimodal_embedder_initialized",
            text_model=self._text_model_name,
            vision_model=self._vision_model_name,
            batch_size=self.batch_size,
        )

    def embed_documents(self, documents: list[Any], session_id: str) -> tuple[list[Any], list[Any]]:
        import time as _time

        if not session_id:
            raise SessionIdRequiredError("SESSION_ID_REQUIRED")
        if not documents:
            return [], []
        start = _time.time()
        docs = documents[: self.max_docs]
        seen_hashes: dict[str, bool] = {}
        unique_docs: list[Any] = []
        deduped_skipped = 0
        for doc in docs:
            h = _sha256_text(getattr(doc, "text", "") or "")
            if h in seen_hashes:
                deduped_skipped += 1
                continue
            seen_hashes[h] = True
            unique_docs.append(doc)
        if not unique_docs:
            raise NoValidDocumentsError(f"ALL_DOCUMENTS_DEDUPED: {deduped_skipped} duplicates")
        text_docs, vision_docs = _route_documents(unique_docs)
        embedded_text: list[Any] = []
        embedded_vision: list[Any] = []
        if text_docs:
            try:
                embedded_text = self._embed_text_batched(text_docs, session_id)
            except Exception as exc:
                logger.error(
                    event="multimodal_text_embed_failed", count=len(text_docs), error=str(exc)
                )
        if vision_docs:
            try:
                embedded_vision = self._embed_vision_batched(vision_docs, session_id)
            except Exception as exc:
                logger.error(
                    event="multimodal_vision_embed_failed", count=len(vision_docs), error=str(exc)
                )
        logger.info(
            event="multimodal_embed_success",
            text_embedded=len(embedded_text),
            vision_embedded=len(embedded_vision),
            deduped_skipped=deduped_skipped,
            latency=round(_time.time() - start, 3),
            session_id=session_id,
        )
        return embedded_text, embedded_vision

    def _embed_text_batched(self, docs: list[Any], session_id: str) -> list[Any]:
        import time as _time

        results: list[Any] = []
        for i in range(0, len(docs), self.batch_size):
            batch = docs[i : i + self.batch_size]
            try:
                embedded = self.text_embedder.embed_documents(batch, session_id=session_id)
                for doc in embedded:
                    emb = getattr(doc, "embedding", None)
                    if not _mm_valid_embedding(emb, settings.TEXT_EMBEDDING_DIM):
                        continue
                    struct = dict(getattr(doc, "structure", {}) or {})
                    struct["embedding_space"] = "text"
                    struct["embedding_model"] = self._text_model_name
                    struct["embedding_dim"] = settings.TEXT_EMBEDDING_DIM
                    struct["embedded_at"] = _time.time()
                    doc.structure = struct
                    results.append(doc)
            except Exception as exc:
                logger.error(event="multimodal_text_batch_failed", batch_start=i, error=str(exc))
        return results

    def _embed_vision_batched(self, docs: list[Any], session_id: str) -> list[Any]:
        import time as _time

        results: list[Any] = []
        seen_paths: dict[str, bool] = {}
        for i in range(0, len(docs), self.batch_size):
            batch = docs[i : i + self.batch_size]
            paths: list[str] = []
            valid_docs: list[Any] = []
            for doc in batch:
                asset_path = _resolve_asset_path(doc)
                if not asset_path:
                    continue
                ph = _sha256_text(asset_path)
                if ph in seen_paths:
                    continue
                seen_paths[ph] = True
                paths.append(asset_path)
                valid_docs.append(doc)
            if not paths:
                continue
            try:
                emb_results = self.image_embedder.embed_batch(paths, session_id=session_id)
                for doc, emb_result in zip(valid_docs, emb_results, strict=False):
                    emb_list = (
                        emb_result.embedding if hasattr(emb_result, "embedding") else emb_result
                    )
                    if not _mm_valid_embedding(emb_list, settings.VISION_EMBEDDING_DIM):
                        continue
                    doc.embedding = emb_list
                    struct = dict(getattr(doc, "structure", {}) or {})
                    struct["embedding_space"] = "vision"
                    struct["embedding_model"] = self._vision_model_name
                    struct["embedding_dim"] = settings.VISION_EMBEDDING_DIM
                    struct["embedded_at"] = _time.time()
                    if hasattr(emb_result, "checksum_sha256"):
                        struct["asset_checksum_sha256"] = emb_result.checksum_sha256
                    doc.structure = struct
                    results.append(doc)
            except Exception as exc:
                logger.error(event="multimodal_vision_batch_failed", batch_start=i, error=str(exc))
        return results

    async def embed_documents_async(
        self, documents: list[Any], session_id: str
    ) -> tuple[list[Any], list[Any]]:
        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.embed_documents(documents, session_id))

    def cross_modal_similarity(
        self, query_text: str, image_embedding: list[float], session_id: str = "default"
    ) -> float:
        if not query_text or not image_embedding:
            return 0.0
        try:
            import numpy as np

            from app.core.model_loader import model_loader

            clip_emb = model_loader.get_siglip_text_embedder().embed_query(
                query_text, session_id=session_id
            )
            if not clip_emb or len(clip_emb) != len(image_embedding):
                return 0.0
            a = np.array(clip_emb, dtype=float)
            b = np.array(image_embedding, dtype=float)
            val = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))
            return 0.0 if (math.isnan(val) or math.isinf(val)) else round(val, 6)
        except Exception:
            return 0.0

    def health_check(self) -> dict[str, Any]:
        return {
            "text_embedder_ok": self.text_embedder is not None,
            "vision_embedder_ok": self.image_embedder is not None,
            "text_model": self._text_model_name,
            "vision_model": self._vision_model_name,
            "batch_size": self.batch_size,
        }
