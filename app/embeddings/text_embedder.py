from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
import unicodedata
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Finance number normalization ───────────────────────────────────────────────
# Appends all equivalent phrasings of finance numbers after the original text.
# "$4.3B" → also appends "4.3 billion 4300 million 4.3 billion dollars".
# Applied to both document text (index time) and queries (query time) so
# scale-variant phrases match regardless of how the number was written.

_SCALE_EXPAND: Dict[str, Tuple[float, str, str]] = {
    # suffix_lower → (multiplier_to_base, full_word, cross_word)
    "b":       (1e9,  "billion",  "million"),
    "bn":      (1e9,  "billion",  "million"),
    "t":       (1e12, "trillion", "billion"),
    "tn":      (1e12, "trillion", "billion"),
    "m":       (1e6,  "million",  "billion"),
    "mn":      (1e6,  "million",  "billion"),
    "k":       (1e3,  "thousand", "million"),
}

_FIN_NUM_RE = re.compile(
    r'([$€£¥₹]?)'              # optional currency symbol
    r'([\d,]+\.?\d*)'          # number with optional commas/decimal
    r'\s*([BMKTbmkt]n?)\b'     # scale suffix
)
_PCT_RE   = re.compile(r'([\d.]+)\s*%')
_BPS_RE   = re.compile(r'([\d.]+)\s*bps\b', re.IGNORECASE)
_QTR_RE   = re.compile(r'\bQ([1-4])\s*(?:FY)?\s*(\d{2,4})\b', re.IGNORECASE)
_HY_RE    = re.compile(r'\bH([12])\s+(\d{4})\b', re.IGNORECASE)


def _normalize_finance_numbers(text: str) -> str:
    """Append expanded forms of finance numbers to *text* without replacing the
    original tokens.  Returns the original text with variant phrases appended
    after a separator so both forms are present in the embedding input."""
    extras: List[str] = []

    # Dollar/currency scale amounts: "$4.3B" → "4.3 billion 4300 million"
    for m in _FIN_NUM_RE.finditer(text):
        raw_num = m.group(2).replace(",", "")
        suffix  = m.group(3).lower().rstrip("n")  # "bn" → "b", "mn" → "m"
        if suffix not in _SCALE_EXPAND:
            continue
        try:
            num_val = float(raw_num)
        except ValueError:
            continue
        mult, full_word, cross_word = _SCALE_EXPAND[suffix]
        cross_val = num_val * mult / (1e9 if cross_word == "billion"
                                      else 1e6 if cross_word == "million"
                                      else 1e3)
        extras.append(f"{num_val} {full_word}")
        extras.append(f"{num_val} {full_word} dollars")
        cross_str = f"{cross_val:.1f}".rstrip("0").rstrip(".")
        extras.append(f"{cross_str} {cross_word}")

    # Percentages: "23.5%" → "23.5 percent"
    for m in _PCT_RE.finditer(text):
        extras.append(f"{m.group(1)} percent")

    # Basis points: "350 bps" → "350 basis points"
    for m in _BPS_RE.finditer(text):
        extras.append(f"{m.group(1)} basis points")

    # Quarter references: "Q3FY24" → "Q3 fiscal year 2024 third quarter 2024"
    _QTR_WORDS = {"1": "first", "2": "second", "3": "third", "4": "fourth"}
    for m in _QTR_RE.finditer(text):
        q, yr = m.group(1), m.group(2)
        if len(yr) == 2:
            yr = "20" + yr
        extras.append(f"Q{q} fiscal year {yr}")
        extras.append(f"{_QTR_WORDS.get(q, 'Q'+q)} quarter {yr}")

    # Half-year: "H1 2025" → "first half 2025"
    _HY_WORDS = {"1": "first", "2": "second"}
    for m in _HY_RE.finditer(text):
        h, yr = m.group(1), m.group(2)
        extras.append(f"{_HY_WORDS.get(h, 'H'+h)} half {yr}")

    if not extras:
        return text
    return text + " " + " ".join(extras)


# REDIS EMBEDDING CACHE — SECTION 4.3 (SHA-256 key, TTL 30 days)

class _EmbeddingCache:

    def __init__(self) -> None:
        self._local: Dict[str, List[float]] = {}
        self._redis = None

    def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            from app.core.infra_registry import infra
            self._redis = infra.get_memory()
        except Exception:
            self._redis = None
        return self._redis

    def _key(self, text: str, model: str, dim: int) -> str:
        raw = f"{model}:{dim}:{text[:500]}"
        return "emb:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, model: str, dim: int) -> Optional[List[float]]:
        key = self._key(text, model, dim)

        # LOCAL LRU FIRST
        if key in self._local:
            return self._local[key]

        # REDIS CACHE — SECTION 4.3
        redis = self._get_redis()
        if redis:
            try:
                cached = redis.cache_get(key)
                if cached and isinstance(cached, list):
                    # POPULATE LOCAL
                    self._local[key] = cached
                    return cached
            except Exception:
                pass

        return None

    def set(self, text: str, model: str, dim: int, embedding: List[float]) -> None:
        key = self._key(text, model, dim)
        self._local[key] = embedding

        # REDIS STORE
        redis = self._get_redis()
        if redis:
            try:
                redis.cache_set(
                    key,
                    embedding,
                    ttl=settings.EMBEDDING_CACHE_TTL,
                )
            except Exception:
                pass


_cache = _EmbeddingCache()


# EMBEDDING VALIDATION

def _valid_embedding(emb: List[float], expected_dim: int) -> bool:
    if not isinstance(emb, list):
        return False
    if len(emb) != expected_dim:
        return False
    if any(math.isnan(v) or math.isinf(v) for v in emb):
        return False
    return True


# SANITIZE TEXT — SECTION 2.3

def _sanitize(text: str) -> Optional[str]:
    if not text:
        return None

    # NFC NORMALIZATION
    text = unicodedata.normalize("NFC", text.strip())

    # STRIP NULL BYTES
    if settings.STRIP_NULL_BYTES and "\x00" in text:
        text = text.replace("\x00", "")

    # STRIP BOM
    if settings.STRIP_BOM:
        text = text.lstrip("\ufeff\ufffe")

    if not text:
        return None

    # PROMPT INJECTION SANITIZATION — delegates to unified guardrail (Phase 26)
    from app.guardrails.input_guard import sanitize as _guard_sanitize
    text = _guard_sanitize(text, surface="text_embedder") or text

    if len(text) > settings.MAX_PROMPT_CHARS:
        text = text[:settings.MAX_PROMPT_CHARS]

    return text if text else None


# MODALITY PREFIX — context enrichment for embedding quality

def _prefix(doc: Any) -> str:
    m  = getattr(doc, "modality", "")
    st = getattr(doc, "subtype", "")
    s  = getattr(doc, "structure", {}) or {}

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


# CONTEXT ENRICHMENT

def _enrich(doc: Any, text: str) -> str:
    context = []
    if getattr(doc, "source_type", None):
        context.append(f"[{doc.source_type.upper()}]")
    if getattr(doc, "modality", None):
        context.append(f"[{doc.modality.upper()}]")
    if getattr(doc, "page", None):
        context.append(f"[Page {doc.page}]")
    if getattr(doc, "source", None):
        context.append(f"[{doc.source}]")

    _structure = getattr(doc, "structure", {}) or {}
    _modality  = (getattr(doc, "modality", "") or "").lower()
    _source_type = (getattr(doc, "source_type", "") or "").lower()
    _content_type = (_structure.get("content_type") or "").lower()

    # Include section title when available — two chunks from different sections
    # of the same document would otherwise produce identical prefixes, causing
    # their embeddings to collapse together in vector space.
    _section_title = _structure.get("section_title") or ""
    if _section_title and len(_section_title) <= 120:
        context.append(f"[Section: {_section_title}]")

    # PDF sub-chunk position — disambiguates adjacent chunks on the same page
    _sub_idx   = _structure.get("sub_chunk_index")
    _sub_total = _structure.get("total_sub_chunks")
    if _sub_idx is not None and _sub_total and int(_sub_total) > 1:
        context.append(f"[Part {int(_sub_idx) + 1}/{int(_sub_total)}]")

    # EXCEL row range — grounds the chunk in a precise cell window
    _row_start = _structure.get("row_start")
    _row_end   = _structure.get("row_end")
    if _row_start is not None and _row_end is not None:
        context.append(f"[Rows {_row_start}-{_row_end}]")

    # Audio/video temporal window — anchors chunk to the media timeline
    _ts_start  = _structure.get("timestamp_start")
    _ts_end    = _structure.get("timestamp_end")
    if _modality in ("audio", "video") and _ts_start is not None:
        if _ts_end is not None:
            context.append(f"[{float(_ts_start):.1f}s-{float(_ts_end):.1f}s]")
        else:
            context.append(f"[{float(_ts_start):.1f}s]")

    # Video frame index — unique frame identifier within the video
    _frame_idx = _structure.get("frame_index")
    if _modality == "video" and _frame_idx is not None:
        context.append(f"[Frame {_frame_idx}]")

    context_prefix = " ".join(context) + " " + _prefix(doc)

    # ── Phase 2.2: DOCX — section hierarchy prefix ────────────────────────────
    # Bakes the full heading path into the vector so "financial projections"
    # queries retrieve the right subsection even when its text doesn't repeat
    # the section name.
    if _source_type == "word" or _content_type.startswith("docx_"):
        hierarchy = _structure.get("section_hierarchy") or []
        if hierarchy:
            hier_str = " > ".join(str(h) for h in hierarchy)
            text = f"Section: {hier_str} | {text}"

    # ── Phase 2.3: XLSX — sheet name + unit scale prefix ─────────────────────
    # "Sheet: Income Statement" baked into vector so "income statement revenue"
    # queries hit the right sheet even across documents.
    if _source_type == "excel" or _content_type.startswith("excel_"):
        sheet = (_structure.get("sheet") or _structure.get("sheet_name") or "").strip()
        unit_scale = (_structure.get("unit_scale") or "").strip()
        prefix_parts = []
        if sheet:
            prefix_parts.append(f"Sheet: {sheet}")
        if unit_scale:
            prefix_parts.append(f"({unit_scale})")
        if prefix_parts:
            text = " | ".join(prefix_parts) + " | " + text

    # ── Phase 2.4: Audio / Video — speaker prefix + finance entity tokens ─────
    # "[SPEAKER: Luca Maestri - CFO] [1842s-1924s] transcript [ENTITIES: ...]"
    if _modality in ("audio", "video") and (_structure.get("content_type") or "").endswith(
        ("speech", "audio_speech_segment", "video_speech")
    ):
        speaker_name = (_structure.get("speaker") or _structure.get("speaker_role") or "").strip()
        ts_s = _ts_start
        ts_e = _ts_end
        speaker_header = ""
        if speaker_name:
            speaker_header = f"[SPEAKER: {speaker_name}]"
        if ts_s is not None and ts_e is not None:
            speaker_header += f" [{float(ts_s):.0f}s-{float(ts_e):.0f}s]"
        finance_entities = _structure.get("finance_entities") or {}
        entity_tokens: List[str] = []
        if isinstance(finance_entities, dict):
            for v in finance_entities.values():
                if isinstance(v, list):
                    entity_tokens.extend(str(x) for x in v[:3])
        entity_suffix = (f" [ENTITIES: {', '.join(entity_tokens)}]"
                         if entity_tokens else "")
        if speaker_header:
            text = f"{speaker_header} {text}{entity_suffix}"
        elif entity_tokens:
            text = f"{text}{entity_suffix}"

    # ── Phase 2.5: Video — use combined_text (transcript + frame captions) ────
    # combined_text includes "[VISUAL AT Xs]: <caption>" so the video chunk
    # embeds both spoken content AND what was visually shown.
    if _modality == "video":
        combined = (_structure.get("combined_text") or "").strip()
        if combined and len(combined) > len(text):
            text = combined

    # ── Phase 2.6: Image — image type prefix ─────────────────────────────────
    # "Bar chart: Revenue by Segment..." retrieves better than plain caption.
    if _modality == "image" or _source_type == "image":
        image_type = (_structure.get("image_type") or "").strip()
        if image_type:
            text = f"{image_type.replace('_', ' ')}: {text}"

    enriched = context_prefix + text
    return enriched.strip()[:settings.MAX_PROMPT_CHARS]


# TEXT EMBEDDER CLASS

class TextEmbedder:

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        device: str,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name    = model_name
        self.batch_size    = batch_size
        self.device        = device
        self.expected_dim  = settings.TEXT_EMBEDDING_DIM
        self.max_text_len  = settings.MAX_PROMPT_CHARS

        self.model = SentenceTransformer(model_name, device=device)

        # Instruction-tuned embedders (Qwen3-Embedding, E5-Instruct, etc.) need a
        # query-side prompt to activate their instruction-following capacity.
        # Document embedding uses no prompt — the model card is explicit about this.
        # We detect by checking the model's own registered prompt names; if "query"
        # is present we use it, otherwise fall back to a raw string prefix.
        self._query_prompt_name: Optional[str] = None
        self._query_prompt_text: Optional[str] = None
        _prompts = getattr(self.model, "prompts", {}) or {}
        if "query" in _prompts:
            self._query_prompt_name = "query"
        elif any(k in model_name.lower() for k in ("qwen3", "e5-instruct", "gte-qwen")):
            # Financial-domain instruction for Qwen3-class instruction-tuned embedders.
            # Tested against the Qwen3-Embedding-0.6B model card guidance.
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

    # ENCODE WITH TENACITY RETRY

    def _encode_with_retry(
        self,
        model,
        texts: List[str],
        max_retries: int = 3,
    ) -> List[List[float]]:
        wait = 1.0
        for attempt in range(max_retries):
            try:
                t_start = time.time()
                embs = model.encode(
                    texts,
                    batch_size=len(texts),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                elapsed_ms = round((time.time() - t_start) * 1000, 1)

                if elapsed_ms > settings.LATENCY_TARGET_EMBED_BATCH_MS:
                    logger.warning(
                        event="embed_batch_latency_exceeded",
                        latency_ms=elapsed_ms,
                        target_ms=settings.LATENCY_TARGET_EMBED_BATCH_MS,
                        batch_size=len(texts),
                    )

                return [e.tolist() for e in embs]

            except Exception as e:
                logger.warning(
                    event="embed_encode_retry",
                    attempt=attempt,
                    error=str(e),
                )
                if attempt >= max_retries - 1:
                    raise
                time.sleep(wait)
                wait = min(wait * 2, 10.0)

        return []

    # SINGLE TEXT EMBED

    def embed_text(
        self,
        text: str,
        session_id: str = "default",
    ) -> List[float]:

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        clean = _sanitize(text)
        if not clean:
            raise ValueError("EMPTY_TEXT")

        # Phase 2.1 — expand finance number forms in queries too, so a query
        # for "4.3 billion" also matches chunks that stored "4300 million".
        clean = _normalize_finance_numbers(clean)

        # Query embeddings use a prompt; use a distinct cache key so a text
        # that appears in both a query and a document doesn't collide.
        _cache_model_key = (
            self.model_name + ":q"
            if (self._query_prompt_name or self._query_prompt_text)
            else self.model_name
        )

        # CACHE CHECK — SECTION 4.3
        cached = _cache.get(clean, _cache_model_key, self.expected_dim)
        if cached is not None:
            logger.debug(event="embed_cache_hit", session_id=session_id)
            return cached

        t_start = time.time()
        _encode_kwargs: dict = dict(convert_to_numpy=True, normalize_embeddings=True)
        if self._query_prompt_name:
            _encode_kwargs["prompt_name"] = self._query_prompt_name
            _text_to_encode = clean
        elif self._query_prompt_text:
            _text_to_encode = self._query_prompt_text + clean
        else:
            _text_to_encode = clean
        emb = self.model.encode(_text_to_encode, **_encode_kwargs).tolist()

        if not _valid_embedding(emb, self.expected_dim):
            raise ValueError("INVALID_EMBEDDING_SINGLE")

        _cache.set(clean, _cache_model_key, self.expected_dim, emb)

        logger.debug(
            event="embed_single_success",
            latency_ms=round((time.time() - t_start) * 1000, 1),
            session_id=session_id,
        )

        return emb

    # RAW STRING LIST EMBED — BATCH WITH CACHE — SECTION 4.3

    def embed_texts(
        self,
        texts: List[str],
        session_id: str = "default",
    ) -> List[List[float]]:

        if not texts:
            return []

        results: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            raw_batch = texts[i:i + self.batch_size]
            clean_batch: List[str] = []
            cached_results: List[Optional[List[float]]] = []

            for t in raw_batch:
                c = _sanitize(t)
                if not c:
                    cached_results.append(None)
                    clean_batch.append("")
                    continue

                hit = _cache.get(c, self.model_name, self.expected_dim)
                if hit is not None:
                    cached_results.append(hit)
                    clean_batch.append("")
                else:
                    cached_results.append(None)
                    clean_batch.append(c)

            # ENCODE ONLY UNCACHED
            to_encode = [(idx, t) for idx, t in enumerate(clean_batch) if t]
            encoded: Dict[int, List[float]] = {}

            if to_encode:
                indices, batch_texts = zip(*to_encode)
                try:
                    embs = self._encode_with_retry(self.model, list(batch_texts))
                    for idx, emb in zip(indices, embs):
                        if _valid_embedding(emb, self.expected_dim):
                            encoded[idx] = emb
                            _cache.set(clean_batch[idx], self.model_name, self.expected_dim, emb)
                except Exception as e:
                    logger.error(
                        event="embed_texts_batch_failed",
                        error=str(e),
                        session_id=session_id,
                    )

            for idx, cached_emb in enumerate(cached_results):
                if cached_emb is not None:
                    results.append(cached_emb)
                elif idx in encoded:
                    results.append(encoded[idx])

        return results

    # DOCUMENT BATCH EMBED — SECTION 4.3

    def embed_documents(
        self,
        documents: List[Any],
        session_id: str = "default",
    ) -> List[Any]:

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        if not documents:
            return []

        start = time.time()
        texts:     List[str] = []
        valid_docs:  List[Any] = []
        seen: Dict[str, bool] = {}

        for doc in documents:
            try:
                raw  = getattr(doc, "text", "") or ""
                clean = _sanitize(raw)
                if not clean:
                    continue

                # Phase 2.1 — append expanded number forms before enrichment
                # so "$4.3B" also encodes "4.3 billion 4300 million" in the vector.
                clean = _normalize_finance_numbers(clean)

                enriched = _enrich(doc, clean)
                h = hashlib.sha256(enriched.encode("utf-8")).hexdigest()

                if h in seen:
                    continue

                seen[h] = True
                texts.append(enriched)
                valid_docs.append(doc)

            except Exception as e:
                logger.warning(
                    event="embed_doc_skip",
                    error=str(e),
                    session_id=session_id,
                )

        if not valid_docs:
            return []

        # SAFETY CAP
        cap        = settings.INGESTION_BATCH_SIZE * 10
        texts      = texts[:cap]
        valid_docs = valid_docs[:cap]

        results: List[Any] = []
        t_target = settings.LATENCY_TARGET_EMBED_BATCH_MS / 1000.0

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_docs  = valid_docs[i:i + self.batch_size]

            t_batch = time.time()

            try:
                embs = self._encode_with_retry(self.model, batch_texts)

                batch_latency = time.time() - t_batch

                if batch_latency > t_target:
                    logger.warning(
                        event="embed_batch_latency_exceeded",
                        latency=round(batch_latency, 3),
                        target=t_target,
                        batch_size=len(batch_texts),
                        session_id=session_id,
                    )

                for doc, emb in zip(batch_docs, embs):
                    if not _valid_embedding(emb, self.expected_dim):
                        continue

                    doc.embedding = emb

                    structure = dict(doc.structure or {})
                    structure["embedding_space"] = "text"
                    structure["embedding_model"] = self.model_name
                    doc.structure = structure

                    # CACHE SET — SECTION 4.3
                    text_for_cache = _sanitize(getattr(doc, "text", "") or "")
                    if text_for_cache:
                        _cache.set(text_for_cache, self.model_name, self.expected_dim, emb)

                    results.append(doc)

            except Exception as e:
                logger.error(
                    event="embed_batch_failed",
                    batch_start=i,
                    error=str(e),
                    session_id=session_id,
                )

        total_latency = round(time.time() - start, 3)
        throughput    = round(len(results) / max(total_latency, 1e-6), 1)

        logger.info(
            event="embed_documents_success",
            embedded=len(results),
            total=len(valid_docs),
            throughput_per_sec=throughput,
            latency=total_latency,
            session_id=session_id,
        )

        return results

    # QUERY EMBED — ALIAS

    def embed_query(
        self,
        query: str,
        session_id: str = "default",
    ) -> List[float]:
        return self.embed_text(query, session_id)

    # MATRYOSHKA EMBED — SECTION 4.3 (short dim + full dim)

    def embed_matryoshka(
        self,
        text: str,
        session_id: str = "default",
    ) -> Tuple[List[float], List[float]]:
        full = self.embed_text(text, session_id)
        short = full[:settings.MATRYOSHKA_SHORT_DIM]
        # L2 NORMALIZE SHORT
        norm = math.sqrt(sum(v * v for v in short)) + 1e-10
        short = [v / norm for v in short]
        return short, full

    # SPARSE VECTOR (BM25-STYLE PROXY) — SECTION 4.3

    def embed_sparse(
        self,
        text: str,
        vocab_size: int = 30000,
    ) -> Dict[int, float]:
        clean = _sanitize(text)
        if not clean:
            return {}
        import re
        tokens = re.findall(r"\b[a-z0-9]+\b", clean.lower())
        tf: Dict[int, float] = {}
        for tok in tokens:
            idx = hash(tok) % vocab_size
            tf[idx] = tf.get(idx, 0.0) + 1.0
        total = sum(tf.values()) + 1e-10
        return {k: round(v / total, 6) for k, v in tf.items()}

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        return {
            "model":       self.model_name,
            "device":      self.device,
            "dim":         self.expected_dim,
            "batch_size":  self.batch_size,
            "cache_local": len(_cache._local),
        }


