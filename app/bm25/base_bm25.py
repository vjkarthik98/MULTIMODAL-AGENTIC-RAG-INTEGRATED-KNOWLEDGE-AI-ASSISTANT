"""base_bm25.py

Shared base for all per-modality BM25 indexes (Phase 4).

Contains the finance tokenizer, BM25Plus machinery, save/load, search, and
BM25Document. Each per-modality subclass overrides _build_indexed_text() to
add its specific field enrichment on top of the base text.

Index version 5 — bump whenever tokenizer behaviour changes.
"""
from __future__ import annotations

import asyncio
import hashlib
import pickle
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_INDEX_VERSION = 5

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import numpy as np
    _NP = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NP = False

try:
    from rank_bm25 import BM25Plus
    _BM25_OK = True
except ImportError:
    _BM25_OK = False

    class BM25Plus:  # type: ignore[no-redef]
        def __init__(self, corpus, **_kw):
            self.corpus = corpus
        def get_scores(self, tokens):
            q = set(tokens)
            return [float(len(q & set(d))) for d in self.corpus]

try:
    import Stemmer as _PyStemmer
    _stemmer = _PyStemmer.Stemmer("english")
    _STEM_OK = True
except ImportError:
    _stemmer = None  # type: ignore[assignment]
    _STEM_OK = False

try:
    import pybreaker
    _breaker = pybreaker.CircuitBreaker(
        fail_max=settings.CIRCUIT_BREAKER_MAX_FAILURES,
        reset_timeout=settings.CIRCUIT_BREAKER_RESET_TIMEOUT,
    )
    _CB_OK = True
except ImportError:
    _CB_OK = False
    class _DummyBreaker:
        def __call__(self, fn): return fn
    _breaker = _DummyBreaker()  # type: ignore[assignment]

# ── Stopwords ─────────────────────────────────────────────────────────────────
STOPWORDS: Set[str] = {
    "the", "is", "and", "a", "an", "of", "to", "in", "on", "for",
    "at", "by", "with", "from", "this", "that", "it", "be", "as",
    "are", "was", "were", "has", "have", "had", "do", "does", "did",
    "but", "or", "not", "so", "if", "its", "our", "we", "he", "she",
    "they", "you", "i", "me", "my", "your", "their", "what", "which",
    "who", "when", "where", "how", "all", "been", "will", "would",
    "could", "should", "may", "might", "can", "any", "some", "no",
    "also", "than", "more", "other", "than", "then", "up", "out",
    "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "each", "few", "further", "once", "very",
    "just", "because", "while", "although", "however", "therefore",
    "since", "whether", "both", "either", "neither", "per",
}

FINANCE_KEEP: Set[str] = {
    "not", "no", "loss", "losses", "deficit", "decline", "decrease",
    "negative", "below", "under", "miss", "missed", "shortfall",
    "risk", "risks", "uncertain", "uncertainty",
}

FIN_ABBR: Dict[str, List[str]] = {
    "eps":     ["earnings", "share"],
    "ebitda":  ["earnings", "interest", "taxes", "depreciation", "amortization"],
    "ebit":    ["earnings", "interest", "taxes"],
    "yoy":     ["year", "growth"],
    "fy":      ["fiscal", "year"],
    "q1":      ["first", "quarter"],
    "q2":      ["second", "quarter"],
    "q3":      ["third", "quarter"],
    "q4":      ["fourth", "quarter"],
    "ceo":     ["chief", "executive", "officer"],
    "cfo":     ["chief", "financial", "officer"],
    "coo":     ["chief", "operating", "officer"],
    "capex":   ["capital", "expenditure"],
    "opex":    ["operating", "expense"],
    "cogs":    ["cost", "goods", "sold"],
    "sga":     ["selling", "general", "administrative"],
    "r&d":     ["research", "development"],
    "ttm":     ["trailing", "twelve", "months"],
    "pe":      ["price", "earnings"],
    "pb":      ["price", "book"],
    "ps":      ["price", "sales"],
    "roe":     ["return", "equity"],
    "roa":     ["return", "assets"],
    "roi":     ["return", "investment"],
    "fcf":     ["free", "cash", "flow"],
    "gaap":    ["accounting", "principles"],
    "buyback": ["repurchase", "share"],
    "ltv":     ["lifetime", "value"],
    "arpu":    ["average", "revenue", "user"],
    "mom":     ["month", "month"],
    "qoq":     ["quarter", "quarter"],
    "bps":     ["basis", "points"],
    "nii":     ["net", "interest", "income"],
    "nim":     ["net", "interest", "margin"],
    "npv":     ["net", "present", "value"],
    "irr":     ["internal", "rate", "return"],
    "wacc":    ["weighted", "average", "cost", "capital"],
}

FIN_BIGRAMS: Set[str] = {
    "net income", "net sales", "gross margin", "operating income",
    "earnings per share", "diluted eps", "basic eps",
    "cash flow", "free cash flow", "total revenue", "total assets",
    "total liabilities", "shareholders equity", "return on equity",
    "return on assets", "research development", "capital expenditure",
    "year over year", "fiscal year", "first quarter", "second quarter",
    "third quarter", "fourth quarter", "annual report", "form 10k",
    "income statement", "balance sheet", "cash flow statement",
    "interest expense", "effective tax", "share repurchase",
    "stock buyback", "dividend per share", "book value",
    "operating expense", "cost of goods", "gross profit",
    "net revenue", "organic growth", "adjusted ebitda",
    "free cash", "working capital", "debt to equity",
    "price earnings", "market cap", "market capitalization",
    "net loss", "operating loss", "revenue decline", "revenue decrease",
    "below consensus", "missed estimates", "below expectations",
    "not profitable", "negative growth", "risk factors",
}

_CONTRACTIONS: Dict[str, str] = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will",
    "'d": " would", "'m": " am", "'s": " is",
}

_CURRENCY_RE  = re.compile(r'[$€£¥₹₩]')
_SCALE_RE     = re.compile(r'(\d[\d,.]*)\s*([bBmMkKtT])\b')
_SCALE_MAP    = {"b": "billion", "m": "million", "k": "thousand", "t": "trillion"}
_PERCENT_RE   = re.compile(r'(\d[\d.]*)\s*%')
_COMMA_NUM_RE = re.compile(r'(\d),(\d)')


def _normalize_text(text: str) -> str:
    for c, e in _CONTRACTIONS.items():
        text = text.replace(c, e)
    text = _PERCENT_RE.sub(r'\1 percent', text)
    text = _COMMA_NUM_RE.sub(r'\1\2', text)

    def _expand(m: re.Match) -> str:
        return f"{m.group(1).replace(',','')} {_SCALE_MAP[m.group(2).lower()]}"
    text = _SCALE_RE.sub(_expand, text)
    return _CURRENCY_RE.sub("", text)


def _expand_scale_variants(tokens: List[str]) -> List[str]:
    _TO_MULT = {"billion": 1e9, "million": 1e6, "thousand": 1e3, "trillion": 1e12}
    _TO_NAME = {1e9: "billion", 1e6: "million", 1e3: "thousand", 1e12: "trillion"}
    extras: List[str] = []
    for i in range(len(tokens) - 1):
        scale = tokens[i + 1]
        if scale not in _TO_MULT:
            continue
        try:
            val = float(tokens[i].replace(",", ""))
        except (ValueError, TypeError):
            continue
        extras.append(f"{tokens[i]}_{scale}")
        absolute = val * _TO_MULT[scale]
        for mult, name in _TO_NAME.items():
            if mult == _TO_MULT[scale]:
                continue
            cv = absolute / mult
            if 0.001 <= cv <= 1e9:
                fs = f"{cv:.3f}".rstrip("0").rstrip(".")
                extras.append(f"{fs}_{name}")
    return tokens + extras


def _stem(tokens: List[str]) -> List[str]:
    if _STEM_OK and _stemmer and tokens:
        return _stemmer.stemWords(tokens)
    return tokens


# ── BM25Document ──────────────────────────────────────────────────────────────

class BM25Document:
    __slots__ = [
        "text", "structure", "modality", "subtype",
        "source", "source_type", "chunk_id", "page",
        "sub_chunk_index", "total_sub_chunks",
        "heading_level",
        "sheet_name", "row_start", "row_end",
        "timestamp_start", "timestamp_end", "speaker", "frame_index",
        "caption",
    ]

    def __init__(self) -> None:
        for slot in self.__slots__:
            setattr(self, slot, None)

    @classmethod
    def from_payload(cls, p: Dict[str, Any]) -> "BM25Document":
        obj              = cls()
        obj.text         = p.get("text") or p.get("content") or ""
        obj.modality     = p.get("modality", "text")
        obj.subtype      = p.get("subtype")
        obj.source       = p.get("source")
        obj.source_type  = p.get("source_type")
        obj.chunk_id     = p.get("chunk_id")
        obj.page         = p.get("page")
        obj.sub_chunk_index  = p.get("sub_chunk_index")
        obj.total_sub_chunks = p.get("total_sub_chunks")
        obj.heading_level    = p.get("heading_level")
        obj.sheet_name   = p.get("sheet_name") or p.get("section_title")
        obj.row_start    = p.get("row_start")
        obj.row_end      = p.get("row_end")
        obj.timestamp_start = p.get("timestamp_start")
        obj.timestamp_end   = p.get("timestamp_end")
        obj.speaker      = p.get("speaker")
        obj.frame_index  = p.get("frame_index")
        obj.caption      = p.get("caption")
        obj.structure    = {
            "doc_id":             p.get("doc_id"),
            "chunk_id":           p.get("chunk_id"),
            "session_id":         p.get("session_id"),
            "content_type":       p.get("content_type"),
            "language":           p.get("language"),
            "section_id":         p.get("section_id"),
            "section_title":      p.get("section_title"),
            "timestamp_start":    p.get("timestamp_start"),
            "timestamp_end":      p.get("timestamp_end"),
            "ingestion_time":     p.get("ingestion_time"),
            "checksum_sha256":    p.get("checksum_sha256"),
            "section_number":     p.get("section_number"),
            "is_forward_looking": p.get("is_forward_looking", False),
            "embedding_space":    "text",
            "sub_chunk_index":    p.get("sub_chunk_index"),
            "total_sub_chunks":   p.get("total_sub_chunks"),
            "heading_level":      p.get("heading_level"),
            "sheet":              p.get("sheet_name") or p.get("section_title"),
            "row_start":          p.get("row_start"),
            "row_end":            p.get("row_end"),
            "frame_index":        p.get("frame_index"),
            "caption":            p.get("caption"),
            "speaker":            p.get("speaker"),
        }
        return obj


# ── BaseBM25 ──────────────────────────────────────────────────────────────────

class BaseBM25(ABC):
    """Abstract base for all per-modality BM25 indexes.

    Subclasses override _build_indexed_text() to add modality-specific field
    enrichment. All tokenization, save/load, search, and lifecycle logic lives
    here.
    """

    #: Modality tag — used to name the per-user index file (e.g. "pdf.pkl")
    modality: str = "base"

    def __init__(self, user_id: Optional[str] = None) -> None:
        self.user_id            = user_id
        self.documents:           List[Any]        = []
        self.tokenized_corpus:    List[List[str]]  = []
        self.bm25:                Optional[BM25Plus] = None
        self.modality_filter:     Optional[str]    = None
        self.max_docs:            int              = settings.BM25_MAX_DOCS
        self._index_loaded:       bool             = False
        self._loaded_user_id:     Optional[str]    = None

    # ── Index file path ───────────────────────────────────────────────────────

    def _index_file(self, user_id: Optional[str] = None) -> Path:
        from app.utils.paths import user_dir
        uid = user_id or self.user_id
        if uid:
            p = user_dir(uid) / "bm25_index"
        else:
            from app.core.config import settings as _s
            p = (
                Path(_s.BM25_INDEX_DIR)
                if hasattr(_s, "BM25_INDEX_DIR")
                else _s.DATA_DIR / "bm25_index"
            )
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{self.modality}.pkl"

    # ── Hashing ───────────────────────────────────────────────────────────────

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ── Indexed text (base — subclasses extend) ───────────────────────────────

    @abstractmethod
    def _build_indexed_text(self, doc: Any) -> str:
        """Return the full multi-field text to tokenize for this doc."""
        ...

    def _base_text(self, doc: Any) -> List[str]:
        """Shared base fields present in every modality: text + section_title ×2."""
        parts: List[str] = []
        text = (getattr(doc, "text", "") or "").strip()
        if text:
            parts.append(text[:settings.BM25_MAX_TEXT_CHARS])
        s             = getattr(doc, "structure", {}) or {}
        section_title = (s.get("section_title") or "").strip()
        if section_title:
            parts.append(section_title)
            parts.append(section_title)
        caption = (s.get("caption") or "").strip()
        if caption:
            parts.append(caption)
        return parts

    # ── Tokenizer ─────────────────────────────────────────────────────────────

    def tokenize(self, text: str) -> List[str]:
        text = str(text or "").lower()
        text = _normalize_text(text)

        bigram_tokens: List[str] = [
            bigram.replace(" ", "_")
            for bigram in FIN_BIGRAMS
            if bigram in text
        ]

        raw = re.findall(r'\d+\.\d+|\b[a-z0-9]+\b', text)
        raw = [t for t in raw if t not in STOPWORDS and len(t) > 1]

        expanded: List[str] = []
        for tok in raw:
            expanded.append(tok)
            if tok in FIN_ABBR:
                expanded.extend(FIN_ABBR[tok])

        stemmed = _stem(expanded)

        # Counter deduplicates and counts term frequencies in one O(n) pass.
        # list(Counter(...)) preserves first-occurrence order (Python 3.7+)
        # and gives unique tokens — same semantics as the prior seen+clean loop.
        freq = Counter(
            tok for tok in stemmed
            if tok and len(tok) > 1 and (tok in FINANCE_KEEP or tok not in STOPWORDS)
        )
        clean = _expand_scale_variants(list(freq))
        return (bigram_tokens + clean)[:settings.BM25_MAX_TOKENS]

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _metadata(self, doc: Any) -> Dict[str, Any]:
        s = dict(getattr(doc, "structure", {}) or {})
        return {
            "modality":           getattr(doc, "modality", "text"),
            "subtype":            getattr(doc, "subtype", None),
            "source":             getattr(doc, "source", None),
            "source_type":        getattr(doc, "source_type", None),
            "doc_id":             s.get("doc_id"),
            "chunk_id":           getattr(doc, "chunk_id", None),
            "session_id":         s.get("session_id"),
            "content_type":       s.get("content_type"),
            "page":               getattr(doc, "page", None),
            "language":           s.get("language"),
            "section_id":         s.get("section_id"),
            "section_title":      s.get("section_title"),
            "heading":            s.get("heading"),
            "is_forward_looking": s.get("is_forward_looking", False),
            "section_number":     s.get("section_number"),
            "sub_chunk_index":    s.get("sub_chunk_index"),
            "total_sub_chunks":   s.get("total_sub_chunks"),
            "heading_level":      s.get("heading_level"),
            "sheet_name":         s.get("sheet") or s.get("sheet_name"),
            "row_start":          s.get("row_start"),
            "row_end":            s.get("row_end"),
            # audio_chunker.py stores "start_timestamp"/"end_timestamp" (reversed
            # word order) and "speaker_name"/"speaker_label"/"speaker_role" (no
            # single "speaker" key) — read both shapes defensively (accuracy
            # phase 2026-07, same class of bug as the sheet_name fix above).
            "timestamp_start":    s.get("timestamp_start") or s.get("start_timestamp"),
            "timestamp_end":      s.get("timestamp_end") or s.get("end_timestamp"),
            "start_timestamp":    s.get("start_timestamp") or s.get("timestamp_start"),
            "end_timestamp":      s.get("end_timestamp") or s.get("timestamp_end"),
            "speaker":            s.get("speaker") or s.get("speaker_name") or s.get("speaker_label"),
            "speaker_name":       s.get("speaker_name") or s.get("speaker_label") or s.get("speaker"),
            "speaker_label":      s.get("speaker_label"),
            "speaker_role":       s.get("speaker_role"),
            "call_section":       s.get("call_section"),
            "frame_index":        s.get("frame_index"),
            "caption":            s.get("caption"),
            "ingestion_time":     s.get("ingestion_time"),
            "checksum_sha256":    s.get("checksum_sha256"),
        }

    # ── Save / load ───────────────────────────────────────────────────────────

    def _save(self, user_id: Optional[str] = None) -> None:
        path = self._index_file(user_id)

        def _do() -> None:
            payload = {
                "index_version":    _INDEX_VERSION,
                "modality":         self.modality,
                "documents":        self.documents,
                "tokenized_corpus": self.tokenized_corpus,
                "saved_at":         time.time(),
                "doc_count":        len(self.documents),
            }
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
            logger.info(
                event="bm25_saved",
                modality=self.modality,
                docs=len(self.documents),
                path=str(path),
            )

        try:
            if _CB_OK:
                _breaker(_do)()
            else:
                _do()
        except Exception as exc:
            logger.error(event="bm25_save_failed", modality=self.modality, error=str(exc))

    def _load(self, user_id: Optional[str] = None) -> None:
        eff_uid = user_id or self.user_id
        if self._index_loaded and self._loaded_user_id == eff_uid:
            return
        # User switched — reset
        self._index_loaded    = False
        self.documents        = []
        self.tokenized_corpus = []
        self.bm25             = None

        path = self._index_file(user_id)
        if not path.exists():
            logger.info(event="bm25_no_index", modality=self.modality, path=str(path))
            return

        def _do() -> None:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if payload.get("index_version", 0) != _INDEX_VERSION:
                logger.warning(
                    event="bm25_version_mismatch",
                    modality=self.modality,
                    saved=payload.get("index_version"),
                    current=_INDEX_VERSION,
                )
                return
            self.documents        = payload.get("documents", [])
            self.tokenized_corpus = payload.get("tokenized_corpus", [])
            if self.tokenized_corpus:
                self.bm25 = BM25Plus(self.tokenized_corpus, k1=1.5, b=0.75)
            self._index_loaded   = True
            self._loaded_user_id = eff_uid
            logger.info(
                event="bm25_loaded",
                modality=self.modality,
                docs=len(self.documents),
            )

        try:
            if _CB_OK:
                _breaker(_do)()
            else:
                _do()
        except Exception as exc:
            logger.error(event="bm25_load_failed", modality=self.modality, error=str(exc))

    # ── Build (full rebuild) ──────────────────────────────────────────────────

    def build_index(self, documents: List[Any], user_id: Optional[str] = None) -> None:
        if not documents:
            return
        start = time.time()
        self.documents        = []
        self.tokenized_corpus = []
        self.bm25             = None
        seen: Set[str]        = set()

        for doc in documents[:self.max_docs]:
            try:
                text = getattr(doc, "text", "")
                if not text:
                    continue
                h = self._hash(text[:settings.BM25_MAX_TEXT_CHARS])
                if h in seen:
                    continue
                seen.add(h)
                tokens = self.tokenize(self._build_indexed_text(doc))
                if not tokens:
                    continue
                self.documents.append(doc)
                self.tokenized_corpus.append(tokens)
            except Exception as exc:
                logger.warning(event="bm25_doc_skip", modality=self.modality, error=str(exc))

        if not self.tokenized_corpus:
            return
        self.bm25 = BM25Plus(self.tokenized_corpus, k1=1.5, b=0.75)
        self._save(user_id)
        logger.info(
            event="bm25_index_built",
            modality=self.modality,
            docs=len(self.documents),
            latency=round(time.time() - start, 2),
        )

    # ── Incremental add ───────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[Any],
        session_id: str = "",
        user_id: Optional[str] = None,
    ) -> None:
        if not documents:
            return
        start = time.time()
        added = 0
        seen: Set[str] = {
            self._hash(getattr(d, "text", "")[:settings.BM25_MAX_TEXT_CHARS])
            for d in self.documents
        }

        for doc in documents:
            try:
                text = getattr(doc, "text", "")
                if not text:
                    continue
                h = self._hash(text[:settings.BM25_MAX_TEXT_CHARS])
                if h in seen:
                    continue
                tokens = self.tokenize(self._build_indexed_text(doc))
                if not tokens:
                    continue
                self.documents.append(doc)
                self.tokenized_corpus.append(tokens)
                seen.add(h)
                added += 1
                if len(self.documents) >= self.max_docs:
                    break
            except Exception as exc:
                logger.warning(event="bm25_add_skip", modality=self.modality, error=str(exc))

        if not added:
            return
        self.bm25 = BM25Plus(self.tokenized_corpus, k1=1.5, b=0.75)
        self._save(user_id)
        logger.info(
            event="bm25_docs_added",
            modality=self.modality, added=added, total=len(self.documents),
            latency=round(time.time() - start, 2), session_id=session_id,
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def _norm_scores(self, raw: Any) -> Any:
        if _NP:
            s      = np.nan_to_num(np.asarray(raw, dtype=float))
            lo, hi = float(s.min()), float(s.max())
            if hi - lo < 1e-9:
                # All scores equal (single-doc or perfectly tied corpus).
                # Normalize by max so a matching doc scores 1.0 instead of 0.
                return s / (hi + 1e-10)
            return (s - lo) / (hi - lo)
        s  = [float(x) for x in raw]
        lo = min(s) if s else 0.0
        hi = max(s) if s else 1e-6
        d  = hi - lo
        if d < 1e-9:
            return [x / (hi + 1e-10) for x in s]
        return [(x - lo) / d for x in s]

    def _topk(self, scores: Any, k: int) -> List[int]:
        if _NP:
            if len(scores) <= k:
                idxs = list(range(len(scores)))
            else:
                idxs = list(np.argpartition(scores, -k)[-k:])
            return sorted(idxs, key=lambda i: float(scores[i]), reverse=True)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return idxs[:k]

    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.bm25:
            self._load(user_id)
        if not self.bm25 or not query:
            return []

        start  = time.time()
        top_k  = min(top_k or settings.BM25_TOP_K, len(self.documents))
        if top_k <= 0:
            return []

        tokens = self.tokenize(query[:settings.MAX_PROMPT_CHARS])
        if not tokens:
            return []

        try:
            raw = self.bm25.get_scores(tokens)
        except Exception as exc:
            logger.error(event="bm25_score_failed", modality=self.modality, error=str(exc))
            return []

        norm   = self._norm_scores(raw)
        idxs   = self._topk(norm, top_k)
        _mw    = getattr(settings, "BM25_MODALITY_WEIGHTS", {
            "text": 1.0, "table": 1.1, "image": 0.9, "audio": 1.0, "video": 1.0,
        })
        results: List[Dict[str, Any]] = []

        for idx in idxs:
            if len(results) >= top_k or idx >= len(self.documents):
                break
            doc  = self.documents[idx]
            meta = self._metadata(doc)

            if filters:
                if filters.get("modality") and meta.get("modality") != filters["modality"]:
                    continue
                if filters.get("source_type") and meta.get("source_type") != filters["source_type"]:
                    continue

            text = getattr(doc, "text", "").strip()
            if not text:
                continue

            score = float(norm[idx]) * _mw.get(meta.get("modality", "text"), 1.0)
            if score < settings.BM25_MIN_SCORE:
                continue

            results.append({
                "id":       f"bm25_{self.modality}_{idx}",
                "text":     text[:settings.RAG_DOC_MAX_CHARS],
                "score":    round(score, 5),
                "metadata": meta,
            })

        logger.info(
            event="bm25_search_done",
            modality=self.modality,
            results=len(results),
            latency=round(time.time() - start, 3),
            session_id=session_id,
        )
        return results

    async def async_search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.search, query, session_id, top_k, filters, user_id)

    # ── Delete / purge / clear ────────────────────────────────────────────────

    def delete_by_source(self, filename: str, user_id: Optional[str] = None) -> int:
        before = len(self.documents)
        keep_d, keep_t = [], []
        for doc, toks in zip(self.documents, self.tokenized_corpus):
            if filename not in (getattr(doc, "source", "") or ""):
                keep_d.append(doc)
                keep_t.append(toks)
        self.documents        = keep_d
        self.tokenized_corpus = keep_t
        removed = before - len(self.documents)
        if removed:
            self.bm25 = BM25Plus(self.tokenized_corpus, k1=1.5, b=0.75) if self.tokenized_corpus else None
            self._save(user_id)
        return removed

    def purge_by_session(self, session_id: str) -> int:
        before = len(self.documents)
        keep_d, keep_t = [], []
        for doc, toks in zip(self.documents, self.tokenized_corpus):
            s = getattr(doc, "structure", {}) or {}
            if s.get("session_id") != session_id:
                keep_d.append(doc)
                keep_t.append(toks)
        self.documents        = keep_d
        self.tokenized_corpus = keep_t
        removed = before - len(self.documents)
        if removed:
            self.bm25 = BM25Plus(self.tokenized_corpus, k1=1.5, b=0.75) if self.tokenized_corpus else None
            self._save()
        return removed

    def clear(self, user_id: Optional[str] = None) -> None:
        self.documents        = []
        self.tokenized_corpus = []
        self.bm25             = None
        self._index_loaded    = False
        path = self._index_file(user_id)
        if path.exists():
            try:
                path.unlink()
            except Exception as exc:
                logger.error(event="bm25_clear_failed", modality=self.modality, error=str(exc))

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        path = self._index_file(user_id)
        return {
            "modality":    self.modality,
            "ready":       self.bm25 is not None,
            "doc_count":   len(self.documents),
            "index_path":  str(path),
            "index_exists": path.exists(),
            "version":     _INDEX_VERSION,
            "bm25_ok":     _BM25_OK,
            "numpy_ok":    _NP,
            "stemmer_ok":  _STEM_OK,
        }

    def set_modality_filter(self, modality: Optional[str]) -> None:
        self.modality_filter = modality
