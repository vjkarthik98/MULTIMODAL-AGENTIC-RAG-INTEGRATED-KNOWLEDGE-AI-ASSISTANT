from __future__ import annotations

import asyncio
import hashlib
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings
from app.utils.logger import get_logger
from app.utils.paths import user_bm25_path

logger = get_logger(__name__)

# ── Index schema version ──────────────────────────────────────────────────────
# Bump whenever tokenizer behaviour changes (stemmer, bigrams, field weights).
# On load, if the pickled version ≠ _INDEX_VERSION the index is discarded so
# the next add_documents() call rebuilds from a clean state rather than mixing
# tokens from two different schemas.
_INDEX_VERSION = 4

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NP_AVAILABLE = False

try:
    from rank_bm25 import BM25Plus
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

    class BM25Plus:  # type: ignore[no-redef]
        def __init__(self, corpus: List[List[str]], **_kw: Any) -> None:
            self.corpus = corpus

        def get_scores(self, tokens: List[str]) -> List[float]:
            query = set(tokens)
            return [float(len(query & set(doc))) for doc in self.corpus]

try:
    import Stemmer as _PyStemmer
    _stemmer = _PyStemmer.Stemmer("english")
    _STEMMER_AVAILABLE = True
except ImportError:
    _stemmer = None  # type: ignore[assignment]
    _STEMMER_AVAILABLE = False

try:
    import pybreaker
    _breaker = pybreaker.CircuitBreaker(
        fail_max=settings.CIRCUIT_BREAKER_MAX_FAILURES,
        reset_timeout=settings.CIRCUIT_BREAKER_RESET_TIMEOUT,
    )
    _PYBREAKER_AVAILABLE = True
except ImportError:
    _PYBREAKER_AVAILABLE = False

    class _DummyBreaker:
        def __call__(self, fn):
            return fn

    _breaker = _DummyBreaker()  # type: ignore[assignment]


# ── Legacy global index path (kept for backward-compat) ───────────────────────
_LEGACY_INDEX_DIR = (
    Path(settings.BM25_INDEX_DIR)
    if hasattr(settings, "BM25_INDEX_DIR")
    else settings.DATA_DIR / "bm25_index"
)
_LEGACY_INDEX_FILE = _LEGACY_INDEX_DIR / "bm25_index.pkl"


# ── Stopwords ─────────────────────────────────────────────────────────────────
_STOPWORDS: Set[str] = {
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

# ── Finance-critical words — MUST NOT be removed as stopwords ─────────────────
# Removing "not", "no", "loss" etc. destroys queries like "net loss",
# "not profitable", "below consensus", "revenue decline".
_FINANCE_KEEP_SET: Set[str] = {
    "not", "no", "loss", "losses", "deficit", "decline", "decrease",
    "negative", "below", "under", "miss", "missed", "shortfall",
    "risk", "risks", "uncertain", "uncertainty",
}

# ── Financial abbreviation expansion ─────────────────────────────────────────
# Applied at both index time and query time so abbreviated queries match full-
# form text and vice-versa. Keys must be single tokens (multi-word keys are
# never matched by the unigram tokenizer). Expansion tokens are stemmed before
# being added to the index, ensuring consistent matching.
_FIN_ABBR: Dict[str, List[str]] = {
    "eps":      ["earnings", "share"],
    "ebitda":   ["earnings", "interest", "taxes", "depreciation", "amortization"],
    "ebit":     ["earnings", "interest", "taxes"],
    "yoy":      ["year", "growth"],
    "fy":       ["fiscal", "year"],
    "q1":       ["first", "quarter"],
    "q2":       ["second", "quarter"],
    "q3":       ["third", "quarter"],
    "q4":       ["fourth", "quarter"],
    "ceo":      ["chief", "executive", "officer"],
    "cfo":      ["chief", "financial", "officer"],
    "coo":      ["chief", "operating", "officer"],
    "capex":    ["capital", "expenditure"],
    "opex":     ["operating", "expense"],
    "cogs":     ["cost", "goods", "sold"],
    "sga":      ["selling", "general", "administrative"],
    "r&d":      ["research", "development"],
    "ttm":      ["trailing", "twelve", "months"],
    "pe":       ["price", "earnings"],
    "pb":       ["price", "book"],
    "ps":       ["price", "sales"],
    "roe":      ["return", "equity"],
    "roa":      ["return", "assets"],
    "roi":      ["return", "investment"],
    "fcf":      ["free", "cash", "flow"],
    "gaap":     ["accounting", "principles"],
    "buyback":  ["repurchase", "share"],
    "ltv":      ["lifetime", "value"],
    "arpu":     ["average", "revenue", "user"],
    "mom":      ["month", "month"],
    "qoq":      ["quarter", "quarter"],
    "bps":      ["basis", "points"],
    "nii":      ["net", "interest", "income"],
    "nim":      ["net", "interest", "margin"],
    "npv":      ["net", "present", "value"],
    "irr":      ["internal", "rate", "return"],
    "wacc":     ["weighted", "average", "cost", "capital"],
}

# ── High-value financial bigrams ──────────────────────────────────────────────
# Detected in the raw text (before unigram tokenisation) and inserted as single
# compound tokens so phrases like "net income" score higher than individual
# words that happen to co-occur by chance. The underscore form is used as the
# BM25 token to avoid any further splitting.
_FIN_BIGRAMS: Set[str] = {
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
    # Finance-negative phrases — critical for "miss" / "decline" queries
    "net loss", "operating loss", "revenue decline", "revenue decrease",
    "below consensus", "missed estimates", "below expectations",
    "not profitable", "negative growth", "risk factors",
}

# ── English contraction expansion ────────────────────────────────────────────
_CONTRACTIONS: Dict[str, str] = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will",
    "'d": " would", "'m": " am", "'s": " is",
}

# ── Currency / unit normalisation patterns ────────────────────────────────────
# Strip leading currency symbols; convert trailing scale suffixes to words so
# "$6.13B" → "6.13 billion" and the number token "6.13" indexes alongside the
# magnitude word "billion".
_CURRENCY_RE   = re.compile(r'[$€£¥₹₩]')
_SCALE_RE      = re.compile(r'(\d[\d,.]*)\s*([bBmMkKtT])\b')
_SCALE_MAP     = {"b": "billion", "m": "million", "k": "thousand", "t": "trillion"}
_PERCENT_RE    = re.compile(r'(\d[\d.]*)\s*%')
_COMMA_NUM_RE  = re.compile(r'(\d),(\d)')   # 1,000 → 1000


def _normalize_text(text: str) -> str:
    """Light normalisation before tokenisation: contractions, currencies,
    scale suffixes, percentage signs."""
    # Contractions
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    # Percentages: "15%" → "15 percent"
    text = _PERCENT_RE.sub(r'\1 percent', text)

    # Remove commas in large numbers: "1,234,567" → "1234567"
    text = _COMMA_NUM_RE.sub(r'\1\2', text)

    # Scale suffixes: "45B" → "45 billion", "3.2M" → "3.2 million"
    def _expand_scale(m: re.Match) -> str:
        num    = m.group(1).replace(",", "")
        scale  = _SCALE_MAP[m.group(2).lower()]
        return f"{num} {scale}"
    text = _SCALE_RE.sub(_expand_scale, text)

    # Strip currency symbols
    text = _CURRENCY_RE.sub("", text)

    return text


def _expand_scale_variants(tokens: List[str]) -> List[str]:
    """For consecutive (number, scale_word) token pairs, emit cross-scale tokens.

    Example: ["4312", "million"] → also adds "4.3_billion", "4312_million".
    This bridges queries that phrase amounts in different scales, e.g. a chunk
    storing "4,312 million" will also match a query for "4.3 billion".
    """
    _SCALE_TO_MULT: Dict[str, float] = {
        "billion": 1e9, "million": 1e6, "thousand": 1e3, "trillion": 1e12,
    }
    _MULT_TO_SCALE: Dict[float, str] = {
        1e9: "billion", 1e6: "million", 1e3: "thousand", 1e12: "trillion",
    }
    extras: List[str] = []
    for i in range(len(tokens) - 1):
        scale_word = tokens[i + 1]
        if scale_word not in _SCALE_TO_MULT:
            continue
        try:
            num_val = float(tokens[i].replace(",", ""))
        except (ValueError, TypeError):
            continue
        # Always emit a compound token for the exact scale so phrases match
        # even when the scale word gets separated after stemming.
        extras.append(f"{tokens[i]}_{scale_word}")
        # Cross-scale variants
        absolute = num_val * _SCALE_TO_MULT[scale_word]
        for mult, scale in _MULT_TO_SCALE.items():
            if mult == _SCALE_TO_MULT[scale_word]:
                continue
            cross_val = absolute / mult
            if 0.001 <= cross_val <= 1e9:
                formatted = f"{cross_val:.3f}".rstrip("0").rstrip(".")
                extras.append(f"{formatted}_{scale}")
    return tokens + extras


def _stem_tokens(tokens: List[str]) -> List[str]:
    """Batch-stem a list of tokens using Snowball (PyStemmer) if available,
    otherwise return the list unchanged. Batch call is O(N) not O(N²)."""
    if _STEMMER_AVAILABLE and _stemmer and tokens:
        return _stemmer.stemWords(tokens)
    return tokens


class BM25Document:
    """Picklable document wrapper carrying all locator fields needed by the
    source-chip / citation layer so BM25 results are citation-complete."""
    __slots__ = [
        # core
        "text", "structure", "modality", "subtype",
        "source", "source_type", "chunk_id", "page",
        # PDF locators
        "sub_chunk_index", "total_sub_chunks",
        # DOCX locators
        "heading_level",
        # XLSX locators
        "sheet_name", "row_start", "row_end",
        # audio / video locators
        "timestamp_start", "timestamp_end", "speaker", "frame_index",
        # image / video content
        "caption",
    ]

    def __init__(self) -> None:
        # Initialise all slots to None so pickling never fails.
        for slot in self.__slots__:
            setattr(self, slot, None)

    @classmethod
    def from_payload(cls, p: Dict[str, Any]) -> "BM25Document":
        obj = cls()
        obj.text        = p.get("text") or p.get("content") or ""
        obj.modality    = p.get("modality", "text")
        obj.subtype     = p.get("subtype")
        obj.source      = p.get("source")
        obj.source_type = p.get("source_type")
        obj.chunk_id    = p.get("chunk_id")
        obj.page        = p.get("page")

        # Locators — PDF
        obj.sub_chunk_index  = p.get("sub_chunk_index")
        obj.total_sub_chunks = p.get("total_sub_chunks")

        # Locators — DOCX
        obj.heading_level = p.get("heading_level")

        # Locators — XLSX
        obj.sheet_name = p.get("sheet_name") or p.get("section_title")
        obj.row_start  = p.get("row_start")
        obj.row_end    = p.get("row_end")

        # Locators — audio / video
        obj.timestamp_start = p.get("timestamp_start")
        obj.timestamp_end   = p.get("timestamp_end")
        obj.speaker         = p.get("speaker")
        obj.frame_index     = p.get("frame_index")

        # Content — image / video
        obj.caption = p.get("caption")

        obj.structure = {
            "doc_id":              p.get("doc_id"),
            "chunk_id":            p.get("chunk_id"),
            "session_id":          p.get("session_id"),
            "content_type":        p.get("content_type"),
            "language":            p.get("language"),
            "section_id":          p.get("section_id"),
            "section_title":       p.get("section_title"),
            "timestamp_start":     p.get("timestamp_start"),
            "timestamp_end":       p.get("timestamp_end"),
            "ingestion_time":      p.get("ingestion_time"),
            "checksum_sha256":     p.get("checksum_sha256"),
            "section_number":      p.get("section_number"),
            "is_forward_looking":  p.get("is_forward_looking", False),
            "embedding_space":     "text",
            # Locators forwarded into structure so _metadata() can read them
            "sub_chunk_index":     p.get("sub_chunk_index"),
            "total_sub_chunks":    p.get("total_sub_chunks"),
            "heading_level":       p.get("heading_level"),
            "sheet":               p.get("sheet_name") or p.get("section_title"),
            "row_start":           p.get("row_start"),
            "row_end":             p.get("row_end"),
            "frame_index":         p.get("frame_index"),
            "caption":             p.get("caption"),
            "speaker":             p.get("speaker"),
        }
        return obj


class BM25Retriever:

    def __init__(self, user_id: Optional[str] = None) -> None:
        self.user_id: Optional[str] = user_id
        self.documents: List[Any]           = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Plus]       = None
        self.modality_filter: Optional[str] = None
        self.max_docs: int                  = settings.BM25_MAX_DOCS
        self._index_loaded: bool            = False
        self._loaded_user_id: Optional[str] = None

    # ── Index file path ───────────────────────────────────────────────────────

    def _index_file(self, user_id: Optional[str] = None) -> Path:
        uid = user_id or self.user_id
        if uid:
            return user_bm25_path(uid)
        return _LEGACY_INDEX_FILE

    # ── Hash ─────────────────────────────────────────────────────────────────

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ── Multi-field indexed text ──────────────────────────────────────────────

    def _build_indexed_text(self, doc: Any) -> str:
        """Build the text submitted to the tokenizer. Concatenates the primary
        content with high-signal metadata fields (section heading, sheet name,
        caption, speaker) so metadata-based queries also retrieve the chunk.
        High-value fields are repeated to boost their term frequency within the
        BM25 document representation."""
        parts: List[str] = []

        text = (getattr(doc, "text", "") or "").strip()
        if text:
            parts.append(text[:settings.BM25_MAX_TEXT_CHARS])

        s = getattr(doc, "structure", {}) or {}
        modality    = (getattr(doc, "modality", "") or "").lower()
        source_type = (getattr(doc, "source_type", "") or "").lower()
        content_type = (s.get("content_type") or "").lower()

        # Section heading — 2× repetition = 2× term frequency → stronger signal
        section_title = (s.get("section_title") or "").strip()
        if section_title:
            parts.append(section_title)
            parts.append(section_title)

        # ── Phase 3.2: PDF — page number token + table title ─────────────────
        # Enables "what is on page 47" and "balance sheet table" queries.
        if source_type == "pdf" or content_type.startswith("pdf_"):
            page_num = getattr(doc, "page", None) or s.get("page_number")
            if page_num is not None:
                parts.append(f"page_{page_num}")
            table_title = (s.get("table_title") or "").strip()
            if table_title:
                parts.append(table_title)
                parts.append(table_title)

        # ── Phase 3.4: DOCX — heading hierarchy ──────────────────────────────
        # "financial projections section EBITDA" query matches via heading tokens.
        if source_type == "word" or content_type.startswith("docx_"):
            section_hierarchy = s.get("section_hierarchy") or []
            if section_hierarchy:
                hierarchy_text = "heading " + " ".join(section_hierarchy).lower()
                parts.append(hierarchy_text)

        # ── Phase 3.5: XLSX — sheet name + semantic group ─────────────────────
        # Enables "income statement sheet revenue" and "revenue lines group" queries.
        if source_type == "excel" or content_type.startswith("excel_"):
            sheet = (s.get("sheet") or s.get("sheet_name") or "").strip()
            if sheet and sheet != section_title:
                sheet_tok = "sheet " + sheet.lower().replace(" ", "_")
                parts.append(sheet_tok)
                parts.append(sheet_tok)  # 2× boost for sheet name
            semantic_group = (s.get("semantic_group") or "").strip()
            if semantic_group:
                parts.append("group " + semantic_group.lower())

        # ── Phase 3.3: Audio / Video — speaker role, call section, timestamp ──
        # Enables "what did the CFO say about margins at 30min" queries.
        if modality in ("audio", "video", "mp3", "mp4"):
            speaker_role = (s.get("speaker_role") or "").strip()
            call_section = (s.get("call_section") or "").strip()
            ts_start = s.get("timestamp_start")
            # Prepend speaker/section context
            if speaker_role:
                parts.insert(0, f"role {speaker_role.lower()}")
            if call_section:
                parts.insert(0, f"section {call_section.replace('_', ' ')}")
            # Timestamp token: enables "at 30 minutes" style queries
            if ts_start is not None:
                try:
                    ts = float(ts_start)
                    mins = int(ts // 60)
                    secs = int(ts % 60)
                    parts.append(f"at_{mins}min{secs:02d}sec")
                except (ValueError, TypeError):
                    pass
            # Finance entities in audio/video metadata
            finance_entities = s.get("finance_entities") or {}
            if isinstance(finance_entities, dict):
                for entity_list in finance_entities.values():
                    if isinstance(entity_list, list):
                        parts.extend(str(e) for e in entity_list[:5])

        # ── Phase 3.6: Image — image type prefix + extracted numbers ──────────
        # Enables "revenue bar chart" and exact number queries on image content.
        if modality in ("image", "jpg") or source_type == "image":
            image_type = (s.get("image_type") or "").strip()
            if image_type:
                parts.insert(0, image_type.replace("_", " "))
            extracted_numbers = s.get("extracted_numbers") or []
            if extracted_numbers:
                parts.append("extracted " + " ".join(str(n) for n in extracted_numbers[:10]))
                # Repeat numbers for higher BM25 weight — they are the primary
                # retrievable content in finance charts.
                parts.append(" ".join(str(n) for n in extracted_numbers[:10]))

        # Image / video caption (existing, kept)
        caption = (s.get("caption") or "").strip()
        if caption:
            parts.append(caption)

        # Audio / video speaker name (existing, kept)
        speaker = (s.get("speaker") or "").strip()
        if speaker:
            parts.append(speaker)

        return " ".join(parts)

    # ── Tokenizer ─────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        text = str(text or "").lower()

        # ── Pre-normalise ────────────────────────────────────────────────────
        text = _normalize_text(text)

        # ── Bigram extraction ────────────────────────────────────────────────
        # Detected BEFORE splitting so "net income" becomes "net_income" as one
        # token, giving the phrase a higher BM25 score than the individual words.
        bigram_tokens: List[str] = []
        for bigram in _FIN_BIGRAMS:
            if bigram in text:
                bigram_tokens.append(bigram.replace(" ", "_"))

        # ── Unigram extraction ───────────────────────────────────────────────
        # Keep decimal/dollar figures intact ("6.13" not ["6", "13"]).
        raw_tokens = re.findall(r'\d+\.\d+|\b[a-z0-9]+\b', text)
        raw_tokens = [t for t in raw_tokens if t not in _STOPWORDS and len(t) > 1]

        # ── Abbreviation expansion ───────────────────────────────────────────
        # For every known abbreviation, add its expansion so abbreviated queries
        # match full-form text and vice-versa. Expansion tokens go through
        # stemming below, just like every other token.
        expanded: List[str] = []
        for tok in raw_tokens:
            expanded.append(tok)
            if tok in _FIN_ABBR:
                expanded.extend(_FIN_ABBR[tok])

        # ── Stemming ─────────────────────────────────────────────────────────
        # Batch-stem all unigrams (Snowball / PyStemmer). Applied symmetrically
        # at index time and query time so "revenues" and "revenue" both stem to
        # "revenu" and match each other.
        stemmed = _stem_tokens(expanded)

        # Re-filter stopwords from stemmed tokens (stemmer may produce stop-like
        # forms) and deduplicate within a fixed window to reduce noise.
        # _FINANCE_KEEP_SET overrides _STOPWORDS — these words are semantically
        # critical in finance ("not profitable", "net loss", "below consensus").
        seen_in_window: Set[str] = set()
        clean: List[str] = []
        for tok in stemmed:
            if not tok or len(tok) <= 1:
                continue
            if tok in _FINANCE_KEEP_SET or tok not in _STOPWORDS:
                if tok not in seen_in_window:
                    clean.append(tok)
                    seen_in_window.add(tok)

        # ── Cross-scale number variants ───────────────────────────────────────
        # Emit "4312_million" and "4.3_billion" from the same chunk so queries
        # at different scales still match via BM25 overlap.
        clean = _expand_scale_variants(clean)

        all_tokens = bigram_tokens + clean
        return all_tokens[:settings.BM25_MAX_TOKENS]

    # ── Metadata extraction ───────────────────────────────────────────────────

    def _metadata(self, doc: Any) -> Dict[str, Any]:
        s = dict(getattr(doc, "structure", {}) or {})
        return {
            "modality":          getattr(doc, "modality", "text"),
            "subtype":           getattr(doc, "subtype", None),
            "source":            getattr(doc, "source", None),
            "source_type":       getattr(doc, "source_type", None),
            "doc_id":            s.get("doc_id"),
            "chunk_id":          getattr(doc, "chunk_id", None),
            "session_id":        s.get("session_id"),
            "content_type":      s.get("content_type"),
            "page":              getattr(doc, "page", None),
            "language":          s.get("language"),
            "section_id":        s.get("section_id"),
            "section_title":     s.get("section_title"),
            "is_forward_looking": s.get("is_forward_looking", False),
            "section_number":    s.get("section_number"),
            # ── PDF locators ──────────────────────────────────────────────────
            "sub_chunk_index":   s.get("sub_chunk_index"),
            "total_sub_chunks":  s.get("total_sub_chunks"),
            # ── DOCX locators ─────────────────────────────────────────────────
            "heading":           s.get("heading"),
            "heading_level":     s.get("heading_level"),
            # ── XLSX locators ─────────────────────────────────────────────────
            # xlsx_chunker.py's structure dict uses "sheet_name" (not "sheet")
            # and a combined "row_range" list (not separate row_start/row_end) —
            # the old key names here never matched, so every XLSX source lost
            # its sheet/row citation once routed through BM25 (accuracy phase
            # 2026-07). Read both shapes defensively.
            "sheet_name":        s.get("sheet_name") or s.get("sheet") or s.get("section_title"),
            "row_range":         s.get("row_range"),
            "row_start":         s.get("row_start") or (s.get("row_range") or [None])[0],
            "row_end":           s.get("row_end") or (s.get("row_range") or [None, None])[-1],
            # ── Audio / video locators ────────────────────────────────────────
            # audio_chunker.py's structure dict uses "start_timestamp"/
            # "end_timestamp" (reversed word order) and "speaker_name"/
            # "speaker_label"/"speaker_role" (not a single "speaker" key) — the
            # old key names here never matched, so every audio source lost its
            # speaker/timestamp citation once routed through BM25 (accuracy
            # phase 2026-07, same class of bug as the XLSX sheet_name fix
            # above). Read both shapes defensively.
            "timestamp_start":   s.get("timestamp_start") or s.get("start_timestamp"),
            "timestamp_end":     s.get("timestamp_end") or s.get("end_timestamp"),
            "start_timestamp":   s.get("start_timestamp") or s.get("timestamp_start"),
            "end_timestamp":     s.get("end_timestamp") or s.get("timestamp_end"),
            "speaker":           s.get("speaker") or s.get("speaker_name") or s.get("speaker_label"),
            "speaker_name":      s.get("speaker_name") or s.get("speaker_label") or s.get("speaker"),
            "speaker_label":     s.get("speaker_label"),
            "speaker_role":      s.get("speaker_role"),
            "call_section":      s.get("call_section"),
            "frame_index":       s.get("frame_index"),
            # ── Image / video caption ─────────────────────────────────────────
            "caption":           s.get("caption"),
            # ── Misc ─────────────────────────────────────────────────────────
            "ingestion_time":    s.get("ingestion_time"),
            "checksum_sha256":   s.get("checksum_sha256"),
        }

    # ── Modality filter setter ────────────────────────────────────────────────

    def set_modality_filter(self, modality: Optional[str]) -> None:
        self.modality_filter = modality

    # ── Circuit-broken save ───────────────────────────────────────────────────

    def _save_index(self, user_id: Optional[str] = None) -> None:
        index_file = self._index_file(user_id)

        def _do_save() -> None:
            index_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "index_version":    _INDEX_VERSION,
                "documents":        self.documents,
                "tokenized_corpus": self.tokenized_corpus,
                "saved_at":         time.time(),
                "doc_count":        len(self.documents),
            }
            tmp_path = index_file.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(index_file)
            logger.info(
                event="bm25_index_saved",
                path=str(index_file),
                docs=len(self.documents),
                version=_INDEX_VERSION,
            )

        try:
            if _PYBREAKER_AVAILABLE:
                _breaker(_do_save)()
            else:
                _do_save()
        except Exception as exc:
            logger.error(event="bm25_index_save_failed", error=str(exc))

    # ── Circuit-broken load ───────────────────────────────────────────────────

    def _load_index(self, user_id: Optional[str] = None) -> None:
        effective_uid = user_id or self.user_id
        if self._index_loaded and self._loaded_user_id == effective_uid:
            return
        if self._index_loaded and self._loaded_user_id != effective_uid:
            self._index_loaded = False
            self.documents = []
            self.tokenized_corpus = []
            self.bm25 = None

        index_file = self._index_file(user_id)

        if not index_file.exists():
            logger.info(event="bm25_no_saved_index", path=str(index_file))
            return

        def _do_load() -> None:
            with open(index_file, "rb") as f:
                payload = pickle.load(f)

            # Discard stale index — tokenizer changed since it was built
            saved_version = payload.get("index_version", 0)
            if saved_version != _INDEX_VERSION:
                logger.warning(
                    event="bm25_index_version_mismatch",
                    saved=saved_version,
                    current=_INDEX_VERSION,
                    path=str(index_file),
                )
                return

            self.documents        = payload.get("documents", [])
            self.tokenized_corpus = payload.get("tokenized_corpus", [])
            if self.tokenized_corpus:
                self.bm25 = BM25Plus(self.tokenized_corpus)
                logger.info(
                    event="bm25_index_loaded",
                    docs=len(self.documents),
                    path=str(index_file),
                    version=_INDEX_VERSION,
                )
            else:
                logger.warning(event="bm25_saved_index_empty")
            self._index_loaded    = True
            self._loaded_user_id  = effective_uid

        try:
            if _PYBREAKER_AVAILABLE:
                _breaker(_do_load)()
            else:
                _do_load()
        except Exception as exc:
            logger.error(event="bm25_index_load_failed", error=str(exc))
            self.documents = []
            self.tokenized_corpus = []
            self.bm25 = None

    # ── Full-rebuild index ────────────────────────────────────────────────────

    def build_index(self, documents: List[Any], user_id: Optional[str] = None) -> None:
        if not documents:
            logger.warning(event="bm25_empty_input")
            return

        start = time.time()
        self.documents        = []
        self.tokenized_corpus = []
        self.bm25             = None
        seen: Set[str]        = set()

        for doc in documents[:self.max_docs]:
            try:
                text = getattr(doc, "text", "")
                structure = getattr(doc, "structure", {}) or {}

                if not text:
                    continue
                if structure.get("embedding_space", "text") != "text":
                    continue

                h = self._hash(text[:settings.BM25_MAX_TEXT_CHARS])
                if h in seen:
                    continue
                seen.add(h)

                tokens = self._tokenize(self._build_indexed_text(doc))
                if not tokens:
                    continue

                self.documents.append(doc)
                self.tokenized_corpus.append(tokens)

            except Exception as exc:
                logger.warning(event="bm25_doc_skip", error=str(exc))

        if not self.tokenized_corpus:
            logger.warning(event="bm25_no_corpus")
            return

        self.bm25 = BM25Plus(self.tokenized_corpus)
        self._save_index(user_id)

        logger.info(
            event="bm25_index_built",
            docs=len(self.documents),
            latency=round(time.time() - start, 2),
            stemmer=_STEMMER_AVAILABLE,
            version=_INDEX_VERSION,
        )

    # ── Single-document incremental add ──────────────────────────────────────

    def add_document(self, text: str, metadata: Dict[str, Any], user_id: Optional[str] = None) -> None:
        if not text or not text.strip():
            return
        h = self._hash(text[:settings.BM25_MAX_TEXT_CHARS])
        seen_existing: Set[str] = {
            self._hash(getattr(d, "text", "")[:settings.BM25_MAX_TEXT_CHARS])
            for d in self.documents
        }
        if h in seen_existing:
            return

        doc = BM25Document()
        doc.text        = text
        doc.structure   = metadata
        doc.modality    = metadata.get("modality", "text")
        doc.subtype     = metadata.get("subtype")
        doc.source      = metadata.get("source")
        doc.source_type = metadata.get("source_type")
        doc.chunk_id    = metadata.get("chunk_id")
        doc.page        = metadata.get("page")
        # Locators
        doc.sub_chunk_index  = metadata.get("sub_chunk_index")
        doc.total_sub_chunks = metadata.get("total_sub_chunks")
        doc.heading_level    = metadata.get("heading_level")
        doc.sheet_name       = metadata.get("sheet_name")
        doc.row_start        = metadata.get("row_start")
        doc.row_end          = metadata.get("row_end")
        doc.timestamp_start  = metadata.get("timestamp_start")
        doc.timestamp_end    = metadata.get("timestamp_end")
        doc.speaker          = metadata.get("speaker")
        doc.frame_index      = metadata.get("frame_index")
        doc.caption          = metadata.get("caption")

        tokens = self._tokenize(self._build_indexed_text(doc))
        if not tokens:
            return

        self.documents.append(doc)
        self.tokenized_corpus.append(tokens)
        self.bm25 = BM25Plus(self.tokenized_corpus)
        self._save_index(user_id)
        logger.info(
            event="bm25_document_added",
            session_id=metadata.get("session_id"),
            user_id=user_id or self.user_id,
            total=len(self.documents),
        )

    # ── Batch incremental add ─────────────────────────────────────────────────

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

        seen_existing: Set[str] = {
            self._hash(getattr(d, "text", "")[:settings.BM25_MAX_TEXT_CHARS])
            for d in self.documents
        }

        for doc in documents:
            try:
                text      = getattr(doc, "text", "")
                structure = getattr(doc, "structure", {}) or {}

                if not text:
                    continue
                if structure.get("embedding_space", "text") != "text":
                    continue

                h = self._hash(text[:settings.BM25_MAX_TEXT_CHARS])
                if h in seen_existing:
                    continue

                tokens = self._tokenize(self._build_indexed_text(doc))
                if not tokens:
                    continue

                self.documents.append(doc)
                self.tokenized_corpus.append(tokens)
                seen_existing.add(h)
                added += 1

                if len(self.documents) >= self.max_docs:
                    logger.warning(
                        event="bm25_max_docs_reached",
                        max=self.max_docs,
                        session_id=session_id,
                    )
                    break

            except Exception as exc:
                logger.warning(event="bm25_add_doc_skip", error=str(exc))

        if added == 0:
            return

        self.bm25 = BM25Plus(self.tokenized_corpus)
        self._save_index(user_id)

        logger.info(
            event="bm25_documents_added",
            added=added,
            total=len(self.documents),
            latency=round(time.time() - start, 2),
            session_id=session_id,
            user_id=user_id or self.user_id,
        )

    # ── Score normalisation ───────────────────────────────────────────────────

    def _normalize_scores(self, raw_scores: Any) -> Any:
        if _NP_AVAILABLE:
            scores = np.asarray(raw_scores, dtype=float)
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            min_s  = float(scores.min()) if scores.size > 0 else 0.0
            max_s  = float(scores.max()) if scores.size > 0 else 1e-6
            spread = max_s - min_s
            if spread > 1e-6:
                scores = (scores - min_s) / spread   # min-max → [0, 1]
            else:
                scores = np.zeros_like(scores)
            return scores
        else:
            scores = [float(s) for s in raw_scores]
            min_s  = min(scores) if scores else 0.0
            max_s  = max(scores) if scores else 1e-6
            spread = max_s - min_s
            if spread > 1e-6:
                return [(s - min_s) / spread for s in scores]
            return [0.0] * len(scores)

    # ── Top-k indices ─────────────────────────────────────────────────────────

    def _topk_indices(self, norm_scores: Any, top_k: int) -> List[int]:
        if _NP_AVAILABLE:
            if len(norm_scores) <= top_k:
                idxs = list(range(len(norm_scores)))
            else:
                idxs = list(np.argpartition(norm_scores, -top_k)[-top_k:])
            return sorted(idxs, key=lambda i: norm_scores[i], reverse=True)
        else:
            idxs = sorted(range(len(norm_scores)), key=lambda i: norm_scores[i], reverse=True)
            return idxs[:top_k]

    # ── Filter predicate ──────────────────────────────────────────────────────

    def _passes_filters(
        self,
        meta: Dict[str, Any],
        user_id: Optional[str],
        filters: Optional[Dict[str, Any]],
    ) -> bool:
        # Tenant isolation — user_id is the primary security boundary
        if user_id and meta.get("session_id"):
            # BM25 index is per-user (file-isolated), but verify the embedded
            # session belongs to a doc the calling user can access.
            pass

        if self.modality_filter and meta.get("modality") != self.modality_filter:
            return False

        if filters:
            if filters.get("modality") and meta.get("modality") != filters["modality"]:
                return False
            if filters.get("language") and meta.get("language") != filters["language"]:
                return False
            if filters.get("source_type") and meta.get("source_type") != filters["source_type"]:
                return False
            if filters.get("user_id") and meta.get("user_id") != filters["user_id"]:
                return False

        return True

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        if not self.bm25:
            self._load_index(user_id)

        if not self.bm25:
            logger.info(event="bm25_empty_index_returning_empty_list", session_id=session_id)
            return []

        if not query:
            return []

        start  = time.time()
        top_k  = min(top_k or settings.BM25_TOP_K, len(self.documents))

        if top_k <= 0:
            return []

        query  = query[:settings.MAX_PROMPT_CHARS]
        tokens = self._tokenize(query)

        if not tokens:
            return []

        try:
            raw_scores = self.bm25.get_scores(tokens)
        except Exception as exc:
            logger.error(event="bm25_score_failed", error=str(exc), session_id=session_id)
            return []

        norm_scores = self._normalize_scores(raw_scores)
        idxs        = self._topk_indices(norm_scores, top_k)

        modality_weights: Dict[str, float] = getattr(settings, "BM25_MODALITY_WEIGHTS", {
            "text": 1.0, "table": 1.1, "image": 0.9,
            "audio": 1.0, "video": 1.0,
        })

        results: List[Dict[str, Any]] = []

        for idx in idxs:
            if len(results) >= top_k:
                break
            if idx >= len(self.documents):
                continue

            doc  = self.documents[idx]
            meta = self._metadata(doc)

            if not self._passes_filters(meta, user_id, filters):
                continue

            text = getattr(doc, "text", "").strip()
            if not text:
                continue

            raw_score      = float(norm_scores[idx])
            modality_boost = modality_weights.get(meta.get("modality", "text"), 1.0)
            final_score    = raw_score * modality_boost

            if final_score < settings.BM25_MIN_SCORE:
                continue

            results.append({
                "id":       f"bm25_{idx}",
                "text":     text[:settings.RAG_DOC_MAX_CHARS],
                "score":    round(final_score, 5),
                "metadata": meta,
            })

        logger.info(
            event="bm25_search_success",
            query_len=len(query),
            results=len(results),
            top_k=top_k,
            latency=round(time.time() - start, 3),
            session_id=session_id,
        )

        return results

    # ── Async wrapper ─────────────────────────────────────────────────────────

    async def async_search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.search, query, session_id, top_k, filters, user_id)

    # ── Delete by source file ─────────────────────────────────────────────────

    def delete_by_source(self, filename: str, user_id: Optional[str] = None) -> int:
        before         = len(self.documents)
        filtered_docs  = []
        filtered_corp  = []
        for doc, tokens in zip(self.documents, self.tokenized_corpus):
            source = getattr(doc, "source", "") or ""
            if filename not in source:
                filtered_docs.append(doc)
                filtered_corp.append(tokens)
        self.documents        = filtered_docs
        self.tokenized_corpus = filtered_corp
        removed = before - len(self.documents)
        if removed > 0:
            self.bm25 = BM25Plus(self.tokenized_corpus) if self.tokenized_corpus else None
            self._save_index(user_id)
            logger.info(event="bm25_delete_by_source", filename=filename, removed=removed)
        return removed

    # ── GDPR session purge ────────────────────────────────────────────────────

    def purge_by_session(self, session_id: str) -> int:
        before        = len(self.documents)
        filtered_docs = []
        filtered_corp = []
        for doc, tokens in zip(self.documents, self.tokenized_corpus):
            s = getattr(doc, "structure", {}) or {}
            if s.get("session_id") != session_id:
                filtered_docs.append(doc)
                filtered_corp.append(tokens)
        self.documents        = filtered_docs
        self.tokenized_corpus = filtered_corp
        removed = before - len(self.documents)
        if removed > 0:
            self.bm25 = BM25Plus(self.tokenized_corpus) if self.tokenized_corpus else None
            self._save_index()
            logger.info(
                event="bm25_purge_session",
                session_id=session_id,
                removed=removed,
            )
        return removed

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        index_file = self._index_file(user_id)
        return {
            "ready":            self.bm25 is not None,
            "doc_count":        len(self.documents),
            "index_exists":     index_file.exists(),
            "index_size_bytes": index_file.stat().st_size if index_file.exists() else 0,
            "index_path":       str(index_file),
            "index_version":    _INDEX_VERSION,
            "user_id":          user_id or self.user_id,
            "modality_filter":  self.modality_filter,
            "bm25_available":   _BM25_AVAILABLE,
            "numpy_available":  _NP_AVAILABLE,
            "stemmer_enabled":  _STEMMER_AVAILABLE,
            "circuit_breaker":  _PYBREAKER_AVAILABLE,
        }

    # ── Clear ─────────────────────────────────────────────────────────────────

    def clear(self, user_id: Optional[str] = None) -> None:
        self.documents        = []
        self.tokenized_corpus = []
        self.bm25             = None
        self._index_loaded    = False
        index_file            = self._index_file(user_id)
        if index_file.exists():
            try:
                index_file.unlink()
                logger.info(event="bm25_index_cleared", path=str(index_file))
            except Exception as exc:
                logger.error(event="bm25_index_clear_failed", error=str(exc))


# ── Multi-index aggregator (Phase 4) ──────────────────────────────────────────
# Routes documents to the correct per-modality index, then fuses search results
# from all 7 indexes using Reciprocal Rank Fusion (RRF).  Exposes the same
# public API as BM25Retriever for backward-compatible hot-swap.

from app.bm25.txt_bm25   import TxtBM25
from app.bm25.pdf_bm25   import PdfBM25
from app.bm25.docx_bm25  import DocxBM25
from app.bm25.xlsx_bm25  import XlsxBM25
from app.bm25.image_bm25 import ImageBM25
from app.bm25.audio_bm25 import AudioBM25
from app.bm25.video_bm25 import VideoBM25

# Map modality string → index class
_MODALITY_TO_CLASS = {
    "txt":   TxtBM25,   "text": TxtBM25,
    "pdf":   PdfBM25,
    "word":  DocxBM25,  "docx": DocxBM25,
    "excel": XlsxBM25,  "xlsx": XlsxBM25,
    "image": ImageBM25,
    "audio": AudioBM25, "mp3":  AudioBM25, "mp4a": AudioBM25,
    "video": VideoBM25, "mp4":  VideoBM25,
}

_RRF_K = 60  # standard RRF constant — higher = gentler decay


def _rrf_fuse(
    ranked_lists: List[List[Dict[str, Any]]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion across multiple ranked result lists."""
    scores: Dict[str, float] = {}
    items:  Dict[str, Dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            key = item.get("id") or item.get("metadata", {}).get("chunk_id") or str(rank)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if key not in items:
                items[key] = item

    fused = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
    results = []
    for key in fused:
        hit = dict(items[key])
        hit["score"] = round(scores[key], 5)
        hit["_rrf_fused"] = True
        results.append(hit)
    return results


class BM25AggregatorRetriever:
    """Multi-index BM25 aggregator.

    Holds one per-modality index instance per user.  Documents are routed to
    the matching modality index on insert; search queries all indexes (or a
    filtered subset) and merges via RRF.

    Public API mirrors BM25Retriever for drop-in replacement.
    """

    def __init__(self, user_id: Optional[str] = None) -> None:
        self.user_id = user_id
        # Lazily instantiated per-modality indexes
        self._indexes: Dict[str, Any] = {}

    def _get_index(self, modality: str, user_id: Optional[str] = None) -> Any:
        uid = user_id or self.user_id
        cls = _MODALITY_TO_CLASS.get(modality)
        if cls is None:
            cls = TxtBM25  # default to txt for unknown modalities
        key = f"{uid}:{cls.modality}"
        if key not in self._indexes:
            self._indexes[key] = cls(user_id=uid)
        return self._indexes[key]

    def _all_indexes(self, user_id: Optional[str] = None) -> List[Any]:
        uid = user_id or self.user_id
        idxs = []
        for cls in set(_MODALITY_TO_CLASS.values()):
            key = f"{uid}:{cls.modality}"
            if key not in self._indexes:
                self._indexes[key] = cls(user_id=uid)
            idxs.append(self._indexes[key])
        return idxs

    # ── Add ──────────────────────────────────────────────────────────────────

    def add_document(self, text: str, metadata: Dict[str, Any], user_id: Optional[str] = None) -> None:
        modality = metadata.get("modality") or metadata.get("source_type") or "text"
        doc = BM25Document.from_payload({**metadata, "text": text})
        self._get_index(modality, user_id).add_documents([doc], user_id=user_id)

    def add_documents(
        self,
        documents: List[Any],
        session_id: str = "",
        user_id: Optional[str] = None,
    ) -> None:
        if not documents:
            return
        buckets: Dict[str, List[Any]] = {}
        for doc in documents:
            s        = getattr(doc, "structure", {}) or {}
            modality = (getattr(doc, "modality", None) or s.get("modality") or
                        getattr(doc, "source_type", None) or "text")
            cls      = _MODALITY_TO_CLASS.get(modality, TxtBM25)
            key      = cls.modality
            buckets.setdefault(key, []).append(doc)

        for mod_key, batch in buckets.items():
            uid = user_id or self.user_id
            idx_key = f"{uid}:{mod_key}"
            cls = next(c for c in set(_MODALITY_TO_CLASS.values()) if c.modality == mod_key)
            if idx_key not in self._indexes:
                self._indexes[idx_key] = cls(user_id=uid)
            self._indexes[idx_key].add_documents(batch, session_id=session_id, user_id=uid)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or settings.BM25_TOP_K

        # If modality filter specified, query only that index
        if filters and filters.get("modality"):
            idx = self._get_index(filters["modality"], user_id)
            return idx.search(query, session_id=session_id, top_k=k,
                               filters=filters, user_id=user_id)

        # Otherwise query all indexes and fuse
        ranked_lists: List[List[Dict[str, Any]]] = []
        for idx in self._all_indexes(user_id):
            try:
                results = idx.search(query, session_id=session_id, top_k=k,
                                     filters=filters, user_id=user_id)
                if results:
                    ranked_lists.append(results)
            except Exception as exc:
                logger.warning(event="bm25_agg_index_search_failed",
                               modality=idx.modality, error=str(exc))

        if not ranked_lists:
            return []
        if len(ranked_lists) == 1:
            return ranked_lists[0][:k]
        return _rrf_fuse(ranked_lists, k)

    async def async_search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self.search, query, session_id, top_k, filters, user_id
        )

    # ── Delete / purge / clear ────────────────────────────────────────────────

    def delete_by_source(self, filename: str, user_id: Optional[str] = None) -> int:
        removed = 0
        for idx in self._all_indexes(user_id):
            try:
                removed += idx.delete_by_source(filename, user_id)
            except Exception as exc:
                logger.warning(event="bm25_agg_delete_failed",
                               modality=idx.modality, error=str(exc))
        return removed

    def purge_by_session(self, session_id: str) -> int:
        removed = 0
        for idx in self._all_indexes():
            try:
                removed += idx.purge_by_session(session_id)
            except Exception as exc:
                logger.warning(event="bm25_agg_purge_failed",
                               modality=idx.modality, error=str(exc))
        return removed

    def clear(self, user_id: Optional[str] = None) -> None:
        for idx in self._all_indexes(user_id):
            try:
                idx.clear(user_id)
            except Exception as exc:
                logger.warning(event="bm25_agg_clear_failed",
                               modality=idx.modality, error=str(exc))
        self._indexes = {}

    # ── Health check ─────────────────────────────────────────────────────────

    def health_check(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        results = {}
        total_docs = 0
        for idx in self._all_indexes(user_id):
            h = idx.health_check(user_id)
            results[idx.modality] = h
            total_docs += h.get("doc_count", 0)
        return {
            "aggregator": True,
            "total_doc_count": total_docs,
            "modality_indexes": results,
        }

    # ── Legacy compat ─────────────────────────────────────────────────────────

    def set_modality_filter(self, modality: Optional[str]) -> None:
        """No-op on aggregator — pass modality via search(filters={'modality': ...}) instead."""
        logger.debug(event="bm25_agg_set_modality_filter_ignored", modality=modality)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from app.core.config import settings
    from app.vectorstore.qdrant_store import QdrantVectorStore
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    parser = argparse.ArgumentParser(description="Rebuild BM25 index from Qdrant data")
    parser.add_argument("--user_id", default="eval_default")
    _args = parser.parse_args()
    _user_id = _args.user_id

    _qs = QdrantVectorStore()
    _bm25 = BM25Retriever(user_id=_user_id)
    _filter = Filter(
        must=[FieldCondition(key="user_id", match=MatchValue(value=_user_id))]
    ) if _user_id else None
    _points, _ = _qs.client.scroll(
        collection_name=settings.TEXT_COLLECTION_NAME,
        with_payload=True,
        limit=5000,
        scroll_filter=_filter,
    )
    _docs = [BM25Document.from_payload(pt.payload or {}) for pt in _points]
    _docs = [d for d in _docs if d.text]
    _bm25.build_index(_docs, user_id=_user_id)
    print(f"Rebuilt BM25 index for user '{_user_id}': {len(_bm25.documents)} docs indexed.")
