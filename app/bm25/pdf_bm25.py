"""pdf_bm25.py — BM25 index for PDF document chunks."""

from __future__ import annotations

import re
from typing import Any

from prometheus_client import Counter

from app.bm25.base_bm25 import BaseBM25
from app.utils.logger import get_logger

logger = get_logger(__name__)

_BM25_INDEXED = Counter(
    "magik_pdf_bm25_indexed_total",
    "Documents indexed in pdf BM25",
)
_BM25_INDEX_ERRORS = Counter(
    "magik_pdf_bm25_index_errors_total",
    "Errors in pdf BM25 _build_indexed_text",
)

# Filename stem → short company token for BM25 boosting
_SOURCE_COMPANY_TOKENS: dict[str, str] = {
    "apple": "apple",
    "microsoft": "microsoft",
    "google": "google alphabet",
    "alphabet": "alphabet google",
    "amazon": "amazon",
    "meta": "meta facebook",
    "nvidia": "nvidia",
    "tesla": "tesla",
    "berkshire": "berkshire hathaway",
    "jpmorgan": "jpmorgan chase",
}

# Financial-statement table types mapped to BM25-boosted token strings
_TABLE_BM25_TOKENS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"income statement|statement.{0,10}operation", re.I),
        "income statement revenue operating expenses earnings",
    ),
    (
        re.compile(r"balance sheet|financial position", re.I),
        "balance sheet assets liabilities equity",
    ),
    (re.compile(r"cash flow", re.I), "cash flow operating investing financing activities"),
    (
        re.compile(r"net sales|revenue.{0,10}product", re.I),
        "net sales revenue product category segment",
    ),
    (re.compile(r"gross margin", re.I), "gross margin profit"),
    (re.compile(r"income tax", re.I), "income tax provision effective rate"),
    (
        re.compile(r"capital return|repurchase|dividend", re.I),
        "capital return share repurchase dividend",
    ),
    (re.compile(r"shareholders?.{0,6}equity", re.I), "shareholders equity retained earnings"),
    (re.compile(r"segment", re.I), "segment results geography"),
]

_SEC_ITEM_RE = re.compile(r'^Item\s+(\d+[A-Z]?)\.?\s*(.*)', re.IGNORECASE)
_FY_RE = re.compile(r'\bFY?(20\d{2})\b')


def _company_tokens(source: str) -> str:
    stem = re.sub(r'\.[^.]+$', '', source or '').lower()
    for key, tokens in _SOURCE_COMPANY_TOKENS.items():
        if key in stem:
            return tokens
    first = re.split(r'[_\-\s]', stem)[0]
    return first if first else ""


def _table_type_tokens(table_title: str) -> str:
    for pat, tokens in _TABLE_BM25_TOKENS:
        if pat.search(table_title):
            return tokens
    return ""


class PdfBM25(BaseBM25):
    """BM25 index for PDF documents (10-K, 10-Q, annual reports, research).

    Enrichment over base:
    - Company name tokens (apple, microsoft …) from source filename
    - page_N token so "page 12" queries find the right chunk
    - Fiscal year tokens (fy2024, 2024) amplified for table chunks
    - table_title ×2 + semantic table-type tokens for financial tables
    - SEC Item number (item_7, mda) and Part (part_ii) tokens
    - section_hierarchy path as tokens
    - Finance entities (top-5) appended for BM25 lexical matching
    """

    modality = "pdf"

    def _build_indexed_text(self, doc: Any) -> str:
        try:
            s = getattr(doc, "structure", {}) or {}
            parts: list[str] = list(self._base_text(doc))

            # ── Company tokens ──────────────────────────────────────────────
            source = getattr(doc, "source", "") or ""
            co = _company_tokens(source)
            if co:
                parts.append(co)

            # ── Page number token ───────────────────────────────────────────
            page = getattr(doc, "page", None) or s.get("page_number")
            if page is not None:
                parts.append(f"page {page}")

            # ── Section hierarchy path ──────────────────────────────────────
            hierarchy: list[str] = s.get("section_hierarchy") or []
            if hierarchy:
                hier_text = " ".join(str(h) for h in hierarchy)
                parts.append(hier_text)
                # SEC Item number as a short searchable token (item_7, item_1a)
                for seg in hierarchy:
                    m = _SEC_ITEM_RE.match(str(seg))
                    if m:
                        num = m.group(1).lower().replace(" ", "")
                        # short aliases: item_7 → mda, item_8 → financial_statements …
                        parts.append(f"item_{num}")
                        if num == "7":
                            parts.extend(["management discussion analysis", "mda"])
                        elif num == "8":
                            parts.extend(["financial statements"])
                        elif num == "1a":
                            parts.extend(["risk factors"])

            # ── Table-chunk amplification ───────────────────────────────────
            chunk_type = (s.get("chunk_type") or getattr(doc, "subtype", "") or "").lower()
            if "table" in chunk_type:
                table_title = (s.get("table_title") or s.get("section_title") or "").strip()
                if table_title:
                    # Repeat title twice so BM25 TF weights table headings higher
                    parts.append(table_title)
                    parts.append(table_title)
                    # Add semantic table-type keyword bag
                    ttype_tokens = _table_type_tokens(table_title)
                    if ttype_tokens:
                        parts.append(ttype_tokens)

                # Fiscal year tokens — repeat each year so "2024" retrieval is strong
                fiscal_years: list[str] = s.get("fiscal_years") or s.get("column_headers") or []
                for fy in fiscal_years[:4]:
                    fy_str = str(fy)
                    parts.append(fy_str)
                    # Also add bare year number for "2024" queries
                    m_yr = _FY_RE.search(fy_str)
                    if m_yr:
                        parts.append(m_yr.group(1))

            # ── Finance entity amplification ────────────────────────────────
            fin_entities = s.get("finance_entities") or []
            if isinstance(fin_entities, list):
                parts.extend(str(x) for x in fin_entities[:5])
            elif isinstance(fin_entities, dict):
                for v in fin_entities.values():
                    if isinstance(v, list):
                        parts.extend(str(x) for x in v[:3])

            # ── Sub-chunk disambiguation ────────────────────────────────────
            sub_idx = s.get("sub_chunk_index")
            sub_total = s.get("total_sub_chunks")
            if sub_idx is not None and sub_total and int(sub_total) > 1:
                parts.append(f"part {int(sub_idx)+1} of {sub_total}")

            _BM25_INDEXED.inc()
            return " ".join(parts)
        except Exception as _exc:
            _BM25_INDEX_ERRORS.inc()
            logger.warning(
                event="bm25_index_text_failed",
                modality=self.modality,
                error=str(_exc),
            )
            return getattr(doc, "text", "") or ""

    def health_check(self, user_id=None) -> dict:
        base = super().health_check(user_id)
        base["class"] = self.__class__.__name__
        return base
