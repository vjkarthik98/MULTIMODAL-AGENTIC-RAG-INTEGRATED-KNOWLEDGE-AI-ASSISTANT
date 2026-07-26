"""pdf_embedder.py — Finance-grade embedder for PDF chunks."""

from __future__ import annotations

import re
from typing import Any

from prometheus_client import Counter

from app.core.config import settings
from app.embeddings.base_embedder import BaseEmbedder
from app.utils.logger import get_logger

logger = get_logger(__name__)

_EMBED_BUILT = Counter(
    "magik_pdf_embed_text_built_total",
    "Embed texts successfully built for pdf",
)
_EMBED_ERRORS = Counter(
    "magik_pdf_embed_text_errors_total",
    "Errors building embed text for pdf",
)

# Known filename → company name map for common filings.
# Keeps embedding context short and precise without loading a full NER model here.
_SOURCE_TO_COMPANY: dict[str, str] = {
    "apple": "Apple Inc.",
    "microsoft": "Microsoft Corporation",
    "google": "Alphabet Inc.",
    "alphabet": "Alphabet Inc.",
    "amazon": "Amazon.com Inc.",
    "meta": "Meta Platforms Inc.",
    "nvidia": "NVIDIA Corporation",
    "tesla": "Tesla Inc.",
    "berkshire": "Berkshire Hathaway Inc.",
    "jpmorgan": "JPMorgan Chase & Co.",
}

# Canonical table-type labels for the embedding prefix, keyed by table_title keyword
_TABLE_TYPE_LABELS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"statement.{0,10}operation|income statement", re.I), "Income Statement"),
    (re.compile(r"balance sheet|financial position", re.I), "Balance Sheet"),
    (re.compile(r"cash flow", re.I), "Cash Flow Statement"),
    (re.compile(r"comprehensive income", re.I), "Comprehensive Income"),
    (re.compile(r"shareholders?.{0,6}equity", re.I), "Shareholders Equity"),
    (re.compile(r"net sales|revenue.{0,10}product", re.I), "Net Sales by Product"),
    (re.compile(r"gross margin", re.I), "Gross Margin"),
    (re.compile(r"income tax|tax provision", re.I), "Income Tax"),
    (re.compile(r"capital return|repurchase|dividend", re.I), "Capital Return"),
    (re.compile(r"segment", re.I), "Segment Results"),
]


def _company_from_source(source: str) -> str:
    """Derive company name from source filename, e.g. 'apple_10k.pdf' → 'Apple Inc.'."""
    stem = re.sub(r'\.[^.]+$', '', source or '').lower()
    for key, name in _SOURCE_TO_COMPANY.items():
        if key in stem:
            return name
    # Fallback: capitalise first token before '_' or '-'
    first = re.split(r'[_\-\s]', stem)[0]
    return first.capitalize() if first else ""


def _canonical_table_label(table_title: str) -> str:
    """Map free-form table_title to a canonical financial statement name."""
    for pat, label in _TABLE_TYPE_LABELS:
        if pat.search(table_title):
            return label
    return table_title.strip()


class PdfEmbedder(BaseEmbedder):
    """Embedder for PDF chunks (10-Ks, 10-Qs, research reports).

    Embedding-text enrichment strategy:
      Prose: [Company] [Page N] [Section: hierarchy_path] text
      Table: [Company] [FiscalYears] [TableType: label] text
      Footnotes: text only (contextual, low weight)

    Including company name and fiscal years in every embed string ensures
    queries like "Apple revenue FY2024" retrieve both prose and table chunks.
    """

    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        try:
            s = getattr(doc, "structure", {}) or {}
            page = getattr(doc, "page", None) or s.get("page_number")
            source = getattr(doc, "source", "") or ""
            chunk_type = (s.get("chunk_type") or getattr(doc, "subtype", "") or "").lower()

            company = _company_from_source(source)
            parts: list[str] = []

            # Company name first — anchors every chunk to the issuer
            if company:
                parts.append(f"[{company}]")

            if "table" in chunk_type:
                # ── Table chunks ────────────────────────────────────────────
                # Fiscal years replace page number for tables (more useful for retrieval)
                fiscal_years = s.get("fiscal_years") or s.get("column_headers") or []
                if fiscal_years:
                    fy_str = " ".join(str(y) for y in fiscal_years[:4])
                    parts.append(f"[{fy_str}]")
                elif page is not None:
                    parts.append(f"[Page {page}]")

                # Canonical table type label for semantic anchoring
                raw_title = (s.get("table_title") or s.get("section_title") or "").strip()
                if raw_title:
                    label = _canonical_table_label(raw_title)
                    parts.append(f"[{label}]")

                prefix = (" ".join(parts) + " ") if parts else ""
                result = (prefix + cleaned_text)[: settings.MAX_PROMPT_CHARS]
                _EMBED_BUILT.inc()
                return result

            # ── Prose / heading / OCR chunks ────────────────────────────────
            if page is not None:
                parts.append(f"[Page {page}]")

            # Full section hierarchy path: "PART II > Item 7. MD&A > Revenue"
            hierarchy = s.get("section_hierarchy") or []
            if hierarchy:
                hier_str = " > ".join(str(h) for h in hierarchy if h)
                if hier_str:
                    parts.append(f"[Section: {hier_str}]")
            else:
                section = (s.get("section_title") or "").strip()
                if section and len(section) <= 150:
                    parts.append(f"[Section: {section}]")

            # Sub-chunk position within a split page — prevents adjacent chunks
            # from collapsing to the same vector.
            sub_idx = s.get("sub_chunk_index")
            sub_total = s.get("total_sub_chunks")
            if sub_idx is not None and sub_total and int(sub_total) > 1:
                parts.append(f"[Part {int(sub_idx)+1}/{int(sub_total)}]")

            prefix = (" ".join(parts) + " ") if parts else ""
            result = (prefix + cleaned_text)[: settings.MAX_PROMPT_CHARS]
            logger.debug(event="embed_text_built", modality="pdf", chars=len(result))
            _EMBED_BUILT.inc()
            return result
        except Exception as _exc:
            _EMBED_ERRORS.inc()
            logger.error(event="embed_text_build_failed", modality="pdf", error=str(_exc))
            return cleaned_text

    def health_check(self) -> dict:
        return {
            "modality": "pdf",
            "status": "ok",
            "class": self.__class__.__name__,
        }
