"""pdf_bm25.py — BM25 index for PDF document chunks."""
from __future__ import annotations

from typing import Any, List

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


class PdfBM25(BaseBM25):
    """BM25 index for PDF documents (10-K, 10-Q, annual reports, research).

    Enrichment over base:
    - page_N token so "page 12" queries find the right chunk
    - table_title ×2 for table chunks — dense retrieval struggles with tables
    - section_hierarchy path as tokens (e.g. "management discussion liquidity capital")
    - footnote markers stripped out before indexing (footnote text linked to parent chunk)
    """

    modality = "pdf"

    def _build_indexed_text(self, doc: Any) -> str:
        try:
            s = getattr(doc, "structure", {}) or {}
            parts: List[str] = list(self._base_text(doc))

            # Page number token
            page = getattr(doc, "page", None) or s.get("page_number")
            if page is not None:
                parts.append(f"page {page}")

            # Table title amplification
            chunk_type = (s.get("chunk_type") or getattr(doc, "subtype", "") or "").lower()
            if "table" in chunk_type:
                table_title = (s.get("table_title") or s.get("section_title") or "").strip()
                if table_title:
                    parts.append(table_title)
                    parts.append(table_title)

            # Section hierarchy path
            hierarchy: List[str] = s.get("section_hierarchy") or []
            if hierarchy:
                parts.append(" ".join(str(h) for h in hierarchy))

            # Sub-chunk disambiguation
            sub_idx   = s.get("sub_chunk_index")
            sub_total = s.get("total_sub_chunks")
            if sub_idx is not None and sub_total and int(sub_total) > 1:
                parts.append(f"part {int(sub_idx)+1} of {sub_total}")

            # Finance entities amplification (extract_finance_entities returns List[str])
            fin_entities = s.get("finance_entities") or []
            if isinstance(fin_entities, list):
                parts.extend(str(x) for x in fin_entities[:5])
            elif isinstance(fin_entities, dict):
                for v in fin_entities.values():
                    if isinstance(v, list):
                        parts.extend(str(x) for x in v[:3])

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
