"""pdf_bm25.py — BM25 index for PDF document chunks."""
from __future__ import annotations

from typing import Any, List

from app.bm25.base_bm25 import BaseBM25


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

        # Finance entities amplification
        fin_entities = s.get("finance_entities") or {}
        if isinstance(fin_entities, dict):
            for v in fin_entities.values():
                if isinstance(v, list):
                    parts.extend(str(x) for x in v[:3])

        return " ".join(parts)
