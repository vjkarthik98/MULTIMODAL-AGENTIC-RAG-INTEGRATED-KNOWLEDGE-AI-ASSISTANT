"""docx_bm25.py — BM25 index for Word document chunks."""
from __future__ import annotations

from typing import Any, Dict, List

from app.bm25.base_bm25 import BaseBM25


class DocxBM25(BaseBM25):
    """BM25 index for Word/DOCX documents (contracts, board minutes, legal filings).

    Enrichment over base:
    - Full heading hierarchy as tokens
    - Defined-term keys indexed (e.g. "adjusted ebitda means ...") so exact term
      queries resolve to the right definition clause
    - Clause number tokens (4.3.b.ii → "section 4 3 b ii") for legal reference queries
    - Heading level token for heading-weight boosting
    """

    modality = "word"

    def _build_indexed_text(self, doc: Any) -> str:
        s = getattr(doc, "structure", {}) or {}
        parts: List[str] = list(self._base_text(doc))

        # Full heading hierarchy
        hierarchy: List[str] = s.get("heading_hierarchy") or []
        if hierarchy:
            parts.append(" ".join(str(h) for h in hierarchy))

        # Heading level token ("heading level 2" helps weight heading chunks)
        heading_level = s.get("heading_level") or getattr(doc, "heading_level", None)
        if heading_level is not None:
            parts.append(f"heading level {heading_level}")

        # Defined terms: "adjusted ebitda" → indexed as defined term for retrieval
        defined_terms: Dict[str, str] = s.get("defined_terms") or {}
        for term in list(defined_terms.keys())[:10]:
            parts.append(str(term))
            parts.append(str(term))  # amplify ×2 — defined terms are retrieval anchors

        # Clause number indexing: "4.3.b.ii" → "section 4 3 b ii"
        section_number = (s.get("section_number") or "").strip()
        if section_number:
            clause_tokens = "section " + section_number.replace(".", " ").replace("(", " ").replace(")", " ")
            parts.append(clause_tokens)

        # Table heading if table chunk
        chunk_type = (s.get("chunk_type") or "").lower()
        if "table" in chunk_type:
            table_title = (s.get("table_title") or s.get("section_title") or "").strip()
            if table_title:
                parts.append(table_title)
                parts.append(table_title)

        return " ".join(parts)
