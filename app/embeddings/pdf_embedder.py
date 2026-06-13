"""pdf_embedder.py — Finance-grade embedder for PDF chunks."""
from __future__ import annotations

from typing import Any

from app.embeddings.base_embedder import BaseEmbedder
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PdfEmbedder(BaseEmbedder):
    """Embedder for PDF chunks (10-Ks, 10-Qs, research reports).

    Enrichment:
      [Page {n}] [Section: {hierarchy_path}] {text}
      Table chunks: "Table: {section_title} | {text}"
      Footnotes: text only (low weight, no extra prefix)
    """

    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        s    = getattr(doc, "structure", {}) or {}
        page = getattr(doc, "page", None) or s.get("page_number")

        parts = []

        if page is not None:
            parts.append(f"[Page {page}]")

        # Full section hierarchy path: "Results of Operations > Revenue"
        hierarchy = s.get("section_hierarchy") or []
        if hierarchy:
            hier_str = " > ".join(str(h) for h in hierarchy if h)
            if hier_str:
                parts.append(f"[Section: {hier_str}]")
        else:
            section = (s.get("section_title") or "").strip()
            if section and len(section) <= 120:
                parts.append(f"[Section: {section}]")

        # Sub-chunk position — prevents adjacent chunks from collapsing in vector space
        sub_idx   = s.get("sub_chunk_index")
        sub_total = s.get("total_sub_chunks")
        if sub_idx is not None and sub_total and int(sub_total) > 1:
            parts.append(f"[Part {int(sub_idx)+1}/{int(sub_total)}]")

        prefix = (" ".join(parts) + " ") if parts else ""

        chunk_type = (s.get("chunk_type") or getattr(doc, "subtype", "") or "").lower()

        # Table chunks: repeat section title so column-header queries retrieve them
        if chunk_type in ("table", "table_row"):
            table_title = (s.get("table_title") or s.get("section_title") or "").strip()
            if table_title:
                return f"Table: {table_title} | {prefix}{cleaned_text}"[:settings.MAX_PROMPT_CHARS]

        result = (prefix + cleaned_text)[:settings.MAX_PROMPT_CHARS]
        logger.debug(event="embed_text_built", modality="pdf", chars=len(result))
        return result
