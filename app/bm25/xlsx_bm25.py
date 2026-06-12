"""xlsx_bm25.py — BM25 index for Excel/spreadsheet table chunks."""
from __future__ import annotations

from typing import Any, List

from app.bm25.base_bm25 import BaseBM25


class XlsxBM25(BaseBM25):
    """BM25 index for Excel documents (financial models, income statements, P&L).

    Enrichment over base:
    - Sheet name prefix: "sheet income statement" for sheet-level queries
    - Unit scale token: "millions" "billions" so "$4.3 million" vs "$4.3 billion"
      queries land in the right sheet
    - Named range keys indexed for assumption lookups
    - Column headers as tokens (repeat ×2) — table rows lack headers in their text
    - Row range token: "row 10 to 16" for positional references
    - All extracted numbers passed through _expand_scale_variants for cross-scale matching
    """

    modality = "excel"

    def _build_indexed_text(self, doc: Any) -> str:
        s = getattr(doc, "structure", {}) or {}
        parts: List[str] = list(self._base_text(doc))

        # Sheet name prefix
        sheet = (
            getattr(doc, "sheet_name", None)
            or s.get("sheet_name")
            or s.get("sheet")
            or s.get("section_title")
            or ""
        ).strip()
        if sheet:
            parts.append(f"sheet {sheet}")
            parts.append(sheet)  # bare name too

        # Unit scale token — crucial for "$4.3B" vs "$4.3M" disambiguation
        unit_scale = (s.get("unit_scale") or s.get("currency") or "").strip().lower()
        if unit_scale:
            parts.append(f"unit {unit_scale}")
            parts.append(unit_scale)

        # Column headers amplified
        col_headers: List[str] = s.get("column_headers") or []
        if col_headers:
            header_text = " ".join(str(h) for h in col_headers[:8])
            parts.append(header_text)
            parts.append(header_text)  # ×2

        # Named ranges (assumption variables)
        named_ranges: dict = s.get("named_ranges") or {}
        for name in list(named_ranges.keys())[:10]:
            parts.append(str(name).replace("_", " "))

        # Row range token
        row_start = getattr(doc, "row_start", None) or s.get("row_start")
        row_end   = getattr(doc, "row_end",   None) or s.get("row_end")
        if row_start is not None and row_end is not None:
            parts.append(f"row {row_start} to {row_end}")

        # Semantic group label (e.g. "revenue breakdown", "cost structure")
        group = (s.get("semantic_group") or "").strip()
        if group:
            parts.append(group)

        return " ".join(parts)
