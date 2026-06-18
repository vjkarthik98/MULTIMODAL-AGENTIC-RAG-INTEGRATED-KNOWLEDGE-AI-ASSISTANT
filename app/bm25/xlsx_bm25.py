"""xlsx_bm25.py — BM25 index for Excel/spreadsheet table chunks."""
from __future__ import annotations

import re
from typing import Any, List

from prometheus_client import Counter

from app.bm25.base_bm25 import BaseBM25
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SCALE_NUM_RE = re.compile(r'\b(\d[\d,]*\.?\d*)\s*(M|B|K|bn|mn|bln|mln)\b', re.IGNORECASE)
_SCALE_MAP = {"m": ("million", 1e6), "mn": ("million", 1e6), "mln": ("million", 1e6),
              "b": ("billion", 1e9), "bn": ("billion", 1e9), "bln": ("billion", 1e9),
              "k": ("thousand", 1e3)}


def _expand_scale(text: str) -> str:
    """Expand '4.3B' to '4.3 billion 4300 million' so cross-scale queries match."""
    extra: List[str] = []
    for m in _SCALE_NUM_RE.finditer(text):
        raw_num = m.group(1).replace(",", "")
        suffix  = m.group(2).lower()
        try:
            val   = float(raw_num)
            label, factor = _SCALE_MAP.get(suffix, ("", 1))
            abs_val = val * factor
            extra.append(f"{val} {label}")
            # cross-scale: billion ↔ million, thousand ↔ million
            if suffix in ("b", "bn", "bln"):
                extra.append(f"{abs_val / 1e6:.0f} million")
            elif suffix in ("m", "mn", "mln"):
                extra.append(f"{abs_val / 1e9:.3f} billion")
            elif suffix in ("k",):
                extra.append(f"{abs_val / 1e6:.3f} million")
        except (ValueError, TypeError):
            pass
    if extra:
        return text + " " + " ".join(extra)
    return text

_BM25_INDEXED = Counter(
    "magik_xlsx_bm25_indexed_total",
    "Documents indexed in xlsx BM25",
)
_BM25_INDEX_ERRORS = Counter(
    "magik_xlsx_bm25_index_errors_total",
    "Errors in xlsx BM25 _build_indexed_text",
)


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

    modality = "xlsx"

    def _build_indexed_text(self, doc: Any) -> str:
        try:
            s = getattr(doc, "structure", {}) or {}
            base_text = " ".join(self._base_text(doc))
            expanded_base = _expand_scale(base_text)
            parts: List[str] = [expanded_base]

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

            # Named range keys — key fix: chunker stores "named_ranges_in_chunk" (List)
            named_range_keys: List[str] = s.get("named_ranges_in_chunk") or []
            for name in named_range_keys[:10]:
                readable = str(name).replace("_", " ")
                parts.append(readable)
                parts.append(readable)  # ×2 — assumption names are high-value retrieval tokens

            # Finance entities amplification
            fin_entities: List[str] = s.get("finance_entities") or []
            if isinstance(fin_entities, list):
                parts.extend(str(x) for x in fin_entities[:5])

            # Row range token
            row_range = s.get("row_range")
            if row_range and len(row_range) >= 2:
                parts.append(f"row {row_range[0]} to {row_range[1]}")

            # Semantic group label (e.g. "Revenue Lines", "Cost Lines")
            group = (s.get("semantic_group") or "").strip()
            if group:
                parts.append(group)

            # Sheet type (e.g. "income_statement", "assumptions")
            sheet_type = (s.get("sheet_type") or "").strip()
            if sheet_type:
                parts.append(sheet_type.replace("_", " "))

            # Named-range assumption text: "assumption: wacc 8.5 percent"
            if s.get("chunk_type") == "named_ranges":
                parts.append("assumption named range")

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
