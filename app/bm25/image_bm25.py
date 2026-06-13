"""image_bm25.py — BM25 index for image/chart chunks."""
from __future__ import annotations

import re
from typing import Any, List

from app.bm25.base_bm25 import BaseBM25, _SCALE_RE, _SCALE_MAP


class ImageBM25(BaseBM25):
    """BM25 index for image chunks (charts, graphs, scanned pages, figures).

    Enrichment over base:
    - image_type prefix: "bar_chart", "line_chart", "table_image" etc.
    - Caption text (already in base via _base_text)
    - OCR text — often contains exact numbers not in caption
    - extracted_numbers with full scale expansion:
        "$4.3B" → index as "4.3 billion 4300 million 4300000000"
    - time_period token for date-scoped chart queries
    """

    modality = "jpg"

    _NUMBER_RE = re.compile(r'(\d[\d,.]*(?:\.\d+)?)\s*([bBmMkK])\b')

    def _build_indexed_text(self, doc: Any) -> str:
        s = getattr(doc, "structure", {}) or {}
        parts: List[str] = list(self._base_text(doc))

        # Image type prefix — queries like "bar chart revenue" should match charts
        image_type = (
            s.get("image_type")
            or s.get("subtype")
            or getattr(doc, "subtype", None)
            or ""
        ).strip().replace("_", " ")
        if image_type:
            parts.append(image_type)
            parts.append(image_type)  # ×2 — image type is the primary retrieval signal

        # OCR text — may contain numbers not captured in caption
        ocr_text = (s.get("ocr_text") or "").strip()
        if ocr_text:
            parts.append(ocr_text[:500])

        # Extracted numbers with cross-scale variants
        extracted_numbers: List[str] = s.get("extracted_numbers") or []
        for num_str in extracted_numbers[:20]:
            num_str = str(num_str).strip()
            parts.append(num_str)
            # Emit scale-expanded variants
            m = _SCALE_RE.search(num_str.lower())
            if m:
                scale_word = _SCALE_MAP.get(m.group(2).lower(), "")
                try:
                    val = float(m.group(1).replace(",", ""))
                    # Cross-scale: if B, also add M variant
                    _TO_MULT = {"billion": 1e9, "million": 1e6, "thousand": 1e3, "trillion": 1e12}
                    _TO_NAME = {v: k for k, v in _SCALE_MAP.items()}
                    abs_val = val * _TO_MULT.get(scale_word, 1.0)
                    for mult, name in _TO_MULT.items():
                        cv = abs_val / mult
                        if 0.001 <= cv <= 1e6:
                            parts.append(f"{cv:.3f}".rstrip("0").rstrip(".") + " " + name)
                except (ValueError, TypeError):
                    pass

        # Time period (e.g. "FY2023", "Q3 2024")
        time_period = (s.get("time_period") or "").strip()
        if time_period:
            parts.append(time_period)

        # Data series names (e.g. "Revenue", "Net Income")
        data_series: List[str] = s.get("data_series") or []
        for series in data_series[:6]:
            parts.append(str(series))

        return " ".join(parts)
