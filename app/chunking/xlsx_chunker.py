from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.chunking.base_chunker import BaseChunker
from app.chunking.finance_numbers import (
    approx_tokens,
    deterministic_chunk_id,
    extract_finance_entities,
    protect,
    restore,
)
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger, modality_var

import time
from prometheus_client import Counter, Histogram

logger = get_logger(__name__)

_CHUNKS_TOTAL = Counter(
    "magik_xlsx_chunks_total",
    "Total chunks produced by xlsx chunker",
)
_CHUNK_ERRORS = Counter(
    "magik_xlsx_chunk_errors_total",
    "Total errors in xlsx chunker",
)

_UNIT_SCALE_RE = re.compile(r"\b(billions?|millions?|thousands?|units?)\b", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"\b(USD|GBP|EUR|JPY|INR|CAD|AUD)\b")
_TARGET_ROWS = 6       # semantic row-group size
_OVERLAP_ROWS = 2


def _caption_bytes(raw_bytes: bytes) -> str:
    try:
        from PIL import Image
        from app.chunking.image_chunker import blip2_caption
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return blip2_caption(img)
    except Exception as exc:
        logger.warning(event="xlsx_blip2_failed", error=str(exc))
        return ""


def _rows_to_nl(rows: List[str], headers: List[str], unit_scale: str, sheet_name: str) -> str:
    """Serialize a row group to natural language for embedding."""
    lines = [f"Sheet: {sheet_name}" + (f" (in {unit_scale})" if unit_scale else "")]
    if headers:
        lines.append("Columns: " + " | ".join(headers))
    for row in rows:
        cells = [c.strip() for c in row.split("|") if c.strip()]
        if cells:
            if headers and len(cells) == len(headers):
                parts = [f"{h}: {v}" for h, v in zip(headers, cells)]
                lines.append(", ".join(parts))
            else:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def _rows_to_markdown(rows: List[str], headers: List[str]) -> str:
    """Serialize a row group to a markdown table for display."""
    parts: List[str] = []
    if headers:
        parts.append("| " + " | ".join(headers) + " |")
        parts.append("|" + "|".join([" --- "] * len(headers)) + "|")
    for row in rows:
        cells = [c.strip() for c in row.split("|")]
        parts.append("| " + " | ".join(cells) + " |")
    return "\n".join(parts)


class XlsxChunker(BaseChunker):
    """Finance-grade chunker for Excel workbooks — row-group chunking with dual NL+markdown repr."""

    def chunk(
        self,
        extracts: List[RawExtract],
        meta: UniversalMetadata,
    ) -> List[IngestedDocument]:
        source = Path(meta.source_path).name or "unknown.xlsx"
        surface = "xlsx_chunker"
        modality_var.set("xlsx")
        _t0 = time.time()
        logger.info(event="chunking_start", modality="xlsx", source=source, extracts=len(extracts))
        if not extracts:
            logger.warning(event="no_extracts_received", modality="xlsx", source=source)
            return []
        try:
            docs: List[IngestedDocument] = []
            chunk_idx = [0]

            # State per sheet
            current_sheet: Optional[str] = None
            sheet_index = [0]        # sequential sheet number (MD Phase 1.4)
            unit_scale: str = ""
            currency: str = "USD"
            column_headers: List[str] = []
            pending_rows: List[RawExtract] = []
            seen_hashes: set = set()

            _SHEET_TYPE_MAP = {
                "income": "income_statement",
                "p&l": "income_statement",
                "profit": "income_statement",
                "balance": "balance_sheet",
                "cash": "cash_flow",
                "capex": "capex",
                "assump": "assumptions",
                "input": "assumptions",
                "summary": "summary",
                "model": "model",
                "data": "data",
            }

            def _infer_sheet_type(name: Optional[str]) -> str:
                if not name:
                    return "unknown"
                nl = name.lower()
                for key, stype in _SHEET_TYPE_MAP.items():
                    if key in nl:
                        return stype
                return "data"

            def flush_rows(force: bool = False) -> None:
                if not pending_rows:
                    return
                if not force and len(pending_rows) < _TARGET_ROWS:
                    return
                _emit_row_group(pending_rows[:], column_headers[:])
                # Slide window with overlap
                pending_rows[:] = pending_rows[_TARGET_ROWS - _OVERLAP_ROWS:]

            def _emit_row_group(rows: List[RawExtract], headers: List[str]) -> None:
                row_texts = [r.text for r in rows]
                nl_text = _rows_to_nl(row_texts, headers, unit_scale, current_sheet or "")
                md_text = _rows_to_markdown(row_texts, headers)
                _protected, _mapping = protect(nl_text)
                nl_text = restore(_protected, _mapping)
                h = hash(nl_text)
                if h in seen_hashes:
                    return
                seen_hashes.add(h)

                fin_entities = extract_finance_entities(nl_text)
                row_nums = [
                    rows[0].extra.get("row_num", rows[0].extra.get("row_start", 1)),
                    rows[-1].extra.get("row_num", rows[-1].extra.get("row_end", len(rows))),
                ]
                chunk_hash = deterministic_chunk_id(
                    source, f"sheet_{current_sheet}_r{row_nums[0]}_{chunk_idx[0]}", chunk_idx[0]
                )
                first_row_ref = rows[0].extra.get("cell_ref", "")
                last_row_ref  = rows[-1].extra.get("cell_ref", "")
                table_region  = f"{first_row_ref}:{last_row_ref}" if first_row_ref and last_row_ref else ""
                is_hidden_group = any(r.extra.get("is_hidden", False) for r in rows)
                sem_group = rows[0].extra.get("semantic_group", "")
                display_fmt = rows[0].extra.get("display_format", "standard")
                _smeta = sheet_meta_map.get(current_sheet or "", {})
                structure = {
                    "chunk_hash_id":         chunk_hash,
                    "source_file":           source,
                    "chunk_index":           chunk_idx[0],
                    "sheet_name":            current_sheet,
                    "sheet_index":           sheet_index[0],
                    "sheet_type":            _smeta.get("sheet_type") or _infer_sheet_type(current_sheet),
                    "table_region":          table_region,
                    "named_ranges_in_chunk": [],
                    "chunk_type":            "table_row_group",
                    "row_range":             row_nums,
                    "column_headers":        headers,
                    "semantic_group":        sem_group,
                    "unit_scale":            unit_scale,
                    "currency":              currency,
                    "display_format":        display_fmt,
                    "is_hidden":             is_hidden_group,
                    "has_formulas_resolved": True,
                    "markdown_repr":         md_text,
                    "finance_entities":      fin_entities,
                }
                doc = self._make_doc(
                    text=nl_text,
                    modality="xlsx",
                    subtype="table_row_group",
                    source=source,
                    page=None,
                    chunk_idx=chunk_idx[0],
                    structure=structure,
                    meta=meta,
                    surface=surface,
                )
                if doc:
                    docs.append(doc)
                    chunk_idx[0] += 1

            # Track sheet-level metadata from sheet_metadata extracts
            sheet_meta_map: Dict[str, Dict] = {}

            for ext in extracts:
                etype = ext.extract_type

                # Collect sheet metadata (emitted before row data)
                if etype == "sheet_metadata":
                    sn = ext.sheet or ""
                    sheet_meta_map[sn] = ext.extra or {}
                    continue

                # Sheet transition — flush what's pending.
                if ext.sheet and ext.sheet != current_sheet:
                    flush_rows(force=True)
                    current_sheet = ext.sheet
                    # Prefer sheet_index from metadata; fall back to sequential counter
                    _smeta = sheet_meta_map.get(current_sheet, {})
                    sheet_index[0] = _smeta.get("sheet_index", sheet_index[0] + 1)
                    unit_scale = _smeta.get("unit_scale", "")
                    currency = "USD"
                    column_headers = []
                    pending_rows = []

                if etype == "unit_header":
                    text = (ext.text or "").strip()
                    m = _UNIT_SCALE_RE.search(text)
                    if m:
                        unit_scale = m.group(1).lower()
                    mc = _CURRENCY_RE.search(text)
                    if mc:
                        currency = mc.group(1)
                    continue

                if etype == "named_range":
                    flush_rows(force=True)
                    nr_text = (ext.text or "").strip()
                    if not nr_text:
                        continue
                    fin_entities = extract_finance_entities(nr_text)
                    # Named-range keys from the raw dict (workbook-level)
                    _raw_nr: Dict = ext.extra.get("named_ranges") or {}
                    if _raw_nr:
                        nr_keys = list(_raw_nr.keys())
                    elif ext.extra.get("is_stats_summary"):
                        nr_keys = []  # stats summary is not a named-range block
                    else:
                        nr_keys = [nr_text.split("=")[0].strip()] if "=" in nr_text else []
                    chunk_hash = deterministic_chunk_id(source, f"named_range_{chunk_idx[0]}", chunk_idx[0])
                    structure = {
                        "chunk_hash_id":         chunk_hash,
                        "source_file":           source,
                        "chunk_index":           chunk_idx[0],
                        "sheet_name":            current_sheet,
                        "sheet_index":           sheet_index[0],
                        "sheet_type":            _infer_sheet_type(current_sheet),
                        "table_region":          "",
                        "named_ranges_in_chunk": nr_keys[:20],
                        "chunk_type":            "named_ranges",
                        "unit_scale":            unit_scale,
                        "has_formulas_resolved": True,
                        "finance_entities":      fin_entities,
                    }
                    doc = self._make_doc(
                        text=nr_text,
                        modality="xlsx",
                        subtype="assumptions",
                        source=source,
                        page=None,
                        chunk_idx=chunk_idx[0],
                        structure=structure,
                        meta=meta,
                        surface=surface,
                    )
                    if doc:
                        docs.append(doc)
                        chunk_idx[0] += 1
                    continue

                if etype in ("table_row", "table_row_hidden"):
                    row_text = (ext.text or "").strip()
                    if not row_text:
                        continue
                    # Detect header row (first row, no numbers).
                    if not pending_rows and not column_headers:
                        if not re.search(r"\d", row_text):
                            column_headers = [c.strip() for c in row_text.split("|") if c.strip()]
                            continue
                    # Check if row represents a section boundary (bold row / empty).
                    is_section_boundary = ext.extra.get("is_section_boundary", False)
                    if is_section_boundary and pending_rows:
                        flush_rows(force=True)

                    ext.extra["is_hidden"] = etype == "table_row_hidden"
                    pending_rows.append(ext)

                    if len(pending_rows) >= _TARGET_ROWS:
                        flush_rows()
                    continue

                if etype == "chart_image":
                    flush_rows(force=True)
                    raw = ext.raw_bytes or b""
                    cap = _caption_bytes(raw) if raw else (ext.text or "")
                    if not cap:
                        continue
                    fin_entities = extract_finance_entities(cap)
                    chunk_hash = deterministic_chunk_id(source, f"chart_{chunk_idx[0]}", chunk_idx[0])
                    structure = {
                        "chunk_hash_id":         chunk_hash,
                        "source_file":           source,
                        "chunk_index":           chunk_idx[0],
                        "sheet_name":            current_sheet,
                        "sheet_index":           sheet_index[0],
                        "sheet_type":            _infer_sheet_type(current_sheet),
                        "table_region":          "",
                        "named_ranges_in_chunk": [],
                        "chunk_type":            "chart_caption",
                        "caption":               cap,
                        "finance_entities":      fin_entities,
                    }
                    doc = self._make_doc(
                        text=cap,
                        modality="xlsx",
                        subtype="chart_caption",
                        source=source,
                        page=None,
                        chunk_idx=chunk_idx[0],
                        structure=structure,
                        meta=meta,
                        surface=surface,
                    )
                    if doc:
                        docs.append(doc)
                        chunk_idx[0] += 1
                    continue

            flush_rows(force=True)

            logger.info(event="xlsx_chunking_done", source=source, chunks=len(docs))
            _CHUNKS_TOTAL.inc(len(docs))
            return docs
        except Exception as _exc:
            _CHUNK_ERRORS.inc()
            logger.error(event="chunking_failed", modality="xlsx", source=source, error=str(_exc))
            raise

    def health_check(self) -> dict:
        return {
            "modality": "xlsx",
            "status": "ok",
            "class": self.__class__.__name__,
        }
