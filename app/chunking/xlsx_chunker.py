from __future__ import annotations

import io
import re
import time
from pathlib import Path

from prometheus_client import Counter

from app.chunking.base_chunker import BaseChunker
from app.chunking.finance_numbers import (
    deterministic_chunk_id,
    extract_finance_entities,
    protect,
    restore,
)
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger, modality_var

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
_HEADER_TAG_RE = re.compile(r"^\[Sheet:.*?\]$")
# xlsx_ingest.py already batches EXCEL_ROWS_PER_CHUNK (default 25) spreadsheet rows
# into a single RawExtract per "table_row". Grouping further here just recombines
# already-batched extracts, so _TARGET_ROWS counts ingest-level batches, not raw
# rows: 1 means "one final chunk per ingest batch" (~25 rows), which is the
# intended per-chunk granularity for row-level financial lookups (e.g. one
# country's premium row must not be diluted by 100+ unrelated countries sharing
# its embedding — accuracy phase 2026-07).
_TARGET_ROWS = 1
_OVERLAP_ROWS = 0


def _caption_bytes(raw_bytes: bytes) -> str:
    try:
        from PIL import Image

        from app.chunking.image_chunker import blip_caption

        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return blip_caption(img)
    except Exception as exc:
        logger.warning(event="xlsx_blip_failed", error=str(exc))
        return ""


def _explode_batch(blob: str) -> list[str]:
    """Split an ingest-level batch (`[Sheet: X, Rows N-M]` + one spreadsheet row
    per line, cells pipe-joined) back into individual row-lines. Without this,
    naively splitting the whole multi-line blob on "|" merges every row in the
    batch into one flat cell list, losing row boundaries entirely — e.g. one
    country's premium value becomes indistinguishable from the next country's
    (accuracy phase 2026-07)."""
    lines = [ln for ln in blob.split("\n") if ln.strip()]
    return [ln for ln in lines if not _HEADER_TAG_RE.match(ln.strip())]


def _rows_to_nl(rows: list[str], headers: list[str], unit_scale: str, sheet_name: str) -> str:
    """Serialize a row group to natural language for embedding."""
    lines = [f"Sheet: {sheet_name}" + (f" (in {unit_scale})" if unit_scale else "")]

    row_lines: list[str] = []
    for blob in rows:
        row_lines.extend(_explode_batch(blob))

    # xlsx_ingest.py repeats the sheet's header row onto every batch. Lift it
    # once (chunker-level `headers` state is populated too rarely to rely on —
    # it only ever fires when a whole 25-row batch happens to contain no
    # digits at all) and drop the duplicate header line from the row body.
    local_headers = list(headers)
    if not local_headers and row_lines and not re.search(r"\d", row_lines[0]):
        local_headers = [c.strip() for c in row_lines[0].split("|") if c.strip()]

    if local_headers:
        lines.append("Columns: " + " | ".join(local_headers))

    for row_line in row_lines:
        cells = [c.strip() for c in row_line.split("|") if c.strip()]
        if not cells:
            continue
        if local_headers and cells == local_headers:
            continue
        if local_headers and len(cells) == len(local_headers):
            parts = [f"{h}: {v}" for h, v in zip(local_headers, cells, strict=False)]
            lines.append(", ".join(parts))
        else:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _rows_to_markdown(rows: list[str], headers: list[str]) -> str:
    """Serialize a row group to a markdown table for display."""
    row_lines: list[str] = []
    for blob in rows:
        row_lines.extend(_explode_batch(blob))

    local_headers = list(headers)
    if not local_headers and row_lines and not re.search(r"\d", row_lines[0]):
        local_headers = [c.strip() for c in row_lines[0].split("|") if c.strip()]

    parts: list[str] = []
    if local_headers:
        parts.append("| " + " | ".join(local_headers) + " |")
        parts.append("|" + "|".join([" --- "] * len(local_headers)) + "|")
    for row_line in row_lines:
        cells = [c.strip() for c in row_line.split("|")]
        if local_headers and cells == local_headers:
            continue
        parts.append("| " + " | ".join(cells) + " |")
    return "\n".join(parts)


class XlsxChunker(BaseChunker):
    """Finance-grade chunker for Excel workbooks — row-group chunking with dual NL+markdown repr."""

    def chunk(
        self,
        extracts: list[RawExtract],
        meta: UniversalMetadata,
    ) -> list[IngestedDocument]:
        source = Path(meta.source_path).name or "unknown.xlsx"
        surface = "xlsx_chunker"
        modality_var.set("xlsx")
        _t0 = time.time()
        logger.info(event="chunking_start", modality="xlsx", source=source, extracts=len(extracts))
        if not extracts:
            logger.warning(event="no_extracts_received", modality="xlsx", source=source)
            return []
        try:
            docs: list[IngestedDocument] = []
            chunk_idx = [0]

            # State per sheet
            current_sheet: str | None = None
            sheet_index = [0]  # sequential sheet number (MD Phase 1.4)
            unit_scale: str = ""
            currency: str = "USD"
            column_headers: list[str] = []
            pending_rows: list[RawExtract] = []
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

            def _infer_sheet_type(name: str | None) -> str:
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
                pending_rows[:] = pending_rows[_TARGET_ROWS - _OVERLAP_ROWS :]

            def _emit_row_group(rows: list[RawExtract], headers: list[str]) -> None:
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
                # row_num is always set equal to row_start at ingestion (xlsx_ingest.py),
                # so using it for BOTH ends collapses every range to [start, start] —
                # e.g. [1, 1] instead of [1, 25]. Use row_start on the first extract
                # and row_end on the last (accuracy phase 2026-07).
                row_nums = [
                    rows[0].extra.get("row_start", rows[0].extra.get("row_num", 1)),
                    rows[-1].extra.get("row_end", rows[-1].extra.get("row_num", len(rows))),
                ]
                chunk_hash = deterministic_chunk_id(
                    source, f"sheet_{current_sheet}_r{row_nums[0]}_{chunk_idx[0]}", chunk_idx[0]
                )
                first_row_ref = rows[0].extra.get("cell_ref", "")
                last_row_ref = rows[-1].extra.get("cell_ref", "")
                table_region = (
                    f"{first_row_ref}:{last_row_ref}" if first_row_ref and last_row_ref else ""
                )
                is_hidden_group = any(r.extra.get("is_hidden", False) for r in rows)
                sem_group = rows[0].extra.get("semantic_group", "")
                display_fmt = rows[0].extra.get("display_format", "standard")
                _smeta = sheet_meta_map.get(current_sheet or "", {})
                structure = {
                    "chunk_hash_id": chunk_hash,
                    "source_file": source,
                    "chunk_index": chunk_idx[0],
                    "sheet_name": current_sheet,
                    "sheet_index": sheet_index[0],
                    "sheet_type": _smeta.get("sheet_type") or _infer_sheet_type(current_sheet),
                    "table_region": table_region,
                    "named_ranges_in_chunk": [],
                    "chunk_type": "table_row_group",
                    "row_range": row_nums,
                    "column_headers": headers,
                    "semantic_group": sem_group,
                    "unit_scale": unit_scale,
                    "currency": currency,
                    "display_format": display_fmt,
                    "is_hidden": is_hidden_group,
                    "has_formulas_resolved": True,
                    "markdown_repr": md_text,
                    "finance_entities": fin_entities,
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
            sheet_meta_map: dict[str, dict] = {}

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
                    _raw_nr: dict = ext.extra.get("named_ranges") or {}
                    if _raw_nr:
                        nr_keys = list(_raw_nr.keys())
                    elif ext.extra.get("is_stats_summary"):
                        nr_keys = []  # stats summary is not a named-range block
                    else:
                        nr_keys = [nr_text.split("=")[0].strip()] if "=" in nr_text else []
                    chunk_hash = deterministic_chunk_id(
                        source, f"named_range_{chunk_idx[0]}", chunk_idx[0]
                    )
                    structure = {
                        "chunk_hash_id": chunk_hash,
                        "source_file": source,
                        "chunk_index": chunk_idx[0],
                        "sheet_name": current_sheet,
                        "sheet_index": sheet_index[0],
                        "sheet_type": _infer_sheet_type(current_sheet),
                        "table_region": "",
                        "named_ranges_in_chunk": nr_keys[:20],
                        "chunk_type": "named_ranges",
                        "unit_scale": unit_scale,
                        "has_formulas_resolved": True,
                        "finance_entities": fin_entities,
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
                    chunk_hash = deterministic_chunk_id(
                        source, f"chart_{chunk_idx[0]}", chunk_idx[0]
                    )
                    structure = {
                        "chunk_hash_id": chunk_hash,
                        "source_file": source,
                        "chunk_index": chunk_idx[0],
                        "sheet_name": current_sheet,
                        "sheet_index": sheet_index[0],
                        "sheet_type": _infer_sheet_type(current_sheet),
                        "table_region": "",
                        "named_ranges_in_chunk": [],
                        "chunk_type": "chart_caption",
                        "caption": cap,
                        "finance_entities": fin_entities,
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
