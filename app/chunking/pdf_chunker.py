from __future__ import annotations

import io
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.chunking.base_chunker import BaseChunker
from app.chunking.finance_numbers import (
    deterministic_chunk_id,
    extract_finance_entities,
)
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger, modality_var

# SEC 10-K heading patterns — duplicated here to keep per-modality file
# boundaries clean (chunker must not import from pdf_ingest).
_SEC_PART_RE = re.compile(r'^PART\s+([IVX]+)\s*$', re.IGNORECASE)
_SEC_ITEM_RE = re.compile(r'^Item\s+(\d+[A-Z]?)\.?\s*(.*)', re.IGNORECASE)


def _classify_sec_heading(text: str) -> str | None:
    """Return 'part', 'item', or None for the given heading text."""
    t = text.strip()
    if _SEC_PART_RE.match(t):
        return "part"
    if _SEC_ITEM_RE.match(t):
        return "item"
    return None


import time

from prometheus_client import Counter

logger = get_logger(__name__)

_CHUNKS_TOTAL = Counter(
    "magik_pdf_chunks_total",
    "Total chunks produced by pdf chunker",
)
_CHUNK_ERRORS = Counter(
    "magik_pdf_chunk_errors_total",
    "Total errors in pdf chunker",
)

# Each table_row / table_summary RawExtract represents one complete table;
# no row-batching is needed — we emit exactly one chunk per extract.

# BLIP semaphore limits concurrent caption calls to stay within A10G 24 GB budget.
_BLIP_SEMAPHORE = threading.Semaphore(2)


def _ocr_bytes(raw_bytes: bytes) -> str:
    try:
        from PIL import Image

        from app.chunking.image_chunker import ocr as _ocr

        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return _ocr(img)
    except Exception as exc:
        logger.warning(event="pdf_trocr_failed", error=str(exc))
        return ""


def _caption_bytes(raw_bytes: bytes) -> str:
    try:
        from PIL import Image

        from app.chunking.image_chunker import blip_caption

        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return blip_caption(img)
    except Exception as exc:
        logger.warning(event="pdf_blip_failed", error=str(exc))
        return ""


def _make_table_chunk(
    ext: RawExtract,
    chunk_type: str,
    subtype: str,
    section_title: str | None,
    section_hierarchy: list[str],
    source: str,
    meta: UniversalMetadata,
    chunker: PdfChunker,
    chunk_idx_ref: list[int],
    char_offset_ref: list[int],
    surface: str,
) -> IngestedDocument | None:
    """Emit exactly one IngestedDocument from a table_row or table_summary RawExtract."""
    extra = ext.extra or {}
    md_text = extra.get("markdown", ext.text or "")
    nl_text = ext.text or md_text
    if not nl_text.strip():
        return None

    _t_title = extra.get("table_title", "")
    _fiscal_yrs = extra.get("fiscal_years", [])
    _sec = (extra.get("section_title") or section_title or "").strip()
    _bad_sec = any(kw in _sec for kw in ("Exhibit", "Trading Policy", "Appendix"))

    # Prefix: "<TableTitle> (Page N)" so LLM has unambiguous context without
    # needing to read the garbled pdfplumber year-header row.
    _label = _t_title or "Financial Table"
    _page_str = f"Page {ext.page}" if ext.page else ""
    _loc = ", ".join(p for p in [_page_str] if p)
    _prefix = f"{_label} ({_loc})" if _loc else _label

    # For markdown table chunks prefix once; summary chunks already self-describe.
    chunk_text = nl_text if chunk_type == "financial_table_summary" else f"{_prefix}\n\n{nl_text}"

    fin_entities = extract_finance_entities(chunk_text)
    chunk_hash = deterministic_chunk_id(
        source, f"p{ext.page or 0}_{chunk_type}_{chunk_idx_ref[0]}", chunk_idx_ref[0]
    )

    structure = {
        "chunk_hash_id": chunk_hash,
        "source_file": source,
        "chunk_index": chunk_idx_ref[0],
        "page_number": ext.page,
        "page_range": [ext.page, ext.page] if ext.page else None,
        "chunk_type": chunk_type,
        "section_title": _sec if not _bad_sec else None,
        "section_hierarchy": section_hierarchy[:],
        "table_title": _t_title or None,
        "column_headers": _fiscal_yrs,
        "fiscal_years": _fiscal_yrs,
        "row_range": None,
        "is_ocr": False,
        "footnotes": [],
        "footnote_markers": [],
        "has_figure": False,
        "finance_entities": fin_entities,
        "char_start": char_offset_ref[0],
        "char_end": char_offset_ref[0] + len(chunk_text),
    }
    doc = chunker._make_doc(
        text=chunk_text,
        modality="pdf",
        subtype=subtype,
        source=source,
        page=ext.page,
        chunk_idx=chunk_idx_ref[0],
        structure=structure,
        meta=meta,
        surface=surface,
    )
    if doc:
        chunk_idx_ref[0] += 1
        char_offset_ref[0] += len(chunk_text) + 1
    return doc


class PdfChunker(BaseChunker):
    """Finance-grade chunker for PDF files — handles prose, tables, OCR, images."""

    def chunk(
        self,
        extracts: list[RawExtract],
        meta: UniversalMetadata,
    ) -> list[IngestedDocument]:
        source = Path(meta.source_path).name or "unknown.pdf"
        surface = "pdf_chunker"
        modality_var.set("pdf")
        _t0 = time.time()
        logger.info(event="chunking_start", modality="pdf", source=source, extracts=len(extracts))
        if not extracts:
            logger.warning(event="no_extracts_received", modality="pdf", source=source)
            return []

        try:
            docs: list[IngestedDocument] = []
            chunk_idx = [0]
            # Cumulative character offset across all emitted chunks for precise citation
            char_offset = [0]

            section_title: str | None = None
            section_hierarchy: list[str] = []
            prose_buf: str = ""
            prose_page: int | None = None
            prose_footnotes: list[str] = []
            pending_table_rows: list[RawExtract] = []
            seen_hashes: set = set()

            def flush_prose() -> None:
                nonlocal prose_buf, prose_footnotes
                if not prose_buf.strip():
                    return
                for piece in self._split_text(prose_buf):
                    if not piece.strip():
                        continue
                    h = hash(piece)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    fin_entities = extract_finance_entities(piece)
                    chunk_hash = deterministic_chunk_id(
                        source, f"p{prose_page or 0}_prose_{chunk_idx[0]}", chunk_idx[0]
                    )
                    structure = {
                        "chunk_hash_id": chunk_hash,
                        "source_file": source,
                        "chunk_index": chunk_idx[0],
                        "page_number": prose_page,
                        "page_range": [prose_page, prose_page] if prose_page else None,
                        "chunk_type": "paragraph",
                        "section_title": section_title,
                        "section_hierarchy": section_hierarchy[:],
                        "table_title": None,
                        "column_headers": [],
                        "row_range": None,
                        "is_ocr": False,
                        "footnotes": prose_footnotes[:],
                        "footnote_markers": [],
                        "has_figure": False,
                        "figure_path": None,
                        "finance_entities": fin_entities,
                        "char_start": char_offset[0],
                        "char_end": char_offset[0] + len(piece),
                    }
                    doc = self._make_doc(
                        text=piece,
                        modality="pdf",
                        subtype="paragraph",
                        source=source,
                        page=prose_page,
                        chunk_idx=chunk_idx[0],
                        structure=structure,
                        meta=meta,
                        surface=surface,
                    )
                    if doc:
                        docs.append(doc)
                        chunk_idx[0] += 1
                        char_offset[0] += len(piece) + 1
                prose_buf = ""
                prose_footnotes = []

            def flush_table() -> None:
                nonlocal pending_table_rows
                for _ext in pending_table_rows:
                    _doc = _make_table_chunk(
                        ext=_ext,
                        chunk_type="financial_table",
                        subtype="table",
                        section_title=section_title,
                        section_hierarchy=section_hierarchy,
                        source=source,
                        meta=meta,
                        chunker=self,
                        chunk_idx_ref=chunk_idx,
                        char_offset_ref=char_offset,
                        surface=surface,
                    )
                    if _doc:
                        docs.append(_doc)
                pending_table_rows = []

            # Pre-compute BLIP captions + TrOCR for all image_region extracts
            # concurrently before the sequential pass. Keyed by id(ext) so the
            # main loop can look up results without changing its structure.
            _img_cache: dict[int, tuple[str, str]] = {}
            _img_extracts = [
                e for e in extracts if e.extract_type == "image_region" and e.raw_bytes
            ]
            if _img_extracts:

                def _caption_and_ocr_safe(ext_obj: RawExtract) -> tuple[int, str, str]:
                    raw = ext_obj.raw_bytes or b""
                    with _BLIP_SEMAPHORE:
                        cap = _caption_bytes(raw)
                        ocr = _ocr_bytes(raw)
                    return id(ext_obj), cap, ocr

                with ThreadPoolExecutor(max_workers=2) as _img_pool:
                    for _eid, _cap, _ocr in _img_pool.map(_caption_and_ocr_safe, _img_extracts):
                        _img_cache[_eid] = (_cap, _ocr)

            for ext in extracts:
                etype = ext.extract_type

                if etype == "header_footer":
                    continue

                if etype == "heading":
                    flush_prose()
                    flush_table()
                    heading_text = (ext.text or "").strip()
                    if heading_text:
                        # Prefer canonical SEC label from ingestor (e.g. "Item 7. MD&A")
                        sec_label = (ext.extra or {}).get("sec_section") or heading_text
                        sec_kind = _classify_sec_heading(heading_text)
                        font = ext.font_size or 12.0

                        if sec_kind == "part":
                            # PART I / II / III → reset to top level
                            section_hierarchy = [sec_label]
                            section_title = sec_label
                        elif sec_kind == "item":
                            # Item N → second level under current Part
                            section_hierarchy = section_hierarchy[:1] + [sec_label]
                            section_title = sec_label
                        elif font >= 16:
                            section_hierarchy = [heading_text]
                            section_title = heading_text
                        elif font >= 14:
                            section_hierarchy = section_hierarchy[:1] + [heading_text]
                            section_title = heading_text
                        else:
                            # Sub-section (3rd level): keep Part + Item, append this
                            section_hierarchy = section_hierarchy[:2] + [heading_text]
                            section_title = heading_text
                    continue

                if etype == "footnote":
                    prose_footnotes.append((ext.text or "").strip())
                    continue

                if etype == "table_summary":
                    # NL summary chunk — one per table, emitted immediately (no batching).
                    flush_prose()
                    flush_table()
                    _ext_sec = (ext.extra or {}).get("section_title")
                    if _ext_sec:
                        section_title = _ext_sec
                    _doc = _make_table_chunk(
                        ext=ext,
                        chunk_type="financial_table_summary",
                        subtype="table",
                        section_title=section_title,
                        section_hierarchy=section_hierarchy,
                        source=source,
                        meta=meta,
                        chunker=self,
                        chunk_idx_ref=chunk_idx,
                        char_offset_ref=char_offset,
                        surface=surface,
                    )
                    if _doc:
                        docs.append(_doc)
                    continue

                if etype == "table_row":
                    # Each extract = one complete table (clean markdown).
                    # Accumulate per-page then flush on page change or non-table extract.
                    flush_prose()
                    row_text = (ext.text or "").strip()
                    if not row_text:
                        continue
                    _ext_sec = (ext.extra or {}).get("section_title")
                    if _ext_sec:
                        section_title = _ext_sec
                    pending_table_rows.append(ext)
                    continue

                # Non-table-row seen → flush any accumulated table.
                if pending_table_rows:
                    flush_table()

                if etype == "prose":
                    text = (ext.text or "").strip()
                    if not text:
                        continue
                    if prose_page is None:
                        prose_page = ext.page
                    # Flush and reset page tracking when page changes.
                    if ext.page and prose_page and ext.page != prose_page:
                        flush_prose()
                        prose_page = ext.page
                    prose_buf = (prose_buf + "\n\n" + text).strip() if prose_buf else text
                    continue

                if etype == "scanned_page":
                    flush_prose()
                    flush_table()
                    raw = ext.raw_bytes or b""
                    ocr_text = _ocr_bytes(raw) if raw else (ext.text or "")
                    if not ocr_text.strip():
                        continue
                    for piece in self._split_text(ocr_text):
                        if not piece.strip():
                            continue
                        fin_entities = extract_finance_entities(piece)
                        chunk_hash = deterministic_chunk_id(
                            source, f"p{ext.page or 0}_ocr_{chunk_idx[0]}", chunk_idx[0]
                        )
                        structure = {
                            "chunk_hash_id": chunk_hash,
                            "source_file": source,
                            "chunk_index": chunk_idx[0],
                            "page_number": ext.page,
                            "page_range": [ext.page, ext.page] if ext.page else None,
                            "chunk_type": "paragraph",
                            "section_title": section_title,
                            "section_hierarchy": section_hierarchy[:],
                            "is_ocr": True,
                            "footnotes": [],
                            "footnote_markers": [],
                            "has_figure": False,
                            "figure_path": None,
                            "finance_entities": fin_entities,
                            "char_start": char_offset[0],
                            "char_end": char_offset[0] + len(piece),
                        }
                        doc = self._make_doc(
                            text=piece,
                            modality="pdf",
                            subtype="ocr",
                            source=source,
                            page=ext.page,
                            chunk_idx=chunk_idx[0],
                            structure=structure,
                            meta=meta,
                            surface=surface,
                        )
                        if doc:
                            docs.append(doc)
                            chunk_idx[0] += 1
                            char_offset[0] += len(piece) + 1
                    continue

                if etype == "image_region":
                    flush_prose()
                    flush_table()
                    raw = ext.raw_bytes or b""
                    caption_text, ocr_text = _img_cache.get(id(ext), ("", ""))
                    combined = f"{caption_text}\n{ocr_text}".strip()
                    if not combined:
                        continue
                    fin_entities = extract_finance_entities(combined)
                    chunk_hash = deterministic_chunk_id(
                        source, f"p{ext.page or 0}_img_{chunk_idx[0]}", chunk_idx[0]
                    )
                    structure = {
                        "chunk_hash_id": chunk_hash,
                        "source_file": source,
                        "chunk_index": chunk_idx[0],
                        "page_number": ext.page,
                        "page_range": [ext.page, ext.page] if ext.page else None,
                        "chunk_type": "figure_caption",
                        "section_title": section_title,
                        "caption": caption_text,
                        "ocr_text": ocr_text,
                        "footnotes": [],
                        "footnote_markers": [],
                        "has_figure": True,
                        "figure_path": None,
                        "finance_entities": fin_entities,
                        "char_start": char_offset[0],
                        "char_end": char_offset[0] + len(combined),
                    }
                    doc = self._make_doc(
                        text=combined,
                        modality="pdf",
                        subtype="figure_caption",
                        source=source,
                        page=ext.page,
                        chunk_idx=chunk_idx[0],
                        structure=structure,
                        meta=meta,
                        surface=surface,
                    )
                    if doc:
                        docs.append(doc)
                        chunk_idx[0] += 1
                        char_offset[0] += len(combined) + 1
                    continue

            flush_prose()
            flush_table()

            logger.info(event="pdf_chunking_done", source=source, chunks=len(docs))
            _CHUNKS_TOTAL.inc(len(docs))
            return docs
        except Exception as _exc:
            _CHUNK_ERRORS.inc()
            logger.error(event="chunking_failed", modality="pdf", source=source, error=str(_exc))
            raise

    def health_check(self) -> dict:
        return {
            "modality": "pdf",
            "status": "ok",
            "class": self.__class__.__name__,
        }
