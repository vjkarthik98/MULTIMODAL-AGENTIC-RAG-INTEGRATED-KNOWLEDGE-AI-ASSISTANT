from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.chunking.base_chunker import BaseChunker
from app.chunking.finance_numbers import (
    deterministic_chunk_id,
    extract_finance_entities,
)
from app.core.config import settings
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TABLE_GROUP_SIZE = 5       # target rows per table chunk
_TABLE_OVERLAP_ROWS = 2     # rows carried into next table chunk


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
        from app.chunking.image_chunker import blip2_caption
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return blip2_caption(img)
    except Exception as exc:
        logger.warning(event="pdf_blip2_failed", error=str(exc))
        return ""


def _split_table_rows(
    rows: List[RawExtract],
    headers: List[str],
    page: Optional[int],
    section_title: Optional[str],
    table_title: Optional[str],
    source: str,
    meta: UniversalMetadata,
    chunker: "PdfChunker",
    chunk_idx_ref: List[int],
    surface: str,
    section_hierarchy: List[str],
) -> List[IngestedDocument]:
    docs: List[IngestedDocument] = []
    step = _TABLE_GROUP_SIZE
    overlap = _TABLE_OVERLAP_ROWS
    i = 0
    while i < len(rows):
        group = rows[i: i + step]
        row_texts = [r.text for r in group]
        if headers:
            header_line = " | ".join(headers)
            nl_text = f"{table_title or 'Table'} ({section_title or ''})\n{header_line}\n"
        else:
            nl_text = f"{table_title or 'Table'} ({section_title or ''})\n"
        nl_text += "\n".join(row_texts)
        fin_entities = extract_finance_entities(nl_text)
        row_nums = (rows[i].extra.get("row_num", i + 1), rows[min(i + step - 1, len(rows) - 1)].extra.get("row_num", i + step))
        chunk_hash = deterministic_chunk_id(source, f"p{page or 0}_table_r{row_nums[0]}", chunk_idx_ref[0])

        structure = {
            "chunk_hash_id":      chunk_hash,
            "source_file":        source,
            "chunk_index":        chunk_idx_ref[0],
            "page_number":        page,
            "page_range":         [page, page] if page else None,
            "chunk_type":         "table",
            "section_title":      section_title,
            "section_hierarchy":  section_hierarchy[:],
            "table_title":        table_title,
            "column_headers":     headers,
            "row_range":          list(row_nums),
            "is_ocr":             False,
            "footnotes":          [],
            "footnote_markers":   [],
            "has_figure":         False,
            "figure_path":        None,
            "finance_entities":   fin_entities,
            "char_start":         0,
            "char_end":           len(nl_text),
        }
        doc = chunker._make_doc(
            text=nl_text,
            modality="pdf",
            subtype="table",
            source=source,
            page=page,
            chunk_idx=chunk_idx_ref[0],
            structure=structure,
            meta=meta,
            surface=surface,
        )
        if doc:
            docs.append(doc)
            chunk_idx_ref[0] += 1
        i += step - overlap
    return docs


class PdfChunker(BaseChunker):
    """Finance-grade chunker for PDF files — handles prose, tables, OCR, images."""

    def chunk(
        self,
        extracts: List[RawExtract],
        meta: UniversalMetadata,
    ) -> List[IngestedDocument]:
        source = Path(meta.source_path).name or "unknown.pdf"
        surface = "pdf_chunker"

        docs: List[IngestedDocument] = []
        chunk_idx = [0]

        section_title: Optional[str] = None
        section_hierarchy: List[str] = []
        prose_buf: str = ""
        prose_page: Optional[int] = None
        prose_footnotes: List[str] = []
        pending_table_rows: List[RawExtract] = []
        table_headers: List[str] = []
        table_title: Optional[str] = None
        table_page: Optional[int] = None
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
                chunk_hash = deterministic_chunk_id(source, f"p{prose_page or 0}_prose_{chunk_idx[0]}", chunk_idx[0])
                structure = {
                    "chunk_hash_id":     chunk_hash,
                    "source_file":       source,
                    "chunk_index":       chunk_idx[0],
                    "page_number":       prose_page,
                    "page_range":        [prose_page, prose_page] if prose_page else None,
                    "chunk_type":        "paragraph",
                    "section_title":     section_title,
                    "section_hierarchy": section_hierarchy[:],
                    "table_title":       None,
                    "column_headers":    [],
                    "row_range":         None,
                    "is_ocr":            False,
                    "footnotes":         prose_footnotes[:],
                    "footnote_markers":  [],
                    "has_figure":        False,
                    "figure_path":       None,
                    "finance_entities":  fin_entities,
                    "char_start":        0,
                    "char_end":          len(piece),
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
            prose_buf = ""
            prose_footnotes = []

        def flush_table() -> None:
            nonlocal pending_table_rows, table_headers, table_title, table_page
            if not pending_table_rows:
                return
            table_docs = _split_table_rows(
                pending_table_rows, table_headers, table_page,
                section_title, table_title, source, meta, self, chunk_idx, surface, section_hierarchy,
            )
            docs.extend(table_docs)
            pending_table_rows = []
            table_headers = []
            table_title = None

        for ext in extracts:
            etype = ext.extract_type

            if etype == "header_footer":
                continue

            if etype == "heading":
                flush_prose()
                flush_table()
                heading_text = (ext.text or "").strip()
                if heading_text:
                    section_title = heading_text
                    # Manage hierarchy stack by font size proxy (larger font = higher level)
                    font = ext.font_size or 12.0
                    if font >= 16:
                        section_hierarchy = [heading_text]
                    elif font >= 14:
                        section_hierarchy = section_hierarchy[:1] + [heading_text]
                    else:
                        section_hierarchy = section_hierarchy[:2] + [heading_text]
                continue

            if etype == "footnote":
                prose_footnotes.append((ext.text or "").strip())
                continue

            if etype == "table_row":
                flush_prose()
                row_text = (ext.text or "").strip()
                if not row_text:
                    continue
                # First row in a new table batch = detect if it's headers
                if not pending_table_rows:
                    table_page = ext.page
                    # Heuristic: if the text has no numbers it's a header row
                    import re
                    if not re.search(r"\d", row_text):
                        table_headers = [c.strip() for c in row_text.split("|") if c.strip()]
                        continue
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
                    chunk_hash = deterministic_chunk_id(source, f"p{ext.page or 0}_ocr_{chunk_idx[0]}", chunk_idx[0])
                    structure = {
                        "chunk_hash_id":     chunk_hash,
                        "source_file":       source,
                        "chunk_index":       chunk_idx[0],
                        "page_number":       ext.page,
                        "page_range":        [ext.page, ext.page] if ext.page else None,
                        "chunk_type":        "paragraph",
                        "section_title":     section_title,
                        "section_hierarchy": section_hierarchy[:],
                        "is_ocr":            True,
                        "footnotes":         [],
                        "footnote_markers":  [],
                        "has_figure":        False,
                        "figure_path":       None,
                        "finance_entities":  fin_entities,
                        "char_start":        0,
                        "char_end":          len(piece),
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
                continue

            if etype == "image_region":
                flush_prose()
                flush_table()
                raw = ext.raw_bytes or b""
                caption_text = _caption_bytes(raw) if raw else ""
                ocr_text = _ocr_bytes(raw) if raw else ""
                combined = f"{caption_text}\n{ocr_text}".strip()
                if not combined:
                    continue
                fin_entities = extract_finance_entities(combined)
                chunk_hash = deterministic_chunk_id(source, f"p{ext.page or 0}_img_{chunk_idx[0]}", chunk_idx[0])
                structure = {
                    "chunk_hash_id":    chunk_hash,
                    "source_file":      source,
                    "chunk_index":      chunk_idx[0],
                    "page_number":      ext.page,
                    "page_range":       [ext.page, ext.page] if ext.page else None,
                    "chunk_type":       "figure_caption",
                    "section_title":    section_title,
                    "caption":          caption_text,
                    "ocr_text":         ocr_text,
                    "footnotes":        [],
                    "footnote_markers": [],
                    "has_figure":       True,
                    "figure_path":      None,
                    "finance_entities": fin_entities,
                    "char_start":       0,
                    "char_end":         len(combined),
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
                continue

        flush_prose()
        flush_table()

        logger.info(event="pdf_chunking_done", source=source, chunks=len(docs))
        return docs
