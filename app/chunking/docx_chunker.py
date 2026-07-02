from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, List, Optional

from app.chunking.base_chunker import BaseChunker
from app.chunking.finance_numbers import (
    approx_tokens,
    deterministic_chunk_id,
    extract_finance_entities,
)
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger, modality_var

import time
from prometheus_client import Counter, Histogram

logger = get_logger(__name__)

_CHUNKS_TOTAL = Counter(
    "magik_docx_chunks_total",
    "Total chunks produced by docx chunker",
)
_CHUNK_ERRORS = Counter(
    "magik_docx_chunk_errors_total",
    "Total errors in docx chunker",
)

_DEFINED_TERM_RE = re.compile(
    r'"?([A-Z][a-zA-Z\s]{2,40})"?\s+(?:means?|shall mean|is defined as)\s+',
    re.IGNORECASE,
)
_BOLD_TERM_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_TERM_RE = re.compile(r"\*(.+?)\*|_(.+?)_")
_CLAUSE_NUM_RE = re.compile(
    r'\b(?:Section|Clause|Article|§)\s*\d+(?:\.\d+)*(?:\([a-z]\))*(?:\([ivx]+\))*\b',
    re.IGNORECASE,
)

_TABLE_GROUP_SIZE = 5
_TABLE_OVERLAP_ROWS = 2

# A finance table (DCF assumptions, income statement, comps) is a single logical
# unit: splitting it into overlapping 5-row windows meant "Implied Price Target |
# $245 | $285 | $190" and "WACC | 8.5% | 8.0% | 9.5%" landed in DIFFERENT chunks,
# so a query retrieving one window got only half the table and the LLM answered
# "I could not find the price targets." Any table that fits this token budget is
# now emitted as ONE chunk; only genuinely large tables fall back to windowing.
_DOCX_TABLE_MAX_TOKENS = 420

# settings.CHUNK_TARGET_TOKENS defaults to 120 — well under the 200-500 token
# target documented for this codebase (CLAUDE.md), and small enough that a
# short coherent passage (e.g. "our thesis rests on three pillars: ...")
# gets split across chunk boundaries, so a query needing all of it only
# retrieves part. Overridden here rather than changing the shared default,
# which other modalities (pdf_chunker, txt_chunker) also read.
_DOCX_PROSE_TARGET_TOKENS = 280


def _caption_bytes(raw_bytes: bytes) -> str:
    try:
        from PIL import Image
        from app.chunking.image_chunker import blip_caption
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return blip_caption(img)
    except Exception as exc:
        logger.warning(event="docx_blip_failed", error=str(exc))
        return ""


def _heading_level_from_style(style_name: Optional[str]) -> int:
    """Extract numeric heading level from DOCX style name (e.g. 'Heading 2' → 2)."""
    if not style_name:
        return 0
    m = re.search(r"heading\s*(\d)", style_name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


_SECTION_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")


def _section_number(heading: Optional[str]) -> Optional[str]:
    """Short section id for the LLM-facing citation tag ("4.1 DCF Model Key
    Assumptions" -> "4.1"). A 60+ char heading is a lot for a small quantized
    model to copy verbatim into a citation; a short number is far more
    reliable, and the full heading text is still stored separately in
    heading/section_title for UI display. Falls back to the full heading when
    there is no numeric prefix (e.g. cover-page titles).
    """
    if not heading:
        return heading
    m = _SECTION_NUMBER_RE.match(heading)
    return m.group(1) if m else heading


class DocxChunker(BaseChunker):
    """Finance-grade chunker for Word documents (investment memos, term sheets, agreements)."""

    def chunk(
        self,
        extracts: List[RawExtract],
        meta: UniversalMetadata,
    ) -> List[IngestedDocument]:
        source = Path(meta.source_path).name or "unknown.docx"
        surface = "docx_chunker"
        modality_var.set("docx")
        _t0 = time.time()
        logger.info(event="chunking_start", modality="docx", source=source, extracts=len(extracts))
        if not extracts:
            logger.warning(event="no_extracts_received", modality="docx", source=source)
            return []
        try:
            docs: List[IngestedDocument] = []
            chunk_idx = [0]
            char_offset = [0]  # cumulative character offset through document

            # Section hierarchy stack: index = level-1, value = heading text
            hierarchy: List[str] = []
            prose_buf: str = ""
            pending_table_rows: List[RawExtract] = []
            table_headers: List[str] = []
            current_table_index: Optional[int] = None
            defined_terms: Dict[str, str] = {}
            seen_hashes: set = set()
            paragraph_index = [0]   # paragraph counter within current section (MD Phase 1.3)
            table_index = [0]       # table counter in document (MD Phase 1.3)

            def current_heading() -> Optional[str]:
                return hierarchy[-1] if hierarchy else None

            def flush_prose() -> None:
                nonlocal prose_buf
                if not prose_buf.strip():
                    return
                for piece in self._split_text(prose_buf, target_tokens=_DOCX_PROSE_TARGET_TOKENS):
                    if not piece.strip():
                        continue
                    h = hash(piece)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    bold_terms = _BOLD_TERM_RE.findall(piece)
                    italic_terms = [g1 or g2 for g1, g2 in _ITALIC_TERM_RE.findall(piece)]
                    clause_nums = _CLAUSE_NUM_RE.findall(piece)
                    # Capture defined terms from this prose chunk
                    for m in _DEFINED_TERM_RE.finditer(piece):
                        term = m.group(1).strip()
                        defined_terms[term] = piece[m.end(): m.end() + 80].strip()

                    fin_entities = extract_finance_entities(piece)
                    paragraph_index[0] += 1
                    chunk_hash = deterministic_chunk_id(source, f"para_{chunk_idx[0]}", chunk_idx[0])
                    structure = {
                        "chunk_hash_id":    chunk_hash,
                        "source_file":      source,
                        "chunk_index":      chunk_idx[0],
                        "paragraph_index":  paragraph_index[0],
                        "heading":          current_heading(),
                        "section_id":       _section_number(current_heading()),
                        "section_title":    current_heading(),
                        "heading_hierarchy": hierarchy[:],
                        "heading_level":    len(hierarchy),
                        "chunk_type":       "paragraph",
                        "has_bold_terms":   bold_terms,
                        "has_italic_terms": italic_terms,
                        "clause_numbers":   clause_nums,
                        "defined_terms":    {},
                        "finance_entities": fin_entities,
                        "char_start":       char_offset[0],
                        "char_end":         char_offset[0] + len(piece),
                    }
                    char_offset[0] += len(piece) + 1
                    doc = self._make_doc(
                        text=piece,
                        modality="docx",
                        subtype="paragraph",
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
                prose_buf = ""

            def _emit_table_chunk(row_texts: List[str], row_range: List[int], is_first: bool) -> None:
                header_line = " | ".join(table_headers) if table_headers else ""
                nl_text = (f"{header_line}\n" if header_line else "") + "\n".join(row_texts)
                fin_entities = extract_finance_entities(nl_text)
                tbl_bold = _BOLD_TERM_RE.findall(nl_text)
                tbl_italic = [g1 or g2 for g1, g2 in _ITALIC_TERM_RE.findall(nl_text)]
                tbl_clauses = _CLAUSE_NUM_RE.findall(nl_text)
                chunk_hash = deterministic_chunk_id(source, f"table_r{row_range[0]}_{chunk_idx[0]}", chunk_idx[0])
                if is_first:
                    table_index[0] += 1
                structure = {
                    "chunk_hash_id":    chunk_hash,
                    "source_file":      source,
                    "chunk_index":      chunk_idx[0],
                    "paragraph_index":  paragraph_index[0],
                    "table_index":      table_index[0],
                    "heading":          current_heading(),
                    "section_id":       _section_number(current_heading()),
                    "section_title":    current_heading(),
                    "heading_hierarchy": hierarchy[:],
                    "heading_level":    len(hierarchy),
                    "chunk_type":       "table",
                    "column_headers":   table_headers[:],
                    "row_range":        row_range,
                    "has_bold_terms":   tbl_bold,
                    "has_italic_terms": tbl_italic,
                    "clause_numbers":   tbl_clauses,
                    "defined_terms":    {},
                    "finance_entities": fin_entities,
                    "char_start":       char_offset[0],
                    "char_end":         char_offset[0] + len(nl_text),
                }
                char_offset[0] += len(nl_text) + 1
                doc = self._make_doc(
                    text=nl_text,
                    modality="docx",
                    subtype="table",
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

            def flush_table() -> None:
                nonlocal pending_table_rows, table_headers, current_table_index
                if not pending_table_rows:
                    return

                all_row_texts = [r.text for r in pending_table_rows]
                header_line = " | ".join(table_headers) if table_headers else ""
                whole_text = (f"{header_line}\n" if header_line else "") + "\n".join(all_row_texts)

                if approx_tokens(whole_text) <= _DOCX_TABLE_MAX_TOKENS:
                    # Whole table fits — keep it as ONE chunk so every row of a
                    # finance table (assumptions, price targets, comps) stays
                    # together and is retrievable as a unit.
                    _emit_table_chunk(all_row_texts, [1, len(pending_table_rows)], is_first=True)
                else:
                    # Large table — fall back to overlapping windows.
                    step = _TABLE_GROUP_SIZE
                    overlap = _TABLE_OVERLAP_ROWS
                    i = 0
                    while i < len(pending_table_rows):
                        group = pending_table_rows[i: i + step]
                        row_texts = [r.text for r in group]
                        row_range = [i + 1, min(i + step, len(pending_table_rows))]
                        _emit_table_chunk(row_texts, row_range, is_first=(i == 0))
                        i += step - overlap

                pending_table_rows = []
                table_headers = []
                current_table_index = None

            for ext in extracts:
                etype = ext.extract_type

                if etype == "header_footer":
                    continue

                if etype == "heading":
                    flush_prose()
                    flush_table()
                    heading_text = (ext.text or "").strip()
                    extra_level = (ext.extra or {}).get("heading_level")
                    level = extra_level or _heading_level_from_style(ext.style_name) or 1
                    # Trim hierarchy to the level above and append new heading.
                    hierarchy = hierarchy[: level - 1]
                    if heading_text:
                        hierarchy.append(heading_text)
                    paragraph_index[0] = 0  # reset per-section paragraph counter
                    continue

                if etype == "table_row":
                    flush_prose()
                    row_text = (ext.text or "").strip()
                    if not row_text:
                        continue
                    ext_table_index = ext.extra.get("table_index") if ext.extra else None
                    if pending_table_rows and ext_table_index != current_table_index:
                        # A new table started without an intervening heading/prose
                        # extract — flush the previous table so rows never merge
                        # across table boundaries.
                        flush_table()
                    if not pending_table_rows:
                        current_table_index = ext_table_index
                        # First row — check if it's a header (no digits usually).
                        if not re.search(r"\d", row_text):
                            table_headers = [c.strip() for c in row_text.split("|") if c.strip()]
                            continue
                    pending_table_rows.append(ext)
                    continue

                if pending_table_rows and etype != "table_row":
                    flush_table()

                if etype == "prose":
                    text = (ext.text or "").strip()
                    if not text:
                        continue
                    prose_buf = (prose_buf + "\n\n" + text).strip() if prose_buf else text
                    continue

                if etype == "comment":
                    flush_prose()
                    flush_table()
                    comment_text = (ext.text or "").strip()
                    if not comment_text:
                        continue
                    chunk_hash = deterministic_chunk_id(source, f"comment_{chunk_idx[0]}", chunk_idx[0])
                    structure = {
                        "chunk_hash_id":    chunk_hash,
                        "source_file":      source,
                        "chunk_index":      chunk_idx[0],
                        "heading":          current_heading(),
                        "section_id":       _section_number(current_heading()),
                        "section_title":    current_heading(),
                        "heading_hierarchy": hierarchy[:],
                        "chunk_type":       "annotation",
                        "finance_entities": extract_finance_entities(comment_text),
                        "char_start":       char_offset[0],
                        "char_end":         char_offset[0] + len(comment_text),
                    }
                    char_offset[0] += len(comment_text) + 1
                    doc = self._make_doc(
                        text=comment_text,
                        modality="docx",
                        subtype="annotation",
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

                if etype == "image_region":
                    flush_prose()
                    flush_table()
                    raw = ext.raw_bytes or b""
                    cap = _caption_bytes(raw) if raw else ""
                    if not cap:
                        continue
                    chunk_hash = deterministic_chunk_id(source, f"img_{chunk_idx[0]}", chunk_idx[0])
                    structure = {
                        "chunk_hash_id":   chunk_hash,
                        "source_file":     source,
                        "chunk_index":     chunk_idx[0],
                        "heading":         current_heading(),
                        "section_id":      _section_number(current_heading()),
                        "section_title":   current_heading(),
                        "heading_hierarchy": hierarchy[:],
                        "chunk_type":      "figure_caption",
                        "caption":         cap,
                        "finance_entities": extract_finance_entities(cap),
                        "char_start":      char_offset[0],
                        "char_end":        char_offset[0] + len(cap),
                    }
                    char_offset[0] += len(cap) + 1
                    doc = self._make_doc(
                        text=cap,
                        modality="docx",
                        subtype="paragraph",
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

            flush_prose()
            flush_table()

            # Emit a definitions chunk if we collected any defined terms.
            if defined_terms:
                def_text = "\n".join(f"{k}: {v}" for k, v in defined_terms.items())
                def_bold = _BOLD_TERM_RE.findall(def_text)
                def_italic = [g1 or g2 for g1, g2 in _ITALIC_TERM_RE.findall(def_text)]
                def_clauses = _CLAUSE_NUM_RE.findall(def_text)
                chunk_hash = deterministic_chunk_id(source, "definitions", chunk_idx[0])
                structure = {
                    "chunk_hash_id":    chunk_hash,
                    "source_file":      source,
                    "chunk_index":      chunk_idx[0],
                    "heading":          current_heading(),
                    "section_id":       _section_number(current_heading()),
                    "section_title":    current_heading(),
                    "heading_hierarchy": hierarchy[:],
                    "heading_level":    len(hierarchy),
                    "paragraph_index":  paragraph_index[0],
                    "chunk_type":       "definitions",
                    "has_bold_terms":   def_bold,
                    "has_italic_terms": def_italic,
                    "clause_numbers":   def_clauses,
                    "defined_terms":    defined_terms,
                    "finance_entities": extract_finance_entities(def_text),
                    "char_start":       char_offset[0],
                    "char_end":         char_offset[0] + len(def_text),
                }
                char_offset[0] += len(def_text) + 1
                doc = self._make_doc(
                    text=def_text,
                    modality="docx",
                    subtype="definition",
                    source=source,
                    page=None,
                    chunk_idx=chunk_idx[0],
                    structure=structure,
                    meta=meta,
                    surface=surface,
                )
                if doc:
                    docs.append(doc)

            logger.info(event="docx_chunking_done", source=source, chunks=len(docs))
            _CHUNKS_TOTAL.inc(len(docs))
            return docs
        except Exception as _exc:
            _CHUNK_ERRORS.inc()
            logger.error(event="chunking_failed", modality="docx", source=source, error=str(_exc))
            raise

    def health_check(self) -> dict:
        return {
            "modality": "docx",
            "status": "ok",
            "class": self.__class__.__name__,
        }
