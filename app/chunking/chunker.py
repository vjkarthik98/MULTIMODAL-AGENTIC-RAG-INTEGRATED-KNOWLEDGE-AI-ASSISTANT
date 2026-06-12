from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# PER-MODALITY CHUNKER DISPATCH (Phase 2)
# Called by ingestion_pipeline when the new ingest→chunk→embed chain is used.
# The legacy chunk_documents() below remains for backward compatibility.

_CHUNKER_MAP = None


def _get_chunker_map():
    global _CHUNKER_MAP
    if _CHUNKER_MAP is None:
        from app.chunking.txt_chunker import TxtChunker
        from app.chunking.pdf_chunker import PdfChunker
        from app.chunking.docx_chunker import DocxChunker
        from app.chunking.xlsx_chunker import XlsxChunker
        from app.chunking.image_chunker import ImageChunker
        from app.chunking.audio_chunker import AudioChunker
        from app.chunking.video_chunker import VideoChunker
        _CHUNKER_MAP = {
            "text":  TxtChunker,
            "pdf":   PdfChunker,
            "word":  DocxChunker,
            "excel": XlsxChunker,
            "image": ImageChunker,
            "audio": AudioChunker,
            "video": VideoChunker,
        }
    return _CHUNKER_MAP


def chunk_raw_extracts(extracts, meta, modality: str) -> List[Any]:
    """Route List[RawExtract] through the appropriate per-modality chunker."""
    chunker_cls = _get_chunker_map().get(modality)
    if chunker_cls is None:
        logger.warning(event="chunker_unknown_modality", modality=modality)
        return []
    chunker = chunker_cls()
    return chunker.chunk(extracts, meta)


# DOCUMENT CHUNKER — CALLED BY INGESTION PIPELINE
#
# Ingestors emit one document per natural unit (a PDF page, a DOCX section, a
# spreadsheet row-block, an audio segment, an image caption). Some of those —
# notably PDF pages and long DOCX/TXT sections — are far larger than the
# embedding model's effective window, which dilutes retrieval (one giant vector
# for a whole page). This step splits ONLY long prose into token-aware windows
# with overlap, preserving every locator (page / sheet / section_title /
# timestamp / caption) on each sub-chunk. Structured/temporal/visual units are
# left intact so row-blocks, transcript segments and captions keep their
# integrity and their citation locators.

# Modalities/subtypes that must never be split (atomic units).
_ATOMIC_MODALITIES = {"image", "audio", "video", "table"}
_ATOMIC_SUBTYPES = {"structured", "caption", "speech", "heading", "frame", "table", "chart"}


def _approx_tokens(text: str) -> int:
    # Cheap, model-agnostic proxy: ~1.3 tokens per whitespace word.
    return int(len(text.split()) * 1.3)


def _split_text(text: str, size: int, overlap: int) -> List[str]:
    """Recursive, boundary-aware splitter: pack paragraph→sentence units into
    windows of at most `size` characters, carrying an `overlap`-char tail into
    the next window for cross-boundary context. Hard-splits any single unit
    longer than `size`."""
    text = (text or "").strip()
    if len(text) <= size:
        return [text] if text else []

    units: List[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            if sent:
                units.append(sent)
    if not units:
        units = [text]

    chunks: List[str] = []
    cur = ""
    step = max(size - overlap, 1)
    for u in units:
        if cur and len(cur) + 1 + len(u) > size:
            chunks.append(cur)
            if overlap > 0 and len(cur) > overlap:
                tail = cur[-overlap:]
                sp = tail.find(" ")
                cur = tail[sp + 1:] if sp != -1 else tail
            else:
                cur = ""
        if len(u) > size:                       # a single oversized unit
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(u), step):
                chunks.append(u[i:i + size])
            continue
        cur = (cur + " " + u).strip() if cur else u
    if cur.strip():
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


def _is_atomic(doc: Any) -> bool:
    modality = (getattr(doc, "modality", "") or "").lower()
    subtype = (getattr(doc, "subtype", "") or "").lower()
    return modality in _ATOMIC_MODALITIES or subtype in _ATOMIC_SUBTYPES


def _clone_with_text(doc: Any, text: str, chunk_id: int, part_index: int, total_parts: int) -> Any:
    """Clone an IngestedDocument-like object with new text, preserving every
    locator field (structure carries page/sheet/section_title/timestamps)."""
    try:
        clone = doc.model_copy(deep=True)        # pydantic v2
    except Exception:
        try:
            clone = doc.copy(deep=True)          # pydantic v1 fallback
        except Exception:
            return None
    clone.text = text
    try:
        clone.chunk_id = chunk_id
    except Exception:
        pass
    struct = getattr(clone, "structure", None)
    if isinstance(struct, dict):
        struct["chunk_index"] = part_index
        struct["chunk_parts"] = total_parts
    # Re-run validation/normalization so the clone is a first-class document.
    if hasattr(clone, "finalize"):
        try:
            clone = clone.finalize()
        except Exception:
            return None
    return clone


def chunk_documents(docs: List[Any]) -> List[Any]:
    """Split long prose documents into token-aware overlapping windows while
    leaving structured/temporal/visual units intact. Assigns a unique chunk_id
    to every output chunk and preserves all locator metadata."""
    start = time.time()
    output: List[Any] = []
    skipped = 0
    split_docs = 0
    total_tokens = 0
    modality_breakdown: Dict[str, int] = {}

    size = max(int(settings.CHUNK_SIZE), 256)
    overlap = max(min(int(settings.CHUNK_OVERLAP), size // 2), 0)
    max_chunks = max(int(settings.MAX_CHUNKS), 1)
    next_chunk_id = 0

    for doc in docs:
        text = getattr(doc, "text", None) or ""
        if not text.strip():
            skipped += 1
            continue

        modality = getattr(doc, "modality", "text") or "text"

        # Decide whether to split: only long, non-atomic prose.
        if _is_atomic(doc) or len(text) <= size:
            pieces = [text]
        else:
            pieces = _split_text(text, size, overlap)
            if len(pieces) > 1:
                split_docs += 1

        if len(pieces) <= 1:
            # Keep the original object; just ensure it has a chunk_id.
            if getattr(doc, "chunk_id", None) is None:
                try:
                    doc.chunk_id = next_chunk_id
                except Exception:
                    pass
            next_chunk_id += 1
            total_tokens += _approx_tokens(text)
            modality_breakdown[modality] = modality_breakdown.get(modality, 0) + 1
            output.append(doc)
        else:
            total = len(pieces)
            for idx, piece in enumerate(pieces):
                clone = _clone_with_text(doc, piece, next_chunk_id, idx, total)
                next_chunk_id += 1
                if clone is None:
                    continue
                total_tokens += _approx_tokens(piece)
                modality_breakdown[modality] = modality_breakdown.get(modality, 0) + 1
                output.append(clone)

        if len(output) >= max_chunks:
            logger.warning(event="chunking_max_chunks_reached", limit=max_chunks)
            break

    logger.info(
        event="chunking_success",
        input=len(docs),
        output=len(output),
        skipped=skipped,
        docs_split=split_docs,
        total_tokens_est=total_tokens,
        chunk_size=size,
        chunk_overlap=overlap,
        modality_breakdown=modality_breakdown,
        latency=round(time.time() - start, 3),
    )

    return output
