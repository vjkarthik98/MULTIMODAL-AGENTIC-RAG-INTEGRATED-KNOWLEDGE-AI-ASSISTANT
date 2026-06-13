from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.chunking.finance_numbers import approx_tokens, protect, restore
from app.core.config import settings
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Priority-ordered separators for finance text (MAGIK spec Phase 1.1).
# Single-newline speaker patterns come after paragraph break so they
# catch transcript formats that use \nCEO: without a blank line.
_FINANCE_SEPARATORS = [
    "\n\n\n",
    "\n\n",
    "\nOPERATOR:", "\nCEO:", "\nCFO:", "\nANALYST:",
    ". ", "? ", "! ", "; ",
    ", ",    # last resort — protected finance numbers are safe from splitting
]

_HEADING_RE = re.compile(r"^(?:[A-Z][A-Z\s&,]{4,}|(?:\d+\.)+\s+.{3,}|#{1,3}\s+.{3,})$")


class BaseChunker(ABC):

    @abstractmethod
    def chunk(
        self,
        extracts: List[RawExtract],
        meta: UniversalMetadata,
    ) -> List[IngestedDocument]:
        ...

    # ── Guardrail ─────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize(text: str, surface: str) -> str:
        try:
            from app.guardrails.input_guard import sanitize as _g
            return _g(text, surface=surface)
        except Exception:
            return text

    # ── Finance-safe text splitting ───────────────────────────────────────────

    @staticmethod
    def _split_text(text: str, target_tokens: Optional[int] = None) -> List[str]:
        target = target_tokens or settings.CHUNK_TARGET_TOKENS
        protected, mapping = protect(text)
        pieces = _recursive_split(protected, target)
        return [restore(p, mapping).strip() for p in pieces if p.strip()]

    # ── IngestedDocument factory ──────────────────────────────────────────────

    @staticmethod
    def _make_doc(
        *,
        text: str,
        modality: str,
        subtype: Optional[str],
        source: str,
        page: Optional[int],
        chunk_idx: int,
        structure: Dict,
        meta: UniversalMetadata,
        surface: str,
    ) -> Optional[IngestedDocument]:
        text = BaseChunker._sanitize(text, surface)
        if not text.strip():
            return None
        struct: Dict = {
            "content_type": subtype or "unknown",
            "embedding_space": "text",
            "token_count": approx_tokens(text),
            **structure,
        }
        struct.setdefault("session_id", meta.custom_fields.get("session_id", "default"))
        struct.setdefault("doc_id", str(meta.file_id))
        try:
            return IngestedDocument(
                text=text,
                modality=modality,
                subtype=subtype,
                source_type="file",
                source=source,
                page=page,
                chunk_id=chunk_idx,
                structure=struct,
                universal_metadata=meta,
            ).finalize()
        except Exception as exc:
            logger.warning(
                event="chunk_build_failed",
                error=str(exc),
                source=source,
                idx=chunk_idx,
            )
            return None

    # ── Heading detector ──────────────────────────────────────────────────────

    @staticmethod
    def _is_heading_line(line: str) -> bool:
        return bool(_HEADING_RE.match(line.strip()))


# ── Legacy chunk_documents helpers (moved from chunker.py) ───────────────────

_ATOMIC_MODALITIES = {"image", "audio", "video", "table"}
_ATOMIC_SUBTYPES   = {"structured", "caption", "speech", "heading", "frame", "table", "chart"}


def _is_atomic(doc: Any) -> bool:
    modality = (getattr(doc, "modality", "") or "").lower()
    subtype  = (getattr(doc, "subtype",  "") or "").lower()
    return modality in _ATOMIC_MODALITIES or subtype in _ATOMIC_SUBTYPES


def _clone_with_text(doc: Any, text: str, chunk_id: int, part_index: int, total_parts: int) -> Optional[Any]:
    """Clone an IngestedDocument-like object with new text, preserving all locator fields."""
    try:
        clone = doc.model_copy(deep=True)
    except Exception:
        try:
            clone = doc.copy(deep=True)
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
    if hasattr(clone, "finalize"):
        try:
            clone = clone.finalize()
        except Exception:
            return None
    return clone


def _legacy_approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _legacy_split_text(text: str, size: int, overlap: int) -> List[str]:
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
        if len(u) > size:
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


def chunk_documents(docs: List[Any]) -> List[Any]:
    """Legacy splitter: splits long prose into overlapping token windows.
    Kept for backward compatibility — new code uses chunk_raw_extracts()."""
    start = time.time()
    output: List[Any] = []
    skipped = 0
    split_docs = 0
    total_tokens = 0
    modality_breakdown: Dict[str, int] = {}

    size    = max(int(settings.CHUNK_SIZE),    256)
    overlap = max(min(int(settings.CHUNK_OVERLAP), size // 2), 0)
    max_chunks = max(int(settings.MAX_CHUNKS), 1)
    next_chunk_id = 0

    for doc in docs:
        text = getattr(doc, "text", None) or ""
        if not text.strip():
            skipped += 1
            continue
        modality = getattr(doc, "modality", "text") or "text"
        if _is_atomic(doc) or len(text) <= size:
            pieces = [text]
        else:
            pieces = _legacy_split_text(text, size, overlap)
            if len(pieces) > 1:
                split_docs += 1

        if len(pieces) <= 1:
            if getattr(doc, "chunk_id", None) is None:
                try:
                    doc.chunk_id = next_chunk_id
                except Exception:
                    pass
            next_chunk_id += 1
            total_tokens += _legacy_approx_tokens(text)
            modality_breakdown[modality] = modality_breakdown.get(modality, 0) + 1
            output.append(doc)
        else:
            total = len(pieces)
            for idx, piece in enumerate(pieces):
                clone = _clone_with_text(doc, piece, next_chunk_id, idx, total)
                next_chunk_id += 1
                if clone is None:
                    continue
                total_tokens += _legacy_approx_tokens(piece)
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
        modality_breakdown=modality_breakdown,
        latency=round(time.time() - start, 3),
    )
    return output


# ── Core recursive splitter ───────────────────────────────────────────────────

def _recursive_split(text: str, target: int) -> List[str]:
    if approx_tokens(text) <= target:
        return [text]
    for sep in _FINANCE_SEPARATORS:
        parts = text.split(sep)
        if len(parts) > 1:
            return _pack(parts, sep, target)
    # Last resort: split at the midpoint word boundary.
    mid = len(text) // 2
    sp = text.rfind(" ", 0, mid)
    cut = sp if sp > 0 else mid
    return _recursive_split(text[:cut], target) + _recursive_split(text[cut:], target)


def _pack(parts: List[str], sep: str, target: int) -> List[str]:
    """Greedily accumulate parts into token-bounded windows."""
    chunks: List[str] = []
    buf = ""
    for p in parts:
        candidate = (buf + sep + p) if buf else p
        if approx_tokens(candidate) <= target:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            if approx_tokens(p) > target:
                chunks.extend(_recursive_split(p, target))
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks
