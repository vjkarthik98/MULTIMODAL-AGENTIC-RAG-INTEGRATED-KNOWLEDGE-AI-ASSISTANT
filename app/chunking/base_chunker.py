from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from app.chunking.finance_numbers import approx_tokens, protect, restore
from app.core.config import settings
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Priority-ordered separators for finance text.
_FINANCE_SEPARATORS = [
    "\n\n\n",
    "\n\nOPERATOR:", "\n\nCEO:", "\n\nCFO:", "\n\nANALYST:",
    "\n\n",
    ". ", "? ", "! ", "; ",
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
