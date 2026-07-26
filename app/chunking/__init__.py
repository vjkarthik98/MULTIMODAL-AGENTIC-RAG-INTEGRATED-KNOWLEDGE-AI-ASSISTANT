"""Chunking package — per-modality dispatch entry point."""

from __future__ import annotations

from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-loaded so heavy imports (whisper, cv2, etc.) only happen when needed.
_CHUNKER_MAP: dict | None = None


def _get_chunker_map() -> dict:
    global _CHUNKER_MAP
    if _CHUNKER_MAP is None:
        from app.chunking.audio_chunker import AudioChunker
        from app.chunking.docx_chunker import DocxChunker
        from app.chunking.image_chunker import ImageChunker
        from app.chunking.pdf_chunker import PdfChunker
        from app.chunking.txt_chunker import TxtChunker
        from app.chunking.video_chunker import VideoChunker
        from app.chunking.xlsx_chunker import XlsxChunker

        _CHUNKER_MAP = {
            "text": TxtChunker,
            "txt": TxtChunker,
            "pdf": PdfChunker,
            "word": DocxChunker,
            "docx": DocxChunker,
            "excel": XlsxChunker,
            "xlsx": XlsxChunker,
            "image": ImageChunker,
            "audio": AudioChunker,
            "mp3": AudioChunker,
            "video": VideoChunker,
            "mp4": VideoChunker,
        }
    return _CHUNKER_MAP


def chunk_raw_extracts(extracts: list[Any], meta: Any, modality: str) -> list[Any]:
    """Route List[RawExtract] through the correct per-modality chunker.

    Each modality's chunker is isolated — an error in one modality cannot
    affect another modality's chunks.
    """
    chunker_cls = _get_chunker_map().get(modality)
    if chunker_cls is None:
        logger.warning(event="chunker_unknown_modality", modality=modality)
        return []
    return chunker_cls().chunk(extracts, meta)


# Re-export legacy function so any existing caller of
# `from app.chunking import chunk_documents` keeps working.
from app.chunking.base_chunker import chunk_documents  # noqa: E402,F401
