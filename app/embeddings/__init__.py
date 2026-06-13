"""Embeddings package — per-modality dispatch entry point."""
from __future__ import annotations

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-loaded so heavy imports only happen when the embedder is first used.
_EMBEDDER_MAP: dict | None = None


def _get_embedder_map() -> dict:
    global _EMBEDDER_MAP
    if _EMBEDDER_MAP is None:
        from app.embeddings.txt_embedder import TxtEmbedder
        from app.embeddings.pdf_embedder import PdfEmbedder
        from app.embeddings.docx_embedder import DocxEmbedder
        from app.embeddings.xlsx_embedder import XlsxEmbedder
        from app.embeddings.image_embedder import ImageDocEmbedder
        from app.embeddings.audio_embedder import AudioEmbedder
        from app.embeddings.video_embedder import VideoEmbedder
        _EMBEDDER_MAP = {
            "text":  TxtEmbedder,
            "txt":   TxtEmbedder,
            "pdf":   PdfEmbedder,
            "word":  DocxEmbedder,
            "docx":  DocxEmbedder,
            "excel": XlsxEmbedder,
            "xlsx":  XlsxEmbedder,
            "image": ImageDocEmbedder,
            "audio": AudioEmbedder,
            "mp3":   AudioEmbedder,
            "video": VideoEmbedder,
            "mp4":   VideoEmbedder,
        }
    return _EMBEDDER_MAP


def get_embedder(modality: str):
    """Return the per-modality embedder instance. Falls back to TxtEmbedder."""
    cls = _get_embedder_map().get(modality)
    if cls is None:
        logger.warning(event="embedder_unknown_modality", modality=modality)
        from app.embeddings.txt_embedder import TxtEmbedder
        cls = TxtEmbedder
    return cls()
