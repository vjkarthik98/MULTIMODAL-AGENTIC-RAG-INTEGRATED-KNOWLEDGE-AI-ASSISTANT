"""
Base class for all per-modality ingestors.

Responsibility contract:
  - extract() does: file I/O, format parsing, security gates on content, returns List[RawExtract]
  - extract() does NOT: chunk splitting, embeddings, ML inference (captioning/OCR/ASR)
  - Security gates on file-level (path traversal, malware scan) stay in router.py

Phase 1 of the MAGIK upgrade plan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from app.ingestion.schema import RawExtract, UniversalMetadata


class BaseIngestor(ABC):
    """Abstract base for all per-modality ingestors."""

    @abstractmethod
    async def extract(
        self,
        path: Path,
        metadata: UniversalMetadata,
    ) -> List[RawExtract]:
        """Extract raw content from file.

        Args:
            path: Resolved, validated file path (security gates already passed).
            metadata: UniversalMetadata populated by router before calling extract.

        Returns:
            List of RawExtract objects — one per logical content unit
            (page, section, speaker turn, frame, etc.).
        """
        ...

    # Shared helper: apply Phase-26 injection guard
    @staticmethod
    def _sanitize(text: str, surface: str) -> str:
        try:
            from app.guardrails.input_guard import sanitize as _g
            return _g(text, surface=surface)
        except Exception:
            return text

    # Shared helper: apply Phase-26 PII scrub.
    # Gated on settings.PII_DETECTION_ENABLED (same flag _redact_pii honors) so the
    # spaCy/Presidio NER pass (~1.4s/doc) is skipped when PII detection is disabled.
    # Set PII_DETECTION_ENABLED=true in .env to re-enable ingest-time PII scrubbing.
    @staticmethod
    def _scrub_pii(text: str, surface: str) -> str:
        from app.core.config import settings
        if not getattr(settings, "PII_DETECTION_ENABLED", False):
            return text
        try:
            from app.guardrails.pii import scrub_pii
            clean, _ = scrub_pii(text)
            return clean
        except Exception:
            return text
