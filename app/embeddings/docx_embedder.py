"""docx_embedder.py — Finance-grade embedder for DOCX chunks."""
from __future__ import annotations

from typing import Any

from app.embeddings.base_embedder import BaseEmbedder
from app.core.config import settings
from app.utils.logger import get_logger
from prometheus_client import Counter

logger = get_logger(__name__)

_EMBED_BUILT = Counter(
    "magik_docx_embed_text_built_total",
    "Embed texts successfully built for docx",
)
_EMBED_ERRORS = Counter(
    "magik_docx_embed_text_errors_total",
    "Errors building embed text for docx",
)


class DocxEmbedder(BaseEmbedder):
    """Embedder for DOCX chunks (contracts, investor letters, board decks).

    Enrichment:
      Section: {h1} > {h2} > {h3} | {text}

    Defined-term amplification: if the chunk contains defined terms (bold text
    followed by "means" / "shall mean"), repeat the term keys so clause-number
    queries can reach them.
    """

    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        try:
            s = getattr(doc, "structure", {}) or {}

            # Full heading hierarchy → "1. Financial Statements > 1.2 Balance Sheet"
            hierarchy = s.get("heading_hierarchy") or s.get("section_hierarchy") or []
            if hierarchy:
                hier_str = " > ".join(str(h) for h in hierarchy if h)
                if hier_str:
                    text = f"Section: {hier_str} | {cleaned_text}"
                else:
                    text = cleaned_text
            else:
                heading = (s.get("heading") or s.get("section_title") or "").strip()
                text    = (f"Section: {heading} | {cleaned_text}") if heading else cleaned_text

            # Defined-term amplification — legal finance docs use precise clause names
            defined_terms: dict = s.get("defined_terms") or {}
            if defined_terms:
                term_tokens = " ".join(str(k) for k in list(defined_terms.keys())[:10])
                text = f"{text} [Terms: {term_tokens}]"

            result = text[:settings.MAX_PROMPT_CHARS]
            logger.debug(event="embed_text_built", modality="docx", chars=len(result))
            _EMBED_BUILT.inc()
            return result
        except Exception as _exc:
            _EMBED_ERRORS.inc()
            logger.error(event="embed_text_build_failed", modality="docx", error=str(_exc))
            return cleaned_text  # safe fallback to unenriched text

    def health_check(self) -> dict:
        return {
            "modality": "docx",
            "status": "ok",
            "class": self.__class__.__name__,
        }
