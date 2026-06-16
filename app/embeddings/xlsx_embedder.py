"""xlsx_embedder.py — Finance-grade embedder for Excel chunks.

Most complex modality embedder: produces two embeddings per table chunk —
  doc.embedding      = BGE of the natural-language summary (semantic retrieval)
  doc.embedding_alt  = BGE of the markdown table repr  (structural retrieval)

Phase 5 hybrid retrieval queries both and takes max(score).
"""
from __future__ import annotations

import time
from typing import Any, List

from app.embeddings.base_embedder import (
    BaseEmbedder,
    normalize_finance_numbers,
    sanitize_text,
    valid_embedding,
)
from app.core.config import settings
from app.utils.logger import get_logger
from prometheus_client import Counter

logger = get_logger(__name__)

_EMBED_BUILT = Counter(
    "magik_xlsx_embed_text_built_total",
    "Embed texts successfully built for xlsx",
)
_EMBED_ERRORS = Counter(
    "magik_xlsx_embed_text_errors_total",
    "Errors building embed text for xlsx",
)


class XlsxEmbedder(BaseEmbedder):
    """Embedder for XLSX chunks (income statements, balance sheets, models).

    Primary enrichment:
      Sheet: {sheet_name} | ({unit_scale}) | {NL summary text}

    Secondary (embedding_alt, table chunks only):
      Sheet: {sheet_name} | {markdown_repr}
    """

    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        try:
            s = getattr(doc, "structure", {}) or {}

            sheet      = (s.get("sheet_name") or s.get("sheet") or "").strip()
            unit_scale = (s.get("unit_scale") or "").strip()

            parts = []
            if sheet:
                parts.append(f"Sheet: {sheet}")
            if unit_scale:
                parts.append(f"({unit_scale})")

            prefix = (" | ".join(parts) + " | ") if parts else ""

            # Named-range chunks include assumption names as tokens
            if (s.get("chunk_type") or getattr(doc, "subtype", "")) in ("named_range", "assumptions"):
                named = (s.get("named_range") or s.get("semantic_group") or "").strip()
                if named:
                    prefix = f"Assumptions: {named} | " + prefix

            result = (prefix + cleaned_text)[:settings.MAX_PROMPT_CHARS]
            _EMBED_BUILT.inc()
            return result
        except Exception as _exc:
            _EMBED_ERRORS.inc()
            logger.error(event="embed_text_build_failed", modality="xlsx", error=str(_exc))
            return cleaned_text  # safe fallback to unenriched text

    def _build_alt_embed_text(self, doc: Any) -> str:
        """Build markdown table text for embedding_alt (structural path)."""
        s          = getattr(doc, "structure", {}) or {}
        sheet      = (s.get("sheet_name") or s.get("sheet") or "").strip()
        unit_scale = (s.get("unit_scale") or "").strip()
        markdown   = (s.get("markdown_repr") or "").strip()

        prefix = f"Sheet: {sheet}" if sheet else ""
        if unit_scale:
            prefix += f" | ({unit_scale})"
        if prefix:
            return (prefix + " | " + markdown)[:settings.MAX_PROMPT_CHARS]
        return markdown[:settings.MAX_PROMPT_CHARS]

    def embed_documents(
        self,
        docs: List[Any],
        session_id: str = "default",
    ) -> List[Any]:
        # Step 1: primary NL embedding via parent
        embedded = super().embed_documents(docs, session_id=session_id)

        # Step 2: secondary markdown embedding for table chunks that have markdown_repr
        embedder = self._get_model()
        alt_texts: List[str]  = []
        alt_docs:  List[Any]  = []

        for doc in embedded:
            s        = getattr(doc, "structure", {}) or {}
            markdown = (s.get("markdown_repr") or "").strip()
            if not markdown:
                continue
            alt_raw = self._build_alt_embed_text(doc)
            clean   = sanitize_text(alt_raw)
            if not clean:
                continue
            clean = normalize_finance_numbers(clean)
            alt_texts.append(clean[:settings.MAX_PROMPT_CHARS])
            alt_docs.append(doc)

        if alt_docs:
            try:
                t = time.time()
                for i in range(0, len(alt_texts), embedder.batch_size):
                    batch_texts = alt_texts[i:i + embedder.batch_size]
                    batch_docs  = alt_docs[i:i + embedder.batch_size]
                    embs = embedder._encode_with_retry(embedder.model, batch_texts)
                    for doc, emb in zip(batch_docs, embs):
                        if valid_embedding(emb, embedder.expected_dim):
                            doc.embedding_alt = emb
                logger.info(
                    event="xlsx_alt_embed_done",
                    count=len(alt_docs),
                    latency=round(time.time() - t, 3),
                    session_id=session_id,
                )
            except Exception as exc:
                logger.warning(
                    event="xlsx_alt_embed_failed",
                    error=str(exc),
                    session_id=session_id,
                )

        return embedded

    def health_check(self) -> dict:
        return {
            "modality": "xlsx",
            "status": "ok",
            "class": self.__class__.__name__,
        }
