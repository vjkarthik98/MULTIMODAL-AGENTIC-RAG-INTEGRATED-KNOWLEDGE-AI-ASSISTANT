from __future__ import annotations

import time
from typing import Any, Dict, List

from app.utils.logger import get_logger

logger = get_logger(__name__)


# DOCUMENT CHUNKER — CALLED BY INGESTION PIPELINE

def chunk_documents(docs: List[Any]) -> List[Any]:
    """
    Post-ingestion chunk normalization step.
    Documents are already split by the ingestion layer; this step
    assigns chunk_id where missing, estimates token counts, and logs stats.
    """
    start    = time.time()
    output:  List[Any] = []
    skipped  = 0
    total_tokens = 0
    modality_breakdown: Dict[str, int] = {}

    for idx, doc in enumerate(docs):
        text = getattr(doc, "text", None) or ""
        if not text.strip():
            skipped += 1
            continue

        # ASSIGN CHUNK ID IF NOT SET
        if getattr(doc, "chunk_id", None) is None:
            try:
                doc.chunk_id = idx
            except Exception:
                pass

        # ROUGH TOKEN ESTIMATE (words × 1.3)
        total_tokens += int(len(text.split()) * 1.3)

        modality = getattr(doc, "modality", "text") or "text"
        modality_breakdown[modality] = modality_breakdown.get(modality, 0) + 1

        output.append(doc)

    logger.info(
        event="chunking_success",
        input=len(docs),
        output=len(output),
        skipped=skipped,
        total_tokens_est=total_tokens,
        modality_breakdown=modality_breakdown,
        latency=round(time.time() - start, 3),
    )

    return output
