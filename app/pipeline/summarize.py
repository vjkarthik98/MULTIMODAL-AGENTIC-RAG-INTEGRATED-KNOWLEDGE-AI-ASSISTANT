from __future__ import annotations

import time
from typing import Any

from app.core.config import settings
from app.core.infra_registry import infra
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Leaves headroom in MAX_PROMPT_CHARS for the instruction text + generated
# output, so a batch never triggers PromptBuilder-style truncation mid-chunk.
_BATCH_CHAR_BUDGET = int(settings.MAX_PROMPT_CHARS * 0.6)
_PARTIAL_SUMMARY_MAX_TOKENS = 220
_FINAL_SUMMARY_MAX_TOKENS = 420
# Caps worst-case latency/cost on a very large ingested document (a 10-K can
# span 50+ chunks) — the map stage stays bounded rather than growing unbounded
# with document size.
_MAX_BATCHES = 12

_PARTIAL_PROMPT = (
    "You are summarizing PART of a longer document for someone who has not "
    "read it. Summarize ONLY the information in this excerpt — do not "
    "speculate about content not shown. Preserve specific figures, names, "
    "and dates verbatim. Write 3-6 sentences of flowing prose, no bullet "
    "points, no headings.\n\nEXCERPT:\n{chunk_text}\n\nSUMMARY:\n"
)

_FINAL_PROMPT = (
    "You are writing the final summary of a document from partial summaries "
    "of its sections, for someone who has not read it. Combine them into one "
    "coherent overview in natural, conversational prose (not a list of "
    "section recaps) that captures the document's main points, key figures, "
    "and conclusions. Do not repeat the same fact twice. Write 4-8 "
    "sentences.\n\nPARTIAL SUMMARIES:\n{partials}\n\nFINAL SUMMARY:\n"
)


def _batch_chunks(chunks: list[dict[str, Any]], char_budget: int) -> list[str]:
    """Pack chunk texts (already in document order) into batches that each
    fit under char_budget, without splitting a single chunk across batches."""
    batches: list[str] = []
    current: list[str] = []
    current_len = 0
    for c in chunks:
        text = str(c.get("text", "") or "")
        if not text:
            continue
        if current_len + len(text) > char_budget and current:
            batches.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(text)
        current_len += len(text)
    if current:
        batches.append("\n\n".join(current))
    return batches


def summarize_document(source: str, session_id: str, user_id: str) -> str:
    """Whole-document summary for `source` (a filename already resolved by the
    caller — see api_routes.py's _is_summarize_request). Pulls every chunk of
    the file via QdrantVectorStore.get_all_chunks_by_source (original reading
    order, not semantic top-k) and map-reduces over the LLM's small context
    budget for documents too large to summarize in one call.
    """
    start = time.time()

    store = infra.get_vector_store()
    if not store:
        return "The knowledge base is currently unavailable — please try again shortly."

    chunks = store.get_all_chunks_by_source(source, user_id=user_id)
    if not chunks:
        return f'I could not find "{source}" in your knowledge base to summarize.'

    llm = model_loader.get_llm()
    if not llm:
        return "LLM unavailable."

    batches = _batch_chunks(chunks, _BATCH_CHAR_BUDGET)

    if len(batches) == 1:
        prompt = _PARTIAL_PROMPT.format(chunk_text=batches[0])
        summary = llm.generate(
            prompt,
            max_tokens=_FINAL_SUMMARY_MAX_TOKENS,
            temperature=0.2,
            top_p=settings.LLM_TOP_P,
            session_id=session_id,
        )
        logger.info(
            event="summarize_document_single_batch",
            source=source,
            chunks=len(chunks),
            duration_ms=round((time.time() - start) * 1000, 1),
            session_id=session_id,
        )
        return (summary or "").strip() or f'I could not generate a summary for "{source}".'

    # MAP — one partial summary per batch (bounded by _MAX_BATCHES).
    partials: list[str] = []
    for batch in batches[:_MAX_BATCHES]:
        prompt = _PARTIAL_PROMPT.format(chunk_text=batch)
        try:
            partial = llm.generate(
                prompt,
                max_tokens=_PARTIAL_SUMMARY_MAX_TOKENS,
                temperature=0.2,
                top_p=settings.LLM_TOP_P,
                session_id=session_id,
            )
            if partial and partial.strip():
                partials.append(partial.strip())
        except Exception as exc:
            logger.warning(event="summarize_partial_failed", source=source, error=str(exc))

    if not partials:
        return f'I could not generate a summary for "{source}".'

    # REDUCE — combine partial summaries into one final answer.
    combined = "\n\n".join(partials)
    final_prompt = _FINAL_PROMPT.format(partials=combined[: settings.MAX_PROMPT_CHARS - 600])
    try:
        final_summary = llm.generate(
            final_prompt,
            max_tokens=_FINAL_SUMMARY_MAX_TOKENS,
            temperature=0.2,
            top_p=settings.LLM_TOP_P,
            session_id=session_id,
        )
    except Exception as exc:
        logger.warning(event="summarize_final_reduce_failed", source=source, error=str(exc))
        final_summary = None

    logger.info(
        event="summarize_document_map_reduce",
        source=source,
        chunks=len(chunks),
        batches=len(batches),
        partials_used=len(partials),
        duration_ms=round((time.time() - start) * 1000, 1),
        session_id=session_id,
    )

    return (final_summary or "").strip() or combined
