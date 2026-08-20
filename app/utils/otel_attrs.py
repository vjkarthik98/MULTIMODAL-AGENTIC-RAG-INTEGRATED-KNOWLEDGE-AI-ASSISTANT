"""OpenInference semantic-convention helpers for span enrichment (monitoring
Phase 4: Arize Phoenix).

Centralizes the flattened-document-attribute pattern so the span wrappers in
query_pipeline.py / rag_pipeline.py / hybrid_retriever.py / reranker.py
(all added in Phase 0 — see those files' own OTel span docstrings) don't
each hand-roll it slightly differently. This module adds ATTRIBUTES to
spans those files already open; it does not open spans itself and does not
change what gets exported to Tempo — Phoenix and Tempo read the same spans
(see monitoring/otel/collector-config.yaml's dual otlp/tempo + otlp/phoenix
exporters), these attributes just give Phoenix's RAG-native UI (retrieved
documents, per-span input/output) something to render.

200-char truncation on every text field — same limit app/eval/jobs/
shadow_sampler.py already uses for context snippets, and for the same
reason: enough to recognize the query/chunk, not a duplicate copy of the
corpus/conversation sitting in a second, less-access-controlled store
(Tempo/Phoenix are both self-hosted on this same box, but still a second
place the data lives — see the monitoring layer's privacy rule against
becoming a data-leak vector).

Every function here swallows its own exceptions. A telemetry-attribute
failure must never taint the span status of the real operation it's
decorating — the span wrappers' own try/except (which DOES set
Status.ERROR) exists for failures in the actual retrieval/generation work,
not for a bug in this module.
"""

from __future__ import annotations

from typing import Any

_SNIPPET_CHARS = 200
_MAX_DOCUMENTS = (
    10  # cap attribute count — 200 chunks attached to one span is heavier, not more useful
)


def set_span_kind(span: Any, kind: str) -> None:
    try:
        from openinference.semconv.trace import SpanAttributes

        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind)
    except Exception:
        pass


def set_input_output(
    span: Any, *, input_value: str | None = None, output_value: str | None = None
) -> None:
    try:
        from openinference.semconv.trace import SpanAttributes

        if input_value:
            span.set_attribute(SpanAttributes.INPUT_VALUE, str(input_value)[:_SNIPPET_CHARS])
        if output_value:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(output_value)[:_SNIPPET_CHARS])
    except Exception:
        pass


def set_retrieval_documents(span: Any, documents: list[dict[str, Any]] | None) -> None:
    """Attach a flattened OpenInference document list. `documents` is
    whatever shape the caller already has — either a raw HybridRetriever/
    Reranker result list (`text`/`score`/`metadata.source`) or the final
    built sources array (`text`/`score`/`source`) — read defensively since
    the two shapes differ slightly and this must never raise either way."""
    try:
        from openinference.semconv.trace import DocumentAttributes, SpanAttributes

        for i, doc in enumerate((documents or [])[:_MAX_DOCUMENTS]):
            if not isinstance(doc, dict):
                continue
            prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}."

            text = doc.get("text") or doc.get("snippet") or ""
            if text:
                span.set_attribute(
                    prefix + DocumentAttributes.DOCUMENT_CONTENT, str(text)[:_SNIPPET_CHARS]
                )

            score = doc.get("score")
            if score is None:
                score = doc.get("final_score")
            if isinstance(score, (int, float)):
                span.set_attribute(prefix + DocumentAttributes.DOCUMENT_SCORE, float(score))

            meta = doc.get("metadata") or {}
            source = meta.get("source") or doc.get("source")
            if source:
                span.set_attribute(prefix + DocumentAttributes.DOCUMENT_ID, str(source))
    except Exception:
        pass
