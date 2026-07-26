"""docx_bm25.py — BM25 index for Word document chunks."""

from __future__ import annotations

from typing import Any

from prometheus_client import Counter

from app.bm25.base_bm25 import BaseBM25
from app.utils.logger import get_logger

logger = get_logger(__name__)

_BM25_INDEXED = Counter(
    "magik_docx_bm25_indexed_total",
    "Documents indexed in docx BM25",
)
_BM25_INDEX_ERRORS = Counter(
    "magik_docx_bm25_index_errors_total",
    "Errors in docx BM25 _build_indexed_text",
)


class DocxBM25(BaseBM25):
    """BM25 index for Word/DOCX documents (contracts, board minutes, legal filings).

    Enrichment over base:
    - Full heading hierarchy as tokens
    - Defined-term keys indexed ×2 (exact term queries → definition clause)
    - Clause number tokens ("Section 4.3(b)(ii)" → "section 4 3 b ii")
    - Finance entity amplification
    - Heading level token for heading-weight boosting
    - Separate definitions sub-index (bm25_docx_definitions.pkl) for
      "what is the definition of X" queries
    """

    modality = "docx"

    def __init__(self) -> None:
        super().__init__()
        # Separate small BM25 for definition chunks only
        self._def_docs: list[Any] = []
        self._def_index: Any | None = None

    def _build_indexed_text(self, doc: Any) -> str:
        try:
            s = getattr(doc, "structure", {}) or {}
            parts: list[str] = list(self._base_text(doc))

            # Full heading hierarchy
            hierarchy: list[str] = s.get("heading_hierarchy") or []
            if hierarchy:
                parts.append(" ".join(str(h) for h in hierarchy))

            # Heading level token ("heading level 2" helps weight heading chunks)
            heading_level = s.get("heading_level") or getattr(doc, "heading_level", None)
            if heading_level is not None:
                parts.append(f"heading level {heading_level}")

            # Defined terms: "adjusted ebitda" → indexed as defined term for retrieval
            defined_terms: dict[str, str] = s.get("defined_terms") or {}
            for term in list(defined_terms.keys())[:10]:
                parts.append(str(term))
                parts.append(str(term))  # amplify ×2 — defined terms are retrieval anchors

            # Clause number indexing: "Section 4.3(b)(ii)" → "section 4 3 b ii"
            clause_numbers: list[str] = s.get("clause_numbers") or []
            for cn in clause_numbers[:5]:
                clause_tokens = cn.replace(".", " ").replace("(", " ").replace(")", " ").lower()
                parts.append(clause_tokens)

            # Finance entities amplification (List[str] from extract_finance_entities)
            fin_entities: list[str] = s.get("finance_entities") or []
            if isinstance(fin_entities, list):
                parts.extend(str(x) for x in fin_entities[:5])

            # Table heading if table chunk
            chunk_type = (s.get("chunk_type") or "").lower()
            if "table" in chunk_type:
                table_title = (
                    s.get("table_title") or s.get("section_title") or s.get("heading") or ""
                ).strip()
                if table_title:
                    parts.append(table_title)
                    parts.append(table_title)

            _BM25_INDEXED.inc()
            return " ".join(parts)
        except Exception as _exc:
            _BM25_INDEX_ERRORS.inc()
            logger.warning(
                event="bm25_index_text_failed",
                modality=self.modality,
                error=str(_exc),
            )
            return getattr(doc, "text", "") or ""

    def add_documents(self, docs: list[Any], user_id: str | None = None) -> None:
        """Index documents; route definition chunks to the definitions sub-index too."""
        super().add_documents(docs, user_id=user_id)
        def_docs = [
            d
            for d in docs
            if (getattr(d, "structure", {}) or {}).get("chunk_type")
            in ("definitions", "definition")
        ]
        if def_docs:
            self._def_docs.extend(def_docs)
            try:
                from rank_bm25 import BM25Okapi

                tokenized = [self.tokenize(self._build_definitions_text(d)) for d in self._def_docs]
                self._def_index = BM25Okapi(tokenized) if tokenized else None
            except Exception as exc:
                logger.warning(event="docx_def_index_rebuild_failed", error=str(exc))

    def _build_definitions_text(self, doc: Any) -> str:
        """Build indexed text for a definition chunk: term keys + definition text."""
        s = getattr(doc, "structure", {}) or {}
        defined_terms: dict[str, str] = s.get("defined_terms") or {}
        parts = [getattr(doc, "text", "") or ""]
        for term, meaning in list(defined_terms.items())[:20]:
            parts.append(f"{term} means {meaning}")
        return " ".join(parts)

    def search_definitions(self, query: str, top_k: int = 5) -> list[Any]:
        """Search the definitions sub-index. Falls back to main index if empty."""
        if self._def_index is None or not self._def_docs:
            return self.search(query, top_k=top_k)
        try:
            tokens = self.tokenize(query)
            scores = self._def_index.get_scores(tokens)
            import heapq

            top_indices = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])
            return [self._def_docs[i] for i in top_indices if scores[i] > 0]
        except Exception as exc:
            logger.warning(event="docx_def_search_failed", error=str(exc))
            return self.search(query, top_k=top_k)

    def health_check(self, user_id=None) -> dict:
        base = super().health_check(user_id)
        base["class"] = self.__class__.__name__
        base["definitions_indexed"] = len(self._def_docs)
        return base
