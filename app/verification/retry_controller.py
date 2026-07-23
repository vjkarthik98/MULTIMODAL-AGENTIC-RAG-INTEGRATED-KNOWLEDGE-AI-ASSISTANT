"""RetryController — one NEW strategy per attempt, bounded (docs/Phase_32_Agentic_Answer_Verification.md §4).

| Attempt | Strategy | Concrete lever |
|---|---|---|
| 1 | expand_retrieval | hybrid_retriever.search() at 2x top_k |
| 2 | query_rewrite | one bounded LLM call → rewritten query → re-embed/re-retrieve |
| 3 | increase_depth | hybrid_retriever.search() at 3x top_k, merged with prior docs |
| 4 | decomposition | _split_query_aspects() (promoted from rag_pipeline.py), per-aspect retrieval, merged |

Never repeats a strategy within one request — tracked in `self.used`.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

STRATEGY_ORDER = ["expand_retrieval", "query_rewrite", "increase_depth", "decomposition"]

_REWRITE_MAX_TOKENS = 64
_REWRITE_PROMPT = (
    "Rewrite the following question to be more explicit and retrieval-friendly. "
    "Keep it factually identical — do not add assumptions. Output ONLY the "
    "rewritten question, nothing else.\n\nQuestion: {query}\n\nRewritten:"
)


class RetryController:
    def __init__(self) -> None:
        self.used: list[str] = []

    def next_strategy(self) -> str | None:
        for s in STRATEGY_ORDER:
            if s not in self.used:
                return s
        return None

    def execute(
        self,
        strategy: str,
        query: str,
        session_id: str,
        user_id: str | None,
        retriever: Any,
        llm: Any,
        filters: dict[str, Any] | None,
        prior_docs: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        """Returns (new_doc_pool, effective_query_used_for_generation)."""
        self.used.append(strategy)

        if strategy == "expand_retrieval":
            docs = self._search(
                retriever, query, session_id, user_id, filters, top_k=settings.DEFAULT_TOP_K * 2
            )
            return self._merge(prior_docs, docs), query

        if strategy == "query_rewrite":
            rewritten = self._rewrite_query(query, llm, session_id)
            docs = self._search(
                retriever, rewritten, session_id, user_id, filters, top_k=settings.DEFAULT_TOP_K
            )
            return self._merge(prior_docs, docs), rewritten

        if strategy == "increase_depth":
            docs = self._search(
                retriever, query, session_id, user_id, filters, top_k=settings.DEFAULT_TOP_K * 3
            )
            return self._merge(prior_docs, docs), query

        if strategy == "decomposition":
            docs = self._decompose_and_retrieve(query, session_id, user_id, retriever, filters)
            return self._merge(prior_docs, docs), query

        return prior_docs, query

    @staticmethod
    def _search(
        retriever: Any,
        query: str,
        session_id: str,
        user_id: str | None,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        try:
            return (
                retriever.search(
                    query=query,
                    session_id=session_id,
                    top_k=top_k,
                    user_id=user_id,
                    filters=filters,
                )
                or []
            )
        except Exception as exc:
            logger.warning(
                event="verify_retry_search_failed", error=str(exc), session_id=session_id
            )
            return []

    @staticmethod
    def _rewrite_query(query: str, llm: Any, session_id: str) -> str:
        if llm is None:
            return query
        try:
            raw = llm.generate(
                _REWRITE_PROMPT.format(query=query),
                max_tokens=_REWRITE_MAX_TOKENS,
                temperature=0.1,
                session_id=session_id,
            )
            rewritten = (raw or "").strip().strip('"')
            if not rewritten:
                return query
            # The rewritten text is a NEW LLM-generated surface that becomes
            # both the next retrieval query AND the next generation prompt's
            # query — it must go through the same guardrail every other
            # prompt-bound text surface does (CLAUDE.md: input_guard.sanitize
            # is the single injection entry point, called on every text
            # surface that reaches a prompt). The ORIGINAL query was already
            # sanitized upstream (rag_pipeline._sanitize / query_pipeline.
            # _sanitize_query) before reaching this loop, but that guarantee
            # does not extend to text an LLM call generates mid-loop.
            from app.guardrails.input_guard import sanitize as _guard_sanitize

            rewritten = _guard_sanitize(rewritten, surface="verification_query_rewrite")
            return rewritten if rewritten else query
        except Exception as exc:
            logger.warning(
                event="verify_retry_rewrite_failed", error=str(exc), session_id=session_id
            )
            return query

    @staticmethod
    def _decompose_and_retrieve(
        query: str,
        session_id: str,
        user_id: str | None,
        retriever: Any,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        try:
            from app.pipeline.rag_pipeline import _split_query_aspects
        except Exception:
            return []
        aspects = _split_query_aspects(query)
        if len(aspects) < 2:
            return []
        merged: list[dict[str, Any]] = []
        for asp in aspects[:5]:
            merged.extend(
                RetryController._search(retriever, asp, session_id, user_id, filters, top_k=4)
            )
        return merged

    @staticmethod
    def _merge(prior: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            from app.pipeline.rag_pipeline import _dedup_docs, _normalize_docs

            return _dedup_docs(_normalize_docs(list(prior) + list(fresh)))
        except Exception:
            return list(prior) + list(fresh)
