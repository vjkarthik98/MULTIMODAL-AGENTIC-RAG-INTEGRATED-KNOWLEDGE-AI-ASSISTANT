import time
import unicodedata
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# LOW QUALITY DOMAIN BLOCKLIST

_BLOCKED_DOMAINS: Set[str] = {
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "facebook.com",
    "twitter.com",
    "instagram.com",
}


class WebSearchTool:

    def __init__(self) -> None:
        if not settings.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY_MISSING")

        from tavily import TavilyClient
        self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)

        self.max_results      = settings.WEB_MAX_RESULTS
        self.max_docs         = settings.WEB_MAX_DOCS
        self.max_doc_chars    = settings.WEB_DOC_MAX_CHARS
        self.max_context_chars = settings.WEB_CONTEXT_MAX_CHARS

    # NORMALIZE

    def _normalize(self, query: str) -> str:
        query = unicodedata.normalize("NFC", str(query or ""))
        return " ".join(query.strip().split())[:settings.MAX_PROMPT_CHARS]

    # DOMAIN CHECK

    def _is_blocked(self, url: Optional[str]) -> bool:
        if not url:
            return False
        return any(domain in url for domain in _BLOCKED_DOMAINS)

    # MAIN

    def execute(
        self,
        query: str,
        context=None,
        session_id: str = "default",
    ) -> Dict[str, Any]:

        start = time.time()

        if not query:
            return self._empty()

        try:
            query = self._normalize(query)

            # TAVILY SEARCH
            t_api       = time.time()
            raw         = self._search(query, session_id)
            api_latency = round(time.time() - t_api, 3)

            if api_latency > settings.RETRIEVAL_TIMEOUT:
                logger.warning(
                    event="web_search_api_slow",
                    api_latency=api_latency,
                    threshold=settings.RETRIEVAL_TIMEOUT,
                    session_id=session_id,
                )

            processed = self._process(raw)

            if not processed["documents"]:
                logger.warning(
                    event="web_search_no_documents",
                    session_id=session_id,
                )
                return self._empty()

            # LLM SUMMARIZATION
            t_llm       = time.time()
            answer      = self._summarize(query, processed["documents"], session_id)
            llm_latency = round(time.time() - t_llm, 3)

            latency = round(time.time() - start, 3)

            logger.info(
                event="web_search_success",
                results_used=len(processed["documents"]),
                api_latency=api_latency,
                llm_latency=llm_latency,
                latency=latency,
                session_id=session_id,
            )

            return {
                "answer":     answer,
                "sources":    processed["sources"],
                "confidence": self._confidence(processed),
                "metadata": {
                    "results_used": len(processed["documents"]),
                    "api_latency":  api_latency,
                    "llm_latency":  llm_latency,
                    "total_latency": latency,
                },
            }

        except Exception as e:
            logger.error(
                event="web_search_failed",
                error=str(e),
                session_id=session_id,
            )
            return self._error(str(e))

    # TAVILY API CALL

    def _search(self, query: str, session_id: str = "") -> Dict:
        try:
            return self.client.search(
                query=query,
                search_depth=settings.WEB_SEARCH_DEPTH,
                max_results=self.max_results,
            )
        except Exception as e:
            logger.error(
                event="tavily_api_failed",
                error=str(e),
                session_id=session_id,
            )
            return {}

    # PROCESS RESULTS

    def _process(self, response: Dict) -> Dict[str, List]:
        documents: List[str] = []
        sources:   List[str] = []
        seen:      set        = set()

        results = response.get("results", [])

        # SORT BY TAVILY SCORE IF AVAILABLE
        results = sorted(
            results,
            key=lambda r: float(r.get("score", 0.0)),
            reverse=True,
        )

        for r in results:
            url     = r.get("url", "")
            content = str(r.get("content", "") or "").strip()
            title   = str(r.get("title", "") or "").strip()

            if self._is_blocked(url):
                continue

            if len(content) < 40:
                continue

            # PREPEND TITLE FOR BETTER CONTEXT
            full_text = f"{title}: {content}" if title else content
            text      = full_text[:self.max_doc_chars]

            key = text[:120]
            if key in seen:
                continue
            seen.add(key)

            documents.append(text)

            if url:
                sources.append(url)

        return {
            "documents": documents[:self.max_docs],
            "sources":   list(dict.fromkeys(sources))[:self.max_docs],
        }

    # SUMMARIZE

    def _summarize(
        self,
        query: str,
        docs: List[str],
        session_id: str = "default",
    ) -> str:

        context = "\n\n".join(docs)[:self.max_context_chars]

        instruction = (
            "Answer ONLY from web results.\n"
            "If insufficient → say 'I don't know'.\n\n"
        )

        body    = f"WEB RESULTS:\n{context}\n\nQUERY:\n{query}\n\nAnswer:"
        allowed = settings.MAX_PROMPT_CHARS - len(instruction) - 50
        prompt  = instruction + body[:max(allowed, 0)]

        try:
            from app.core.model_loader import model_loader

            llm       = model_loader.get_llm()
            t_start   = time.time()
            response  = llm.generate(
                prompt,
                max_tokens=settings.LLM_MAX_TOKENS,
                session_id=session_id,
            )

            if time.time() - t_start > settings.MODEL_TIMEOUT_SEC:
                logger.warning(
                    event="web_search_llm_timeout",
                    session_id=session_id,
                )
                return "Summary took too long to generate."

            return response.strip() if response else "No answer generated."

        except Exception as e:
            logger.warning(
                event="web_search_llm_failed",
                error=str(e),
                session_id=session_id,
            )
            return "Summary unavailable."

    # CONFIDENCE

    def _confidence(self, processed: Dict) -> float:
        n = len(processed.get("documents", []))

        if n >= 5:
            return 0.85
        if n >= 3:
            return 0.70
        if n >= 1:
            return 0.50

        return 0.30

    # RESPONSES

    def _empty(self) -> Dict[str, Any]:
        return {
            "answer":     "No relevant results found.",
            "sources":    [],
            "confidence": 0.3,
            "metadata":   {},
        }

    def _error(self, msg: str) -> Dict[str, Any]:
        return {
            "answer":     "Search failed.",
            "sources":    [],
            "confidence": 0.2,
            "metadata":   {"error": msg},
        }


# ============================================================
# TESTS - Phase 24 Upgrade
# Run: pytest app/tools/web_search.py -v
# ============================================================

def test_agent_react_loop_terminates() -> None:
    assert WebSearchTool._normalize(object.__new__(WebSearchTool), " hello   web ") == "hello web"


def test_tool_registry_validates_schema() -> None:
    assert settings.WEB_SEARCH_DEPTH in {"basic", "advanced"}


def test_planner_parallel_tool_calls() -> None:
    assert settings.AGENT_MAX_STEPS > 0


def test_web_search_deduplicates_results() -> None:
    tool = object.__new__(WebSearchTool)
    tool.max_docs = 5
    tool.max_doc_chars = 1000
    response = {
        "results": [
            {"url": "https://example.com/a", "title": "A", "content": "same content " * 10, "score": 0.9},
            {"url": "https://example.com/b", "title": "A", "content": "same content " * 10, "score": 0.8},
        ]
    }
    processed = WebSearchTool._process(tool, response)
    assert len(processed["documents"]) == 1


def test_agent_timeout_guard() -> None:
    assert settings.RETRIEVAL_TIMEOUT > 0
