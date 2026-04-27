import time
from typing import Dict, Any, List

from tavily import TavilyClient

from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger


logger = get_logger(__name__)


class WebSearchTool:

    def __init__(self):
        logger.info("[WebSearchTool] initializing")

        if not settings.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY missing in config")

        self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)

        self.max_results = settings.WEB_MAX_RESULTS
        self.max_docs = settings.WEB_MAX_DOCS
        self.max_doc_chars = settings.WEB_DOC_MAX_CHARS
        self.max_context_chars = settings.WEB_CONTEXT_MAX_CHARS

        logger.info("[WebSearchTool] initialized")

    # MAIN 
    def execute(
        self,
        query: str,
        context: Dict[str, Any] = None,
        session_id: str = None
    ) -> Dict[str, Any]:

        start = time.time()

        if not query or not query.strip():
            return self._empty_response()

        try:
            query = self._sanitize(query)

            logger.info("[WebSearchTool] search started")

            raw = self._search_api(query)
            processed = self._process_results(raw)

            if not processed["documents"]:
                return self._empty_response()

            answer = self._summarize(query, processed["documents"])

            latency = round(time.time() - start, 2)

            return {
                "answer": answer,
                "sources": processed["sources"],
                "confidence": 0.7,
                "metadata": {
                    "results_used": len(processed["documents"]),
                    "latency": latency
                }
            }

        except Exception as e:
            logger.error("[WebSearchTool] failed | %s", str(e))

            return {
                "answer": "Search failed. Please try again.",
                "sources": [],
                "confidence": 0.2,
                "metadata": {"error": str(e)}
            }

    # SANITIZE 
    def _sanitize(self, query: str) -> str:
        query = query.strip()

        if len(query) > settings.MAX_PROMPT_CHARS:
            query = query[:settings.MAX_PROMPT_CHARS]

        return query

    # API 
    def _search_api(self, query: str) -> Dict:

        return self.client.search(
            query=query,
            search_depth=settings.WEB_SEARCH_DEPTH,
            max_results=self.max_results
        )

    # PROCESS 
    def _process_results(self, response: Dict) -> Dict[str, List]:

        documents = []
        sources = []

        for r in response.get("results", []):

            content = r.get("content")
            url = r.get("url")

            if not content:
                continue

            text = str(content)[:self.max_doc_chars]

            documents.append(text)

            if url:
                sources.append(url)

        return {
            "documents": documents[:self.max_docs],
            "sources": list(set(sources))[:self.max_docs]
        }

    # SUMMARIZE 
    def _summarize(self, query: str, docs: List[str]) -> str:

        combined = "\n\n".join(docs)

        # Truncate context
        combined = combined[:self.max_context_chars]

        prompt = (
            "You are a highly accurate AI assistant.\n\n"
            "Use ONLY the provided web results.\n"
            "Do not hallucinate.\n\n"

            "WEB RESULTS:\n"
            f"{combined}\n\n"

            "QUERY:\n"
            f"{query}\n\n"

            "Return:\n"
            "- Clear answer\n"
            "- Fact-based\n"
            "- Concise\n\n"

            "Answer:"
        )

        # Global safety
        if len(prompt) > settings.MAX_PROMPT_CHARS:
            logger.warning("[WebSearchTool] prompt truncated")
            prompt = prompt[-settings.MAX_PROMPT_CHARS:]

        try:
            llm = model_loader.get_llm()

            response = llm.generate(prompt)

        except Exception as e:
            logger.warning("[WebSearchTool] LLM unavailable | %s", str(e))

            return "Summary unavailable due to model error"

    # EMPTY 
    def _empty_response(self) -> Dict[str, Any]:

        logger.warning("[WebSearchTool] no results")

        return {
            "answer": "No relevant search results found.",
            "sources": [],
            "confidence": 0.3,
            "metadata": {}
        }