import time
from typing import Dict, Any, List

from tavily import TavilyClient

from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchTool:

    def __init__(self):

        if not settings.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY_MISSING")

        self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)

        self.max_results = settings.WEB_MAX_RESULTS
        self.max_docs = settings.WEB_MAX_DOCS
        self.max_doc_chars = settings.WEB_DOC_MAX_CHARS
        self.max_context_chars = settings.WEB_CONTEXT_MAX_CHARS

    #  MAIN 
    def execute(self, query: str, context=None, session_id=None) -> Dict[str, Any]:

        start = time.time()

        if not query:
            return self._empty()

        try:
            query = self._normalize(query)

            #  SEARCH 
            t_api = time.time()
            raw = self._search(query)
            api_latency = round(time.time() - t_api, 3)

            processed = self._process(raw)

            if not processed["documents"]:
                return self._empty()

            #  SUMMARIZE 
            t_llm = time.time()
            answer = self._summarize(query, processed["documents"])
            llm_latency = round(time.time() - t_llm, 3)

            latency = round(time.time() - start, 3)

            return {
                "answer": answer,
                "sources": processed["sources"],
                "confidence": self._confidence(processed),
                "metadata": {
                    "results_used": len(processed["documents"]),
                    "api_latency": api_latency,
                    "llm_latency": llm_latency,
                    "total_latency": latency
                }
            }

        except Exception as e:
            logger.error(event="web_search_failed", error=str(e))
            return self._error(str(e))

    #  NORMALIZE 
    def _normalize(self, query: str) -> str:
        return " ".join(query.strip().split())[:settings.MAX_PROMPT_CHARS]

    #  API 
    def _search(self, query: str) -> Dict:

        try:
            return self.client.search(
                query=query,
                search_depth=settings.WEB_SEARCH_DEPTH,
                max_results=self.max_results
            )
        except Exception as e:
            logger.error(event="tavily_error", error=str(e))
            return {}

    #  PROCESS 
    def _process(self, response: Dict) -> Dict[str, List]:

        documents = []
        sources = []
        seen = set()

        for r in response.get("results", []):

            content = str(r.get("content", "")).strip()
            url = r.get("url")

            if len(content) < 40:
                continue

            text = content[:self.max_doc_chars]

            key = text[:120]
            if key in seen:
                continue
            seen.add(key)

            documents.append(text)

            if url:
                sources.append(url)

        return {
            "documents": documents[:self.max_docs],
            "sources": list(set(sources))[:self.max_docs]
        }

    #  SUMMARIZE 
    def _summarize(self, query: str, docs: List[str]) -> str:

        context = "\n\n".join(docs)[:self.max_context_chars]

        instruction = (
            "Answer ONLY from web results.\n"
            "If insufficient → say 'I don't know'.\n\n"
        )

        body = f"WEB RESULTS:\n{context}\n\nQUERY:\n{query}\n\nAnswer:"

        max_chars = settings.MAX_PROMPT_CHARS
        allowed = max_chars - len(instruction) - 50

        prompt = instruction + body[:allowed]

        try:
            llm = model_loader.get_llm()
            response = llm.generate(prompt)

            return response.strip() if response else "No answer generated."

        except Exception as e:
            logger.warning(event="web_llm_failed", error=str(e))
            return "Summary unavailable."

    #  CONFIDENCE 
    def _confidence(self, processed: Dict) -> float:

        n = len(processed.get("documents", []))

        if n >= 5:
            return 0.85
        if n >= 3:
            return 0.7
        if n >= 1:
            return 0.5

        return 0.3

    #  RESPONSES 
    def _empty(self) -> Dict[str, Any]:
        return {
            "answer": "No relevant results found.",
            "sources": [],
            "confidence": 0.3,
            "metadata": {}
        }

    def _error(self, msg: str) -> Dict[str, Any]:
        return {
            "answer": "Search failed.",
            "sources": [],
            "confidence": 0.2,
            "metadata": {"error": msg}
        }