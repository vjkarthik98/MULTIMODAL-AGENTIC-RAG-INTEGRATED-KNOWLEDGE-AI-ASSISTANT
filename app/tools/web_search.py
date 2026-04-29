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

    #  MAIN 
    def execute(self, query: str, context=None, session_id=None) -> Dict[str, Any]:

        start = time.time()

        if not query or not query.strip():
            return self._empty_response()

        try:
            query = self._normalize(query)

            #  SEARCH 
            t_api = time.time()
            raw = self._search_api(query)
            api_latency = round(time.time() - t_api, 2)

            processed = self._process_results(raw)

            if not processed["documents"]:
                return self._empty_response()

            #  SUMMARIZE 
            t_llm = time.time()
            answer = self._summarize(query, processed["documents"])
            llm_latency = round(time.time() - t_llm, 2)

            confidence = self._compute_confidence(processed)

            latency = round(time.time() - start, 2)

            return {
                "answer": answer,
                "sources": processed["sources"],
                "confidence": confidence,
                "metadata": {
                    "results_used": len(processed["documents"]),
                    "api_latency": api_latency,
                    "llm_latency": llm_latency,
                    "total_latency": latency
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

    #  NORMALIZE 
    def _normalize(self, query: str) -> str:
        return " ".join(query.strip().split())[:settings.MAX_PROMPT_CHARS]

    #  API 
    def _search_api(self, query: str) -> Dict:

        return self.client.search(
            query=query,
            search_depth=settings.WEB_SEARCH_DEPTH,
            max_results=self.max_results
        )

    #  PROCESS 
    def _process_results(self, response: Dict) -> Dict[str, List]:

        documents = []
        sources = []
        seen = set()

        for r in response.get("results", []):

            content = r.get("content", "")
            url = r.get("url", "")

            if not content or len(content) < 30:
                continue

            text = content.strip()[:self.max_doc_chars]

            key = text[:100]
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

        combined = "\n\n".join(docs)[:self.max_context_chars]

        instruction = (
            "You are a highly reliable assistant.\n"
            "Answer ONLY from provided web results.\n"
            "If unsure, say 'I don't know'.\n\n"
        )

        body = f"WEB RESULTS:\n{combined}\n\nQUERY:\n{query}\n\n"

        output = "Answer:"

        # SAFE PROMPT BUILD
        max_chars = settings.MAX_PROMPT_CHARS
        available = max_chars - len(instruction) - len(output) - 50

        body = body[:available]

        prompt = instruction + body + output

        try:
            llm = model_loader.get_llm()
            response = llm.generate(prompt)

            return response.strip() if response else "No answer generated."

        except Exception as e:
            logger.warning("[WebSearchTool] LLM failed | %s", str(e))
            return "Summary unavailable due to model error."

    #  CONFIDENCE 
    def _compute_confidence(self, processed: Dict) -> float:

        doc_count = len(processed.get("documents", []))

        if doc_count >= 5:
            return 0.85
        if doc_count >= 3:
            return 0.7
        if doc_count >= 1:
            return 0.5

        return 0.3

    #  EMPTY 
    def _empty_response(self) -> Dict[str, Any]:

        return {
            "answer": "No relevant search results found.",
            "sources": [],
            "confidence": 0.3,
            "metadata": {}
        }