from tavily import TavilyClient
from app.core.model_loader import model_loader
from dotenv import load_dotenv
import os
from typing import Dict, Any, List
from app.utils.logger import get_logger
import time

# Logger
logger = get_logger(__name__)

load_dotenv()


class WebSearchTool:
    def __init__(self):
        logger.info("[WebSearchTool] Initializing Tavily client")

        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in environment")
        
        self.client = TavilyClient(api_key=api_key)


        logger.info("[WebSearchTool] Tavily client initialized")

    # MAIN EXECUTION
    def execute(
        self,
        query: str,
        context: Dict[str, Any] = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        
        start_time = time.time()

        try:
            logger.info(f"[WebSearchTool] Search started | query={query}")

            raw_results = self._search_api(query)

            processed = self._process_results(raw_results)

            if not processed["documents"]:
                return self._empty_response()
            
            summary = self._summarize(query, processed["documents"])

            latency = round(time.time() - start_time, 2)

            return {
                "answer": summary,
                "sources": processed["sources"],
                "confidence": 0.7,
                "metadata": {
                    "results_used": len(processed["documents"]),
                    "latency": latency
                }
            }
        
        except Exception as e:

            logger.error(f"[WebSearchTool] Failed | {str(e)}")

            return {
                "answer": "Search failed. Please try again.",
                "sources": [],
                "confidence": 0.2,
                "metadata": {"error": str(e)}
            }
    
    # Api Call
    def _search_api(self, query: str) -> Dict:

        return self.client.search(
            query=query,
            search_depth="advanced",
            max_results=7
        )
    

    # Process Results
    def _process_results(self, response: Dict) -> Dict[str, List]:

        documents = []
        sources = []

        for r in response.get("results", []): 

            content = r.get("content")
            url = r.get("url")

            if not content:
                continue

            documents.append(content[:400])

            if url:
                sources.append(url)

        return {
            "documents": documents[:5],
            "sources": list(set(sources))[:5]
        }
    
    # Summarization

    def _summarize(self, query: str, docs: List[str]) -> str:

        combined = "\n\n".join(docs)[:2000]

        prompt = f"""
You are a highly accurate AI assistant.

Your task:
- Answer the query using the search results
- Focus on factual correctness
- Prioritize recent and relevant information
- Avoid speculation

Search Results:
{combined}

User Query:
{query}

Return:
- Clear, concise answer
- No fluff
- No hallucination

Final Answer:
"""
        return model_loader.generate(prompt).strip()
    
    # Empty Response
    def _empty_response(self) -> Dict[str, Any]:

        logger.warning("[WebSearchTool] No useful results found")

        return {
            "answer": "No relevant search results found.",
            "sources": [],
            "confidence": 0.3,
            "metadata": {}
        }