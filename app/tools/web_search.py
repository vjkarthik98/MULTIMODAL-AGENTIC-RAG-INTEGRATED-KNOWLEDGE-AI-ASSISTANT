from tavily import TavilyClient
from app.core.model_loader import model_loader
from dotenv import load_dotenv
import os
import logging

# ✅ Logger
logger = logging.getLogger(__name__)

load_dotenv()


class WebSearchTool:
    def __init__(self):
        logger.info("[WebSearchTool] Initializing Tavily client")

        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        logger.info("[WebSearchTool] Tavily client initialized")

    def search(self, query: str) -> str:
        try:
            logger.info(f"[WebSearchTool] Search started | query={query}")

            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=5
            )

            logger.debug("[WebSearchTool] Raw response received")

            results = []

            for r in response.get("results", [])[:3]:
                if r.get("content"):
                    results.append(r["content"][:500])

            if not results:
                logger.warning("[WebSearchTool] No useful results found")
                return "No useful search results found."

            combined_text = "\n\n".join(results)
            combined_text = combined_text[:1500]

            prompt = f""" 
You are a helpful AI assistant.

Summarize the following search results into a clear, concise answer.

Focus on:
- latest updates
- important developments
- factual accuracy

Search Results:
{combined_text}

User Query:
{query}

Final Answer:
"""

            logger.info("[WebSearchTool] Generating summarized answer")

            return model_loader.generate(prompt)

        except Exception as e:
            logger.error(f"[WebSearchTool] Failed | error={str(e)}")
            return "Search failed. Please try again."