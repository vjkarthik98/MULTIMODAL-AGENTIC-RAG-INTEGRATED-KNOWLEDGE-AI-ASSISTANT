from typing import Dict, Any, Callable
import time

from app.core.config import settings
from app.core.model_loader import model_loader

from app.pipeline.rag_pipeline import RAGPipeline
from app.tools.web_search import WebSearchTool

from app.memory.memory_filter import filter_relevant_history

from app.reasoning.query_decomposer import QueryDecomposer
from app.reasoning.reasoning_engine import ReasoningEngine
from app.reasoning.result_fusion import ResultFusion
from app.core.infra_registry import infra

from app.utils.logger import get_logger


logger = get_logger(__name__)


class Tool:

    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable,
        tool_type: str = "generic",
        cost: str = "medium"
    ):
        self.name = name
        self.description = description
        self.handler = handler
        self.tool_type = tool_type
        self.cost = cost

    #  SAFE EXECUTION 
    def execute(self, query: str, context: Dict = None, session_id: str = None) -> Dict:

        start = time.time()

        try:
            query = " ".join(query.strip().split())[:settings.MAX_PROMPT_CHARS]

            if not query:
                return self._error("empty_query")

            result = self.handler(query, context or {}, session_id)

            if result is None:
                return self._error("empty_result")

            latency = round(time.time() - start, 2)

            return {
                "tool": self.name,
                "result": result,
                "status": "success",
                "latency": latency
            }

        except Exception as e:
            logger.error("[Tool:%s] failed | %s", self.name, str(e))
            return self._error(str(e))

    def _error(self, msg):
        return {
            "tool": self.name,
            "result": None,
            "status": "error",
            "error": msg
        }


class ToolRegistry:

    def __init__(self):

        self.tools: Dict[str, Tool] = {}

        #  SAFE INIT 
        try:
            self.rag_pipeline = RAGPipeline()
        except Exception as e:
            logger.warning("[ToolRegistry] RAG init failed | %s", str(e))
            self.rag_pipeline = None

        self.web_search = WebSearchTool()
        self.memory = infra.get_memory()  

        self.embedder = model_loader.get_embedder()
        self.llm = model_loader.get_llm()

        self.decomposer = QueryDecomposer(self.llm)
        self.reasoning_engine = ReasoningEngine(self.llm)
        self.fusion = ResultFusion()

        self._register_default_tools()

    #  REGISTRATION 
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        logger.debug("[ToolRegistry] registered=%s", tool.name)

    def get(self, tool_name: str) -> Tool:
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        return tool

    #  TOOLS 
    def _register_default_tools(self):

        #  RAG 
        def rag_tool(query, context, session_id):
            if not self.rag_pipeline:
                return []
            return self.rag_pipeline.run(query, session_id=session_id)

        self.register(Tool("rag", "Retrieve from knowledge base", rag_tool, "retrieval"))

        #  SEARCH 
        def search_tool(query, context, session_id):
            return self.web_search.execute(query, context, session_id)

        self.register(Tool("search", "External search", search_tool, "external", "high"))

        #  MEMORY 
        def memory_tool(query, context, session_id):

            history = self.memory.get_history(session_id)

            if not history:
                return []

            filtered = filter_relevant_history(
                query=query,
                history=history,
                embedder=self.embedder
            )

            return filtered 

        self.register(Tool("memory", "Fetch memory", memory_tool, "memory", "low"))

        #  DECOMPOSE 
        def decompose_tool(query, context, session_id):
            return self.decomposer.decompose(query)

        self.register(Tool("decompose", "Query decomposition", decompose_tool))

        #  FUSION 
        def fusion_tool(query, context, session_id):
            return self.fusion.fuse(context.get("results", []))

        self.register(Tool("fusion", "Result fusion", fusion_tool))

        #  REASON 
        def reasoning_tool(query, context, session_id):

            return self.reasoning_engine.generate_answer(
                query=query,
                retrieved_docs=context.get("docs", []),
                memory_context=context.get("memory", "")
            )

        self.register(Tool("reason", "Final reasoning", reasoning_tool, "reasoning", "high"))

        logger.info("[ToolRegistry] tools registered")