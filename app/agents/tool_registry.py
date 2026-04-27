from typing import Dict, Any, Callable

from app.core.config import settings
from app.core.model_loader import model_loader

from app.pipeline.rag_pipeline import RAGPipeline
from app.tools.web_search import WebSearchTool

from app.memory.redis_memory import RedisMemory
from app.memory.memory_filter import filter_relevant_history

from app.reasoning.query_decomposer import QueryDecomposer
from app.reasoning.reasoning_engine import ReasoningEngine
from app.reasoning.result_fusion import ResultFusion

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

    def execute(self, query: str, context: Dict = None, session_id: str = None) -> Dict:
        try:
            logger.info("[Tool:%s] start", self.name)

            if not query:
                return {
                    "tool": self.name,
                    "result": None,
                    "status": "error",
                    "error": "empty_query"
                }

            result = self.handler(
                query[:settings.MAX_PROMPT_CHARS],
                context or {},
                session_id
            )

            return {
                "tool": self.name,
                "result": result,
                "status": "success"
            }

        except Exception as e:
            logger.error("[Tool:%s] failed | %s", self.name, str(e))

            return {
                "tool": self.name,
                "result": None,
                "status": "error",
                "error": str(e)
            }


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

        # Shared instances (important fix)
        try:
            self.rag_pipeline = RAGPipeline()
        except Exception as e:
            logger.warning("[ToolRegistry] RAG pipeline init failed | %s", str(e))
            self.rag_pipeline = None

        self.web_search = WebSearchTool()
        self.memory = ()

        
        self.embedder = model_loader.get_embedder()
        self.llm = model_loader.get_llm()

        self.decomposer = QueryDecomposer(self.llm)
        self.reasoning_engine = ReasoningEngine(self.llm)
        self.fusion = ResultFusion()

        self._register_default_tools()

    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        logger.debug("[ToolRegistry] registered=%s", tool.name)

    def get(self, tool_name: str) -> Tool:
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        return tool

    def list_tools(self) -> Dict[str, str]:
        return {name: tool.description for name, tool in self.tools.items()}

    def _register_default_tools(self):

        # RAG
        def rag_tool(query, context, session_id):
            result = self.rag_pipeline.run(query, session_id=session_id)
            return result

        self.register(
            Tool(
                name="rag",
                description="Retrieve from internal multimodal knowledge base",
                handler=rag_tool,
                tool_type="retrieval",
                cost="medium"
            )
        )

        # SEARCH
        def search_tool(query, context, session_id):
            return self.web_search.execute(query, context, session_id)

        self.register(
            Tool(
                name="search",
                description="Search external real-time information",
                handler=search_tool,
                tool_type="external",
                cost="high"
            )
        )

        # MEMORY
        def memory_tool(query, context, session_id):
            history = self.memory.get_history(session_id)

            if not history:
                return ""

            filtered = filter_relevant_history(
                query=query,
                history=history,
                embedder=self.embedder
            )

            texts = [m.get("content", "") for m in filtered if m.get("content")]

            merged = "\n".join(texts)

            return merged[:settings.MEMORY_MAX_CONTEXT_CHARS]

        self.register(
            Tool(
                name="memory",
                description="Retrieve relevant past conversation context",
                handler=memory_tool,
                tool_type="memory",
                cost="low"
            )
        )

        # DECOMPOSE
        def decompose_tool(query, context, session_id):
            return self.decomposer.decompose(query)

        self.register(
            Tool(
                name="decompose",
                description="Break complex query into sub-queries",
                handler=decompose_tool,
                tool_type="reasoning",
                cost="low"
            )
        )

        # REASON
        def reasoning_tool(query, context, session_id):
            return self.reasoning_engine.generate_answer(
                query=query,
                retrieved_docs=context.get("docs", []),
                memory_context=context.get("memory", "")
            )

        self.register(
            Tool(
                name="reason",
                description="Generate final answer using reasoning engine",
                handler=reasoning_tool,
                tool_type="reasoning",
                cost="high"
            )
        )

        # FUSION
        def fusion_tool(query, context, session_id):
            results = context.get("results", [])
            return self.fusion.fuse(results)

        self.register(
            Tool(
                name="fusion",
                description="Fuse and rank results",
                handler=fusion_tool,
                tool_type="postprocessing",
                cost="low"
            )
        )

        logger.info("[ToolRegistry] all tools registered")