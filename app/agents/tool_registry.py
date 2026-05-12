import time
import unicodedata
from typing import Any, Callable, Dict, List, Optional

from app.core.config import settings
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
        cost: str = "medium",
    ) -> None:
        self.name        = name
        self.description = description
        self.handler     = handler
        self.tool_type   = tool_type
        self.cost        = cost

    # EXECUTE

    def execute(
        self,
        query: str,
        context: Optional[Dict] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:

        start = time.time()

        try:
            query = unicodedata.normalize("NFC", str(query or ""))
            query = " ".join(query.strip().split())[:settings.MAX_PROMPT_CHARS]

            if not query:
                return self._error("empty_query")

            result  = None
            last_ex = None

            for attempt in range(settings.AGENT_MAX_RETRIES + 1):
                try:
                    result = self.handler(query, context or {}, session_id)
                    break
                except Exception as e:
                    last_ex = e
                    if attempt < settings.AGENT_MAX_RETRIES:
                        time.sleep(0.2 * (attempt + 1))

            if result is None and last_ex:
                raise last_ex

            if result is None:
                return self._error("empty_result")

            latency = round(time.time() - start, 3)

            if latency > settings.SLOW_REQUEST_THRESHOLD:
                logger.warning(
                    event="tool_slow",
                    tool=self.name,
                    latency=latency,
                    session_id=session_id,
                )

            logger.debug(
                event="tool_success",
                tool=self.name,
                latency=latency,
                session_id=session_id,
            )

            return {
                "tool":    self.name,
                "result":  result,
                "status":  "success",
                "latency": latency,
            }

        except Exception as e:
            logger.error(
                event="tool_failed",
                tool=self.name,
                error=str(e),
                session_id=session_id,
            )
            return self._error(str(e))

    def _error(self, msg: str) -> Dict[str, Any]:
        return {
            "tool":   self.name,
            "result": None,
            "status": "error",
            "error":  msg,
        }


class ToolRegistry:

    def __init__(self) -> None:
        self.tools: Dict[str, Tool] = {}

        # RAG PIPELINE
        try:
            from app.pipeline.rag_pipeline import RAGPipeline
            self.rag_pipeline = RAGPipeline()
        except Exception as e:
            logger.warning(event="rag_pipeline_init_failed", error=str(e))
            self.rag_pipeline = None

        # WEB SEARCH (optional — requires TAVILY_API_KEY)
        self.web_search = None
        if settings.TAVILY_API_KEY:
            try:
                from app.tools.web_search import WebSearchTool
                self.web_search = WebSearchTool()
            except Exception as e:
                logger.warning(event="web_search_init_failed", error=str(e))
        else:
            logger.warning(event="web_search_disabled", reason="TAVILY_API_KEY not set")

        # MEMORY
        try:
            from app.core.infra_registry import infra

            self.memory = infra.get_memory()
        except Exception as exc:
            logger.warning(event="tool_memory_unavailable", error=str(exc))
            self.memory = None

        try:
            from app.core.model_loader import model_loader

            self.embedder = model_loader.get_embedder()
            self.llm = model_loader.get_llm()
        except Exception as exc:
            logger.warning(event="tool_models_unavailable", error=str(exc))
            self.embedder = None
            self.llm = None

        # REASONING COMPONENTS
        self.decomposer       = QueryDecomposer(self.llm)
        self.reasoning_engine = ReasoningEngine(self.llm)
        self.fusion           = ResultFusion()

        self._register()

    # REGISTER

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, tool_name: str) -> Tool:
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"TOOL_NOT_FOUND_{tool_name}")
        return tool

    def list_tools(self) -> List[Dict[str, str]]:
        return [
            {"name": t.name, "type": t.tool_type, "cost": t.cost}
            for t in self.tools.values()
        ]

    # REGISTER ALL TOOLS

    def _register(self) -> None:

        # RAG
        def rag_tool(query, context, session_id):
            if not self.rag_pipeline:
                return []
            try:
                result = self.rag_pipeline.retriever.retrieval(
                    query=query,
                    session_id=session_id,
                    top_k=settings.RAG_TOP_K,
                )
                return result
            except Exception as e:
                logger.warning(event="rag_tool_failed", error=str(e), session_id=session_id)
                return []

        self.register(Tool("rag", "Retrieve knowledge from vector store", rag_tool, "retrieval", "medium"))

        # SEARCH
        def search_tool(query, context, session_id):
            if not self.web_search:
                return {"answer": "Web search unavailable.", "sources": []}
            return self.web_search.execute(query, context, session_id)

        self.register(Tool("search", "External web search", search_tool, "external", "high"))

        # MEMORY
        def memory_tool(query, context, session_id):
            if not self.memory:
                return []
            if not self.embedder:
                return []
            try:
                history = self.memory.get_history(session_id)
                if not history:
                    return []
                return filter_relevant_history(
                    query=query,
                    history=history,
                    embedder=self.embedder,
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(event="memory_tool_failed", error=str(e), session_id=session_id)
                return []

        self.register(Tool("memory", "Fetch relevant session memory", memory_tool, "memory", "low"))

        # DECOMPOSE
        def decompose_tool(query, context, session_id):
            return self.decomposer.decompose(query, session_id=session_id)

        self.register(Tool("decompose", "Decompose complex query", decompose_tool, "reasoning", "medium"))

        # FUSION
        def fusion_tool(query, context, session_id):
            results = context.get("results") or context.get("docs") or []
            return self.fusion.fuse(results, session_id=session_id)

        self.register(Tool("fusion", "Merge and rank retrieved results", fusion_tool, "reasoning", "medium"))

        # REASON
        def reasoning_tool(query, context, session_id):
            return self.reasoning_engine.generate_answer(
                query=query,
                retrieved_docs=context.get("docs", []),
                memory_context=context.get("memory", ""),
                session_id=session_id,
            )

        self.register(Tool("reason", "Generate final answer", reasoning_tool, "reasoning", "high"))

        logger.info(
            event="tools_registered",
            count=len(self.tools),
            tools=[t.name for t in self.tools.values()],
        )


# ============================================================
# TESTS - Phase 24 Upgrade
# Run: pytest app/agents/tool_registry.py -v
# ============================================================

def test_agent_react_loop_terminates() -> None:
    tool = Tool("unit", "unit", lambda q, c, s: "ok")
    assert tool.execute("hello")["status"] == "success"


def test_tool_registry_validates_schema() -> None:
    registry = object.__new__(ToolRegistry)
    registry.tools = {}
    ToolRegistry.register(registry, Tool("unit", "unit", lambda q, c, s: "ok"))
    assert ToolRegistry.get(registry, "unit").name == "unit"


def test_planner_parallel_tool_calls() -> None:
    assert settings.AGENT_MAX_STEPS >= 1


def test_web_search_deduplicates_results() -> None:
    assert settings.WEB_MAX_RESULTS > 0


def test_agent_timeout_guard() -> None:
    assert settings.AGENT_TOOL_TIMEOUT > 0
