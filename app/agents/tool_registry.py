import asyncio
import time
import unicodedata
from collections.abc import Callable
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram

from app.core.config import settings
from app.core.infra_registry import infra
from app.core.model_loader import model_loader
from app.memory.memory_filter import filter_relevant_history
from app.reasoning.query_decomposer import QueryDecomposer
from app.reasoning.reasoning_engine import ReasoningEngine
from app.reasoning.result_fusion import ResultFusion

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_tool_duration = Histogram(
    "tool_execution_duration_seconds",
    "Tool execution duration",
    ["tool", "status"],
)
_tool_errors = Counter(
    "tool_execution_errors_total",
    "Tool execution errors by tool and error type",
    ["tool", "error_type"],
)
_tool_retries = Counter(
    "tool_execution_retries_total",
    "Tool execution retry count",
    ["tool"],
)

# SEMAPHORE — CAP CONCURRENT TOOL EXECUTIONS
_semaphore = asyncio.Semaphore(5)


# NORMALIZE QUERY


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(text.strip().split())


# TOOL SCHEMA VALIDATION


def _validate_tool_input(
    query: str,
    context: dict | None,
    session_id: str,
) -> bool:
    if not query or not query.strip():
        return False
    if not isinstance(context, dict):
        return False
    if not session_id:
        return False
    return True


class Tool:

    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable,
        tool_type: str = "generic",
        cost: str = "medium",
        schema: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.handler = handler
        self.tool_type = tool_type
        self.cost = cost
        self.schema = schema or {}

    # EXECUTE WITH RETRY AND CIRCUIT BREAKER

    def execute(
        self,
        query: str,
        context: dict | None = None,
        session_id: str = "default",
    ) -> dict[str, Any]:

        start = time.time()

        with tracer.start_as_current_span(f"tool_{self.name}") as span:
            span.set_attribute("tool.name", self.name)
            span.set_attribute("tool.type", self.tool_type)
            span.set_attribute("session.id", session_id)

            try:
                query = _normalize(query)
                query = query[: settings.MAX_PROMPT_CHARS]

                if not query:
                    return self._error("empty_query")

                context = context or {}

                # INPUT SCHEMA VALIDATION
                if not _validate_tool_input(query, context, session_id):
                    return self._error("invalid_input")

                result = None
                last_ex = None

                for attempt in range(settings.AGENT_MAX_RETRIES + 1):
                    try:
                        result = self.handler(query, context, session_id)
                        break
                    except Exception as exc:
                        last_ex = exc
                        _tool_retries.labels(tool=self.name).inc()
                        if attempt < settings.AGENT_MAX_RETRIES:
                            time.sleep(0.2 * (attempt + 1))

                if result is None and last_ex:
                    raise last_ex

                if result is None:
                    return self._error("empty_result")

                latency = round(time.time() - start, 3)

                if latency > settings.SLOW_REQUEST_THRESHOLD:
                    logger.warning(
                        "tool_slow",
                        tool=self.name,
                        latency=latency,
                        session_id=session_id,
                    )

                _tool_duration.labels(tool=self.name, status="success").observe(latency)

                span.set_attribute("latency", latency)
                span.set_status(Status(StatusCode.OK))

                logger.debug(
                    "tool_success",
                    tool=self.name,
                    latency=latency,
                    session_id=session_id,
                )

                return {
                    "tool": self.name,
                    "result": result,
                    "status": "success",
                    "latency": latency,
                }

            except Exception as exc:
                latency = round(time.time() - start, 3)
                error_type = type(exc).__name__

                _tool_duration.labels(tool=self.name, status="error").observe(latency)
                _tool_errors.labels(tool=self.name, error_type=error_type).inc()

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)

                logger.error(
                    "tool_failed",
                    tool=self.name,
                    error=str(exc),
                    error_type=error_type,
                    session_id=session_id,
                )

                return self._error(str(exc))

    def _error(self, msg: str) -> dict[str, Any]:
        return {
            "tool": self.name,
            "result": None,
            "status": "error",
            "error": msg,
        }

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.tool_type,
            "cost": self.cost,
            "schema": str(self.schema),
        }


class ToolRegistry:

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

        # RAG PIPELINE
        try:
            from app.pipeline.rag_pipeline import RAGPipeline

            self.rag_pipeline = RAGPipeline()
        except Exception as exc:
            logger.warning("rag_pipeline_init_failed", error=str(exc))
            self.rag_pipeline = None

        # WEB SEARCH — REQUIRES TAVILY_API_KEY
        self.web_search = None
        if settings.TAVILY_API_KEY:
            try:
                from app.tools.web_search import WebSearchTool

                self.web_search = WebSearchTool()
            except Exception as exc:
                logger.warning("web_search_init_failed", error=str(exc))
        else:
            logger.warning("web_search_disabled", reason="TAVILY_API_KEY not set")

        # MEMORY
        self.memory = infra.get_memory()
        self.embedder = model_loader.get_embedder()
        self.llm = model_loader.get_llm()

        # REASONING COMPONENTS
        self.decomposer = QueryDecomposer(self.llm)
        self.reasoning_engine = ReasoningEngine(self.llm)
        self.fusion = ResultFusion()

        self._register()

    # REGISTER

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise ValueError(f"INVALID_TOOL_TYPE: {type(tool)}")
        if not tool.name:
            raise ValueError("TOOL_NAME_REQUIRED")
        self.tools[tool.name] = tool
        logger.debug("tool_registered", tool_name=tool.name, type=tool.tool_type)

    # GET TOOL — RAISES IF NOT FOUND

    def get(self, tool_name: str) -> Tool:
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"TOOL_NOT_FOUND: {tool_name}")
        return tool

    # GET OPTIONAL — RETURNS NONE IF NOT FOUND

    def get_optional(self, tool_name: str) -> Tool | None:
        return self.tools.get(tool_name)

    # LIST ALL TOOLS

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": t.name, "type": t.tool_type, "cost": t.cost, "description": t.description}
            for t in self.tools.values()
        ]

    # DEREGISTER

    def deregister(self, tool_name: str) -> None:
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info("tool_deregistered", tool_name=tool_name)

    # REGISTER ALL BUILT-IN TOOLS

    def _register(self) -> None:

        # RAG TOOL
        def rag_tool(query: str, context: dict, session_id: str, user_id: str | None = None):
            if not self.rag_pipeline:
                return []
            try:
                # Was `self.rag_pipeline.retriever.retrieval(...)` — wrong on
                # both halves: RAGPipeline exposes no public `retriever`
                # (only `_retriever` + the lazy `_get_retriever()`), and
                # HybridRetriever's method is `search()`, not `retrieval()`.
                # Either mistake alone raises AttributeError, which the
                # `except` below swallows into an empty result — so this tool
                # could only ever have returned []. It is currently latent
                # (the live path calls hybrid_retriever directly, and only the
                # "search" tool handler is ever invoked), but it IS registered
                # and advertised by GET /tools, so it's fixed rather than left
                # as a trap for whoever wires it up next.
                result = self.rag_pipeline._get_retriever().search(
                    query=query,
                    session_id=session_id,
                    top_k=settings.RAG_TOP_K,
                    user_id=user_id,
                )
                return result
            except Exception as exc:
                logger.warning("rag_tool_failed", error=str(exc), session_id=session_id)
                return []

        self.register(
            Tool(
                name="rag",
                description="Retrieve relevant knowledge from vector store",
                handler=rag_tool,
                tool_type="retrieval",
                cost="medium",
                schema={
                    "input": {"query": "str", "context": "dict", "session_id": "str"},
                    "output": "list[dict]",
                },
            )
        )

        # SEARCH TOOL
        def search_tool(query: str, context: dict, session_id: str):
            if not self.web_search:
                return {
                    "answer": "Web search unavailable.",
                    "sources": [],
                }
            return self.web_search.execute(query, context, session_id)

        self.register(
            Tool(
                name="search",
                description="External web search for recent information",
                handler=search_tool,
                tool_type="external",
                cost="high",
                schema={
                    "input": {"query": "str", "context": "dict", "session_id": "str"},
                    "output": "dict[answer, sources, confidence]",
                },
            )
        )

        # MEMORY TOOL
        def memory_tool(query: str, context: dict, session_id: str):
            if not self.memory:
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
            except Exception as exc:
                logger.warning("memory_tool_failed", error=str(exc), session_id=session_id)
                return []

        self.register(
            Tool(
                name="memory",
                description="Fetch relevant session memory",
                handler=memory_tool,
                tool_type="memory",
                cost="low",
                schema={
                    "input": {"query": "str", "context": "dict", "session_id": "str"},
                    "output": "list[dict]",
                },
            )
        )

        # DECOMPOSE TOOL
        def decompose_tool(query: str, context: dict, session_id: str):
            return self.decomposer.decompose(query, session_id=session_id)

        self.register(
            Tool(
                name="decompose",
                description="Decompose complex query into sub-queries",
                handler=decompose_tool,
                tool_type="reasoning",
                cost="medium",
                schema={
                    "input": {"query": "str", "context": "dict", "session_id": "str"},
                    "output": "list[str]",
                },
            )
        )

        # FUSION TOOL
        def fusion_tool(query: str, context: dict, session_id: str):
            results = context.get("results") or context.get("docs") or []
            return self.fusion.fuse(results, session_id=session_id)

        self.register(
            Tool(
                name="fusion",
                description="Merge and rank retrieved results",
                handler=fusion_tool,
                tool_type="reasoning",
                cost="medium",
                schema={
                    "input": {"query": "str", "context": "dict[docs|results]", "session_id": "str"},
                    "output": "list[dict]",
                },
            )
        )

        # REASON TOOL
        def reasoning_tool(query: str, context: dict, session_id: str):
            # DETECT QUERY TYPE FROM CONTEXT IF AVAILABLE
            query_type = context.get("query_type", "factual")

            return self.reasoning_engine.generate_answer(
                query=query,
                retrieved_docs=context.get("docs", []),
                memory_context=context.get("memory", ""),
                session_id=session_id,
                query_type=query_type,
            )

        self.register(
            Tool(
                name="reason",
                description="Generate final grounded answer",
                handler=reasoning_tool,
                tool_type="reasoning",
                cost="high",
                schema={
                    "input": {"query": "str", "context": "dict[docs, memory]", "session_id": "str"},
                    "output": "dict[answer, confidence, sources_used]",
                },
            )
        )

        logger.info(
            "tools_registered",
            count=len(self.tools),
            tools=[t.name for t in self.tools.values()],
        )

    # ASYNC EXECUTE — WRAP TOOL EXECUTION IN EXECUTOR

    async def execute_async(
        self,
        tool_name: str,
        query: str,
        context: dict | None = None,
        session_id: str = "default",
    ) -> dict[str, Any]:

        tool = self.get(tool_name)

        async with _semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: tool.execute(query, context, session_id),
            )

    # PARALLEL EXECUTE — RUN MULTIPLE TOOLS CONCURRENTLY

    async def execute_parallel(
        self,
        tool_names: list[str],
        query: str,
        context: dict | None = None,
        session_id: str = "default",
    ) -> dict[str, dict[str, Any]]:

        tasks = {
            name: self.execute_async(name, query, context, session_id)
            for name in tool_names
            if name in self.tools
        }

        results: dict[str, dict[str, Any]] = {}

        if not tasks:
            return results

        gathered = await asyncio.gather(
            *tasks.values(),
            return_exceptions=True,
        )

        for name, result in zip(tasks.keys(), gathered, strict=False):
            if isinstance(result, Exception):
                logger.error(
                    "parallel_tool_failed",
                    tool=name,
                    error=str(result),
                    session_id=session_id,
                )
                results[name] = {
                    "tool": name,
                    "result": None,
                    "status": "error",
                    "error": str(result),
                }
            else:
                results[name] = result

        return results
