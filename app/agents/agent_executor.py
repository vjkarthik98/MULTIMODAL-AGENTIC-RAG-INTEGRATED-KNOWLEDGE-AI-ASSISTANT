import time
import unicodedata
from typing import Any, Dict, List, Optional

from app.agents.agent_router import AgentRouter
from app.agents.agent_schema import AgentDecision
from app.agents.planner import Planner
from app.agents.tool_registry import ToolRegistry
from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AgentExecutor:

    def __init__(self) -> None:
        self.router       = AgentRouter()
        self.planner      = Planner()
        self.registry     = ToolRegistry()
        self.max_steps    = settings.AGENT_MAX_STEPS
        self.step_timeout = settings.AGENT_STEP_TIMEOUT_SEC

    # NORMALIZE

    def _normalize(self, query: str) -> str:
        query = unicodedata.normalize("NFC", str(query or ""))
        return " ".join(query.strip().split())

    # MAIN

    def run(self, query: str, session_id: str) -> Dict[str, Any]:

        if not query:
            return self._reject()

        start   = time.time()
        query   = self._normalize(query)[:settings.MAX_PROMPT_CHARS]

        try:
            logger.info(event="agent_exec_start", session_id=session_id)

            # ROUTING
            decision = self.router.route(query, session_id)

            # PLANNING
            plan = self.planner.create_plan(decision, query, session_id=session_id)

            if not plan or not plan.steps:
                raise ValueError("INVALID_PLAN")

            steps = plan.steps[:self.max_steps]

            context:         Dict[str, Any] = {}
            execution_trace: List[Dict]     = []
            final_output                    = None
            steps_executed                  = 0
            steps_skipped                   = 0

            # STEP EXECUTION
            for step in steps:
                step_start = time.time()

                try:
                    tool = self.registry.get(step.tool)

                    result = self._execute_step(
                        tool=tool,
                        query=query,
                        context=context,
                        session_id=session_id,
                        step_start=step_start,
                    )

                    steps_executed += 1

                    trace_entry = {
                        "tool":    step.tool,
                        "status":  result.get("status"),
                        "latency": result.get("latency"),
                    }
                    execution_trace.append(trace_entry)

                    if not self._valid_result(result):
                        if step.optional:
                            steps_skipped += 1
                        continue

                    output = result.get("result")
                    self._update_context(step.tool, context, output, query, session_id)

                    if step.tool == "reason":
                        final_output = output

                except Exception as e:
                    steps_skipped += 1
                    log_level = "warning" if step.optional else "warning"
                    logger.warning(
                        event="step_failed",
                        tool=step.tool,
                        optional=step.optional,
                        error=str(e),
                        session_id=session_id,
                    )
                    continue

            # FINAL OUTPUT GUARD
            if not final_output:
                final_output = self._fallback_llm(query, session_id)

            latency = round(time.time() - start, 3)

            logger.info(
                event="agent_exec_success",
                decision=decision.action,
                steps_executed=steps_executed,
                steps_skipped=steps_skipped,
                latency=latency,
                session_id=session_id,
            )

            return self._format(final_output, decision, latency, execution_trace)

        except Exception as e:
            logger.error(
                event="agent_exec_failed",
                error=str(e),
                session_id=session_id,
            )
            return self._fallback(query, start, session_id)

    # STEP EXECUTION

    def _execute_step(
        self,
        tool,
        query: str,
        context: Dict,
        session_id: str,
        step_start: float,
    ) -> Dict[str, Any]:

        result = tool.execute(
            query=query,
            context=self._truncate_context(context),
            session_id=session_id,
        )

        elapsed = time.time() - step_start
        if elapsed > self.step_timeout:
            raise TimeoutError(f"STEP_TIMEOUT_{tool.name}_{elapsed:.1f}s")

        return result

    # RESULT VALIDATION

    def _valid_result(self, result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        if result.get("status") != "success":
            return False
        if "result" not in result:
            return False
        return True

    # CONTEXT UPDATE

    def _update_context(
        self,
        tool: str,
        context: Dict,
        output: Any,
        query: str,
        session_id: str,
    ) -> None:

        try:
            if tool == "memory":
                context["memory"] = output

            elif tool == "rag":
                docs = output if isinstance(output, list) else []
                context["docs"] = docs[:settings.MAX_CHUNKS]

            elif tool == "decompose":
                sub_queries = output if isinstance(output, list) else []
                context["sub_queries"] = sub_queries

                # EXPAND RAG FOR EACH SUB-QUERY
                if sub_queries:
                    try:
                        rag_tool   = self.registry.get("rag")
                        extra_docs = []
                        for sq in sub_queries[:settings.MAX_SUBQUERIES]:
                            res = rag_tool.execute(
                                query=sq,
                                context=context,
                                session_id=session_id,
                            )
                            if self._valid_result(res) and isinstance(res.get("result"), list):
                                extra_docs.extend(res["result"])

                        existing = context.get("docs", [])
                        merged   = existing + extra_docs
                        context["docs"] = merged[:settings.MAX_CHUNKS]

                    except Exception as e:
                        logger.warning(
                            event="sub_query_rag_failed",
                            error=str(e),
                            session_id=session_id,
                        )

            elif tool == "fusion":
                results = output if isinstance(output, list) else []
                context["results"] = results[:settings.MAX_CHUNKS]
                if results:
                    context["docs"] = results[:settings.MAX_CHUNKS]

            elif tool == "search":
                context["search"] = output
                if isinstance(output, dict) and output.get("answer"):
                    context["docs"] = [{
                        "text":     output.get("answer", ""),
                        "score":    0.9,
                        "metadata": {"source": "web_search", "modality": "text"},
                    }]

        except Exception as e:
            logger.warning(
                event="context_update_failed",
                tool=tool,
                error=str(e),
                session_id=session_id,
            )

    # CONTEXT TRUNCATION

    def _truncate_context(self, context: Dict) -> Dict:
        truncated: Dict = {}
        for k, v in context.items():
            if isinstance(v, list):
                truncated[k] = v[:settings.MAX_CHUNKS]
            else:
                truncated[k] = v
        return truncated

    # FORMAT OUTPUT

    def _format(
        self,
        output: Any,
        decision: AgentDecision,
        latency: float,
        trace: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:

        if isinstance(output, dict):
            answer   = output.get("answer", "")
            metadata = {
                "confidence":  output.get("confidence", 0.5),
                "sources_used": output.get("sources_used", 0),
            }
        else:
            answer   = str(output)
            metadata = {}

        metadata["execution_trace"] = trace or []

        return {
            "response": answer,
            "source":   "agent",
            "decision": decision.action,
            "reason":   decision.reason,
            "latency":  latency,
            "metadata": metadata,
        }

    # FALLBACK LLM

    def _fallback_llm(self, query: str, session_id: str = "") -> str:
        try:
            llm = model_loader.get_llm()
            return llm.generate(
                f"Answer clearly:\n{query}",
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=0.2,
                top_p=settings.LLM_TOP_P,
                session_id=session_id,
            )
        except Exception as e:
            logger.error(
                event="fallback_llm_failed",
                error=str(e),
                session_id=session_id,
            )
            return "Unable to generate a response."

    def _fallback(self, query: str, start_time: float, session_id: str = "") -> Dict[str, Any]:
        response = self._fallback_llm(query, session_id)
        return {
            "response": response,
            "source":   "fallback",
            "decision": "fallback",
            "reason":   "executor_failure",
            "latency":  round(time.time() - start_time, 3),
            "metadata": {},
        }

    # REJECT

    def _reject(self) -> Dict[str, Any]:
        return {
            "response": "Query cannot be empty.",
            "source":   "validation",
            "decision": "reject",
            "reason":   "empty_query",
            "metadata": {},
        }