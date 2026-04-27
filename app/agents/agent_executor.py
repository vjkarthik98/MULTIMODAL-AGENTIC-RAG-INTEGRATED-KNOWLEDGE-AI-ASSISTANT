import time
from typing import Dict, Any

from app.core.config import settings
from app.core.model_loader import model_loader

from app.agents.agent_router import AgentRouter
from app.agents.planner import Planner
from app.agents.tool_registry import ToolRegistry

from app.utils.logger import get_logger


logger = get_logger(__name__)


class AgentExecutor:
    def __init__(self):
        self.router = AgentRouter()
        self.planner = Planner()
        self.registry = ToolRegistry()

    def run(self, query: str, session_id: str) -> Dict[str, Any]:
        if not query or not query.strip():
            return {
                "response": "Query cannot be empty.",
                "source": "validation",
                "decision": "reject",
                "reason": "empty_query",
                "metadata": {}
            }

        start_time = time.time()

        logger.info("[AgentExecutor][START] session_id=%s", session_id)

        try:
            decision = self.router.route(query, session_id)

            logger.info(
                "[AgentExecutor][ROUTE] action=%s | reason=%s",
                decision.action,
                decision.reason
            )

            plan = self.planner.create_plan(decision, query)

            if not plan or not getattr(plan, "steps", None):
                raise ValueError("Invalid execution plan")

            logger.info(
                "[AgentExecutor] Plan=%s",
                [step.tool for step in plan.steps]
            )

            context: Dict[str, Any] = {}
            final_output = None

            for step in plan.steps:
                logger.info("[AgentExecutor] Step=%s", step.tool)

                try:
                    tool = self.registry.get(step.tool)

                    result = tool.execute(
                        query=query[:settings.MAX_PROMPT_CHARS],
                        context=context,
                        session_id=session_id
                    )

                    if not isinstance(result, dict):
                        raise ValueError("Invalid tool output")

                    if result.get("status") != "success":
                        logger.warning("[AgentExecutor] Tool failed=%s", step.tool)
                        continue

                    output = result.get("result")

                    # Controlled context propagation
                    if step.tool == "memory":
                        context["memory"] = output

                    elif step.tool == "rag":
                        context["docs"] = output

                    elif step.tool == "decompose":
                        context["sub_queries"] = output

                    elif step.tool == "fusion":
                        context["results"] = output

                    elif step.tool == "search":
                        context["search"] = output

                    elif step.tool == "reason":
                        final_output = output

                except Exception as e:
                    logger.error(
                        "[AgentExecutor][STEP_FAIL] tool=%s | error=%s",
                        step.tool,
                        str(e)
                    )
                    continue

            # Fallback if reasoning failed
            if not final_output:
                logger.warning("[AgentExecutor] No final output, fallback LLM")

                llm = model_loader.get_llm()

                final_output = llm.generate(
                    query[:settings.MAX_PROMPT_CHARS],
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=settings.LLM_TEMPERATURE,
                    top_p=settings.LLM_TOP_P,
                )

            latency = round(time.time() - start_time, 2)

            if isinstance(final_output, dict):
                response_text = final_output.get("answer", "")
                metadata = {
                    "confidence": final_output.get("confidence"),
                    "sources_used": final_output.get("sources_used")
                }
            else:
                response_text = str(final_output)
                metadata = {}

            return {
                "response": response_text,
                "source": "agent",
                "decision": decision.action,
                "reason": decision.reason,
                "latency": latency,
                "metadata": metadata
            }

        except Exception as e:
            logger.error("[AgentExecutor][FAILED] session_id=%s | error=%s", session_id, str(e))

            return self._fallback(query, start_time)

    def _fallback(self, query: str, start_time: float) -> Dict[str, Any]:
        try:
            llm = model_loader.get_llm()

            response = llm.generate(
                query[:settings.MAX_PROMPT_CHARS],
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
            )

        except Exception as e:
            logger.error("[AgentExecutor][FALLBACK_FAIL] %s", str(e))
            response = "I'm unable to process your request right now."

        return {
            "response": response,
            "source": "fallback",
            "decision": "fallback",
            "reason": "executor_failure",
            "latency": round(time.time() - start_time, 2),
            "metadata": {}
        }