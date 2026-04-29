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

        self.max_steps = getattr(settings, "AGENT_MAX_STEPS", 10)

    #  NORMALIZE 
    def _normalize(self, query: str) -> str:
        return " ".join(query.strip().split())

    #  MAIN 
    def run(self, query: str, session_id: str) -> Dict[str, Any]:

        if not query or not query.strip():
            return self._reject()

        start_time = time.time()
        query = self._normalize(query)[:settings.MAX_PROMPT_CHARS]

        logger.info("[AgentExecutor][START] session_id=%s", session_id)

        try:
            #  ROUTER 
            decision = self.router.route(query, session_id)

            logger.info(
                "[AgentExecutor][ROUTE] action=%s | reason=%s",
                decision.action,
                decision.reason
            )

            #  PLAN 
            plan = self.planner.create_plan(decision, query)

            if not plan or not getattr(plan, "steps", None):
                raise ValueError("Invalid execution plan")

            steps = plan.steps[:self.max_steps]

            logger.info("[AgentExecutor] Plan=%s", [s.tool for s in steps])

            context: Dict[str, Any] = {}
            final_output = None

            #  EXECUTION 
            for idx, step in enumerate(steps):

                step_start = time.time()

                logger.info("[AgentExecutor] Step=%s", step.tool)

                try:
                    tool = self.registry.get(step.tool)

                    result = self._safe_tool_execute(
                        tool,
                        query,
                        context,
                        session_id
                    )

                    if not self._validate_tool_result(result):
                        continue

                    output = result.get("result")

                    self._update_context(step.tool, context, output)

                    if step.tool == "reason":
                        final_output = output

                    logger.info(
                        "[AgentExecutor][STEP_SUCCESS] tool=%s latency=%.2fs",
                        step.tool,
                        time.time() - step_start
                    )

                except Exception as e:
                    logger.error(
                        "[AgentExecutor][STEP_FAIL] tool=%s | error=%s",
                        step.tool,
                        str(e)
                    )
                    continue

            #  FINAL OUTPUT 
            if not final_output:
                logger.warning("[AgentExecutor] fallback triggered")
                final_output = self._fallback_llm(query)

            latency = round(time.time() - start_time, 2)

            return self._format_output(
                final_output,
                decision,
                latency
            )

        except Exception as e:
            logger.error(
                "[AgentExecutor][FAILED] session_id=%s | error=%s",
                session_id,
                str(e)
            )

            return self._fallback(query, start_time)

    #  SAFE TOOL EXECUTION 
    def _safe_tool_execute(self, tool, query, context, session_id):

        return tool.execute(
            query=query,
            context=self._truncate_context(context),
            session_id=session_id
        )

    #  VALIDATE TOOL RESULT 
    def _validate_tool_result(self, result: Dict[str, Any]) -> bool:

        if not isinstance(result, dict):
            return False

        if result.get("status") != "success":
            return False

        if "result" not in result:
            return False

        return True

    # CONTEXT UPDATE
    def _update_context(self, tool: str, context: Dict, output):

        if tool == "memory":
            context["memory"] = output

        elif tool == "rag":
            if isinstance(output, list):
                context["docs"] = output[:settings.MAX_CHUNKS]
            else:
                logger.warning("[AgentExecutor] rag output not list, skipping")
                context["docs"] = []

        elif tool == "decompose":
            context["sub_queries"] = output if isinstance(output, list) else []

        elif tool == "fusion":
            if isinstance(output, list):
                context["results"] = output[:settings.MAX_CHUNKS]
            else:
                logger.warning("[AgentExecutor] fusion output not list, skipping")
                context["results"] = []

        elif tool == "search":
            context["search"] = output
            
    #  TRUNCATE CONTEXT 
    def _truncate_context(self, context: Dict) -> Dict:

        truncated = {}

        for k, v in context.items():
            if isinstance(v, list):
                truncated[k] = v[:settings.MAX_CHUNKS]
            else:
                truncated[k] = v

        return truncated

    #  FORMAT OUTPUT 
    def _format_output(self, final_output, decision, latency):

        if isinstance(final_output, dict):

            answer = final_output.get("answer", "")
            metadata = {
                "confidence": final_output.get("confidence", 0.5),
                "sources_used": final_output.get("sources_used", 0)
            }

        else:
            answer = str(final_output)
            metadata = {}

        return {
            "response": answer,
            "source": "agent",
            "decision": decision.action,
            "reason": decision.reason,
            "latency": latency,
            "metadata": metadata
        }

    #  FALLBACK LLM 
    def _fallback_llm(self, query: str):

        llm = model_loader.get_llm()

        safe_prompt = f"Answer clearly:\n\n{query}"

        return llm.generate(
            safe_prompt,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=0.3,
            top_p=settings.LLM_TOP_P,
        )

    #  FULL FALLBACK 
    def _fallback(self, query: str, start_time: float):

        response = self._fallback_llm(query)

        return {
            "response": response,
            "source": "fallback",
            "decision": "fallback",
            "reason": "executor_failure",
            "latency": round(time.time() - start_time, 2),
            "metadata": {}
        }

    #  REJECT 
    def _reject(self):
        return {
            "response": "Query cannot be empty.",
            "source": "validation",
            "decision": "reject",
            "reason": "empty_query",
            "metadata": {}
        }