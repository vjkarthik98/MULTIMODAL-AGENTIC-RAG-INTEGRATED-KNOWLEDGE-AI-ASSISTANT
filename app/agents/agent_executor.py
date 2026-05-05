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
        self.step_timeout = getattr(settings, "AGENT_STEP_TIMEOUT_SEC", 5)

    #  NORMALIZE 
    def _normalize(self, query: str) -> str:
        return " ".join(query.strip().split())

    #  MAIN 
    def run(self, query: str, session_id: str) -> Dict[str, Any]:

        if not query:
            return self._reject()

        start_time = time.time()
        query = self._normalize(query)[:settings.MAX_PROMPT_CHARS]

        try:
            logger.info(event="agent_exec_start", session_id=session_id)

            #  ROUTING 
            decision = self.router.route(query, session_id)

            #  PLANNING 
            plan = self.planner.create_plan(decision, query)

            if not plan or not getattr(plan, "steps", None):
                raise ValueError("INVALID_PLAN")

            steps = plan.steps[:self.max_steps]

            context: Dict[str, Any] = {}
            final_output = None

            #  EXECUTION 
            for step in steps:

                step_start = time.time()

                try:
                    tool = self.registry.get(step.tool)

                    result = self._execute_step(
                        tool,
                        query,
                        context,
                        session_id,
                        step_start
                    )

                    if not self._valid_result(result):
                        continue

                    output = result.get("result")

                    self._update_context(step.tool, context, output)

                    if step.tool == "reason":
                        final_output = output

                except Exception as e:
                    logger.warning(event="step_failed", tool=step.tool, error=str(e))
                    continue

            #  FINAL 
            if not final_output:
                final_output = self._fallback_llm(query)

            latency = round(time.time() - start_time, 3)

            return self._format(final_output, decision, latency)

        except Exception as e:
            logger.error(event="agent_exec_failed", error=str(e))
            return self._fallback(query, start_time)

    #  STEP EXEC 
    def _execute_step(self, tool, query, context, session_id, start_time):

        result = tool.execute(
            query=query,
            context=self._truncate_context(context),
            session_id=session_id
        )

        if time.time() - start_time > self.step_timeout:
            raise TimeoutError("STEP_TIMEOUT")

        return result

    #  VALIDATION 
    def _valid_result(self, result: Dict[str, Any]) -> bool:

        if not isinstance(result, dict):
            return False

        if result.get("status") != "success":
            return False

        if "result" not in result:
            return False

        return True

    #  CONTEXT 
    def _update_context(self, tool: str, context: Dict, output):

        try:
            if tool == "memory":
                context["memory"] = output

            elif tool == "rag":
                context["docs"] = output[:settings.MAX_CHUNKS] if isinstance(output, list) else []

            elif tool == "decompose":
                context["sub_queries"] = output if isinstance(output, list) else []

            elif tool == "fusion":
                context["results"] = output[:settings.MAX_CHUNKS] if isinstance(output, list) else []

            elif tool == "search":
                context["search"] = output

        except Exception as e:
            logger.warning(event="context_update_failed", error=str(e))

    #  CONTEXT LIMIT 
    def _truncate_context(self, context: Dict) -> Dict:

        truncated = {}

        for k, v in context.items():
            if isinstance(v, list):
                truncated[k] = v[:settings.MAX_CHUNKS]
            else:
                truncated[k] = v

        return truncated

    #  FORMAT 
    def _format(self, output, decision, latency):

        if isinstance(output, dict):
            answer = output.get("answer", "")
            metadata = {
                "confidence": output.get("confidence", 0.5),
                "sources_used": output.get("sources_used", 0)
            }
        else:
            answer = str(output)
            metadata = {}

        return {
            "response": answer,
            "source": "agent",
            "decision": decision.action,
            "reason": decision.reason,
            "latency": latency,
            "metadata": metadata
        }

    #  FALLBACK 
    def _fallback_llm(self, query: str):

        llm = model_loader.get_llm()

        return llm.generate(
            f"Answer clearly:\n{query}",
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=0.2,
            top_p=settings.LLM_TOP_P,
        )

    def _fallback(self, query: str, start_time: float):

        response = self._fallback_llm(query)

        return {
            "response": response,
            "source": "fallback",
            "decision": "fallback",
            "reason": "executor_failure",
            "latency": round(time.time() - start_time, 3),
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