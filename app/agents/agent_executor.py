from app.agents.agent_router import AgentRouter
from app.pipeline.rag_pipeline import RAGPipeline
from app.tools.web_search import WebSearchTool
from app.memory.redis_memory import RedisMemory
from app.memory.memory_filter import filter_relevant_history
from app.agents.planner import Planner
from app.agents.tool_registry import ToolRegistry
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

import time
from typing import Dict, Any

# Logger
logger = get_logger(__name__)

class AgentExecutor:
    def __init__(self):
        self.router = AgentRouter()
        self.planner = Planner()
        self.registry = ToolRegistry()
        



    # MAIN ENTRY
    def run(self, query: str, session_id: str) -> Dict[str, Any]:

        start_time = time.time()


        logger.info(f"[AgentExecutor][START] session_id={session_id}")

        # STEP 1: Routing

        decision = self.router.route(query, session_id)

        logger.info(
            f"[AgentExecutor][ROUTE] action={decision.action} | reason={decision.reason}"
        )

        # STEP 2: Create Plan
        plan = self.planner.create_plan(decision, query)

        logger.info(f"[AgentExecutor] Plan: {[step.tool for step in plan.steps]}")

        # STEP 3: Execution Loop
        context = {}
        final_output = None

        for step in plan.steps:

            logger.info(f"[AgentExecutor] Executing Step: {step.tool}")

            tool = self.registry.get(step.tool)

            result = tool.execute(
                query = query,
                context=context,
                session_id=session_id
            )

            if result["status"] != "success":
                logger.warning(f"[AgentExecutor] Tool failed: {step.tool}")
                continue

            output = result["result"]

            # Context Flow 

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


        # STEP 3: Final Response
        latency = round(time.time() - start_time, 2)

        if isinstance(final_output, dict):
            response_text = final_output.get("answer")
            metadata = {
                "confidence": final_output.get("confidence"),
                "sources_used": final_output.get("sources_used")
            }
        else:
            response_text = final_output
            metadata = {}


        return {
            "response": response_text,
            "source": "agent",
            "decision": decision.action,
            "reason": decision.reason,
            "latency": latency,
            "metadata": metadata
        }
    
    # HANDLERS
    def _handle_rag(self, query: str, session_id: str):
        logger.info("[Executor] RAG execution")

        rag = RAGPipeline()
        result = rag.run(query, session_id=session_id)

        return {
            "response": result,
            "source": "rag",
            "metadata": {"type": "retrieval"}
        }
    
    def _handle_search(self, query: str):
        logger.info("[Executor] SEARCH execution")

        tool = WebSearchTool()
        result = tool.search(query)

        return {
            "response": result,
            "source": "search",
            "metadata": {"type": "external"}
        }
    
    def _handle_memory(self, query: str, session_id: str): 
        logger.info("[Executor]  MEMORY EXECUTION")

        memory = RedisMemory()
        history = memory.get_history(session_id)

        if not history:
            return {
                "response": "No relevant past context found.",
                "source": "memory"
            }
        embedder = model_loader.get_embedder()

        filtered = filter_relevant_history(
            query=query,
            history=history,
            embedder=embedder
        )

        context = "\n".join([m["context"] for m in filtered])

        prompt = f"""
Use past conversation context to answer:

Context:
{context}

Query:
{query}
"""
        response = model_loader.generate(prompt)

        return {
            "response": response,
            "source": "memory",
            "metadata": {"messages_used": len(filtered)}
        }
    
    def _handle_hybrid(self, query: str, session_id: str):
        logger.info("[Executor] HYBRID EXECUTION")

        # Step 1: RAG
        rag_result = self._handle_rag(query, session_id)

        # Step 2: Memory
        memory_result = self._handle_memory(query, session_id)

        # Step 3: Fusion
        prompt = f"""
    Combine the following into a single, coherent answer:

    RAG Result:
    {rag_result["response"]}

    Memory Context:
    {memory_result["response"]}

    User Query:
    {query}
    """
        
        final = model_loader.generate(prompt)

        return {
            "response": final,
            "source": "hybrid",
            "metadata": {
                "rag_used": True,
                "memory_used": True,
            }
        }
    
    def _handle_direct(self, query: str):
        logger.info("[Executor] DIRECT EXECUTION")

        response = model_loader.generate(query)

        return {
            "response": response,
            "source": "llm",
            "metadata": {"type": "direct"}
        }




