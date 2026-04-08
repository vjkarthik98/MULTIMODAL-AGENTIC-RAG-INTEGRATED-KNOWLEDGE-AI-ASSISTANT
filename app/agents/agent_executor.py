from app.agents.agent_router import AgentRouter
from app.core.model_loader import model_loader
import logging

logger = logging.getLogger(__name__)

class AgentExecutor:
    """
    Executes actions based on agent decision.
    """

    def __init__(self):
        self.router = AgentRouter()

    def run(self, query: str, session_id: str):
        logger.info(f"[AgentExecutor] session_id={session_id} | Recieved query")

        decision = self.router.route(query)

        logger.info(
            f"[AgentExecutor] session_id={session_id} | Decision: {decision.action} | Reason: {decision.reason}"
        )

        if decision.action == "rag":
            return self._handle_rag(query)

        elif decision.action == "search":
            return self._handle_search(query)

        else:
            return self._handle_direct(query)

    # ---------------------------
    # ACTION HANDLERS
    # ---------------------------

    def _handle_rag(self, query: str, session_id: str):
        logger.info(f"[AgentExecutor] session_id={session_id} | Executing RAG pipeline")

        from app.pipeline.rag_pipeline import RAGPipeline 
        rag = RAGPipeline()

        result = rag.run(query)
        
        logger.info(f"[AgentExecutor] session_id={session_id} | RAG completed")

        return result

    def _handle_search(self, query: str, session_id: str):
        logger.info(f"[AgentExecutor] session_id={session_id} | Executing SEARCH tool")
        
        from app.tools.web_search import WebSearchTool

        tool = WebSearchTool()
        result = tool.search(query)

        logger.info(f"[AgentExecutor] session_id={session_id} | SEARCH completed")

        return {
            "response": result,
            "source": "search"
        }

    def _handle_direct(self, query: str, session_id: str):
        logger.info(f"[AgentExecutor] session_id={session_id} | Executing DIRECT LLM")


        response_text = model_loader.generate(query)

        logger.info(f"[AgentExecutor] session_id={session_id} | DIRECT LLM completed")

        return {
            "response": response_text,
            "source": "llm"
        }