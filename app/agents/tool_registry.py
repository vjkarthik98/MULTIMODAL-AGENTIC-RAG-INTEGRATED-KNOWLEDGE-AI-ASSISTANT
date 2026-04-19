from typing import Dict, Any, Callable
from app.pipeline.rag_pipeline import RAGPipeline
from app.tools.web_search import WebSearchTool
from app.memory.redis_memory import RedisMemory
from app.memory.memory_filter import filter_relevant_history
from app.reasoning.query_decomposer import QueryDecomposer
from app.reasoning.reasoning_engine import ReasoningEngine
from app.reasoning.result_fusion import ResultFusion
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

# Logger
logger = get_logger(__name__)

class Tool:
    def __init__(
        self,
        name:str,
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
            logger.info(f"[Tool:{self.name}] Execution started")

            result = self.handler(query, context or {}, session_id)

            logger.info(f"[Tool:{self.name}] Execution completed")

            return {
                "tool": self.name,
                "result": result,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"[Tool:{self.name}] Failed | {str(e)}")

            return {
                "tool": self.name,
                "result": None,
                "status": "error",
                "error": str(e)
            }
        
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    # Tool Registration
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        logger.debug(f"[ToolRegistry] Registered tool: {tool.name}")

    def get(self, tool_name: str) -> Tool:
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
        return self.tools[tool_name]
    
    def list_tools(self) -> Dict[str, str]:
        return {
            name: tool.description
            for name, tool in self.tools.items()
        }
    

    # Default Tools
    def _register_default_tools(self):

        # RAG TOOL
        def rag_tool(query, context, session_id):

            rag = RAGPipeline()
            return rag.run(query, session_id=session_id)
        
        self.register(
            Tool(
                name="rag",
                description="Retrieve from internal multimodal knowledge base",
                handler=rag_tool,
                tool_type="retrieval",
                cost="medium"
            )
        )

        # Search Tool
        def search_tool(query, context, session_id):
            tool = WebSearchTool()
            return tool.execute(query, context, session_id)
        
        self.register(
            Tool(
                name="search",
                description="Search external real-time information",
                handler=search_tool,
                tool_type="external",
                cost="high"
            )
        )

        # Memory Tool
        def memory_tool(query, context, session_id):

            memory = RedisMemory()
            history = memory.get_history(session_id)

            if not history:
                return "No relevant memory found."
            
            embedder = model_loader.get_embedder()

            filtered = filter_relevant_history(
                query=query,
                history=history,
                embedder=embedder
            )

            return "\n".join([m["content"] for m in filtered])
        
        self.register(
            Tool(
                name="memory",
                description = "Retriever relevant past conversation context",
                handler=memory_tool,
                tool_type="memory",
                cost="low"
            )
        )
        
        # Query Decomposer
        def decompose_tool(query, context, session_id):
            
            decomposer = QueryDecomposer(model_loader)
            return decomposer.decompose(query)
        
        self.register(
            Tool(
                name="decompose",
                description="Break complex query into sub-queries",
                handler=decompose_tool,
                tool_type="reasoning",
                cost="low"
            )
        )

        # Reasoning Engine
        def reasoning_tool(query, context, session_id):

            engine = ReasoningEngine(model_loader)

            return engine.generate_answer(
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
                cost = "high"
            )
        )

        # Result Fusion
        def fusion_tool(query, context, session_id):

            fusion = ResultFusion()

            return fusion.fuse(context.get("results", []))
        
        self.register(
            Tool(
                name="fusion",
                description="Fuse and rank results",
                handler=fusion_tool,
                tool_type="postprocessing",
                cost="low"
            )
        )

        logger.info("[ToolRegistry]  All tools registered successfully")
            

        
