from typing import List, Dict, Any
from app.utils.logger import get_logger
from app.agents.agent_schema import AgentDecision

# Logger
logger = get_logger(__name__)

class ExecutionStep:
    def __init__(self, tool: str, description: str = ""):
        self.tool = tool
        self.description = description

    def to_dict(self):
        return {
            "tool": self.tool,
            "description": self.description
        }
    
class ExecutionPlan:
    def __init__(self, steps: List[ExecutionStep]):
        self.steps = steps

    def to_list(self):
        return [step.to_dict() for step in self.steps]
    
class Planner:
    def create_plan(
        self,
        decision: AgentDecision,
        query: str
    ) -> ExecutionPlan:
        
        logger.info(f"[Planner] Creating plan | action = {decision.action}")

        signals = decision.signals or {}

        # Simple Actions
        if decision.action == "direct":
            return ExecutionPlan([
                ExecutionStep("reason", "Direct reasoning without retrieval")
            ])
        
        if decision.action == "search":
            return ExecutionPlan([
                ExecutionStep("search", "Fetch external data"),
                ExecutionStep("reason", "Summarize and answer")
            ])
        
        if decision.action == "memory":
            return ExecutionPlan([
                ExecutionStep("memory", "Fetch past context"),
                ExecutionStep("reason", "Answer using memory")
            ])
        
        # RAG 
        if decision.action == "rag":

            steps = []

            # Complex query -> decompose
            if signals.get("is_complex"):
                steps.append(
                    ExecutionStep("decompose", "Break into sub_queries")
                )

            steps.append(
                ExecutionStep("rag", "Retrieve Knowledge")
            )

            # Multi-result -> fuse
            if signals.get("is_complex"):
                steps.append(
                    ExecutionStep("fusion", "Merge retrieval results")
                )

            steps.append(
                ExecutionStep("reason", "Generate final answer")
            )

            return ExecutionPlan(steps)
        
        # Hybrid
        if decision.action == "hybrid":

            steps = []

            # Step 1: Memory first
            steps.append(
                ExecutionStep("memory", "Fetch conversation context")
            )

            # Step 2: Decompose if complex
            if signals.get("is_complex"):
                steps.append(
                    ExecutionStep("decompose", "split query")
                )

            # Step 3: Retrieval
            steps.append(
                ExecutionStep("rag", "Retrieve knowledge")
            )

            # Step 4: Fusion
            steps.append(
                ExecutionStep("fusion", "Combine results")
            )

            # Step 5: Reasoning
            steps.append(
                ExecutionStep("reason", "Final reasoning with memory + Knowledge")
            )

            return ExecutionPlan(steps)
        
        # Fallback
        logger.warning("[Planner] Unknown action -> fallback")

        return ExecutionPlan([
            ExecutionStep("reason", "Fallback reasoning")
        ])