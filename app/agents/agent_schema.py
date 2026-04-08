from pydantic import BaseModel
from typing import Literal

class AgentDecision(BaseModel):
    """
    Structured output from LLM router.
    """

    action: Literal["rag", "search", "direct"]
    reason: str