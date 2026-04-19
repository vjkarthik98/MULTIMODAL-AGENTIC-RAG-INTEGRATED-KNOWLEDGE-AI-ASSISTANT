from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List

class AgentDecision(BaseModel):
    
    # CORE DECISION
    action: Literal[
        "rag",
        "search",
        "direct",
        "memory",
        "hybrid"
    ]

    reason: str = Field(..., min_length=3)

    # CONFIDENCE 
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score of decision"
    )

    # SIGNALS (FROM QUERY ANALYSIS)
    signals: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Signals like recency, complexity, modality hints"
    )

    # TOOLS HINTS
    suggested_tools: Optional[List[str]] = Field(
        default_factory=list,
        description="Optional tools suggested by router"
    )

    # DEBUG / TRACE
    trace: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Debug info for observability"
    )

    # VALIDATION
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.7
    
    def requires_fallback(self) -> bool:
        return self.confidence < 0.4