from dataclasses import dataclass
from typing import Dict, Any, List, Optional 

@dataclass
class IngestedDocument:
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
