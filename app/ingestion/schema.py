from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class IngestedDocument:
    text: str
    metadata: Dict[str, Any]