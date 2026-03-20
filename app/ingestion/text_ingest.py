from app.ingestion.schema import IngestedDocument
import os
from datetime import datetime

def ingest(file_path: str) -> IngestedDocument:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    metadata = {
        "source": os.path.basename(file_path),
        "modality": "text",
        "ingestion_time": datetime.utcnow().isoformat()
    }

    return IngestedDocument(
        text = text,
        metadata=metadata
    )