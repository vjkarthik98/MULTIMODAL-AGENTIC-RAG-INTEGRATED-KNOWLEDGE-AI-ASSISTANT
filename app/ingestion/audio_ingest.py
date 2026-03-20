import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel
from app.ingestion.schema import IngestedDocument
import os
from datetime import datetime

model = WhisperModel("base", compute_type="int8")

def ingest(file_path:str) -> IngestedDocument:
    segments, info = model.transcribe(file_path)

    text = ""
    for segment in segments:
        text += segment.text + " "
    
    metadata = {
        "source": os.path.basename(file_path),
        "modality": "audio",
        "ingestion_time": datetime.utcnow().isoformat(),
        "model": "faster-whisper-base",
        "language": info.language
    }
    return IngestedDocument(
        text = text.strip(),
        metadata=metadata
    )

    