import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel
from app.ingestion.schema import IngestedDocument
from datetime import datetime

# Load model once (global)
model = WhisperModel("base", compute_type="int8")

def ingest(file_path:str) -> list[IngestedDocument]:
    """
    Audio ingestion:
    - Coverts audio -> text using faster-whisper
    - Returns segment-wise documents (NOT single text)
    """
    
    try:
        segments, info = model.transcribe(file_path)

        documents = []

        for i, segment in enumerate(segments):
            # skip empty or very small segments (important for quality)
            if not segment.text or len(segment.text.strip()) < 2:
                continue

            documents.append(
                IngestedDocument(
                    text=segment.text.strip(),
                    metadata={
                        "source": os.path.basename(file_path),
                        "modality": "audio",
                        "chunk_id": i,
                        "start_time": segment.start,
                        "end_time": segment.end,
                        "language": info.language,
                        "ingestion_time": datetime.utcnow().isoformat(),
                        "model": "faster-whisper-base"
                    }
                )
            )

            # Validation (important for pipeline safety)
            if not documents:
                raise ValueError("No valid audio segments extracted")
            
            return documents
        
    except Exception as e:
        raise RuntimeError(f"Audio ingestion failed: {str(e)}")

    

    