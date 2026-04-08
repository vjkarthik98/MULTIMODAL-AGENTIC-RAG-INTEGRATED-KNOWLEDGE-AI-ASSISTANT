import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from app.core.model_loader import model_loader
from app.ingestion.schema import IngestedDocument
from datetime import datetime
import logging

# Logger
logger = logging.getLogger(__name__)


def ingest(file_path: str) -> list[IngestedDocument]:
    """
    Audio ingestion:
    - Converts audio -> text using faster-whisper
    - Returns segment-wise documents (NOT single text)
    """

    try:
        logger.info(f"[AudioIngest] Starting ingestion | file={file_path}")

        audio_model = model_loader.get_whisper()

        segments, info = audio_model.transcribe(file_path)

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
                        "model": model_loader.MODEL_CONFIG["whisper"]
                    }
                )
            )

        # Validation (important for pipeline safety)
        if not documents:
            logger.error(f"[AudioIngest] No valid segments | file={file_path}")
            raise ValueError("No valid audio segments extracted")

        logger.info(
            f"[AudioIngest] Completed | file={file_path} | segments={len(documents)}"
        )

        return documents

    except Exception as e:
        logger.error(f"[AudioIngest] Failed | file={file_path} | error={str(e)}")
        raise RuntimeError(f"Audio ingestion failed: {str(e)}")