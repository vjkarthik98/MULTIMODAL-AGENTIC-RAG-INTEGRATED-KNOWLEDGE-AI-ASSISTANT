from app.utils.chunking import chunk_text
from app.ingestion.schema import IngestedDocument

import os
from datetime import datetime
import logging

# Logger
logger = logging.getLogger(__name__)


def ingest(file_path: str) -> list[IngestedDocument]:
    try:
        logger.info(f"[TextIngest] Starting ingestion | file={file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            logger.error(f"[TextIngest] Empty file | file={file_path}")
            raise ValueError("Empty text file")

        metadata = {
            "source": os.path.basename(file_path),
            "modality": "text",
            "ingestion_time": datetime.utcnow().isoformat()
        }

        chunks = chunk_text(text)

        logger.info(
            f"[TextIngest] Chunking completed | file={file_path} | chunks={len(chunks)}"
        )

        documents = [
            IngestedDocument(
                text=chunk,
                metadata={
                    **metadata,
                    "chunk_id": i
                }
            )
            for i, chunk in enumerate(chunks)
        ]

        return documents

    except Exception as e:
        logger.error(f"[TextIngest] Failed | file={file_path} | error={str(e)}")
        raise