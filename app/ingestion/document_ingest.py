import os
import pdfplumber
import docx
import pandas as pd
import logging

from datetime import datetime
from app.ingestion.schema import IngestedDocument
from app.utils.chunking import chunk_text

# ✅ Logger
logger = logging.getLogger(__name__)


def ingest(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    text = ""

    try:
        logger.info(f"[DocumentIngest] Starting ingestion | file={file_path}")

        # PDF
        if ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

        # Word
        elif ext == ".docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"

        # Excel
        elif ext in [".xlsx", ".xls"]:
            try:
                df = pd.read_excel(file_path, engine="openpyxl")
            except Exception:
                df = pd.read_excel(file_path)

            text = df.astype(str).to_string()

        else:
            logger.error(f"[DocumentIngest] Unsupported file type | file={file_path}")
            raise ValueError("Unsupported document type")

    except Exception as e:
        logger.error(f"[DocumentIngest] Parsing failed | file={file_path} | error={str(e)}")
        raise ValueError(f"Document parsing failed: {str(e)}")

    # Extra Safety
    if not text.strip():
        logger.error(f"[DocumentIngest] Empty content | file={file_path}")
        raise ValueError("No readable content extracted from document")

    metadata = {
        "source": os.path.basename(file_path),
        "modality": "document",
        "ingestion_time": datetime.utcnow().isoformat()
    }

    # Chunking
    chunks = chunk_text(text)

    logger.info(
        f"[DocumentIngest] Chunking completed | file={file_path} | chunks={len(chunks)}"
    )

    return [
        IngestedDocument(
            text=chunk,
            metadata={**metadata, "chunk_id": i}
        )
        for i, chunk in enumerate(chunks)
    ]