from app.ingestion.schema import IngestedDocument
from app.embeddings.image_embedder import ImageEmbedder

import pytesseract
from PIL import Image
import os
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\karth\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

embedder = ImageEmbedder()

def ingest(file_path: str):
    image = Image.open(file_path)

    # OCR text
    text = pytesseract.image_to_string(image).strip()

    # CLIP embedding
    embedding = embedder.embed(file_path)

    metadata = {
        "source": os.path.basename(file_path),
        "modality": "image",
        "ingestion_time": datetime.utcnow().isoformat(),
        "ocr": True
    }

    return [
        IngestedDocument(
            text = text,
            metadata=metadata,
            embedding=embedding
        )
    ]