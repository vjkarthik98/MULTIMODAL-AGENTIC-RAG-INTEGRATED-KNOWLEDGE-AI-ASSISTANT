from app.ingestion.schema import IngestedDocument
import pytesseract
from PIL import Image
import os
from datetime import datetime


pytesseract.pytesseract.tesseract_cmd = r"C:\Users\karth\AppData\Local\Programs\Tesseract-OCR\tesseract.exe "


def ingest(file_path: str) -> IngestedDocument:
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image).strip()

    metadata = {
        "source": os.path.basename(file_path),
        "modality": "image",
        "ingestion_time": datetime.utcnow().isoformat(),
        "ocr": True
    }

    return IngestedDocument(
        text = text,
        metadata=metadata
    )