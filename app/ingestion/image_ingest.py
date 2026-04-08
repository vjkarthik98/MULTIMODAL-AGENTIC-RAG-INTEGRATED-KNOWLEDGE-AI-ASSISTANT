from app.ingestion.schema import IngestedDocument
from app.ingestion.frame_captioner import generate_caption

import pytesseract
from PIL import Image, ImageOps
import numpy as np

import os
from datetime import datetime
import logging

# Logger
logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\karth\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


def ingest(file_path: str, session_id: str = "default"):
    try:
        logger.info(f"[ImageIngest] session_id={session_id} | Loading image | file={file_path}")

        image = Image.open(file_path)

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = Image.fromarray(np.array(image))
        image.load()

    except Exception as e:
        logger.error(f"[ImageIngest] session_id={session_id} | Invalid image | error={str(e)}")
        raise ValueError(f"Invalid image file: {e}")

    # Step 1: OCR text
    ocr_text = pytesseract.image_to_string(image).strip()

    # Step 2: Caption
    caption = generate_caption(file_path)

    # Step 3: Fallback
    if not caption:
        caption = "An image (caption unavailable)"

    # Step 4: combine text
    final_text = ""

    if caption:
        final_text += f"Image Description: {caption}\n"

    if ocr_text:
        final_text += f"OCR Text: {ocr_text}"

    if not final_text.strip():
        logger.error(f"[ImageIngest] session_id={session_id} | No usable content extracted")
        raise ValueError("Image ingestion failed: No caption or OCR extracted")

    metadata = {
        "source": os.path.basename(file_path),
        "modality": "image",
        "caption": caption,
        "session_id": session_id,
        "ingestion_time": datetime.utcnow().isoformat(),
        "ocr": True
    }

    logger.info(
        f"[ImageIngest] session_id={session_id} | Completed | file={file_path}"
    )

    return [
        IngestedDocument(
            text=final_text.strip(),
            metadata=metadata,
        )
    ]