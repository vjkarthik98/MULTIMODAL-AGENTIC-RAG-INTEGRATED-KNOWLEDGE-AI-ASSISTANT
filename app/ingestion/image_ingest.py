import hashlib
import os
import uuid

import pytesseract
from PIL import Image, ImageOps

from app.core.config import settings
from app.ingestion.frame_captioner import generate_caption
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger


logger = get_logger(__name__)

DEFAULT_TESSERACT_PATH = r"C:\Users\karth\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

if settings.TESSERACT_CMD and os.path.exists(settings.TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
elif os.path.exists(DEFAULT_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_PATH


def _generate_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as file_handle:
        return hashlib.md5(file_handle.read()).hexdigest()


def ingest(file_path: str, session_id: str = "default") -> list[IngestedDocument]:
    if not session_id:
        raise ValueError("session_id is required")
    if not os.path.exists(file_path):
        raise ValueError(f"{file_path} not found")

    source_name = os.path.basename(file_path)
    source_path = os.path.abspath(file_path)
    doc_id = str(uuid.uuid4())
    file_hash = _generate_file_hash(file_path)

    try:
        logger.info("[ImageIngest][START] session_id=%s | file=%s", session_id, file_path)
        with Image.open(file_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
            image.load()

        width, height = image.size
    except Exception as exc:
        logger.error("[ImageIngest][INVALID] session_id=%s | error=%s", session_id, exc)
        raise ValueError(f"Invalid image file: {exc}") from exc

    ocr_text = ""
    try:
        ocr_text = (pytesseract.image_to_string(image) or "").strip()
    except Exception as exc:
        logger.error("[ImageIngest][OCR_FAIL] session_id=%s | error=%s", session_id, exc)

    try:
        caption = generate_caption(file_path, session_id=session_id)
    except Exception as exc:
        logger.error("[ImageIngest][CAPTION_FAIL] session_id=%s | error=%s", session_id, exc)
        caption = None

    caption = caption or "An image (caption unavailable)"

    base_structure = {
        "doc_id": doc_id,
        "session_id": session_id,
        "file_hash": file_hash,
        "source_path": source_path,
        "asset_path": source_path,
        "image_width": width,
        "image_height": height,
        "modality_source": "image",
    }

    documents = [
        IngestedDocument(
            text=caption,
            modality="image",
            subtype="caption",
            source_type="image",
            source=source_name,
            structure={
                **base_structure,
                "content_type": "semantic_description",
            },
        )
    ]

    if ocr_text:
        documents.append(
            IngestedDocument(
                text=ocr_text,
                modality="image",
                subtype="ocr",
                source_type="image",
                source=source_name,
                structure={
                    **base_structure,
                    "content_type": "extracted_text",
                },
            )
        )

    logger.info("[ImageIngest][SUCCESS] session_id=%s | docs=%s", session_id, len(documents))
    return documents
