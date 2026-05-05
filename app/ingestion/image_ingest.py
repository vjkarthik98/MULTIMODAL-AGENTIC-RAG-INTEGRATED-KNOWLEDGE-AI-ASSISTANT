import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import List

import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageOps

from app.core.config import settings
from app.ingestion.frame_captioner import generate_caption
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)


#  HASH 
def _generate_file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


#  VALIDATION 
def _validate_image(image: Image.Image):
    width, height = image.size

    if width < 32 or height < 32:
        raise ValueError("IMAGE_TOO_SMALL")

    if width == 0 or height == 0:
        raise ValueError("INVALID_IMAGE_DIMENSIONS")


#  PREPROCESS OCR 
def _preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoise = cv2.fastNlMeansDenoising(gray)
    thresh = cv2.adaptiveThreshold(
        denoise,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    return thresh


#  OCR 
def _extract_ocr(image: Image.Image) -> str:
    try:
        processed = _preprocess_for_ocr(image)
        text = pytesseract.image_to_string(processed) or ""

        text = text.strip()

        if len(text) < 10:
            return ""

        return text

    except Exception as e:
        logger.warning(event="ocr_failed", error=str(e))
        return ""


#  CAPTION 
def _generate_caption_safe(file_path: str, session_id: str) -> str:
    try:
        caption = generate_caption(file_path, session_id=session_id)
    except Exception as e:
        logger.warning(event="caption_failed", error=str(e))
        caption = None

    if not caption or len(caption.split()) < 5:
        return getattr(settings, "DEFAULT_IMAGE_CAPTION", "Generic image content")

    return caption.strip()


#  WATERMARK 
def _detect_watermark(text: str) -> bool:
    keywords = ["CONFIDENTIAL", "DRAFT", "WATERMARK"]
    text_upper = text.upper()
    return any(k in text_upper for k in keywords)


#  MAIN 
def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(file_path)

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise ValueError("IMAGE_TOO_LARGE")

    start = time.time()

    try:
        logger.info(event="image_ingest_start", file=file_path)

        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")

        _validate_image(image)

        width, height = image.size

        # resize
        max_dim = getattr(settings, "MAX_IMAGE_DIM", 1024)
        if max(width, height) > max_dim:
            image.thumbnail((max_dim, max_dim))

        source_name = path.name
        source_path = str(path.resolve())

        doc_id = str(uuid.uuid4())
        file_hash = _generate_file_hash(file_path)

        # OCR
        ocr_text = _extract_ocr(image)

        # Caption
        caption = _generate_caption_safe(file_path, session_id)

        watermark_flag = _detect_watermark(ocr_text + " " + caption)

        base_structure = {
            "doc_id": doc_id,
            "session_id": session_id,
            "file_hash": file_hash,
            "source_path": source_path,
            "asset_path": source_path,
            "image_width": width,
            "image_height": height,
            "watermark_detected": watermark_flag,
            "ingestion_time": time.time(),
        }

        documents: List[IngestedDocument] = []

        # caption doc
        documents.append(
            IngestedDocument(
                text=caption,
                modality="image",
                subtype="caption",
                source_type="image",
                source=source_name,
                structure={**base_structure, "content_type": "caption"},
                extra_metadata={
                    "modality_weight": 1.0,
                    "importance_score": 1.0,
                    "data_quality_score": 1.0,
                },
            ).finalize()
        )

        # OCR doc
        if ocr_text:
            documents.append(
                IngestedDocument(
                    text=ocr_text,
                    modality="image",
                    subtype="ocr",
                    source_type="image",
                    source=source_name,
                    structure={**base_structure, "content_type": "ocr"},
                    extra_metadata={
                        "modality_weight": 0.9,
                        "importance_score": 0.8,
                        "data_quality_score": 0.8,
                    },
                ).finalize()
            )

        if not documents:
            raise ValueError("NO_VALID_IMAGE_DOCS")

        latency = round(time.time() - start, 2)

        logger.info(
            event="image_ingest_success",
            docs=len(documents),
            latency=latency
        )

        return documents

    except Exception as e:
        logger.error(event="image_ingest_failed", error=str(e))
        raise