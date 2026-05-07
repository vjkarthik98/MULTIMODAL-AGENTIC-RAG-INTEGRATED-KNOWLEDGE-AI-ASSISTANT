import hashlib
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

from app.core.config import settings
from app.ingestion.frame_captioner import generate_caption
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)


# SUPPORTED FORMATS

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# HASH

def _file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# VALIDATION

def _validate_image(image: Image.Image) -> None:
    width, height = image.size

    if width == 0 or height == 0:
        raise ValueError("INVALID_IMAGE_DIMENSIONS")

    if width < 32 or height < 32:
        raise ValueError("IMAGE_TOO_SMALL")


def _check_aspect_ratio(width: int, height: int) -> Optional[str]:
    if height == 0:
        return None
    ratio = width / height
    if ratio > 20 or ratio < 0.05:
        return f"EXTREME_ASPECT_RATIO_{ratio:.2f}"
    return None


# OCR PREPROCESSING

def _preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    img    = np.array(image)
    gray   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # DENOISE
    denoise = cv2.fastNlMeansDenoising(gray, h=10)
    # ADAPTIVE THRESHOLD
    thresh = cv2.adaptiveThreshold(
        denoise,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return thresh


# OCR

def _extract_ocr(image: Image.Image) -> str:
    try:
        processed = _preprocess_for_ocr(image)
        text      = (pytesseract.image_to_string(processed) or "").strip()

        if len(text) < 10:
            return ""

        return text

    except Exception as e:
        logger.warning(event="ocr_failed", error=str(e))
        return ""


# BLUR DETECTION

def _blur_score(image: Image.Image) -> float:
    try:
        gray      = np.array(image.convert("L"))
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize: > 100 = sharp, < 20 = blurry
        return float(min(laplacian / 100.0, 1.0))
    except Exception:
        return 1.0


# CAPTION

def _generate_caption_safe(file_path: str, session_id: str) -> str:
    try:
        caption = generate_caption(file_path, session_id=session_id)
    except Exception as e:
        logger.warning(event="caption_failed", error=str(e))
        caption = None

    if not caption or len(caption.split()) < 5:
        logger.warning(event="caption_too_short", file=Path(file_path).name)
        return "Generic image content"

    return caption.strip()


# WATERMARK DETECTION

def _detect_watermark(text: str) -> bool:
    keywords   = {"CONFIDENTIAL", "DRAFT", "WATERMARK", "SAMPLE", "RESTRICTED"}
    text_upper = text.upper()
    return any(k in text_upper for k in keywords)


# QUALITY SCORE

def _image_quality_score(
    blur: float,
    has_ocr: bool,
    watermark: bool,
    width: int,
    height: int,
) -> float:
    score = blur

    # RESOLUTION BOOST
    pixel_count = width * height
    if pixel_count > 500_000:
        score = min(score + 0.1, 1.0)

    # PENALTIES
    if watermark:
        score = max(score - 0.2, 0.0)

    if not has_ocr:
        score = max(score - 0.05, 0.0)

    return round(score, 3)


# MAIN

def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(f"UNSUPPORTED_IMAGE_FORMAT: {ext}")

    file_size = path.stat().st_size

    if file_size == 0:
        raise ValueError("EMPTY_FILE")

    if file_size > settings.MAX_FILE_SIZE_IMAGE:
        raise ValueError(
            f"IMAGE_TOO_LARGE: {file_size} bytes exceeds {settings.MAX_FILE_SIZE_IMAGE} bytes"
        )

    start = time.time()

    try:
        logger.info(
            event="image_ingest_start",
            file=path.name,
            size=file_size,
            session_id=session_id,
        )

        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")

        _validate_image(image)

        original_width, original_height = image.size

        # ASPECT RATIO CHECK
        aspect_warning = _check_aspect_ratio(original_width, original_height)
        if aspect_warning:
            logger.warning(event="aspect_ratio_warning", warning=aspect_warning, file=path.name)

        # RESIZE IF NEEDED
        if max(original_width, original_height) > settings.MAX_IMAGE_DIM:
            image.thumbnail((settings.MAX_IMAGE_DIM, settings.MAX_IMAGE_DIM), Image.LANCZOS)

        width, height = image.size

        source_name = path.name
        source_path = str(path.resolve())
        doc_id      = str(uuid.uuid4())
        file_hash   = _file_hash(file_path)

        # ANALYSIS
        ocr_text  = _extract_ocr(image)
        caption   = _generate_caption_safe(file_path, session_id)
        blur      = _blur_score(image)
        watermark = _detect_watermark(ocr_text + " " + caption)
        quality   = _image_quality_score(blur, bool(ocr_text), watermark, width, height)

        base_structure: Dict = {
            "doc_id":            doc_id,
            "session_id":        session_id,
            "file_hash":         file_hash,
            "source_path":       source_path,
            "asset_path":        source_path,
            "original_width":    original_width,
            "original_height":   original_height,
            "image_width":       width,
            "image_height":      height,
            "image_format":      ext.lstrip(".").upper(),
            "watermark_detected": watermark,
            "blur_score":        blur,
            "ingestion_time":    time.time(),
        }

        documents: List[IngestedDocument] = []

        # CAPTION DOCUMENT
        documents.append(
            IngestedDocument(
                text=caption,
                modality="image",
                subtype="caption",
                source_type="image",
                source=source_name,
                structure={**base_structure, "content_type": "image_caption"},
                extra_metadata={
                    "modality_weight":    1.0,
                    "importance_score":   quality,
                    "data_quality_score": quality,
                },
            ).finalize()
        )

        # OCR DOCUMENT
        if ocr_text:
            documents.append(
                IngestedDocument(
                    text=ocr_text,
                    modality="image",
                    subtype="ocr",
                    source_type="image",
                    source=source_name,
                    structure={**base_structure, "content_type": "image_ocr"},
                    extra_metadata={
                        "modality_weight":    0.9,
                        "importance_score":   min(quality + 0.1, 1.0),
                        "data_quality_score": 0.8,
                    },
                ).finalize()
            )

        if not documents:
            raise ValueError("NO_VALID_IMAGE_DOCS")

        latency = round(time.time() - start, 2)

        logger.info(
            event="image_ingest_success",
            file=path.name,
            docs=len(documents),
            blur=blur,
            quality=quality,
            watermark=watermark,
            has_ocr=bool(ocr_text),
            latency=latency,
            session_id=session_id,
        )

        return documents

    except Exception as e:
        logger.error(
            event="image_ingest_failed",
            file=path.name,
            session_id=session_id,
            error=str(e),
            latency=round(time.time() - start, 2),
        )
        raise