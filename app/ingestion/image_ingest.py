import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import List

import pytesseract
from PIL import Image, ImageOps

from app.core.config import settings
from app.ingestion.frame_captioner import generate_caption
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _generate_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise ValueError(f"Image too large: {size_mb:.2f}MB")

    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")

        max_dim = getattr(settings, "MAX_IMAGE_DIM", 1024)
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim))

        image.load()
        return image.copy()


def ingest(file_path: str, session_id: str = "default") -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("session_id is required")

    path = Path(file_path)

    start = time.time()

    try:
        logger.info("[ImageIngest][START] session_id=%s | file=%s", session_id, file_path)

        image = _load_image(path)

        width, height = image.size

        source_name = path.name
        source_path = str(path.resolve())

        doc_id = str(uuid.uuid4())
        file_hash = _generate_file_hash(file_path)

        # OCR
        ocr_text = ""
        try:
            ocr_text = (pytesseract.image_to_string(image) or "").strip()

            if len(ocr_text) > settings.MAX_PROMPT_CHARS:
                logger.warning("[ImageIngest] OCR truncated")
                ocr_text = ocr_text[:settings.MAX_PROMPT_CHARS]

        except Exception as e:
            logger.warning("[ImageIngest][OCR_FAIL] %s", str(e))

        # Caption
        try:
            caption = generate_caption(file_path, session_id=session_id)
        except Exception as e:
            logger.warning("[ImageIngest][CAPTION_FAIL] %s", str(e))
            caption = None

        if not caption:
            caption = getattr(settings, "DEFAULT_IMAGE_CAPTION", "Image content")

        base_structure = {
            "doc_id": doc_id,
            "session_id": session_id,
            "file_hash": file_hash,
            "source_path": source_path,
            "asset_path": source_path,
            "image_width": width,
            "image_height": height,
            "modality_source": "image",
            "ingestion_time": time.time(),
        }

        documents: List[IngestedDocument] = []

        # Caption document
        documents.append(
            IngestedDocument(
                text=caption[:settings.MAX_PROMPT_CHARS],
                modality="image",
                subtype="caption",
                source_type="image",
                source=source_name,
                structure={
                    **base_structure,
                    "content_type": "semantic_description",
                },
            )
        )

        # OCR document
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

        latency = round(time.time() - start, 2)

        logger.info(
            "[ImageIngest][SUCCESS] session_id=%s | docs=%s | latency=%ss",
            session_id,
            len(documents),
            latency
        )

        return documents

    except Exception as e:
        logger.error("[ImageIngest][FAILED] %s", str(e))
        raise