import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

try:
    import torch
except ImportError:
    torch = None

logger = get_logger(__name__)


# WEAK PATTERNS

_WEAK_PREFIXES = [
    "a blurry image of",
    "a close up of",
    "an image of",
    "a picture of",
    "a photo of",
    "photo of",
    "image of",
    "this is a",
    "this is an",
]

_CAPTION_MAX_WORDS = 30
_CAPTION_MIN_WORDS = 5


# REPETITION CLEAN

def _remove_repetition(text: str) -> str:
    words = text.split()
    if len(words) < 6:
        return text

    half  = len(words) // 2
    first = " ".join(words[:half])
    second = " ".join(words[half:])

    if first.strip().lower() == second.strip().lower():
        return first.strip()

    return text


# CAPTION CLEAN

def _clean_caption(text: str) -> Optional[str]:
    if not text:
        return None

    text  = text.strip()
    lower = text.lower()

    for pattern in _WEAK_PREFIXES:
        if lower.startswith(pattern):
            text = text[len(pattern):].strip()
            lower = text.lower()
            break

    text = _remove_repetition(text)

    words = text.split()

    if len(words) < _CAPTION_MIN_WORDS:
        return None

    text = " ".join(words[:_CAPTION_MAX_WORDS])
    text = text[0].upper() + text[1:] if text else text

    return text if len(text) > 10 else None


# CAPTION CONFIDENCE PROXY

def _caption_confidence(caption: str) -> float:
    words      = caption.split()
    unique     = set(w.lower() for w in words)
    word_count = len(words)

    if word_count == 0:
        return 0.0

    diversity  = len(unique) / word_count
    length_score = min(len(caption) / 100.0, 1.0)

    return round((diversity + length_score) / 2.0, 3)


# IMAGE LOADING

def _load_image(image_path: str) -> Image.Image:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"IMAGE_NOT_FOUND: {image_path}")

    if path.stat().st_size > settings.MAX_FILE_SIZE_IMAGE:
        raise ValueError(
            f"IMAGE_TOO_LARGE: {path.stat().st_size} bytes exceeds {settings.MAX_FILE_SIZE_IMAGE}"
        )

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)

        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size

        if w < 32 or h < 32:
            raise ValueError("IMAGE_TOO_SMALL")

        if max(w, h) > settings.MAX_IMAGE_DIM:
            img.thumbnail(
                (settings.MAX_IMAGE_DIM, settings.MAX_IMAGE_DIM),
                Image.LANCZOS,
            )

        return img.copy()


# MAIN

def generate_caption(image_path: str, session_id: str) -> Optional[str]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    if torch is None:
        logger.warning(event="torch_not_available", session_id=session_id)
        return None

    start = time.time()

    try:
        logger.debug(
            event="caption_start",
            image=Path(image_path).name,
            session_id=session_id,
        )

        image = _load_image(image_path)

        processor, model, device = model_loader.get_blip()

        inputs = processor(image, return_tensors="pt").to(device)

        t_infer = time.time()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=settings.BLIP_MAX_TOKENS,
                num_beams=settings.BLIP_NUM_BEAMS,
            )

        infer_latency = round((time.time() - t_infer) * 1000, 1)

        if infer_latency > settings.LATENCY_TARGET_IMAGE_MS:
            logger.warning(
                event="caption_latency_exceeded",
                latency_ms=infer_latency,
                target_ms=settings.LATENCY_TARGET_IMAGE_MS,
                session_id=session_id,
            )

        caption_raw = processor.decode(output[0], skip_special_tokens=True)
        caption     = _clean_caption(caption_raw)

        if not caption:
            logger.warning(
                event="caption_rejected",
                raw=caption_raw[:80] if caption_raw else "",
                session_id=session_id,
            )
            return None

        confidence = _caption_confidence(caption)

        logger.debug(
            event="caption_success",
            length=len(caption),
            words=len(caption.split()),
            confidence=confidence,
            latency_ms=infer_latency,
            session_id=session_id,
        )

        return caption

    except Exception as e:
        logger.error(
            event="caption_failed",
            image=Path(image_path).name,
            session_id=session_id,
            error=str(e),
            latency=round(time.time() - start, 3),
        )
        return None