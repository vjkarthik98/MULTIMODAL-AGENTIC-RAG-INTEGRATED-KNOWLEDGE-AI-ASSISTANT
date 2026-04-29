from typing import Optional
from pathlib import Path

from PIL import Image

from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

try:
    import torch
except ImportError:
    torch = None


logger = get_logger(__name__)


# CLEAN AND NORMALIZE CAPTION TEXT
def _clean_caption(text: str) -> Optional[str]:
    cleaned = (text or "").strip()

    if not cleaned:
        return None

    # REMOVE WEAK PREFIXES
    weak_prefixes = (
        "a blurry image of",
        "a close up of",
        "an image of",
        "a picture of",
    )

    lowered = cleaned.lower()
    for prefix in weak_prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break

    words = cleaned.split()

    if not words:
        return None

    # LIMIT CAPTION LENGTH
    max_words = getattr(settings, "CAPTION_MAX_WORDS", 30)
    cleaned = " ".join(words[:max_words]).strip()

    # CAPITALIZE FIRST LETTER
    cleaned = cleaned[:1].upper() + cleaned[1:]

    # FINAL QUALITY CHECK
    return cleaned if len(cleaned) >= 3 else None


# LOAD AND VALIDATE IMAGE
def _load_image(image_path: str):
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"{image_path} NOT FOUND")

    # FILE SIZE CHECK
    if path.stat().st_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError("IMAGE TOO LARGE")

    with Image.open(path) as img:

        # ENSURE RGB FORMAT
        img = img.convert("RGB")

        # RESIZE IF TOO LARGE
        max_dim = getattr(settings, "MAX_IMAGE_DIM", 1024)
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))

        return img.copy()


# MAIN CAPTION GENERATION FUNCTION
def generate_caption(image_path: str, session_id: str = "default") -> Optional[str]:

    # VALIDATE SESSION
    if not session_id:
        raise ValueError("SESSION_ID REQUIRED")

    # CHECK TORCH AVAILABILITY
    if torch is None:
        logger.warning("[FrameCaptioner] TORCH NOT AVAILABLE")
        return None

    try:
        logger.debug(
            "[FrameCaptioner][START] session_id=%s | image=%s",
            session_id,
            image_path
        )

        # LOAD IMAGE
        image = _load_image(image_path)

        # LOAD MODEL
        processor, model, device = model_loader.get_blip()

        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=settings.BLIP_MAX_TOKENS,
                num_beams=settings.BLIP_NUM_BEAMS,
            )

        caption_raw = processor.decode(output[0], skip_special_tokens=True)

        # CLEAN CAPTION
        caption = _clean_caption(caption_raw)

        # VALIDATE CAPTION QUALITY
        if not caption:
            logger.warning("[FrameCaptioner] WEAK OR EMPTY CAPTION")
            return None

        # GLOBAL SAFETY TRUNCATION
        if len(caption) > settings.MAX_PROMPT_CHARS:
            caption = caption[:settings.MAX_PROMPT_CHARS]

        logger.debug(
            "[FrameCaptioner][SUCCESS] session_id=%s | caption_len=%s",
            session_id,
            len(caption)
        )

        return caption

    except Exception as e:
        logger.error(
            "[FrameCaptioner][FAILED] session_id=%s | image=%s | error=%s",
            session_id,
            image_path,
            str(e)
        )
        return None