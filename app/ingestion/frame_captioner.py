from typing import Optional
from pathlib import Path

from PIL import Image, ImageOps

from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

try:
    import torch
except ImportError:
    torch = None


logger = get_logger(__name__)


#  CLEAN 
def _clean_caption(text: str) -> Optional[str]:
    if not text:
        return None

    text = text.strip()

    weak_patterns = [
        "a blurry image of",
        "a close up of",
        "an image of",
        "a picture of",
        "photo of",
    ]

    lower = text.lower()
    for p in weak_patterns:
        if lower.startswith(p):
            text = text[len(p):].strip()
            break

    words = text.split()

    if len(words) < 3:
        return None

    max_words = getattr(settings, "CAPTION_MAX_WORDS", 30)
    text = " ".join(words[:max_words])

    text = text[0].upper() + text[1:] if text else text

    return text if len(text) > 5 else None


#  IMAGE 
def _load_image(image_path: str) -> Image.Image:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(image_path)

    if path.stat().st_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError("IMAGE_TOO_LARGE")

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")

        if img.size[0] < 32 or img.size[1] < 32:
            raise ValueError("IMAGE_TOO_SMALL")

        max_dim = getattr(settings, "MAX_IMAGE_DIM", 1024)
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))

        return img.copy()


#  MAIN 
def generate_caption(image_path: str, session_id: str) -> Optional[str]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    if torch is None:
        logger.warning(event="torch_not_available")
        return None

    try:
        logger.debug(event="caption_start", image=image_path)

        image = _load_image(image_path)

        processor, model, device = model_loader.get_blip()

        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=settings.BLIP_MAX_TOKENS,
                num_beams=settings.BLIP_NUM_BEAMS,
            )

        caption_raw = processor.decode(output[0], skip_special_tokens=True)

        caption = _clean_caption(caption_raw)

        if not caption:
            logger.warning(event="caption_rejected")
            return None

        logger.debug(
            event="caption_success",
            length=len(caption)
        )

        return caption

    except Exception as e:
        logger.error(event="caption_failed", error=str(e))
        return None