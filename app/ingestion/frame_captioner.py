from typing import Optional

from PIL import Image

from app.core.model_loader import model_loader
from app.utils.logger import get_logger

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


logger = get_logger(__name__)


def _clean_caption(text: str) -> Optional[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    weak_prefixes = (
        "a blurry image of",
        "a close up of",
        "an image of",
        "a picture of",
    )

    lowered = cleaned.lower()
    for prefix in weak_prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break

    words = cleaned.split()
    if not words:
        return None

    cleaned = " ".join(words[:30]).strip()
    cleaned = cleaned[:1].upper() + cleaned[1:]
    return cleaned if len(cleaned) >= 3 else None


def generate_caption(image_path: str, session_id: str = "default") -> Optional[str]:
    try:
        if torch is None:
            logger.warning("[FrameCaptioner] session_id=%s | torch unavailable", session_id)
            return None

        logger.debug(
            "[FrameCaptioner] session_id=%s | Generating caption | image=%s",
            session_id,
            image_path,
        )

        try:
            with Image.open(image_path) as raw_image:
                image = raw_image.convert("RGB")
        except Exception as exc:
            logger.error(
                "[FrameCaptioner] session_id=%s | Invalid image | error=%s",
                session_id,
                exc,
            )
            return None

        processor, model, device = model_loader.get_blip()
        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
            )

        caption = _clean_caption(processor.decode(output[0], skip_special_tokens=True))
        if caption is None:
            logger.warning("[FrameCaptioner] session_id=%s | Weak caption generated", session_id)
            return None

        logger.debug("[FrameCaptioner] session_id=%s | Caption generated", session_id)
        return caption

    except Exception as exc:
        logger.error(
            "[FrameCaptioner] session_id=%s | Failed | image=%s | error=%s",
            session_id,
            image_path,
            exc,
        )
        return None
