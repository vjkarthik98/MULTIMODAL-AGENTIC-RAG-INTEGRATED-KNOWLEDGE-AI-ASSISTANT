from app.core.model_loader import model_loader
from PIL import Image
import torch
import logging

# Logger
logger = logging.getLogger(__name__)


def generate_caption(image_path: str) -> str:
    """
    Generate caption for a single frame
    """

    try:
        logger.debug(f"[FrameCaptioner] Generating caption | image={image_path}")

        processor, model, device = model_loader.get_blip()

        image = Image.open(image_path).convert("RGB")

        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)

        caption = processor.decode(out[0], skip_special_tokens=True)

        logger.debug(f"[FrameCaptioner] Caption generated successfully")

        return caption

    except Exception as e:
        logger.error(f"[FrameCaptioner] Failed | image={image_path} | error={str(e)}")
        raise RuntimeError(f"Frame Captioning failed: {str(e)}")