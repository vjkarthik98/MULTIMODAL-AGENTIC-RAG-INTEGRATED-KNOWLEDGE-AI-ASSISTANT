"""
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from app.core.model_loader import model_loader
import torch
from PIL import Image
import logging

# Logger
logger = logging.getLogger(__name__)


class ImageEmbedder:

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("[ImageEmbedder] Loading CLIP vision model...")

        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, image_path: str):
        try:
            logger.debug(f"[ImageEmbedder] Processing image: {image_path}")

            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.image_embeds[0]

            logger.debug(f"[ImageEmbedder] Embedding generated, dim={len(embedding)}")

            return embedding.cpu().numpy().tolist()

        except Exception as e:
            logger.error(f"[ImageEmbedder] Error processing image: {str(e)}")
            raise
            """