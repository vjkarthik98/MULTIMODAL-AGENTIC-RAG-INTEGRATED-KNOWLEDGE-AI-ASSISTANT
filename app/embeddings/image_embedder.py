from transformers import CLIPProcessor, CLIPVisionModelWithProjection
import torch
from PIL import Image

class ImageEmbedder:
    
    def __init__(self):
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def embed(self, image_path: str):

        print(">>> IMAGE EMBEDDER RUNNING <<<")

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
            
        with torch.no_grad():
            # Use get_image_features instead of vision_model
            outputs = self.model(**inputs)

        # Output will be (1, 512)
        embedding = outputs.image_embeds[0]

        print(f"DEBUG: Image Embedding Size = {len(embedding)}")
        return embedding.cpu().numpy().tolist()
    
        