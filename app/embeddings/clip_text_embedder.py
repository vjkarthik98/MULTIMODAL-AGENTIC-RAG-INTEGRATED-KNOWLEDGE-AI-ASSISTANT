from transformers import CLIPProcessor, CLIPTextModelWithProjection
import torch

class ClipTextEmbedder:
    def __init__(self):
        # We only load the Text portion of the model to avoid projection head interference
        model_id = "openai/clip-vit-large-patch14"
        self.model = CLIPTextModelWithProjection.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
    
    def embed(self, text: str):
        # Tokenize
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # This is now guaranteed to be 768
            embedding = outputs.text_embeds[0]

        print(f"DEBUG: Text Query Size = {len(embedding)}")
        return embedding.cpu().numpy().tolist()
   