from transformers import CLIPProcessor, CLIPTextModelWithProjection
import torch

class ClipTextEmbedder:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CLIPTextModelWithProjection.from_pretrained(model_name).to(self.device)
        
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def embed(self, text: str):
        # Tokenize
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # This is now guaranteed to be 768
            embedding = outputs.text_embeds[0]

        
        return embedding.cpu().numpy().tolist()
   