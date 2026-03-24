import os
from llama_cpp import Llama

class GGUFModel:
    def __init__(self):
        model_path=os.path.join(
            "models",
            "mistral",
            "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        )
        
        self.llm = Llama(
            model_path = model_path,
            n_ctx=2048,
            n_threads=6,
            n_batch=256,
            verbose=False # temporarily enable for debugging
        )

    def generate(self, prompt: str, max_tokens: int= 256):
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>"]
        )
        return response["choices"][0]["text"].strip()
    
    def stream(self, prompt: str, max_tokens: int = 512):
        for chunk in self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stream=True
        ):
            yield chunk["choices"][0]["text"]