import requests

class OllamaGenerator:

    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "mistral"

    def generate(self, prompt):

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(self.url,json=payload)

        result = response.json()

        return result["response"]
    