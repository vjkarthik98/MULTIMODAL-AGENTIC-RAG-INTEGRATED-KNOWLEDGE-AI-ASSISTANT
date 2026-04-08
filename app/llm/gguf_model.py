import os
import logging
from llama_cpp import Llama

# Logger
logger = logging.getLogger(__name__)


class GGUFModel:
    def __init__(self):
        model_path = os.path.join(
            "models",
            "mistral",
            "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        )

        logger.info(f"[GGUFModel] Loading model from path={model_path}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=6,
            n_batch=256,
            verbose=False  # keep False in production
        )

        logger.info("[GGUFModel] Model loaded successfully")

    def generate(self, prompt: str, max_tokens: int = 256):
        try:
            logger.debug("[GGUFModel] Generating response")

            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>"]
            )

            return response["choices"][0]["text"].strip()

        except Exception as e:
            logger.error(f"[GGUFModel] Generation failed | error={str(e)}")
            raise

    def stream(self, prompt: str, max_tokens: int = 512):
        try:
            logger.debug("[GGUFModel] Streaming response started")

            for chunk in self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stream=True
            ):
                yield chunk["choices"][0]["text"]

            logger.debug("[GGUFModel] Streaming completed")

        except Exception as e:
            logger.error(f"[GGUFModel] Streaming failed | error={str(e)}")
            raise