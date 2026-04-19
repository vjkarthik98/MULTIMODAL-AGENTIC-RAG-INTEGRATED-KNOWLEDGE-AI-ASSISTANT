import os
from app.utils.logger import get_logger
from llama_cpp import Llama

# Logger
logger = get_logger(__name__)


class GGUFModel:
    def __init__(self):
        self._llm = None
        self._model_path = os.path.join(
            "models",
            "mistral",
            "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        )

        logger.info(f"[GGUFModel] Initialized (lazy load) | path={self._model_path}")


    # LAZY LOAD
    def _load(self):
        if self._llm is not None:
            return self._llm
        
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"[GGUFModel] Model not found at {self._model_path}"
            )
        
        logger.info("[GGUFModel] Loading model...")

        self._llm = Llama(
            model_path=self._model_path,
            n_ctx=2048,
            n_threads=max(os.cpu_count() // 2, 2),
            n_batch=256,
            verbose=False  
        )

        logger.info("[GGUFModel] Model loaded successfully")

        return self._llm
    
    # GENERATE
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            llm = self._load()

            logger.debug("[GGUFModel] Generating response")

            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9,
                stop=["</s>"]
            )

            text = response["choices"][0]["text"]

            return text.strip() if text else ""
        

        except Exception as e:
            logger.error(f"[GGUFModel] Generation failed | error={str(e)}")
            raise

    # STREAM
    def stream(self, prompt: str, max_tokens: int = 512):
        try:
            llm = self._load()

            logger.debug("[GGUFModel] Streaming started")

            for chunk in self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9,
                stream=True
            ):
                token =  chunk["choices"][0]["text"]

                if token:
                    yield token

            logger.debug("[GGUFModel] Streaming completed")

        except Exception as e:
            logger.error(f"[GGUFModel] Streaming failed | error={str(e)}")
            raise

    # WARMUP
    def warmup(self):
        self._load()