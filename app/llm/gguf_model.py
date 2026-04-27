import os
import threading
import time

from llama_cpp import Llama

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


class GGUFModel:
    def __init__(self, model_path: str = None):
        self._model_path = model_path or settings.LLM_MODEL_PATH
        self._llm = None
        self._lock = threading.RLock()

        logger.info(f"[GGUFModel] Initialized | path={self._model_path}")

    # Internal loader
    def _load(self):
        if self._llm is not None:
            return self._llm

        with self._lock:
            if self._llm is not None:
                return self._llm

            if not os.path.exists(self._model_path):
                raise FileNotFoundError(
                    f"[GGUFModel] Model not found at {self._model_path}"
                )

            start = time.time()
            logger.info("[GGUFModel] Loading GGUF model...")

            try:
                self._llm = Llama(
                    model_path=self._model_path,
                    n_ctx=settings.CONTEXT_MAX_TOKENS,
                    n_threads=max(os.cpu_count() // 2, 2),
                    n_batch=512,
                    verbose=False,
                )

                latency = time.time() - start
                logger.info(f"[GGUFModel] Model loaded | {latency:.2f}s")

            except Exception as e:
                logger.error(f"[GGUFModel] Load failed | error={str(e)}")
                raise

        return self._llm

    # Generate (non-stream)
    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ) -> str:
        llm = self._load()

        try:
            response = llm(
                prompt,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                temperature=temperature or settings.LLM_TEMPERATURE,
                top_p=top_p or settings.LLM_TOP_P,
                stop=["</s>"],
            )

            text = response["choices"][0]["text"]
            return text.strip() if text else ""

        except Exception as e:
            logger.error(f"[GGUFModel] Generation failed | error={str(e)}")
            raise

    # Streaming generation
    def stream(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ):
        llm = self._load()

        try:
            for chunk in llm(
                prompt,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                temperature=temperature or settings.LLM_TEMPERATURE,
                top_p=top_p or settings.LLM_TOP_P,
                stream=True,
            ):
                token = chunk["choices"][0]["text"]
                if token:
                    yield token

        except Exception as e:
            logger.error(f"[GGUFModel] Streaming failed | error={str(e)}")
            raise

    # Warmup
    def warmup(self):
        try:
            self._load()
            logger.info("[GGUFModel] Warmup completed")
        except Exception as e:
            logger.warning(f"[GGUFModel] Warmup failed | {str(e)}")