import os
import threading
import time

from llama_cpp import Llama

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GGUFModel:

    def __init__(self, model_path: str = None, n_gpu_layers: int = 0):
        self._model_path = model_path or settings.LLM_MODEL_PATH
        self._llm = None
        self._lock = threading.RLock()

        self.n_gpu_layers = n_gpu_layers
        self.max_prompt_chars = settings.MAX_PROMPT_CHARS

    #  NORMALIZE 
    def _normalize(self, text: str) -> str:
        return " ".join(str(text or "").strip().split())

    #  LOAD 
    def _load(self):

        if self._llm:
            return self._llm

        with self._lock:

            if self._llm:
                return self._llm

            if not os.path.exists(self._model_path):
                raise FileNotFoundError("MODEL_NOT_FOUND")

            start = time.time()

            try:
                self._llm = Llama(
                    model_path=self._model_path,
                    n_ctx=settings.CONTEXT_MAX_TOKENS,
                    n_threads=max(os.cpu_count() // 2, 2),
                    n_batch=512,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False,
                )

                logger.info(
                    event="gguf_loaded",
                    latency=round(time.time() - start, 2)
                )

            except Exception as e:
                logger.error(event="gguf_load_failed", error=str(e))
                raise

        return self._llm

    #  GENERATE 
    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        retries: int = 1,
    ) -> str:

        prompt = self._normalize(prompt)

        if not prompt:
            return ""

        if len(prompt) > self.max_prompt_chars:
            prompt = prompt[:self.max_prompt_chars]

        llm = self._load()

        for attempt in range(retries + 1):

            try:
                start = time.time()

                res = llm(
                    prompt,
                    max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                    temperature=temperature or settings.LLM_TEMPERATURE,
                    top_p=top_p or settings.LLM_TOP_P,
                    stop=["</s>"],
                )

                text = res["choices"][0]["text"]

                if not text or len(text.strip()) < 2:
                    raise ValueError("EMPTY_RESPONSE")

                logger.debug(
                    event="gguf_generate",
                    latency=round(time.time() - start, 3)
                )

                return text.strip()

            except Exception as e:
                logger.warning(
                    event="gguf_retry",
                    attempt=attempt,
                    error=str(e)
                )

                if attempt >= retries:
                    raise

                time.sleep(0.2)

        return ""

    #  STREAM 
    def stream(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ):

        prompt = self._normalize(prompt)

        if not prompt:
            return

        if len(prompt) > self.max_prompt_chars:
            prompt = prompt[:self.max_prompt_chars]

        llm = self._load()

        try:
            start = time.time()

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

            logger.debug(
                event="gguf_stream_complete",
                latency=round(time.time() - start, 2)
            )

        except Exception as e:
            logger.error(event="gguf_stream_failed", error=str(e))

    #  WARMUP 
    def warmup(self):

        try:
            start = time.time()

            self._load()

            try:
                self.generate("Hello", max_tokens=5)
            except Exception:
                pass

            logger.info(
                event="gguf_warmup",
                latency=round(time.time() - start, 2)
            )

        except Exception as e:
            logger.warning(event="gguf_warmup_failed", error=str(e))