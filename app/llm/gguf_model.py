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

        self.max_prompt_chars = settings.MAX_PROMPT_CHARS

        logger.info(f"[GGUFModel] initialized | path={self._model_path}")

    #  NORMALIZE 
    def _normalize(self, text: str) -> str:
        return " ".join(str(text or "").strip().split())

    #  LOAD 
    def _load(self):

        if self._llm is not None:
            return self._llm

        with self._lock:
            if self._llm is not None:
                return self._llm

            if not os.path.exists(self._model_path):
                raise FileNotFoundError(
                    f"[GGUFModel] model not found at {self._model_path}"
                )

            start = time.time()

            logger.info("[GGUFModel] loading model")

            try:
                self._llm = Llama(
                    model_path=self._model_path,
                    n_ctx=settings.CONTEXT_MAX_TOKENS,
                    n_threads=max(os.cpu_count() // 2, 2),
                    n_batch=512,
                    verbose=False,
                )

                logger.info(
                    "[GGUFModel] loaded | latency=%.2fs",
                    time.time() - start
                )

            except Exception as e:
                logger.error("[GGUFModel] load failed | %s", str(e))
                raise

        return self._llm

    #  SAFE GENERATE 
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
            logger.warning("[GGUFModel] prompt truncated")
            prompt = prompt[:self.max_prompt_chars]

        llm = self._load()

        for attempt in range(retries + 1):

            try:
                start = time.time()

                response = llm(
                    prompt,
                    max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                    temperature=temperature or settings.LLM_TEMPERATURE,
                    top_p=top_p or settings.LLM_TOP_P,
                    stop=["</s>"],
                )

                latency = round(time.time() - start, 2)

                text = response["choices"][0]["text"]

                if not text or len(text.strip()) < 2:
                    raise ValueError("empty_response")

                logger.debug(
                    "[GGUFModel] generate success | latency=%ss",
                    latency
                )

                return text.strip()

            except Exception as e:
                logger.warning(
                    "[GGUFModel] generation attempt %s failed | %s",
                    attempt,
                    str(e)
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
                "[GGUFModel] stream completed | latency=%.2fs",
                time.time() - start
            )

        except Exception as e:
            logger.error("[GGUFModel] streaming failed | %s", str(e))
            return

    #  WARMUP 
    def warmup(self):

        try:
            start = time.time()

            self._load()

            # minimal token generation to warm cache
            try:
                self.generate("Hello", max_tokens=5)
            except Exception:
                pass

            logger.info(
                "[GGUFModel] warmup completed | latency=%.2fs",
                time.time() - start
            )

        except Exception as e:
            logger.warning("[GGUFModel] warmup failed | %s", str(e))