import os
import threading
import time
import unicodedata
from typing import Iterator, Optional

from llama_cpp import Llama

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# STOP TOKENS

_STOP_TOKENS = [
    "</s>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "###",
    "\n\n\n",
]

# ARTIFACT PATTERNS TO STRIP

_STRIP_PREFIXES = [
    "Answer:",
    "OUTPUT:",
    "Assistant:",
]


class GGUFModel:

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_gpu_layers: int = 0,
    ) -> None:
        self._model_path  = model_path or settings.LLM_MODEL_PATH
        self._llm         = None
        self._lock        = threading.RLock()
        self.n_gpu_layers = n_gpu_layers
        self.max_prompt_chars = settings.MAX_PROMPT_CHARS

    # NORMALIZE

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", str(text or ""))
        return " ".join(text.strip().split())

    # CLEAN OUTPUT

    def _clean_output(self, text: str) -> str:
        text = text.strip()

        for stop in _STOP_TOKENS:
            if stop in text:
                text = text.split(stop)[0].strip()

        return text

    # LOAD

    def _load(self) -> Llama:
        if self._llm:
            return self._llm

        with self._lock:
            if self._llm:
                return self._llm

            if not os.path.exists(self._model_path):
                raise FileNotFoundError(f"MODEL_NOT_FOUND: {self._model_path}")

            start = time.time()

            try:
                self._llm = Llama(
                    model_path=self._model_path,
                    n_ctx=settings.CONTEXT_MAX_TOKENS,
                    n_threads=settings.LLM_THREADS,
                    n_batch=settings.LLM_N_BATCH,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False,
                )

                logger.info(
                    event="gguf_loaded",
                    model=os.path.basename(self._model_path),
                    n_ctx=settings.CONTEXT_MAX_TOKENS,
                    n_threads=settings.LLM_THREADS,
                    n_gpu_layers=self.n_gpu_layers,
                    latency=round(time.time() - start, 2),
                )

            except Exception as e:
                logger.error(event="gguf_load_failed", error=str(e))
                raise

        return self._llm

    # GENERATE

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        retries: int = 1,
        session_id: str = "",
    ) -> str:

        prompt = self._normalize(prompt)

        if not prompt:
            return ""

        if len(prompt) > self.max_prompt_chars:
            prompt = prompt[:self.max_prompt_chars]

        llm          = self._load()
        max_tokens_  = max_tokens   or settings.LLM_MAX_TOKENS
        temperature_ = temperature  if temperature  is not None else settings.LLM_TEMPERATURE
        top_p_       = top_p        if top_p        is not None else settings.LLM_TOP_P

        for attempt in range(retries + 1):
            try:
                start = time.time()

                res = llm(
                    prompt,
                    max_tokens=max_tokens_,
                    temperature=temperature_,
                    top_p=top_p_,
                    stop=_STOP_TOKENS,
                )

                raw_text = res["choices"][0]["text"]

                if not raw_text or len(raw_text.strip()) < 2:
                    raise ValueError("EMPTY_RESPONSE")

                text    = self._clean_output(raw_text)
                latency = round(time.time() - start, 3)

                # USAGE STATS
                usage            = res.get("usage", {})
                prompt_tokens    = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                tps              = round(completion_tokens / max(latency, 1e-6), 1)

                logger.debug(
                    event="gguf_generate_success",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    tokens_per_sec=tps,
                    latency=latency,
                    session_id=session_id,
                )

                return text

            except Exception as e:
                logger.warning(
                    event="gguf_generate_retry",
                    attempt=attempt,
                    error=str(e),
                    session_id=session_id,
                )

                if attempt >= retries:
                    raise

                time.sleep(0.2 * (attempt + 1))

        return ""

    # STREAM

    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        session_id: str = "",
    ) -> Iterator[str]:

        prompt = self._normalize(prompt)

        if not prompt:
            return

        if len(prompt) > self.max_prompt_chars:
            prompt = prompt[:self.max_prompt_chars]

        llm = self._load()

        try:
            start       = time.time()
            token_count = 0

            for chunk in llm(
                prompt,
                max_tokens=max_tokens   or settings.LLM_MAX_TOKENS,
                temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
                top_p=top_p             if top_p       is not None else settings.LLM_TOP_P,
                stop=_STOP_TOKENS,
                stream=True,
            ):
                token = chunk["choices"][0]["text"]
                if token:
                    token_count += 1
                    yield token

            latency = round(time.time() - start, 2)
            tps     = round(token_count / max(latency, 1e-6), 1)

            logger.debug(
                event="gguf_stream_complete",
                tokens=token_count,
                tokens_per_sec=tps,
                latency=latency,
                session_id=session_id,
            )

        except Exception as e:
            logger.error(
                event="gguf_stream_failed",
                error=str(e),
                session_id=session_id,
            )

    # WARMUP

    def warmup(self) -> None:
        try:
            start = time.time()
            self._load()

            try:
                self.generate("Hello", max_tokens=5)
            except Exception:
                pass

            logger.info(
                event="gguf_warmup_complete",
                latency=round(time.time() - start, 2),
            )

        except Exception as e:
            logger.warning(event="gguf_warmup_failed", error=str(e))

    # HEALTH CHECK

    def health_check(self) -> dict:
        return {
            "loaded":       self._llm is not None,
            "model_path":   self._model_path,
            "model_exists": os.path.exists(self._model_path),
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads":    settings.LLM_THREADS,
        }