from __future__ import annotations

import os
import threading
import time
import unicodedata
from typing import Any, Dict, Iterator, List, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# PROMETHEUS METRICS — SECTION 6

def _get_metrics():
    try:
        from prometheus_client import Counter, Histogram, Gauge
        llm_latency = Histogram(
            "llm_call_latency_seconds",
            "LLM inference latency",
            ["model", "mode"],
        )
        llm_errors = Counter(
            "llm_errors_total",
            "LLM errors by type",
            ["error_type"],
        )
        llm_tokens = Counter(
            "llm_tokens_total",
            "LLM tokens generated",
            ["model"],
        )
        circuit_state = Gauge(
            "circuit_breaker_state",
            "Circuit breaker state per service (0=closed, 1=open)",
            ["service"],
        )
        return {
            "llm_latency":    llm_latency,
            "llm_errors":     llm_errors,
            "llm_tokens":     llm_tokens,
            "circuit_state":  circuit_state,
        }
    except Exception:
        return {}


_METRICS: Dict[str, Any] = {}

if settings.PROMETHEUS_ENABLED:
    try:
        _METRICS = _get_metrics()
    except Exception:
        pass


def _record_latency(model: str, mode: str, latency: float) -> None:
    try:
        if "llm_latency" in _METRICS:
            _METRICS["llm_latency"].labels(model=model, mode=mode).observe(latency)
    except Exception:
        pass


def _record_error(error_type: str) -> None:
    try:
        if "llm_errors" in _METRICS:
            _METRICS["llm_errors"].labels(error_type=error_type).inc()
    except Exception:
        pass


def _record_tokens(model: str, count: int) -> None:
    try:
        if "llm_tokens" in _METRICS:
            _METRICS["llm_tokens"].labels(model=model).inc(count)
    except Exception:
        pass


def _set_circuit_state(service: str, open_: bool) -> None:
    try:
        if "circuit_state" in _METRICS:
            _METRICS["circuit_state"].labels(service=service).set(1 if open_ else 0)
    except Exception:
        pass


# CIRCUIT BREAKER — SECTION 2.1

class _CircuitBreaker:

    def __init__(
        self,
        fail_max:      int   = 5,
        reset_timeout: float = 60.0,
        name:          str   = "gguf_llm",
    ) -> None:
        self.fail_max      = fail_max
        self.reset_timeout = reset_timeout
        self.name          = name
        self._failures     = 0
        self._opened_at:   Optional[float] = None
        self._lock         = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            elapsed = time.time() - self._opened_at
            if elapsed >= self.reset_timeout:
                # HALF-OPEN: allow one attempt
                self._opened_at = None
                self._failures  = 0
                _set_circuit_state(self.name, False)
                logger.info(event="circuit_breaker_half_open", service=self.name)
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures  = 0
            self._opened_at = None
            _set_circuit_state(self.name, False)

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.fail_max:
                self._opened_at = time.time()
                _set_circuit_state(self.name, True)
                logger.warning(
                    event="circuit_breaker_opened",
                    service=self.name,
                    failures=self._failures,
                )

    def __enter__(self) -> "_CircuitBreaker":
        if self.is_open:
            raise RuntimeError(
                f"CIRCUIT_BREAKER_OPEN: {self.name} — too many failures, "
                f"retry after {self.reset_timeout}s"
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is not None:
            self.record_failure()
        else:
            self.record_success()
        return False


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

# ARTIFACT PREFIXES TO STRIP

_STRIP_PREFIXES = [
    "Answer:",
    "OUTPUT:",
    "Assistant:",
    "ASSISTANT:",
]

# PROMPT INJECTION PATTERNS — consolidated into app/guardrails/policies.yaml (Phase 26)


# GGUF MODEL CLASS

class GGUFModel:

    def __init__(
        self,
        model_path:   Optional[str] = None,
        n_gpu_layers: int           = 0,
    ) -> None:
        self._model_path      = model_path or settings.LLM_MODEL_PATH
        self._llm             = None
        self._lock            = threading.RLock()
        self.n_gpu_layers     = n_gpu_layers
        self.max_prompt_chars = settings.MAX_PROMPT_CHARS
        self._model_name      = os.path.basename(self._model_path)
        self._circuit         = _CircuitBreaker(
            fail_max=5,
            reset_timeout=60.0,
            name="gguf_llm",
        )

    # NORMALIZE — SECTION 2.3

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", str(text or ""))
        # STRIP NULL BYTES
        text = text.replace("\x00", "")
        # STRIP BOM
        text = text.lstrip("\ufeff\ufffe")
        return " ".join(text.strip().split())

    # PROMPT INJECTION SANITIZATION — delegates to unified guardrail (Phase 26)

    def _sanitize_prompt(self, prompt: str) -> str:
        from app.guardrails.input_guard import sanitize as _guard_sanitize
        cleaned = _guard_sanitize(prompt, surface="gguf_model")
        if cleaned != prompt:
            logger.warning(event="gguf_prompt_injection_stripped")
        return cleaned

    # CLEAN OUTPUT

    def _clean_output(self, text: str) -> str:
        text = text.strip()

        # STRIP STOP TOKENS
        for stop in _STOP_TOKENS:
            if stop in text:
                text = text.split(stop)[0].strip()

        # STRIP ARTIFACT PREFIXES
        for prefix in _STRIP_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # STRIP NULL BYTES
        text = text.replace("\x00", "")

        return text

    # LOAD MODEL — SECTION 4.6

    def _load(self):
        if self._llm:
            return self._llm

        with self._lock:
            if self._llm:
                return self._llm

            if not os.path.exists(self._model_path):
                raise FileNotFoundError(f"MODEL_NOT_FOUND: {self._model_path}")

            model_size = os.path.getsize(self._model_path)
            if model_size < 1024 * 1024:
                raise ValueError(
                    f"MODEL_TOO_SMALL: {model_size} bytes — likely corrupt"
                )

            start = time.time()

            try:
                from llama_cpp import Llama
                self._llm = Llama(
                    model_path=self._model_path,
                    n_ctx=settings.CONTEXT_MAX_TOKENS,
                    n_threads=settings.LLM_THREADS,
                    n_batch=settings.LLM_N_BATCH,
                    n_gpu_layers=self.n_gpu_layers,
                    use_mlock=settings.LLM_USE_MLOCK,
                    verbose=False,
                )

                logger.info(
                    event="gguf_loaded",
                    model=self._model_name,
                    n_ctx=settings.CONTEXT_MAX_TOKENS,
                    n_threads=settings.LLM_THREADS,
                    n_gpu_layers=self.n_gpu_layers,
                    size_mb=round(model_size / (1024 * 1024), 1),
                    latency=round(time.time() - start, 2),
                )

            except Exception as e:
                _record_error("model_load_failed")
                logger.error(event="gguf_load_failed", error=str(e))
                raise

        return self._llm

    # TOKEN-SAFE TRUNCATION
    # Character limits are unreliable for financial/numeric text where one token
    # can be as short as 1 char (e.g. "3,787,464%" → ~7 tokens for 10 chars).
    # This method uses the actual SentencePiece tokenizer baked into the loaded
    # model to count tokens precisely, then truncates to fit within the context
    # window. Hard floor of 150 output tokens reserved regardless of max_tokens.
    def _truncate_to_token_budget(self, prompt: str, max_tokens: int) -> str:
        try:
            llm = self._load()
            # tokenize/detokenize touch the shared llama.cpp context — they MUST
            # hold the inference lock so they never run concurrently with an
            # in-flight generate()/stream() on another thread (that races the
            # KV cache and crashes the process with SIGSEGV).
            with self._lock:
                token_ids = llm.tokenize(prompt.encode("utf-8", errors="replace"))
                budget = settings.CONTEXT_MAX_TOKENS - max(max_tokens, 150) - 64  # 64-token safety margin
                if len(token_ids) <= budget:
                    return prompt
                token_ids = token_ids[:budget]
                return llm.detokenize(token_ids).decode("utf-8", errors="replace")
        except Exception as _e:
            # Tokenizer unavailable — fall back to conservative char truncation
            logger.warning(event="gguf_token_truncation_failed", error=str(_e))
            safe_chars = (settings.CONTEXT_MAX_TOKENS - max(max_tokens, 150) - 64) * 3
            return prompt[:safe_chars]

    # GENERATE WITH RETRY + CIRCUIT BREAKER — SECTION 2.1

    def generate(
        self,
        prompt:      str,
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
        top_p:       Optional[float] = None,
        retries:     int             = 2,
        session_id:  str             = "",
    ) -> str:

        prompt = self._normalize(prompt)
        prompt = self._sanitize_prompt(prompt)

        if not prompt:
            return ""

        max_tokens_  = max_tokens   if max_tokens   is not None else settings.LLM_MAX_TOKENS
        temperature_ = temperature  if temperature  is not None else settings.LLM_TEMPERATURE
        top_p_       = top_p        if top_p        is not None else settings.LLM_TOP_P

        # TOKEN-SAFE TRUNCATION — must happen before the llama.cpp call.
        # Character limits fail for financial/numeric text (1–2 chars/token).
        # This uses the model's own tokenizer to guarantee the prompt fits
        # within the context window and never triggers a SIGSEGV in llama.cpp.
        prompt = self._truncate_to_token_budget(prompt, max_tokens_)

        # CIRCUIT BREAKER CHECK — SECTION 2.1
        if self._circuit.is_open:
            _record_error("circuit_breaker_open")
            raise RuntimeError(
                "CIRCUIT_BREAKER_OPEN: LLM is temporarily unavailable"
            )

        last_exc: Optional[Exception] = None
        wait = float(settings.LLM_RETRY_WAIT_MIN)

        for attempt in range(max(retries, 0) + 1):
            try:
                with self._circuit:
                    llm   = self._load()
                    start = time.time()

                    # INFERENCE LOCK — llama.cpp is NOT thread-safe. Concurrent
                    # llm() calls on the shared context corrupt the KV cache and
                    # SIGSEGV the process. Serializing them is the only safe
                    # option on a single shared context.
                    #
                    # HARD LLM CALL TIMEOUT — SECTION 2.1. Enforced BETWEEN
                    # tokens via an internal stream, so a runaway generation is
                    # actually stopped at the deadline (freeing the lock for
                    # queued requests) and the partial text is returned. The old
                    # post-hoc check threw away a COMPLETED >60s answer and then
                    # retried the whole generation up to 2 more times — turning
                    # one slow call into a ~3× latency catastrophe.
                    timed_out = False
                    deadline  = start + settings.LLM_CALL_TIMEOUT_SEC
                    parts: List[str] = []
                    with self._lock:
                        for chunk in llm(
                            prompt,
                            max_tokens=max_tokens_,
                            temperature=temperature_,
                            top_p=top_p_,
                            top_k=settings.LLM_TOP_K_SAMPLING,
                            min_p=settings.LLM_MIN_P,
                            repeat_penalty=settings.LLM_REPEAT_PENALTY,
                            stop=_STOP_TOKENS,
                            stream=True,
                        ):
                            tok = chunk["choices"][0]["text"]
                            if tok:
                                parts.append(tok)
                            if time.time() > deadline:
                                timed_out = True
                                break

                    elapsed  = time.time() - start
                    raw_text = "".join(parts)

                    if timed_out:
                        logger.warning(
                            event="gguf_generate_deadline_hit",
                            elapsed=round(elapsed, 1),
                            timeout_sec=settings.LLM_CALL_TIMEOUT_SEC,
                            partial_tokens=len(parts),
                            session_id=session_id,
                        )

                    if not raw_text or len(raw_text.strip()) < 2:
                        raise ValueError("EMPTY_RESPONSE")

                    text = self._clean_output(raw_text)

                    prompt_tokens     = 0  # not reported in stream mode
                    completion_tokens = len(parts)
                    tps               = round(completion_tokens / max(elapsed, 1e-6), 1)

                    _record_latency(self._model_name, "generate", elapsed)
                    _record_tokens(self._model_name, completion_tokens)

                    logger.debug(
                        event="gguf_generate_success",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        tokens_per_sec=tps,
                        latency=round(elapsed, 3),
                        session_id=session_id,
                    )

                    return text

            except (TimeoutError, ValueError) as e:
                last_exc = e
                _record_error(type(e).__name__)
                logger.warning(
                    event="gguf_generate_retry",
                    attempt=attempt,
                    error=str(e),
                    session_id=session_id,
                )
                if attempt >= retries:
                    break
                time.sleep(min(wait, float(settings.LLM_RETRY_WAIT_MAX)))
                wait *= 2.0

            except RuntimeError as e:
                # CIRCUIT BREAKER OPEN — DO NOT RETRY
                _record_error("circuit_breaker_open")
                raise

            except Exception as e:
                last_exc = e
                _record_error(type(e).__name__)
                logger.warning(
                    event="gguf_generate_retry",
                    attempt=attempt,
                    error=str(e),
                    session_id=session_id,
                )
                if attempt >= retries:
                    break
                time.sleep(min(wait, float(settings.LLM_RETRY_WAIT_MAX)))
                wait *= 2.0

        if last_exc:
            logger.error(
                event="gguf_generate_failed",
                error=str(last_exc),
                session_id=session_id,
            )
            raise last_exc

        return ""

    # STREAM — SECTION 4.6 SSE TOKEN STREAMING

    def stream(
        self,
        prompt:      str,
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
        top_p:       Optional[float] = None,
        session_id:  str             = "",
    ) -> Iterator[str]:

        prompt = self._normalize(prompt)
        prompt = self._sanitize_prompt(prompt)

        if not prompt:
            return

        _max_tok = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
        prompt = self._truncate_to_token_budget(prompt, _max_tok)

        # CIRCUIT BREAKER CHECK
        if self._circuit.is_open:
            _record_error("circuit_breaker_open")
            logger.warning(event="gguf_stream_circuit_open", session_id=session_id)
            return

        try:
            llm = self._load()

            start       = time.time()
            token_count = 0
            buffer      = ""

            # INFERENCE LOCK — held for the full token loop so no concurrent
            # generate()/stream() touches the shared llama.cpp context. The loop
            # is bounded by max_tokens and LLM_CALL_TIMEOUT_SEC, so the lock is
            # held for one bounded generation then freed.
            with self._circuit, self._lock:
                for chunk in llm(
                    prompt,
                    max_tokens=max_tokens   if max_tokens   is not None else settings.LLM_MAX_TOKENS,
                    temperature=temperature if temperature  is not None else settings.LLM_TEMPERATURE,
                    top_p=top_p             if top_p        is not None else settings.LLM_TOP_P,
                    top_k=settings.LLM_TOP_K_SAMPLING,
                    min_p=settings.LLM_MIN_P,
                    repeat_penalty=settings.LLM_REPEAT_PENALTY,
                    stop=_STOP_TOKENS,
                    stream=True,
                ):
                    token = chunk["choices"][0]["text"]

                    if not token:
                        continue

                    # HARD STREAMING TIMEOUT — SECTION 2.1. Same bound as
                    # generate(); FILE_PROCESSING_TIMEOUT_SEC (300s) let a
                    # runaway stream hold the inference lock for 5 minutes.
                    elapsed = time.time() - start
                    if elapsed > settings.LLM_CALL_TIMEOUT_SEC:
                        logger.warning(
                            event="gguf_stream_timeout",
                            elapsed=round(elapsed, 1),
                            session_id=session_id,
                        )
                        break

                    # STRIP NULL BYTES FROM TOKENS
                    token = token.replace("\x00", "")

                    buffer += token
                    token_count += 1
                    yield token

            latency = round(time.time() - start, 2)
            tps     = round(token_count / max(latency, 1e-6), 1)

            _record_latency(self._model_name, "stream", latency)
            _record_tokens(self._model_name, token_count)

            logger.debug(
                event="gguf_stream_complete",
                tokens=token_count,
                tokens_per_sec=tps,
                latency=latency,
                session_id=session_id,
            )

        except RuntimeError as e:
            _record_error("circuit_breaker_open")
            logger.warning(
                event="gguf_stream_circuit_blocked",
                error=str(e),
                session_id=session_id,
            )

        except Exception as e:
            _record_error(type(e).__name__)
            logger.error(
                event="gguf_stream_failed",
                error=str(e),
                session_id=session_id,
            )

    # WARMUP — SECTION 2.1

    def warmup(self) -> None:
        try:
            start = time.time()
            self._load()

            try:
                self.generate(
                    "Hello",
                    max_tokens=5,
                    temperature=0.0,
                )
            except Exception:
                pass

            logger.info(
                event="gguf_warmup_complete",
                latency=round(time.time() - start, 2),
                model=self._model_name,
            )

        except Exception as e:
            logger.warning(event="gguf_warmup_failed", error=str(e))

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        model_exists = os.path.exists(self._model_path)
        model_size   = (
            os.path.getsize(self._model_path)
            if model_exists else 0
        )
        return {
            "loaded":          self._llm is not None,
            "model_path":      self._model_path,
            "model_name":      self._model_name,
            "model_exists":    model_exists,
            "model_size_mb":   round(model_size / (1024 * 1024), 1) if model_size else 0,
            "n_gpu_layers":    self.n_gpu_layers,
            "n_threads":       settings.LLM_THREADS,
            "n_ctx":           settings.CONTEXT_MAX_TOKENS,
            "circuit_open":    self._circuit.is_open,
            "circuit_failures": self._circuit._failures,
        }

    # RESET — FOR TESTING / RELOAD

    def reset(self) -> None:
        with self._lock:
            self._llm = None
            self._circuit.record_success()
            logger.warning(event="gguf_model_reset", model=self._model_name)

    # CONTEXT MANAGER SUPPORT

    def __enter__(self) -> "GGUFModel":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False