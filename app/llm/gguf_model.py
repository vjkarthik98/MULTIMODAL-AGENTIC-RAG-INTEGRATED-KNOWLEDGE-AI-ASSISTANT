from __future__ import annotations

import os
import queue as _queue
import threading
import time
import unicodedata
from collections.abc import Iterator
from typing import Any

from app.core.config import settings
from app.core.metrics import llm_call_latency as _shared_llm_latency
from app.utils.logger import get_logger

logger = get_logger(__name__)


# PROMETHEUS METRICS — SECTION 6
# llm_call_latency_seconds is a shared singleton from app.core.metrics —
# this file used to register its own copy of the same metric name, which
# silently raced (and usually lost) against identical copies in
# rag_pipeline.py/query_pipeline.py/reasoning_engine.py. See app/core/
# metrics.py's comment for the live incident that made this get fixed.


def _get_metrics():
    try:
        from prometheus_client import Counter, Gauge

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
            "llm_errors": llm_errors,
            "llm_tokens": llm_tokens,
            "circuit_state": circuit_state,
        }
    except Exception:
        return {}


_METRICS: dict[str, Any] = {}

if settings.PROMETHEUS_ENABLED:
    try:
        _METRICS = _get_metrics()
    except Exception:
        pass


def _record_latency(model: str, mode: str, latency: float) -> None:
    try:
        _shared_llm_latency.labels(model=model, mode=mode).observe(latency)
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
        fail_max: int = 5,
        reset_timeout: float = 60.0,
        name: str = "gguf_llm",
    ) -> None:
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.name = name
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            elapsed = time.time() - self._opened_at
            if elapsed >= self.reset_timeout:
                # HALF-OPEN: allow one attempt
                self._opened_at = None
                self._failures = 0
                _set_circuit_state(self.name, False)
                logger.info(event="circuit_breaker_half_open", service=self.name)
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
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

    def __enter__(self) -> _CircuitBreaker:
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
#
# Superset across every model family this project has run: the Mistral tokens
# ([INST]/[/INST]/<<SYS>>/</s>) and the ChatML tokens (<|im_end|>/<|im_start|>/
# <|endoftext|>) both stay in the list regardless of LLM_PROMPT_FORMAT — a model
# never spontaneously emits another family's control tokens, so the extras are
# inert. This means switching LLM_MODEL_PATH/LLM_PROMPT_FORMAT never requires
# touching this list again.

_STOP_TOKENS = [
    "</s>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "###",
    "\n\n\n",
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
]


def _seed_kwarg(seed: int | None) -> dict[str, int]:
    """`{"seed": n}` when a caller asked for a specific seed, `{}` otherwise.

    Only regeneration passes one (app/llm/regeneration.py). Every other call
    omits the kwarg entirely rather than sending a default, so the sampler's
    existing behaviour on the normal answer path is byte-for-byte unchanged
    by this parameter existing.
    """
    return {} if seed is None else {"seed": int(seed)}


# ARTIFACT PREFIXES TO STRIP

_STRIP_PREFIXES = [
    "Answer:",
    "OUTPUT:",
    "Assistant:",
    "ASSISTANT:",
]

# CHAT TEMPLATE — every call site in this codebase (app/prompt/prompt_builder.py,
# app/reasoning/reasoning_engine.py) assembles ONE plain-text instruction body
# (system guidance + memory/context + query + output-format rules, all
# concatenated). That body is model-family agnostic. The only thing that needs
# to change when swapping LLM_MODEL_PATH to a different model family is the
# turn markup wrapped around that body — handled centrally here, in
# GGUFModel._format_for_model(), so no prompt-assembly code has to know or care
# which model is currently loaded.
_CHATML_SYSTEM_DEFAULT = (
    "You are a precise financial research assistant. Follow the instructions "
    "and answer using only the given context."
)

# PROMPT INJECTION PATTERNS — consolidated into app/guardrails/policies.yaml (Phase 26)


# LLAMA-SERVER HTTP CLIENT
#
# Drop-in replacement for the in-process llama_cpp.Llama callable. It proxies
# inference to a separate llama-server process (launched by start_server.py)
# that owns its OWN CUDA context. This is what lets the LLM stay on the GPU
# without corrupting PyTorch's CUDA context in the main process (the in-process
# embed-stage SIGSEGV).
#
# It mimics the exact surface GGUFModel uses on the loaded model:
#   - __call__(prompt, max_tokens=, temperature=, ..., stream=True) -> iterator
#       of {"choices":[{"text": <tok>}]} chunks (same shape as llama_cpp.Llama)
#   - tokenize(bytes) -> List[int]   (via the server's /extras/tokenize)
#   - detokenize(List[int]) -> bytes (via /extras/detokenize)
# so generate(), stream() and _truncate_to_token_budget() work UNCHANGED.


class _LlamaServerClient:

    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        import requests  # local import keeps module import cheap

        self._requests = requests

    def __call__(self, prompt: str, stream: bool = True, **kwargs: Any):
        # Map llama_cpp kwargs straight onto the OpenAI-compatible /v1/completions.
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"),
            "min_p": kwargs.get("min_p"),
            "repeat_penalty": kwargs.get("repeat_penalty"),
            "stop": kwargs.get("stop"),
            # Only sent when a caller explicitly asked for one (regeneration);
            # omitted otherwise so the server keeps its own default behaviour.
            "seed": kwargs.get("seed"),
            "stream": True,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        resp = self._requests.post(
            f"{self._base}/v1/completions",
            json=payload,
            stream=True,
            timeout=(10, settings.LLM_CALL_TIMEOUT_SEC + 30),
        )
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw[6:] if raw.startswith("data: ") else raw
            if line.strip() == "[DONE]":
                break
            try:
                import json as _json

                yield _json.loads(line)
            except Exception:
                continue

    def tokenize(self, data: bytes) -> list[int]:
        text = (
            data.decode("utf-8", errors="replace")
            if isinstance(data, (bytes, bytearray))
            else str(data)
        )
        r = self._requests.post(f"{self._base}/extras/tokenize", json={"input": text}, timeout=30)
        r.raise_for_status()
        return list(r.json().get("tokens", []))

    def detokenize(self, tokens: list[int]) -> bytes:
        r = self._requests.post(
            f"{self._base}/extras/detokenize", json={"tokens": list(tokens)}, timeout=30
        )
        r.raise_for_status()
        return str(r.json().get("text", "")).encode("utf-8", errors="replace")

    def health(self) -> bool:
        try:
            r = self._requests.get(f"{self._base}/v1/models", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# GGUF MODEL CLASS


class GGUFModel:

    def __init__(
        self,
        model_path: str | None = None,
        n_gpu_layers: int = 0,
    ) -> None:
        self._model_path = model_path or settings.LLM_MODEL_PATH
        self._llm = None
        self._lock = threading.RLock()
        self.n_gpu_layers = n_gpu_layers
        self.max_prompt_chars = settings.MAX_PROMPT_CHARS
        self._model_name = os.path.basename(self._model_path)
        self._circuit = _CircuitBreaker(
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

    # Finance safety suffix — appended after injection sanitization, before LLM call.
    # Additive: does not affect guardrails, only nudges the model away from
    # unsolicited investment recommendations. (Plan Phase 6 requirement.)
    _FINANCE_SAFETY_SUFFIX = (
        "\n[SAFETY: Do not recommend specific securities or trades. "
        "State all figures as reported.]"
    )

    def _sanitize_prompt(self, prompt: str) -> str:
        from app.guardrails.input_guard import sanitize as _guard_sanitize

        # No `if cleaned != prompt: log a warning` here on purpose — that
        # comparison can't tell a genuine injection strip apart from
        # _normalize_encoding()'s routine, always-on NFKC pass (ligatures
        # like "fi"->"fi", footnote superscripts, smart quotes — all
        # ordinary in typeset financial PDFs, zero malicious content
        # involved). Confirmed live: this fired on nearly every LLM call in
        # a full Tier-2 run, drowning out genuine detections in noise.
        # sanitize() itself already logs the precise signal —
        # input_guard_sanitize_stripped, with pattern/severity detail, only
        # inside its real `if match:` branch, already correctly tagged
        # surface="gguf_model" — so there is nothing this duplicate,
        # imprecise check adds.
        cleaned = _guard_sanitize(prompt, surface="gguf_model")
        # Append finance safety guard (plan Phase 6.6)
        cleaned = cleaned + self._FINANCE_SAFETY_SUFFIX
        return cleaned

    # CHAT TEMPLATE — applied LAST, after sanitize/truncate, so injection
    # scanning and the token-budget truncator both operate on the real
    # semantic content and never see (or accidentally truncate into) the
    # turn-markup tokens.
    def _format_for_model(self, prompt: str) -> str:
        fmt = getattr(settings, "LLM_PROMPT_FORMAT", "raw")
        if fmt == "chatml":
            return (
                "<|im_start|>system\n"
                f"{_CHATML_SYSTEM_DEFAULT}<|im_end|>\n"
                "<|im_start|>user\n"
                f"{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        return prompt  # "raw" — legacy plain-completion prompting, unchanged

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
                text = text[len(prefix) :].strip()

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

            # SEPARATE-PROCESS PATH — proxy to the llama-server (its own CUDA
            # context). No in-process llama.cpp, so PyTorch keeps the GPU to
            # itself in this process and ingestion embeds never SIGSEGV.
            if settings.LLM_USE_SERVER:
                base = f"http://{settings.LLM_SERVER_HOST}:{settings.LLM_SERVER_PORT}"
                client = _LlamaServerClient(base)
                self._llm = client
                logger.info(
                    event="gguf_loaded",
                    model=self._model_name,
                    mode="llama_server",
                    server=base,
                    n_gpu_layers=self.n_gpu_layers,
                )
                return self._llm

            if not os.path.exists(self._model_path):
                raise FileNotFoundError(f"MODEL_NOT_FOUND: {self._model_path}")

            model_size = os.path.getsize(self._model_path)
            if model_size < 1024 * 1024:
                raise ValueError(f"MODEL_TOO_SMALL: {model_size} bytes — likely corrupt")

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
        budget = settings.CONTEXT_MAX_TOKENS - max(max_tokens, 150) - 64  # 64-token safety margin

        # FAST PATH — skip the tokenize() round-trip when the prompt clearly fits.
        # With the LLM in a separate process, tokenize()/detokenize() are HTTP
        # round-trips that block time-to-first-token. English/finance text averages
        # well above 2.5 chars/token; if the prompt is shorter than budget*2.5
        # chars it cannot exceed the token budget, so we return it untouched and
        # avoid the network call entirely. Long prompts still tokenize precisely.
        if len(prompt) <= int(budget * 2.5):
            return prompt

        try:
            llm = self._load()
            # tokenize/detokenize touch the shared llama.cpp context — they MUST
            # hold the inference lock so they never run concurrently with an
            # in-flight generate()/stream() on another thread (that races the
            # KV cache and crashes the process with SIGSEGV).
            with self._lock:
                token_ids = llm.tokenize(prompt.encode("utf-8", errors="replace"))
                if len(token_ids) <= budget:
                    return prompt
                token_ids = token_ids[:budget]
                return llm.detokenize(token_ids).decode("utf-8", errors="replace")
        except Exception as _e:
            # Tokenizer unavailable — fall back to conservative char truncation
            logger.warning(event="gguf_token_truncation_failed", error=str(_e))
            safe_chars = budget * 3
            return prompt[:safe_chars]

    # GENERATE WITH RETRY + CIRCUIT BREAKER — SECTION 2.1

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        retries: int = 2,
        session_id: str = "",
        seed: int | None = None,
    ) -> str:

        prompt = self._normalize(prompt)
        prompt = self._sanitize_prompt(prompt)

        if not prompt:
            return ""

        max_tokens_ = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
        temperature_ = temperature if temperature is not None else settings.LLM_TEMPERATURE
        top_p_ = top_p if top_p is not None else settings.LLM_TOP_P

        # TOKEN-SAFE TRUNCATION — must happen before the llama.cpp call.
        # Character limits fail for financial/numeric text (1–2 chars/token).
        # This uses the model's own tokenizer to guarantee the prompt fits
        # within the context window and never triggers a SIGSEGV in llama.cpp.
        prompt = self._truncate_to_token_budget(prompt, max_tokens_)
        prompt = self._format_for_model(prompt)

        # CIRCUIT BREAKER CHECK — SECTION 2.1
        if self._circuit.is_open:
            _record_error("circuit_breaker_open")
            raise RuntimeError("CIRCUIT_BREAKER_OPEN: LLM is temporarily unavailable")

        last_exc: Exception | None = None
        wait = float(settings.LLM_RETRY_WAIT_MIN)

        for attempt in range(max(retries, 0) + 1):
            try:
                with self._circuit:
                    llm = self._load()
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
                    deadline = start + settings.LLM_CALL_TIMEOUT_SEC
                    parts: list[str] = []
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
                            **_seed_kwarg(seed),
                        ):
                            tok = chunk["choices"][0]["text"]
                            if tok:
                                parts.append(tok)
                            if time.time() > deadline:
                                timed_out = True
                                break

                    elapsed = time.time() - start
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

                    prompt_tokens = 0  # not reported in stream mode
                    completion_tokens = len(parts)
                    tps = round(completion_tokens / max(elapsed, 1e-6), 1)

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

            except RuntimeError:
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
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        session_id: str = "",
        seed: int | None = None,
    ) -> Iterator[str]:

        prompt = self._normalize(prompt)
        prompt = self._sanitize_prompt(prompt)

        if not prompt:
            return

        _max_tok = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
        prompt = self._truncate_to_token_budget(prompt, _max_tok)
        prompt = self._format_for_model(prompt)

        # CIRCUIT BREAKER CHECK
        if self._circuit.is_open:
            _record_error("circuit_breaker_open")
            logger.warning(event="gguf_stream_circuit_open", session_id=session_id)
            return

        try:
            llm = self._load()

            start = time.time()
            token_count = 0

            # THREAD+QUEUE STREAMING TIMEOUT
            # The timeout check inside a plain `for chunk in llm(..., stream=True):`
            # loop only fires AFTER the first token is yielded.  Prefill (processing
            # the full input prompt) runs BEFORE the first token, so a long prefill
            # phase could block the caller indefinitely.
            #
            # Fix: run the llama-cpp iterator on a daemon thread and drain its
            # output via a bounded queue.  The main thread (this generator) calls
            # q.get(timeout=...) which covers both prefill and per-token stalls.
            _SENTINEL = object()
            tok_q: _queue.Queue = _queue.Queue(maxsize=512)

            _max_tok = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
            _temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
            _top_p_v = top_p if top_p is not None else settings.LLM_TOP_P

            def _generate() -> None:
                try:
                    with self._circuit, self._lock:
                        for chunk in llm(
                            prompt,
                            max_tokens=_max_tok,
                            temperature=_temp,
                            top_p=_top_p_v,
                            top_k=settings.LLM_TOP_K_SAMPLING,
                            min_p=settings.LLM_MIN_P,
                            repeat_penalty=settings.LLM_REPEAT_PENALTY,
                            stop=_STOP_TOKENS,
                            stream=True,
                            **_seed_kwarg(seed),
                        ):
                            tok_q.put(chunk)
                except RuntimeError as _re:
                    tok_q.put(_re)
                except Exception as _ex:
                    tok_q.put(_ex)
                finally:
                    tok_q.put(_SENTINEL)

            gen_thread = threading.Thread(target=_generate, daemon=True)
            gen_thread.start()

            # Separate timeouts: longer for first token (covers prefill),
            # shorter per-token guard (catches mid-generation stalls).
            _first_tok_timeout = float(settings.LLM_CALL_TIMEOUT_SEC)
            _per_tok_timeout = min(30.0, _first_tok_timeout)
            _is_first_token = True

            while True:
                _timeout = _first_tok_timeout if _is_first_token else _per_tok_timeout
                try:
                    item = tok_q.get(timeout=_timeout)
                except _queue.Empty:
                    elapsed = round(time.time() - start, 1)
                    phase = "prefill" if _is_first_token else "generation"
                    logger.warning(
                        event="gguf_stream_timeout",
                        phase=phase,
                        elapsed=elapsed,
                        session_id=session_id,
                    )
                    break

                if item is _SENTINEL:
                    break

                if isinstance(item, (RuntimeError, Exception)):
                    raise item

                _is_first_token = False
                token = item["choices"][0]["text"]

                if not token:
                    continue

                token = token.replace("\x00", "")
                token_count += 1
                yield token

            latency = round(time.time() - start, 2)
            tps = round(token_count / max(latency, 1e-6), 1)

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

    def health_check(self) -> dict[str, Any]:
        model_exists = os.path.exists(self._model_path)
        model_size = os.path.getsize(self._model_path) if model_exists else 0
        return {
            "loaded": self._llm is not None,
            "model_path": self._model_path,
            "model_name": self._model_name,
            "model_exists": model_exists,
            "model_size_mb": round(model_size / (1024 * 1024), 1) if model_size else 0,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": settings.LLM_THREADS,
            "n_ctx": settings.CONTEXT_MAX_TOKENS,
            "circuit_open": self._circuit.is_open,
            "circuit_failures": self._circuit._failures,
        }

    # RESET — FOR TESTING / RELOAD

    def reset(self) -> None:
        with self._lock:
            self._llm = None
            self._circuit.record_success()
            logger.warning(event="gguf_model_reset", model=self._model_name)

    # CONTEXT MANAGER SUPPORT

    def __enter__(self) -> GGUFModel:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False
