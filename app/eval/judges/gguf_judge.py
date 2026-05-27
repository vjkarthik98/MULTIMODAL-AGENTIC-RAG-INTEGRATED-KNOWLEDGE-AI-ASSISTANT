"""Wraps the live FastAPI server as a Ragas-compatible LLM judge.

Instead of loading a second GGUF model into VRAM (which would OOM on the T4),
we POST to the running /rag/query endpoint. The server already holds Mistral-7B
Q4_K_M in VRAM; this judge reuses it at low temperature for deterministic scoring.

Ragas 0.1.x expects BaseRagasLLM with both generate_text and agenerate_text.
"""
from __future__ import annotations

import os
import typing as t

import httpx
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue

try:
    from ragas.llms.base import BaseRagasLLM
    from ragas.run_config import RunConfig
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    BaseRagasLLM = object  # type: ignore[assignment,misc]
    RunConfig = object  # type: ignore[assignment,misc]

_SERVER_URL = os.getenv("EVAL_SERVER_URL", "http://127.0.0.1:8000")
_JUDGE_SESSION = "eval_judge"
_JUDGE_USER    = "eval_default"
_HTTP_TIMEOUT  = 300  # seconds — GGUF generation can be slow


class GGUFJudge(BaseRagasLLM if RAGAS_AVAILABLE else object):  # type: ignore[misc]
    """Routes judge calls through the live FastAPI server to avoid double-loading GGUF.

    The server holds Mistral-7B Q4_K_M in VRAM; a second load would OOM on T4.
    We POST to /rag/query with the judge prompt and parse the 'answer' field.
    RunConfig.max_workers is forced to 1 because the server serialises GPU calls.
    """

    def __init__(self, temperature: float = 0.1, run_config: t.Optional[t.Any] = None):
        if RAGAS_AVAILABLE:
            cfg = run_config or RunConfig(max_workers=1, timeout=_HTTP_TIMEOUT)
            super().__init__(run_config=cfg)
        self._temperature = temperature

    def _call_server(self, prompt_text: str) -> str:
        payload = {
            "query":      prompt_text,
            "session_id": _JUDGE_SESSION,
            "user_id":    _JUDGE_USER,
        }
        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
                resp = client.post(f"{_SERVER_URL}/rag/query", json=payload)
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer") or data.get("response") or ""
                return str(answer).strip()
        except Exception as e:
            raise RuntimeError(
                f"GGUFJudge HTTP call failed — is the server running at {_SERVER_URL}? Error: {e}"
            ) from e

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Any = None,
    ) -> LLMResult:
        text_prompt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        text = self._call_server(text_prompt)
        generations = [[Generation(text=text)] for _ in range(n)]
        return LLMResult(generations=generations)

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Any = None,
    ) -> LLMResult:
        import asyncio
        text_prompt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        text = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._call_server(text_prompt)
        )
        generations = [[Generation(text=text)] for _ in range(n)]
        return LLMResult(generations=generations)

    def set_run_config(self, run_config: t.Any) -> None:
        if RAGAS_AVAILABLE and hasattr(super(), "set_run_config"):
            super().set_run_config(run_config)


def get_judge(temperature: float = 0.1) -> "GGUFJudge":
    """Return a GGUFJudge instance. Raises ImportError if ragas unavailable."""
    if not RAGAS_AVAILABLE:
        raise ImportError("ragas is not installed. Run: pip install ragas==0.1.21")
    return GGUFJudge(temperature=temperature)
