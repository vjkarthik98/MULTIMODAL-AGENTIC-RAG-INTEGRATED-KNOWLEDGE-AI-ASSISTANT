"""Wraps app/llm/gguf_model.py as a Ragas-compatible LLM judge.

Ragas 0.1.x expects BaseRagasLLM with an async agenerate_text method.
We wrap the sync GGUF generate() in an executor to satisfy the async interface.
"""
from __future__ import annotations

import asyncio
import typing as t

from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue

try:
    from ragas.llms.base import BaseRagasLLM
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    BaseRagasLLM = object  # type: ignore


class GGUFJudge(BaseRagasLLM if RAGAS_AVAILABLE else object):  # type: ignore[misc]
    """Local GGUF Mistral as Ragas judge. Uses the same model as production inference.

    Configured via EVAL_JUDGE_TEMPERATURE (default 0.1 for deterministic judgement).
    Uses a separate session_id ("eval_judge") to avoid polluting query session memory.
    """

    def __init__(self, temperature: float = 0.1):
        self._temperature = temperature
        self._llm = None
        self._lock = asyncio.Lock()

    def _get_llm(self):
        if self._llm is None:
            try:
                from app.llm.gguf_model import GGUFModel
                self._llm = GGUFModel()
            except Exception as e:
                raise RuntimeError(
                    f"Could not load GGUF model for eval judge: {e}. "
                    "Ensure LLM_MODEL_PATH is set and the GGUF file exists."
                ) from e
        return self._llm

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: object = None,
    ) -> LLMResult:
        """Run GGUF inference in a thread executor so async callers don't block."""
        temp = temperature if temperature is not None else self._temperature
        text_prompt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

        loop = asyncio.get_event_loop()
        llm = self._get_llm()

        def _call():
            return llm.generate(
                prompt=text_prompt,
                temperature=temp,
                session_id="eval_judge",
            )

        text = await loop.run_in_executor(None, _call)
        generations = [[Generation(text=text)] for _ in range(n)]
        return LLMResult(generations=generations)

    def generate(self, *args, **kwargs):
        """Synchronous shim — runs the async path in a new loop if needed."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in an async context — can't nest; use run_in_executor pattern
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(asyncio.run, self.agenerate_text(*args, **kwargs))
                    return future.result()
            return loop.run_until_complete(self.agenerate_text(*args, **kwargs))
        except RuntimeError:
            return asyncio.run(self.agenerate_text(*args, **kwargs))


def get_judge(temperature: float = 0.1) -> GGUFJudge:
    """Return a GGUFJudge instance. Raises RuntimeError if GGUF unavailable."""
    if not RAGAS_AVAILABLE:
        raise ImportError("ragas is not installed. Run: pip install ragas==0.1.21")
    return GGUFJudge(temperature=temperature)
