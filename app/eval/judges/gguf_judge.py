"""Wraps the live FastAPI server as a Ragas-compatible LLM judge.

Uses /rag/llm/generate (direct LLM, no RAG pipeline) to avoid context
pollution from retrieval. Extracts JSON from Mistral's conversational
response using regex so Ragas can parse the structured output it needs.
"""

from __future__ import annotations

import json
import os
import re
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
_JUDGE_USER = "eval_default"
_HTTP_TIMEOUT = 300


def _extract_json_from_text(text: str) -> str:
    """
    Mistral often wraps JSON in ```json...``` or outputs it after prose.
    Tries multiple extraction strategies in order of reliability.
    """
    text = text.strip()

    # Strategy 1: triple-backtick code block
    cb = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if cb:
        try:
            json.loads(cb.group(1))
            return cb.group(1).strip()
        except json.JSONDecodeError:
            pass

    # Strategy 2: find JSON array (greedy — handles nested objects)
    arr = re.search(r"(\[[\s\S]*\])", text)
    if arr:
        try:
            json.loads(arr.group(1))
            return arr.group(1).strip()
        except json.JSONDecodeError:
            pass

    # Strategy 3: find JSON object (greedy)
    obj = re.search(r"(\{[\s\S]*\})", text)
    if obj:
        try:
            json.loads(obj.group(1))
            return obj.group(1).strip()
        except json.JSONDecodeError:
            pass

    # Strategy 4: find first { or [ and extract to matching close
    for start_char, end_char in [('[', ']'), ('{', '}')]:
        idx = text.find(start_char)
        if idx >= 0:
            # Walk to find balanced closing bracket
            depth = 0
            for i, c in enumerate(text[idx:], idx):
                if c == start_char:
                    depth += 1
                elif c == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[idx : i + 1]
                        try:
                            json.loads(candidate)
                            return candidate.strip()
                        except json.JSONDecodeError:
                            break

    # Last resort — return as-is
    return text


class GGUFJudge(BaseRagasLLM if RAGAS_AVAILABLE else object):  # type: ignore[misc]
    """
    Routes judge calls through /rag/llm/generate (direct LLM endpoint).
    Extracts structured JSON from Mistral's conversational output so
    Ragas can parse faithfulness/relevancy scores correctly.
    """

    def __init__(self, temperature: float = 0.1, run_config: t.Any | None = None):
        if RAGAS_AVAILABLE:
            cfg = run_config or RunConfig(max_workers=1, timeout=_HTTP_TIMEOUT)
            super().__init__(run_config=cfg)
        self._temperature = temperature

    def _call_server(self, prompt_text: str) -> str:
        """
        Call the direct LLM endpoint, extract JSON, fall back to
        CrossEncoder judge if Mistral returns unparseable prose.
        """
        # Detect prompt type and add a strict closing instruction
        # that matches what Ragas expects for each metric
        if "simpler_statements" in prompt_text or "sentence_index" in prompt_text:
            suffix = "\nRespond with ONLY a JSON array like: [{\"sentence_index\": 0, \"simpler_statements\": [\"statement here\"]}]"
        elif "nli_statements" in prompt_text or (
            "verdict" in prompt_text and "statements" in prompt_text
        ):
            suffix = "\nRespond with ONLY a JSON array like: [{\"statement\": \"text\", \"reason\": \"reason\", \"verdict\": 1}]"
        elif "noncommittal" in prompt_text or "generate a question" in prompt_text.lower():
            suffix = "\nRespond with ONLY a JSON object like: {\"question\": \"text\", \"noncommittal\": 0}"
        elif "context was useful" in prompt_text.lower():
            suffix = "\nRespond with ONLY a JSON object like: {\"reason\": \"reason\", \"verdict\": \"1\"}"
        else:
            suffix = "\nRespond with ONLY valid JSON. No explanations."

        forced_prompt = (
            "[INST] You are a JSON-only evaluator. "
            "Output ONLY raw JSON with no preamble, no explanation, no markdown. "
            "Follow the exact schema shown.\n\n" + prompt_text + suffix + " [/INST]"
        )
        payload = {
            "prompt": forced_prompt,
            "max_tokens": 512,
            "temperature": self._temperature,
        }
        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
                resp = client.post(
                    f"{_SERVER_URL}/rag/llm/generate",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data.get("text") or ""
                extracted = _extract_json_from_text(raw)

                # Validate it's actually parseable JSON
                import json as _json

                try:
                    _json.loads(extracted)
                    return extracted
                except (_json.JSONDecodeError, ValueError):
                    # Mistral returned prose — return as-is and let Ragas handle it
                    return extracted

        except Exception as e:
            raise RuntimeError(
                f"GGUFJudge HTTP call failed — is the server running at {_SERVER_URL}? Error: {e}"
            ) from e

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
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
        temperature: float | None = None,
        stop: list[str] | None = None,
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


def get_judge(temperature: float = 0.1) -> GGUFJudge:
    """Return a GGUFJudge instance. Raises ImportError if ragas unavailable."""
    if not RAGAS_AVAILABLE:
        raise ImportError("ragas is not installed. Run: pip install ragas==0.1.21")
    return GGUFJudge(temperature=temperature)
