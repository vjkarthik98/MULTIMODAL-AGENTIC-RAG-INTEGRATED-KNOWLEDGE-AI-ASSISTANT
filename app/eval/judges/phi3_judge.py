"""Phi-3-mini-4k-instruct judge for Ragas evaluation.

Loads Phi-3-mini Q4 (~2.3GB VRAM) as a dedicated eval judge.
Phi-3 follows JSON instructions precisely — no prose, no preamble.
Completely separate from Mistral 7B (RAG model) — no interference.

Model: microsoft/Phi-3-mini-4k-instruct-gguf (MIT license)
Path:  /opt/dlami/nvme/hf_cache/Phi-3-mini-4k-instruct-q4.gguf
"""
from __future__ import annotations

import json
import os
import re
import threading
import typing as t

from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue

try:
    from ragas.llms.base import BaseRagasLLM
    from ragas.run_config import RunConfig
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    BaseRagasLLM = object  # type: ignore
    RunConfig = object     # type: ignore

_MODEL_PATH = os.getenv(
    "PHI3_JUDGE_MODEL_PATH",
    "/opt/dlami/nvme/hf_cache/Phi-3-mini-4k-instruct-q4.gguf",
)
_lock = threading.Lock()
_llm = None


def _load_phi3():
    """Load Phi-3 mini once and cache it."""
    global _llm
    if _llm is not None:
        return _llm
    with _lock:
        if _llm is not None:
            return _llm
        try:
            from llama_cpp import Llama
            print(f"[eval] Loading Phi-3 judge from {_MODEL_PATH} ...")
            _llm = Llama(
                model_path=_MODEL_PATH,
                n_ctx=4096,
                n_gpu_layers=10,    # partial GPU offload — fits alongside Mistral
                n_threads=4,
                verbose=False,
            )
            print("[eval] Phi-3 judge loaded ✓")
        except Exception as e:
            raise RuntimeError(f"Failed to load Phi-3 judge: {e}") from e
    return _llm


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Extract first valid JSON object or array from text."""
    text = text.strip()

    # Triple backtick block
    cb = re.search(r"```(?:json)?\s*([\[{][\s\S]*?[\]}])\s*```", text)
    if cb:
        try:
            json.loads(cb.group(1))
            return cb.group(1).strip()
        except json.JSONDecodeError:
            pass

    # Greedy array
    arr = re.search(r"(\[[\s\S]*\])", text)
    if arr:
        try:
            json.loads(arr.group(1))
            return arr.group(1).strip()
        except json.JSONDecodeError:
            pass

    # Greedy object
    obj = re.search(r"(\{[\s\S]*\})", text)
    if obj:
        try:
            json.loads(obj.group(1))
            return obj.group(1).strip()
        except json.JSONDecodeError:
            pass

    # Balanced bracket walk
    for start_char, end_char in [('[', ']'), ('{', '}')]:
        idx = text.find(start_char)
        if idx >= 0:
            depth = 0
            for i, c in enumerate(text[idx:], idx):
                if c == start_char:
                    depth += 1
                elif c == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[idx:i+1]
                        try:
                            json.loads(candidate)
                            return candidate.strip()
                        except json.JSONDecodeError:
                            break
    return text


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(ragas_prompt: str) -> str:
    """
    Wrap the Ragas prompt in Phi-3's chat template.
    Uses exact JSON schemas from Ragas metric definitions.
    Phi-3 template: <|system|>...<|end|>\n<|user|>...<|end|>\n<|assistant|>
    """
    p = ragas_prompt.lower()

    # Faithfulness decomposition — long_form_answer prompt
    if "simpler_statements" in p or "sentence_index" in p or "break down each sentence" in p:
        schema = '[{"sentence_index": 0, "simpler_statements": ["full statement without pronouns"]}]'
        instruction = "Break down each sentence into simpler statements. Output ONLY a JSON array."

    # Faithfulness NLI — nli_statements prompt
    elif "inferred based on the context" in p or ("verdict" in p and "faithfulness" in p):
        schema = '[{"statement": "original statement text", "reason": "reason for verdict", "verdict": 1}]'
        instruction = "Judge each statement faithfulness. verdict=1 if supported by context, 0 if not. Output ONLY a JSON array."

    # Context precision — exact schema: {reason, verdict}
    elif "context was useful" in p or "useful in arriving" in p or "verification task" in p:
        schema = '{"reason": "explanation of why context is or is not useful", "verdict": 1}'
        instruction = "Verify if context was useful. Output ONLY a JSON object with reason and verdict (1=useful, 0=not useful)."

    # Answer relevancy — exact schema: {question, noncommittal}
    elif "noncommittal" in p or "generate a question" in p:
        schema = '{"question": "a question that the answer is responding to", "noncommittal": 0}'
        instruction = "Generate a question the answer responds to. noncommittal=1 if answer is vague/evasive, 0 if committal. Output ONLY a JSON object."

    # Context recall — uses NLI-style attribution
    elif "attributed" in p or "context_recall" in p or ("sentences" in p and "context" in p):
        schema = '[{"statement": "sentence from answer", "reason": "attribution reason", "attributed": 1}]'
        instruction = "For each sentence, check if it can be attributed to the context. attributed=1 if yes. Output ONLY a JSON array."

    else:
        schema = '[{"statement": "text", "reason": "reason", "verdict": 1}]'
        instruction = "Evaluate each statement. Output ONLY valid JSON."

    system = (
        f"You are a JSON-only evaluator. {instruction} "
        f"The exact output schema is: {schema} "
        "Output ONLY raw JSON. No preamble. No explanation. No markdown. No extra text."
    )

    return (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{ragas_prompt}<|end|>\n"
        f"<|assistant|>\n"
    )


# ── Core generate ─────────────────────────────────────────────────────────────

def _generate(prompt_text: str) -> str:
    llm = _load_phi3()
    full_prompt = _build_prompt(prompt_text)
    try:
        output = llm(
            full_prompt,
            max_tokens=256,
            temperature=0.0,    # deterministic JSON output
            stop=["<|end|>", "<|user|>", "<|system|>", "\n\n"],
        )
        raw = output["choices"][0]["text"].strip()
        return _extract_json(raw)
    except Exception as e:
        return json.dumps([{"statement": "error", "reason": str(e), "verdict": 0}])


# ── Ragas-compatible judge ────────────────────────────────────────────────────

class Phi3Judge(BaseRagasLLM if RAGAS_AVAILABLE else object):  # type: ignore
    """
    Phi-3-mini-4k-instruct as Ragas judge.
    Loads ~2.3GB on GPU alongside Mistral. Outputs strict JSON reliably.
    """

    def __init__(self, run_config: t.Optional[t.Any] = None):
        if RAGAS_AVAILABLE:
            cfg = run_config or RunConfig(max_workers=1, timeout=600)
            super().__init__(run_config=cfg)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Any = None,
    ) -> LLMResult:
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        out = _generate(text)
        return LLMResult(generations=[[Generation(text=out)] for _ in range(n)])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Any = None,
    ) -> LLMResult:
        import asyncio
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        out = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _generate(text)
        )
        return LLMResult(generations=[[Generation(text=out)] for _ in range(n)])

    def set_run_config(self, run_config: t.Any) -> None:
        if RAGAS_AVAILABLE and hasattr(super(), "set_run_config"):
            super().set_run_config(run_config)


def get_judge() -> "Phi3Judge":
    if not RAGAS_AVAILABLE:
        raise ImportError("ragas is not installed")
    return Phi3Judge()
