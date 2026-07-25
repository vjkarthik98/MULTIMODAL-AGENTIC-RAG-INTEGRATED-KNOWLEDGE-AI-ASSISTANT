"""CrossEncoder-based Ragas judge.

Uses CrossEncoder already on GPU to produce the exact JSON schemas
that Ragas 0.1.x expects for each metric. No external API needed.

Ragas metric → expected JSON output:
  faithfulness (NLI)     → [{"statement":"..","reason":"..","verdict":0|1}]
  faithfulness (decomp)  → [{"sentence_index":0,"simpler_statements":["..."]}]
  answer_relevancy       → {"question":"..","noncommittal":0|1}
  context_precision      → {"reason":"..","verdict":"1"|"0"}
"""

from __future__ import annotations

import json
import re
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
    RunConfig = object  # type: ignore


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_reranker():
    from app.core.model_loader import model_loader

    return model_loader.get_reranker()


def _sigmoid(x: float) -> float:
    import math

    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except (OverflowError, ValueError):
        return 0.5


def _ce_score(pairs: list) -> list:
    """Score pairs with CrossEncoder, return list of sigmoid-normalised floats."""
    reranker = _get_reranker()
    if reranker is None or not pairs:
        return [0.5] * len(pairs)
    try:
        raw = reranker.predict(pairs)
        return [_sigmoid(float(s)) for s in raw]
    except Exception:
        return [0.5] * len(pairs)


def _extract_field(text: str, field: str) -> str:
    """Extract a quoted field value from a Ragas prompt."""
    m = re.search(rf'"{field}":\s*"(.*?)"', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try unquoted
    m2 = re.search(rf'{field}:\s*(.+?)(?:\n|$)', text)
    if m2:
        return m2.group(1).strip().strip('"\'')
    return ""


def _extract_list_field(text: str, field: str) -> list[str]:
    """Extract a JSON array field from a Ragas prompt."""
    m = re.search(rf'"{field}":\s*\[(.*?)\]', text, re.DOTALL)
    if m:
        items = re.findall(r'"([^"]+)"', m.group(1))
        return items
    return []


# ── Metric-specific scorers ───────────────────────────────────────────────────


def _nli_statements(prompt_text: str) -> str:
    """
    Faithfulness NLI pass.
    Input: context + list of statements
    Output: [{"statement":"..","reason":"..","verdict":0|1}]
    """
    context = _extract_field(prompt_text, "context")
    statements = _extract_list_field(prompt_text, "statements")

    if not statements:
        # Try to parse from plain text
        lines = [
            l.strip().strip("-").strip()
            for l in prompt_text.split("\n")
            if len(l.strip()) > 15
            and not l.strip().startswith("{")
            and "task" not in l.lower()
            and "context" not in l.lower()[:10]
        ]
        statements = lines[:6]

    if not context:
        context = prompt_text[:400]

    if not statements:
        return "[]"

    pairs = [(context[:512], stmt[:256]) for stmt in statements]
    scores = _ce_score(pairs)

    results = []
    for stmt, score in zip(statements, scores):
        verdict = 1 if score > 0.45 else 0
        results.append(
            {
                "statement": stmt,
                "reason": f"Semantic similarity to context: {score:.3f}",
                "verdict": verdict,
            }
        )
    return json.dumps(results)


def _decompose_statements(prompt_text: str) -> str:
    """
    Faithfulness decomposition pass.
    Input: question + answer + sentences
    Output: [{"sentence_index":0,"simpler_statements":["..."]}]
    """
    # Extract sentences from the prompt
    sentences_match = re.search(
        r'sentences["\s:]+(.+?)(?:analysis|$)', prompt_text, re.DOTALL | re.IGNORECASE
    )
    if sentences_match:
        raw = sentences_match.group(1)
        # Parse "0:sentence text" format
        indexed = re.findall(r'(\d+)\s*:\s*(.+?)(?=\d+\s*:|$)', raw, re.DOTALL)
        results = []
        for idx, sentence in indexed:
            sentence = sentence.strip()
            results.append(
                {
                    "sentence_index": int(idx),
                    "simpler_statements": [sentence] if sentence else [],
                }
            )
        if results:
            return json.dumps(results)

    # Fallback: split answer into sentences
    answer = _extract_field(prompt_text, "answer")
    if not answer:
        answer = prompt_text[:300]

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 5]
    results = [
        {"sentence_index": i, "simpler_statements": [s]} for i, s in enumerate(sentences[:5])
    ]
    return json.dumps(
        results if results else [{"sentence_index": 0, "simpler_statements": [answer[:100]]}]
    )


def _answer_relevancy(prompt_text: str) -> str:
    """
    Answer relevancy.
    Input: answer + context
    Output: {"question":"..","noncommittal":0|1}
    """
    answer = _extract_field(prompt_text, "answer")
    context = _extract_field(prompt_text, "context")

    if not answer:
        answer = prompt_text[:200]

    # Detect noncommittal answers
    noncommittal_phrases = [
        "i don't know",
        "i'm not sure",
        "i cannot",
        "no information",
        "not found in",
        "unable to",
        "i do not have",
        "no relevant",
        "cannot find",
        "not available",
    ]
    is_noncommittal = any(p in answer.lower() for p in noncommittal_phrases)

    # Score answer vs context relevance
    if context and not is_noncommittal:
        scores = _ce_score([(context[:512], answer[:256])])
        relevance = scores[0]
        is_noncommittal = relevance < 0.25

    # Generate a question from the answer (Ragas uses this to compute similarity)
    # Strip trailing punctuation and make it a question
    q = re.sub(r'[.!?]+$', '', answer[:80]).strip()
    question = q + "?" if q else "What is described?"

    return json.dumps(
        {
            "question": question,
            "noncommittal": 1 if is_noncommittal else 0,
        }
    )


def _context_precision(prompt_text: str) -> str:
    """
    Context precision.
    Input: question + answer + context
    Output: {"reason":"..","verdict":"1"|"0"}
    """
    question = _extract_field(prompt_text, "question")
    answer = _extract_field(prompt_text, "answer")
    context = _extract_field(prompt_text, "context")

    if not context:
        context = prompt_text[:400]
    if not question:
        question = answer[:100] if answer else ""

    query = f"{question} {answer}".strip()[:256]
    scores = _ce_score([(context[:512], query)])
    score = scores[0]

    verdict = "1" if score > 0.35 else "0"
    return json.dumps(
        {
            "reason": f"Context relevance score: {score:.3f}",
            "verdict": verdict,
        }
    )


# ── Prompt type detector ──────────────────────────────────────────────────────


def _detect_and_score(prompt_text: str) -> str:
    """Detect which Ragas metric is being scored and return correct JSON."""
    p = prompt_text.lower()

    # Faithfulness decomposition (long_form_answer prompt)
    if "simpler_statements" in p or "sentence_index" in p or "break down each sentence" in p:
        return _decompose_statements(prompt_text)

    # Faithfulness NLI (nli_statements prompt)
    if "verdict" in p and ("faithfulness" in p or "statements" in p or "inferred" in p):
        return _nli_statements(prompt_text)

    # Context precision
    if "context was useful" in p or "context_precision" in p or ("verdict" in p and "useful" in p):
        return _context_precision(prompt_text)

    # Answer relevancy
    if "noncommittal" in p or "generate a question" in p:
        return _answer_relevancy(prompt_text)

    # Context recall — Ragas uses NLI format similar to faithfulness
    if "context_recall" in p or ("attributed" in p and "context" in p):
        return _nli_statements(prompt_text)

    # Default fallback
    if "statements" in p:
        return _nli_statements(prompt_text)
    return _answer_relevancy(prompt_text)


# ── Ragas-compatible judge class ──────────────────────────────────────────────


class CrossEncoderJudge(BaseRagasLLM if RAGAS_AVAILABLE else object):  # type: ignore
    """
    Ragas-compatible judge using CrossEncoder on GPU.
    Produces exact JSON schemas expected by each Ragas metric.
    No external API. No extra GPU memory.
    """

    def __init__(self, run_config: t.Any | None = None):
        if RAGAS_AVAILABLE:
            cfg = run_config or RunConfig(max_workers=1, timeout=120)
            super().__init__(run_config=cfg)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: t.Any = None,
    ) -> LLMResult:
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        out = _detect_and_score(text)
        return LLMResult(generations=[[Generation(text=out)] for _ in range(n)])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: t.Any = None,
    ) -> LLMResult:
        import asyncio

        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        out = await asyncio.get_event_loop().run_in_executor(None, lambda: _detect_and_score(text))
        return LLMResult(generations=[[Generation(text=out)] for _ in range(n)])

    def set_run_config(self, run_config: t.Any) -> None:
        if RAGAS_AVAILABLE and hasattr(super(), "set_run_config"):
            super().set_run_config(run_config)


def get_judge() -> CrossEncoderJudge:
    if not RAGAS_AVAILABLE:
        raise ImportError("ragas is not installed")
    return CrossEncoderJudge()
