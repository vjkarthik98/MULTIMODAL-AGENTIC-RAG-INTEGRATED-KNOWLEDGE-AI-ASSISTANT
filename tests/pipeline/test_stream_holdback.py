"""True token streaming (perf pass) — RAGPipeline.stream must:

1. yield tokens INCREMENTALLY (not one completed block at the end),
2. PII-scrub every flushed segment (entities never reach the client raw),
3. end with a \x00REPLACE\x00 sentinel carrying the canonical guarded answer,
4. end with a \x00SOURCES\x00 sentinel carrying the source-chip JSON,
5. suppress refusals entirely — only the \x00REFUSAL\x00 sentinel is sent,
   whether the refusal is short (under the prefix gate) or long.
"""
import json

import pytest

from app.pipeline import rag_pipeline as rp


_CONTEXT = (
    "Quarterly revenue increased by 12 percent compared to the prior year, "
    "driven by strong services growth and stable hardware margins."
)

_ANSWER = (
    "The quarterly revenue increased by 12 percent compared to the prior "
    "year, driven by strong services growth and stable hardware margins "
    "across all operating segments. Contact investor relations at "
    "ir.contact@example.com for details about the upcoming shareholder "
    "meeting and the dividend schedule for institutional holders."
)

_REFUSAL_SHORT = "I could not find this in the provided sources."

_REFUSAL_LONG = (
    "I could not find any relevant information in the provided sources to "
    "answer this question. The documents discuss unrelated topics such as "
    "hardware margins and shareholder meetings, none of which mention the "
    "subject you asked about."
)


class _FakeRetriever:
    def search(self, **kwargs):
        return [{
            "text": _CONTEXT,
            "score": 0.9,
            "metadata": {"source": "report.txt", "modality": "text"},
        }]


class _FakePromptBuilder:
    # **kwargs, not a fixed parameter list: the real PromptBuilder gained
    # `memory` (and later `regenerate`) and this double did not, so every test
    # in this file failed inside stream()'s try/except as
    # "unexpected keyword argument 'memory'" — a broken double reported as a
    # streaming bug. Accept whatever the caller passes; these tests are about
    # the holdback/flush behaviour, not the prompt's shape.
    def build_prompt(self, query, context, session_id=None, **kwargs):
        return f"CONTEXT: {context}\nQUERY: {query}"


class _FakeLLM:
    def __init__(self, answer):
        self._answer = answer

    def stream(self, prompt, **kwargs):
        # ~5-char tokens, mimicking llama.cpp sub-word chunks
        for i in range(0, len(self._answer), 5):
            yield self._answer[i:i + 5]


def _run_stream(monkeypatch, answer, query="What drove revenue growth this quarter?"):
    monkeypatch.setattr(rp, "_get_stream_reranker", lambda: None)
    pipe = rp.RAGPipeline()
    pipe._get_retriever = _FakeRetriever  # bound-attr shadow: called as self._get_retriever()
    pipe._get_prompt_builder = _FakePromptBuilder
    pipe._get_llm = lambda: _FakeLLM(answer)
    return list(pipe.stream(query, session_id="t-stream"))


def _split(events):
    tokens, replace, sources, refusal = [], None, None, False
    for ev in events:
        if ev.startswith("\x00REPLACE\x00"):
            replace = ev[9:]
        elif ev.startswith("\x00SOURCES\x00"):
            sources = json.loads(ev[9:])
        elif ev.startswith("\x00REFUSAL\x00"):
            refusal = True
        else:
            tokens.append(ev)
    return tokens, replace, sources, refusal


def test_tokens_stream_incrementally(monkeypatch):
    events = _run_stream(monkeypatch, _ANSWER)
    tokens, replace, sources, refusal = _split(events)
    assert not refusal
    assert len(tokens) >= 2, "expected multiple incremental flushes, got one block"
    assert replace is not None and len(replace) > 50
    assert isinstance(sources, list) and sources
    assert sources[0]["source"] == "report.txt"


def test_flushed_segments_are_pii_scrubbed(monkeypatch):
    events = _run_stream(monkeypatch, _ANSWER)
    tokens, replace, _, _ = _split(events)
    streamed = "".join(tokens)
    assert "ir.contact@example.com" not in streamed
    assert "ir.contact@example.com" not in replace


def test_replace_carries_canonical_guarded_answer(monkeypatch):
    events = _run_stream(monkeypatch, _ANSWER)
    _, replace, _, _ = _split(events)
    assert "quarterly revenue increased by 12 percent" in replace.lower()


def test_short_refusal_under_prefix_gate_never_streams(monkeypatch):
    events = _run_stream(monkeypatch, _REFUSAL_SHORT)
    tokens, replace, sources, refusal = _split(events)
    assert refusal
    assert tokens == [], "refusal text must never reach the client"
    assert replace is None and sources is None


def test_long_refusal_caught_at_prefix_gate(monkeypatch):
    events = _run_stream(monkeypatch, _REFUSAL_LONG)
    tokens, replace, sources, refusal = _split(events)
    assert refusal
    assert tokens == []
    assert replace is None and sources is None
