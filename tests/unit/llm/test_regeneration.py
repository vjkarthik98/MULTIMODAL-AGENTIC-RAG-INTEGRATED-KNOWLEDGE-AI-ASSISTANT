"""Regeneration: what makes "regenerate" produce a different answer.

The bug these cover: pressing regenerate re-ran the whole pipeline with
no_cache=True and still returned the previous answer character for
character, because retrieval and decoding are both deterministic. Skipping
the cache is not enough — the second run has to sample differently and be
told the previous attempt was rejected.
"""
from __future__ import annotations

import inspect

from app.core.config import settings
from app.llm.gguf_model import _seed_kwarg
from app.llm.regeneration import REGENERATE_DIRECTIVE, regeneration_sampling

# ── regeneration_sampling ─────────────────────────────────────────────────────

class TestRegenerationSampling:

    def test_raises_a_greedy_temperature_to_the_floor(self):
        temp, _ = regeneration_sampling(0.0)
        assert temp == settings.LLM_TEMPERATURE_REGENERATE
        assert temp > 0.0, "temperature 0 is argmax — it would reproduce the same answer"

    def test_floor_never_cools_an_already_hotter_query_type(self):
        hot = settings.LLM_TEMPERATURE_REGENERATE + 0.2
        temp, _ = regeneration_sampling(hot)
        assert temp == hot

    def test_seed_is_fresh_each_call(self):
        seeds = {regeneration_sampling(0.0)[1] for _ in range(20)}
        # Same temperature + same seed = same trajectory, so a constant seed
        # would collapse every regeneration back onto one answer.
        assert len(seeds) > 1

    def test_seed_fits_downstream_integer_fields(self):
        for _ in range(50):
            _, seed = regeneration_sampling(0.0)
            assert 0 <= seed < 2**31


# ── seed plumbing into llama.cpp ──────────────────────────────────────────────

class TestSeedKwarg:

    def test_absent_when_no_seed_requested(self):
        # Must stay absent, not None: the normal answer path's sampling
        # behaviour has to be unchanged by this parameter existing.
        assert _seed_kwarg(None) == {}

    def test_present_when_requested(self):
        assert _seed_kwarg(12345) == {"seed": 12345}

    def test_coerces_to_int(self):
        assert _seed_kwarg(7.0) == {"seed": 7}

    def test_generate_and_stream_accept_a_seed(self):
        from app.llm.gguf_model import GGUFModel

        assert "seed" in inspect.signature(GGUFModel.generate).parameters
        assert "seed" in inspect.signature(GGUFModel.stream).parameters


# ── the prompt directive ──────────────────────────────────────────────────────

class TestRegenerateDirective:

    def test_tells_the_model_not_to_invent(self):
        low = REGENERATE_DIRECTIVE.lower()
        assert "rejected" in low
        assert "same evidence" in low
        # The one thing a regeneration must never do is manufacture a new
        # figure in order to look different from the last attempt.
        assert "do not add any fact that is not in the evidence" in low

    def test_prompt_builder_omits_it_by_default(self):
        from app.prompt.prompt_builder import PromptBuilder

        prompt = PromptBuilder().build_prompt(
            query="What was total revenue?",
            context="Total revenue was $391.0 billion.",
        )
        assert "REGENERATE:" not in prompt

    def test_prompt_builder_places_it_before_the_query(self):
        from app.prompt.prompt_builder import PromptBuilder

        prompt = PromptBuilder().build_prompt(
            query="What was total revenue?",
            context="Total revenue was $391.0 billion.",
            regenerate=True,
        )
        assert "REGENERATE:" in prompt
        # After the answer cue the model would continue the directive text
        # instead of obeying it.
        assert prompt.index("REGENERATE:") < prompt.index("QUERY:")

    def test_cot_prompt_places_it_before_the_query(self):
        from app.reasoning.reasoning_engine import _build_cot_prompt

        plain = _build_cot_prompt("What was total revenue?", "Revenue was $391.0B.", "")
        regen = _build_cot_prompt(
            "What was total revenue?", "Revenue was $391.0B.", "", regenerate=True
        )
        assert "REGENERATE:" not in plain
        assert "REGENERATE:" in regen
        assert regen.index("REGENERATE:") < regen.index("QUERY:")


# ── end to end through the reasoning engine ───────────────────────────────────

class _RecordingLLM:
    """Stands in for GGUFModel — records exactly what sampling it was asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return (
            "Answer: Total revenue was $391.0 billion. [apple_10k.pdf]\n"
            "Answer Tags: [apple_10k.pdf]\nConfidence: 0.9\nSources Used: 1"
        )


_DOCS = [
    {
        "text": "Total revenue was $391.0 billion in fiscal 2024.",
        "metadata": {"source": "apple_10k.pdf", "modality": "pdf"},
        "score": 0.9,
    }
]


def _answer(regenerate: bool) -> tuple[str, dict]:
    from unittest.mock import patch

    from app.reasoning.reasoning_engine import ReasoningEngine

    llm = _RecordingLLM()
    # Entity grounding pulls in the NER model; irrelevant here and slow.
    with patch("app.pipeline.query_pipeline._query_entities_ungrounded", return_value=False):
        ReasoningEngine(llm).generate_answer(
            query="What was total revenue?",
            retrieved_docs=_DOCS,
            regenerate=regenerate,
        )
    prompt, kwargs = llm.calls[0]
    return prompt, kwargs


class TestReasoningEngineRegenerate:

    def test_normal_answer_stays_greedy_and_unseeded(self):
        prompt, kwargs = _answer(regenerate=False)
        assert kwargs["temperature"] == 0.0
        assert kwargs["seed"] is None
        assert "REGENERATE:" not in prompt

    def test_regenerate_samples_off_the_greedy_path(self):
        prompt, kwargs = _answer(regenerate=True)
        assert kwargs["temperature"] == settings.LLM_TEMPERATURE_REGENERATE
        assert isinstance(kwargs["seed"], int)
        assert "REGENERATE:" in prompt

    def test_two_regenerations_do_not_share_a_seed(self):
        _, first = _answer(regenerate=True)
        _, second = _answer(regenerate=True)
        assert first["seed"] != second["seed"]


# ── the flag survives every hop from HTTP request to sampler ──────────────────

class TestRegenerateWiring:
    """The chain is only as good as its weakest hop: a `regenerate` that stops
    being forwarded halfway down silently restores the original bug (identical
    answer, no error anywhere)."""

    def test_request_model_exposes_the_flag(self):
        from app.api.api_routes import QueryRequest

        assert QueryRequest(query="q").regenerate is False
        assert QueryRequest(query="q", regenerate=True).regenerate is True

    def test_every_hop_accepts_it(self):
        from app.pipeline.query_pipeline import query_pipeline
        from app.pipeline.rag_pipeline import RAGPipeline
        from app.prompt.prompt_builder import PromptBuilder
        from app.reasoning.reasoning_engine import ReasoningEngine
        from app.verification.verification_loop import VerificationLoop

        for fn in (
            RAGPipeline.stream,
            query_pipeline,
            VerificationLoop.run,
            VerificationLoop._generate,
            ReasoningEngine.generate_answer,
            ReasoningEngine.generate_answer_async,
            PromptBuilder.build_prompt,
            PromptBuilder.build_prompt_async,
        ):
            params = inspect.signature(fn).parameters
            assert "regenerate" in params, f"{fn.__qualname__} drops the regenerate flag"
            assert params["regenerate"].default is False, (
                f"{fn.__qualname__} must default to off — regeneration sampling is "
                "opt-in, never the default answer path"
            )
