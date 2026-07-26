"""Regression: prompt builder must contain Phase 24 Rule 5 and Rule 6 fixes.

Phase 24 fixes:
- Rule 5 tightened: "Answer only what the QUERY asks. Do NOT add extra sentences..."
- Rule 6 relaxed: removed "if the relevant period chunk is missing, say so" clause
  that caused the LLM to refuse even when the correct figure was present partway through
  a mixed-content chunk.
"""
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def test_rule_5_concise_phrase_present():
    """Prompt builder must contain 'Answer only what the QUERY asks' or equivalent."""
    import app.reasoning.reasoning_engine as re_mod

    source = inspect.getsource(re_mod)
    has_rule5 = (
        "Answer only what the QUERY asks" in source
        or "answer only what the query asks" in source.lower()
        or "Do NOT add extra sentences" in source
    )
    assert has_rule5, (
        "Rule 5 ('Answer only what the QUERY asks') missing from reasoning_engine — "
        "LLM may pad correct answers with hallucinated filler sentences"
    )


def test_rule_6_no_refusal_clause():
    """Prompt builder must NOT contain the old Rule 6 refusal clause that blocked correct answers."""
    import app.reasoning.reasoning_engine as re_mod

    source = inspect.getsource(re_mod)
    # The old problematic clause that caused false refusals
    bad_phrases = [
        "if the relevant period chunk is missing, say so",
        "if chunk is missing, refuse",
    ]
    for phrase in bad_phrases:
        assert phrase.lower() not in source.lower(), (
            f"Old Rule 6 refusal clause '{phrase}' still present in reasoning_engine — "
            "LLM will incorrectly refuse when the correct figure is in a mixed-content chunk"
        )


def test_prompt_builder_importable():
    """reasoning_engine must be importable without errors."""
    try:
        from app.reasoning.reasoning_engine import ReasoningEngine
        assert ReasoningEngine is not None
    except ImportError as e:
        pytest.fail(f"reasoning_engine import failed: {e}")


def test_reasoning_engine_has_generate_or_reason_method():
    """ReasoningEngine must have a method for generating reasoned answers."""
    from app.reasoning.reasoning_engine import ReasoningEngine

    methods = [m for m in dir(ReasoningEngine)
               if not m.startswith("__")
               and any(w in m.lower() for w in ("reason", "generate", "answer", "prompt", "build", "run", "invoke"))]
    assert methods, (
        f"No reasoning/generation method found in ReasoningEngine. "
        f"Available methods: {[m for m in dir(ReasoningEngine) if not m.startswith('__')]}"
    )
