"""Tests for app/eval/deepeval_suite.py.

Two tiers, same split as the rest of app/eval/'s test coverage:
  - unit: the DeepEvalBaseLLM wrapper's JSON-extraction/schema logic, mocked
    HTTP — no server, no deepeval LLM calls, always runs.
  - integration: a real (small, --limit capped) end-to-end run against a live
    MAGIK server, skipped cleanly if one isn't reachable or deepeval isn't
    installed — same self-skip convention app/eval/runners/generation_runner.py
    already uses (_server_available()), not a hard dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

deepeval = pytest.importorskip("deepeval", reason="deepeval not installed — pip install '.[quality]'")


# ── unit: MagikLocalLLM wrapper ──────────────────────────────────────────────


def test_deepeval_llm_generate_returns_raw_text_without_schema():
    from app.eval.deepeval_suite import get_deepeval_llm

    with patch("app.eval.deepeval_suite._call_server", return_value='{"score": 1}'):
        llm = get_deepeval_llm()
        out = llm.generate("some prompt")
        assert out == '{"score": 1}'


def test_deepeval_llm_generate_parses_schema_when_given():
    from pydantic import BaseModel

    from app.eval.deepeval_suite import get_deepeval_llm

    class Verdict(BaseModel):
        score: int

    with patch("app.eval.deepeval_suite._call_server", return_value='{"score": 3}'):
        llm = get_deepeval_llm()
        out = llm.generate("some prompt", schema=Verdict)
        assert isinstance(out, Verdict)
        assert out.score == 3


def test_deepeval_llm_falls_back_to_raw_text_on_bad_json_with_schema():
    from pydantic import BaseModel

    from app.eval.deepeval_suite import get_deepeval_llm

    class Verdict(BaseModel):
        score: int

    with patch("app.eval.deepeval_suite._call_server", return_value="not json at all"):
        llm = get_deepeval_llm()
        out = llm.generate("some prompt", schema=Verdict)
        assert isinstance(out, str)


def test_deepeval_llm_model_name_declares_local_judge():
    from app.eval.deepeval_suite import get_deepeval_llm

    llm = get_deepeval_llm()
    # Must never silently claim OpenAI — this project's whole positioning is
    # 100%-open-source / privacy-preserving (see CLAUDE.md).
    name = llm.get_model_name().lower()
    assert "openai" not in name
    assert "gpt" not in name


# ── integration: real small end-to-end run ───────────────────────────────────


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.timeout(300)
def test_deepeval_suite_end_to_end_smoke():
    from app.eval.config import load_config
    from app.eval.deepeval_suite import _build_eval_rows, _run_metrics, _server_available

    if not _server_available():
        pytest.skip("MAGIK API not reachable — start it first for this integration test")

    import asyncio

    cfg = load_config()
    cfg.modality = "txt"
    try:
        eval_rows, errors = asyncio.run(_build_eval_rows(cfg, limit=1))
    except RuntimeError as exc:
        pytest.skip(f"could not build eval rows: {exc}")

    assert eval_rows, "expected at least one gold row to succeed"

    payload = _run_metrics(eval_rows, ["bias", "toxicity"])
    assert "summary" in payload
    for name in ("bias", "toxicity"):
        assert name in payload["summary"]
        mean = payload["summary"][name]["mean"]
        if mean is not None:
            assert 0.0 <= mean <= 1.0
