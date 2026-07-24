"""Shared fixtures for the whole tests/unit/ tree.

Autouse guards against real heavy/networked dependencies leaking into "unit"
tests (per pyproject.toml's marker doc: "fast unit tests with no external
dependencies"). Found by actually running the suite, not by inspection —
two independent call sites hit this before it was fixed globally:
tests/unit/agents/test_agent_controller.py (via agent_controller._guard_output)
and tests/unit/pipeline/test_rag_pipeline.py (via rag_pipeline._output_guard_check
-> output_guard.check -> _check_toxicity -> _get_detoxify -> Detoxify("original")
-> torch.hub downloads a model checkpoint over the network on first use).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_real_detoxify_download():
    """Force output_guard._get_detoxify() to short-circuit to None (its own
    documented "Detoxify unavailable" path — see the except ImportError /
    except Exception branches in app/guardrails/output_guard.py) instead of
    attempting a real network download. _check_toxicity() already falls back
    to the real, fast, regex-based _HARMFUL_CONTENT_PATTERNS check when the
    model is None, so this keeps real guardrail logic under test — it only
    removes the network-dependent ML layer, not toxicity checking itself.
    """
    with patch("app.guardrails.output_guard._get_detoxify", return_value=None):
        yield
