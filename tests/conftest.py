"""Repo-wide pytest configuration.

Auto-applies the pyproject.toml markers (unit/integration/guardrails/auth/
multimodal) based on test location, so `pytest -m unit`, `-m guardrails`,
`-m auth` etc. (as documented in CLAUDE.md and run in ci.yml / eval-gate.yml)
actually select the intended tests instead of collecting zero — no individual
test file needs its own `@pytest.mark.*` decorator for these directory-level
categories. Explicit per-test markers (e.g. `pytestmark = pytest.mark.skipif(...)`
already used in tests/integration/) are unaffected; this only *adds* markers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Not a pytest test despite the test_*.py name pytest's default discovery
# matches: tests/video/test_video_modality_e2e.py is a standalone CLI script
# (its own argparse, its own exit-code pass/fail contract — see its own
# docstring: "Run: python tests/video/test_video_modality_e2e.py"). It has
# zero `def test_*` functions, so it never contributed a real test case —
# but pytest still imported it during collection, unconditionally executing
# its top-level argparse parsing and os.chdir() as a side effect on every
# `pytest tests/` run. Excluding it from collection entirely, not renaming
# or restructuring it — it's not ours to rewrite, just to stop pytest
# tripping over it.
collect_ignore_glob = ["video/test_video_modality_e2e.py"]

_DIR_MARKERS = {
    "unit": "unit",
    "guardrails": "guardrails",
    "auth": "auth",
    "integration": "integration",
    "video": "multimodal",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    for item in items:
        rel_parts = Path(item.fspath).relative_to(Path(__file__).parent).parts
        if not rel_parts:
            continue
        top_dir = rel_parts[0]
        marker_name = _DIR_MARKERS.get(top_dir)
        if marker_name:
            item.add_marker(getattr(pytest.mark, marker_name))
