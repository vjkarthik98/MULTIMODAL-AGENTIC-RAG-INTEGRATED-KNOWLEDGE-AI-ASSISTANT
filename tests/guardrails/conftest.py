"""Shared fixtures for Phase 26 guardrail tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import pytest


# ---------------------------------------------------------------------------
# JSONL corpus loader
# ---------------------------------------------------------------------------

CORPUS_PATH = Path(__file__).parent / "adversarial" / "red_team_prompts.jsonl"


def load_corpus() -> List[Dict[str, Any]]:
    """Load all cases from the red-team JSONL corpus."""
    cases = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return cases


def corpus_by_action(action: str) -> List[Dict[str, Any]]:
    """Return all corpus entries where expected_action == action."""
    return [c for c in load_corpus() if c.get("expected_action") == action]


def corpus_by_tag(tag: str) -> List[Dict[str, Any]]:
    """Return all corpus entries that include the given tag."""
    return [c for c in load_corpus() if tag in c.get("tags", [])]


# ---------------------------------------------------------------------------
# Parametrize helpers — used by individual test modules via indirect=True
# ---------------------------------------------------------------------------

@pytest.fixture
def block_cases() -> List[Dict[str, Any]]:
    return corpus_by_action("block")


@pytest.fixture
def allow_cases() -> List[Dict[str, Any]]:
    return corpus_by_action("allow")


@pytest.fixture
def injection_block_cases() -> List[Dict[str, Any]]:
    return [c for c in corpus_by_action("block") if "injection" in c.get("tags", [])]


@pytest.fixture
def jailbreak_block_cases() -> List[Dict[str, Any]]:
    return [c for c in corpus_by_action("block") if "jailbreak" in c.get("tags", [])]


@pytest.fixture
def ssrf_block_cases() -> List[Dict[str, Any]]:
    return [c for c in corpus_by_action("block") if "ssrf" in c.get("tags", [])]


@pytest.fixture
def encoding_block_cases() -> List[Dict[str, Any]]:
    return [c for c in corpus_by_action("block") if "encoding" in c.get("tags", [])]


@pytest.fixture
def benign_cases() -> List[Dict[str, Any]]:
    return corpus_by_tag("benign")


# ---------------------------------------------------------------------------
# Session-level setup: force guardrail policy to load without Redis/LLM
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _patch_audit(monkeypatch_session):
    """Make audit_decision a no-op to avoid Redis writes during tests."""
    import app.guardrails.audit as _audit
    monkeypatch_session.setattr(_audit, "audit_decision", lambda **_kw: None)


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch (pytest default is function-scoped)."""
    with pytest.MonkeyPatch.context() as mp:
        yield mp
