"""Unit tests for ModelLoader.health_check()'s llm_ready field.

Regression coverage for a real production bug (2026-08-08): "llm" only
reflected whether the GGUFModel wrapper had been constructed, not whether it
was actually usable — in llama_server mode, that gap let /ready report
readiness while llama-server was still loading, and a real /rag/query call
hit connection-refused during that window. llm_ready must reflect
GGUFModel.health_check()'s real "ready" signal, additively (without
changing "llm"'s existing bool meaning — other consumers already read it).

ModelLoader() is safe to construct directly: __init__ only sets up a lock,
a thread pool, and None placeholders — no GPU/network calls happen until a
getter (get_llm(), get_embedder(), ...) is actually called.
"""

from __future__ import annotations

from app.core.model_loader import ModelLoader


class _FakeGGUFModel:
    def __init__(self, ready: bool):
        self._ready = ready

    def health_check(self) -> dict:
        return {"ready": self._ready, "loaded": True}


class _FakeGGUFModelBroken:
    """Simulates health_check() itself raising — must not propagate."""

    def health_check(self) -> dict:
        raise RuntimeError("boom")


class TestModelLoaderLLMReady:

    def test_llm_not_constructed_is_not_ready(self):
        loader = ModelLoader()
        health = loader.health_check()
        assert health["llm"] is False
        assert health["llm_ready"] is False

    def test_llm_constructed_but_not_ready(self):
        loader = ModelLoader()
        loader._llm = _FakeGGUFModel(ready=False)
        health = loader.health_check()
        assert health["llm"] is True
        assert health["llm_ready"] is False

    def test_llm_constructed_and_ready(self):
        loader = ModelLoader()
        loader._llm = _FakeGGUFModel(ready=True)
        health = loader.health_check()
        assert health["llm"] is True
        assert health["llm_ready"] is True

    def test_llm_health_check_raising_reports_not_ready_not_crash(self):
        loader = ModelLoader()
        loader._llm = _FakeGGUFModelBroken()
        health = loader.health_check()  # must not raise
        assert health["llm"] is True
        assert health["llm_ready"] is False
