"""Unit tests for GET /ready's readiness() handler in app/main.py.

Regression coverage for a real production bug (2026-08-08): all_ready was
computed from models.get("embedder", False) alone — it never looked at the
LLM at all. That let /ready report "ready" while llama-server was still
loading its checkpoint shards, and a real /rag/query call in that window
hit connection-refused and tripped the LLM's circuit breaker with nothing
having reported the system as not-ready.

readiness() is a plain sync function with no injected dependencies, so it's
callable directly — no TestClient/app lifespan needed.
"""

from __future__ import annotations

from unittest.mock import patch

from app.main import readiness


def _patched_readiness(models: dict, infra_status: dict | None = None):
    infra_status = infra_status if infra_status is not None else {}
    with (
        patch("app.core.model_loader.model_loader.health_check", return_value=models),
        patch("app.core.infra_registry.infra.health_check", return_value=infra_status),
    ):
        return readiness()


class TestReadinessEndpoint:

    def test_not_ready_when_embedder_and_llm_both_missing(self):
        result = _patched_readiness({"embedder": False, "llm_ready": False})
        assert result["status"] == "degraded"

    def test_not_ready_when_embedder_ready_but_llm_not_ready(self):
        """The exact bug: embedder alone used to be enough."""
        result = _patched_readiness({"embedder": True, "llm_ready": False})
        assert result["status"] == "degraded"

    def test_not_ready_when_llm_ready_but_embedder_not_ready(self):
        result = _patched_readiness({"embedder": False, "llm_ready": True})
        assert result["status"] == "degraded"

    def test_ready_when_both_embedder_and_llm_ready(self):
        result = _patched_readiness({"embedder": True, "llm_ready": True})
        assert result["status"] == "ready"

    def test_missing_llm_ready_key_defaults_to_not_ready(self):
        """models dict without llm_ready at all (e.g. a stale/mocked caller)
        must fail closed, not silently pass as ready."""
        result = _patched_readiness({"embedder": True})
        assert result["status"] == "degraded"

    def test_response_includes_models_and_infra(self):
        result = _patched_readiness(
            {"embedder": True, "llm_ready": True, "llm": True},
            infra_status={"qdrant": True},
        )
        assert result["models"] == {"embedder": True, "llm_ready": True, "llm": True}
        assert result["infra"] == {"qdrant": True}
