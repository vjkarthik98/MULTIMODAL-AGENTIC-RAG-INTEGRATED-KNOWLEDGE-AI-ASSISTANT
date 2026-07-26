"""Shared fixtures for tests/integration/.

_LLAMA_SERVER_UP / requires_llama_server: several integration tests call the
real end-to-end query pipeline (app.pipeline.query_pipeline), which loads
real models via app.core.model_registry.ensure_for_query() and — with no
llama-server running — blocks on a future.result(timeout=...) that
eventually raises, but only after the full MODEL_TIMEOUT_SEC wait, and in at
least one observed run outlasted pytest-timeout's own kill mechanism
entirely (a real hang, not just a slow test). test_llm_server_smoke.py
already had the right pattern (skip cleanly instead of blocking); this
centralizes it so every file that needs a live server can reuse one check
instead of five copies of the same socket probe.
"""
from __future__ import annotations

import socket

import pytest

from app.core.config import settings


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


LLAMA_SERVER_UP = _port_open(settings.LLM_SERVER_HOST, settings.LLM_SERVER_PORT)

requires_llama_server = pytest.mark.skipif(
    not LLAMA_SERVER_UP,
    reason=f"llama-server not reachable on {settings.LLM_SERVER_HOST}:{settings.LLM_SERVER_PORT}",
)
