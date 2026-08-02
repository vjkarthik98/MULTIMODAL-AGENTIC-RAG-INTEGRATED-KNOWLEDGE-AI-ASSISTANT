"""Shared fixtures for tests/api_contract/.

Schemathesis drives the live FastAPI server over HTTP — never `TestClient(app)`
in-process. tests/conftest.py already documents why: WARMUP_AT_STARTUP defaults
true, so firing the real lifespan kicks off unstoppable GPU model loading on a
background thread that outlasts pytest's own timeout. The same reachability-
probe-then-skip pattern as tests/integration/conftest.py::requires_llama_server
is reused here so a dev machine or CI job with no server running gets a clean
skip instead of a connection-refused failure.

BASE_URL defaults to the local docker-compose `api` service (see
docker-compose.yml). Never defaults to the real deployed URL — pointing this at
production is an explicit, opt-in action (`MAGIK_API_BASE_URL=https://... pytest
tests/api_contract/`), not something a bare `pytest` invocation should ever do
by accident, since a live run also wakes the wake-on-demand AWS box.
"""

from __future__ import annotations

import os

import httpx
import pytest

BASE_URL = os.getenv("MAGIK_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _server_reachable(base_url: str, timeout: float = 3.0) -> bool:
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{base_url}/health")
            return resp.status_code == 200
    except Exception:
        return False


SERVER_UP = _server_reachable(BASE_URL)

requires_api_server = pytest.mark.skipif(
    not SERVER_UP,
    reason=(
        f"MAGIK API not reachable at {BASE_URL}/health — start it first "
        f"(`docker compose up -d api qdrant redis mongo` for local mode, or set "
        f"MAGIK_API_BASE_URL for a deliberate live-mode run)."
    ),
)
