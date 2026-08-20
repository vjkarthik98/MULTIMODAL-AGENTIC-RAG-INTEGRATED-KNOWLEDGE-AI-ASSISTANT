"""A query must be scoped to an explicitly selected file before it runs.

Without this, retrieval falls through to an unscoped whole-KB search and
answers mix content from unrelated documents. `sources` (the UI's @ picker)
is required on both POST /rag/query and POST /rag/query/stream unless the
query is a web-search request (force_web, or — on the stream route only —
one auto-detected by _is_web_request's phrase/real-time heuristics).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth.models import UserPublic, UserRole


def _assembled_sse_text(body: str) -> str:
    """Reassemble the message from a `_stream_chunks`/`_sse` SSE body.

    Canned-message streams (guardrail blocks, gibberish, this scope gate)
    are paced one JSON-encoded character per `data:` line — see
    api_routes.py `_stream_chunks`/`_sse` — so the raw response text never
    contains the message as a contiguous substring. Other event lines (e.g.
    the typed `{"__type__":"sources",...}` event) decode to a dict, not a
    str, and are skipped rather than joined.
    """
    out = []
    for line in body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            piece = json.loads(line[len("data: ") :])
        except json.JSONDecodeError:
            continue
        if isinstance(piece, str):
            out.append(piece)
    return "".join(out)


_USER = UserPublic(
    user_id="u-scope-1",
    email="scope@example.com",
    role=UserRole.USER,
    is_active=True,
    created_at=datetime.now(timezone.utc),
)


@pytest.fixture
def client():
    from app.auth.dependencies import get_current_user
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: _USER
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.pop(get_current_user, None)


def _body(**extra):
    body = {"query": "what was total revenue", "session_id": "s-scope"}
    body.update(extra)
    return body


class TestQueryRouteRequiresScope:

    def test_no_sources_and_no_force_web_is_rejected(self, client):
        r = client.post("/rag/query", json=_body())
        assert r.status_code == 400
        assert "select" in r.json()["detail"].lower()

    def test_empty_sources_list_is_rejected(self, client):
        r = client.post("/rag/query", json=_body(sources=[]))
        assert r.status_code == 400

    def test_force_web_is_exempt(self, client):
        with patch("app.api.api_routes._run_web_search", new=AsyncMock()) as web:
            web.return_value = ("Apple is at $190.", ["https://x.com"], ["title"], None)
            r = client.post("/rag/query", json=_body(force_web=True))
        assert r.status_code == 200

    def test_with_sources_the_kb_pipeline_runs(self, client):
        with patch("app.api.api_routes._get_query_pipeline") as pipeline:
            pipeline.return_value = lambda *a, **k: {
                "answer": "Total revenue was $391.0B.",
                "confidence": 0.8,
                "sources": [],
            }
            r = client.post("/rag/query", json=_body(sources=["10k.pdf"]))
        assert r.status_code == 200
        assert "391.0B" in r.json()["answer"]

    def test_how_are_you_gets_greeting_template_not_400(self, client):
        r = client.post("/rag/query", json=_body(query="How are you?"))
        assert r.status_code == 200
        assert r.json()["decision"] == "greeting"
        assert "select" not in r.json()["answer"].lower()

    def test_meta_capability_question_gets_meta_template_not_400(self, client):
        r = client.post(
            "/rag/query",
            json=_body(query="Do you know about the Financial documents stored in the Knowledge Base?"),
        )
        assert r.status_code == 200
        assert r.json()["decision"] == "meta_capability"

    def test_unrelated_question_without_sources_still_400s(self, client):
        r = client.post("/rag/query", json=_body(query="What is the capital of France?"))
        assert r.status_code == 400


class TestStreamRouteRequiresScope:

    def test_no_sources_and_no_force_web_streams_a_guidance_message(self, client):
        r = client.post("/rag/query/stream", json=_body())
        assert r.status_code == 200  # SSE 200 with an inline message, not a raw 4xx
        assert "select" in _assembled_sse_text(r.text).lower()
        assert "[DONE]" in r.text

    def test_force_web_is_exempt(self, client):
        with patch("app.api.api_routes._run_web_search", new=AsyncMock()) as web:
            web.return_value = ("Apple is at $190.", ["https://x.com"], ["title"], None)
            r = client.post("/rag/query/stream", json=_body(force_web=True))
        assert r.status_code == 200
        assert "select" not in _assembled_sse_text(r.text).lower()

    def test_realtime_heuristic_web_query_is_exempt_even_without_sources(self, client):
        # _is_web_request fires on realtime signals without the force_web
        # toggle — that path must resolve above the scope gate, not be
        # blocked by it.
        with patch("app.api.api_routes._run_web_search", new=AsyncMock()) as web:
            web.return_value = ("It's $190 today.", ["https://x.com"], ["title"], None)
            r = client.post(
                "/rag/query/stream",
                json=_body(query="what is the stock price today"),
            )
        assert r.status_code == 200
        assert "select" not in _assembled_sse_text(r.text).lower()

    def test_how_are_you_streams_greeting_template(self, client):
        r = client.post("/rag/query/stream", json=_body(query="How are you?"))
        assert r.status_code == 200
        text = _assembled_sse_text(r.text).lower()
        assert "select a file to scope" not in text
        assert "@" in text  # points at the file picker

    def test_meta_capability_question_streams_meta_template(self, client):
        r = client.post("/rag/query/stream", json=_body(query="What can you do?"))
        assert r.status_code == 200
        text = _assembled_sse_text(r.text)
        assert "select a file to scope" not in text.lower()

    def test_greeting_no_longer_calls_agent_executor(self, client):
        # Regression guard: the streaming route's greeting branch must be a
        # pure template lookup now, not an LLM call.
        with patch("app.agents.agent_controller.AgentExecutor") as mock_exec:
            r = client.post("/rag/query/stream", json=_body(query="hi"))
        assert r.status_code == 200
        mock_exec.assert_not_called()
