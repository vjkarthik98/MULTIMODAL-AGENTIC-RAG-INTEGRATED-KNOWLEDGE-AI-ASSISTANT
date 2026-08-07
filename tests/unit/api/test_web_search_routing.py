"""Web-mode queries must never be answered from the knowledge base.

The reported symptom: click the web icon, the web answer streams in, it
flickers, and a knowledge-base answer replaces it. Cause was the client's
refusal fallback — it calls POST /rag/query, which accepted `force_web` on
its request model and never read it, so the fallback silently ran the KB
pipeline and its answer overwrote the web one on screen.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.api_routes import (
    _is_web_request,
    _web_failure_message,
    _web_source_payload,
)
from app.auth.models import UserPublic, UserRole

# ── routing decision ──────────────────────────────────────────────────────────

class TestIsWebRequest:

    def test_force_web_alone_is_enough(self):
        assert _is_web_request("what is our gross margin", True) is True

    def test_explicit_phrase_without_the_toggle(self):
        assert _is_web_request("search the web for Apple news", False) is True

    def test_realtime_signal_without_the_toggle(self):
        assert _is_web_request("what is the stock price today", False) is True

    def test_plain_kb_question_is_not_web(self):
        assert _is_web_request("what was total revenue in the 10-K", False) is False

    def test_case_insensitive(self):
        assert _is_web_request("SEARCH ONLINE for the latest filing", False) is True

    def test_empty_query_with_toggle_still_web(self):
        # The toggle is the user's explicit instruction; it does not depend on
        # the wording of the question.
        assert _is_web_request("", True) is True


class TestWebHelpers:

    def test_source_payload_shape_matches_the_kb_contract(self):
        out = _web_source_payload(["https://a.com/x", "https://b.com/y"], ["A title"])
        assert out[0] == {
            "source": "https://a.com/x",
            "modality": "web",
            "title": "A title",
            "page_number": None,
            "start_time": None,
        }
        assert out[1]["title"] == "", "missing title must not shift the pairing"

    def test_source_payload_drops_empty_urls(self):
        assert _web_source_payload(["", None, "https://c.com"], []) == [
            {
                "source": "https://c.com",
                "modality": "web",
                "title": "",
                "page_number": None,
                "start_time": None,
            }
        ]

    def test_failure_message_names_the_reason_and_the_way_out(self):
        msg = _web_failure_message("returned no results")
        assert "returned no results" in msg
        assert "turn off web search" in msg


# ── POST /rag/query with force_web ────────────────────────────────────────────

_USER = UserPublic(
    user_id="u-web-1",
    email="web@example.com",
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


def _post(client, **extra):
    body = {"query": "what is Apple trading at", "session_id": "s-web"}
    body.update(extra)
    return client.post("/rag/query", json=body)


def test_force_web_returns_the_web_answer(client):
    with patch(
        "app.api.api_routes._run_web_search",
        new=AsyncMock(return_value=("Apple is trading at $283.80.", ["https://x.com/a"], ["X"], None)),
    ), patch("app.api.api_routes._get_query_pipeline") as pipeline:
        r = _post(client, force_web=True)

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Apple is trading at $283.80."
    assert body["decision"] == "web"
    assert body["sources"][0]["modality"] == "web"
    # The KB pipeline must not run at all for a web-mode query.
    pipeline.assert_not_called()


def test_force_web_failure_does_not_fall_back_to_the_knowledge_base(client):
    with patch(
        "app.api.api_routes._run_web_search",
        new=AsyncMock(return_value=("", [], [], "returned no results")),
    ), patch("app.api.api_routes._get_query_pipeline") as pipeline:
        r = _post(client, force_web=True)

    assert r.status_code == 200
    body = r.json()
    assert "returned no results" in body["answer"]
    assert "turn off web search" in body["answer"]
    assert body["decision"] == "web_failed"
    assert body["sources"] == []
    pipeline.assert_not_called()


def test_without_force_web_the_knowledge_base_pipeline_still_runs(client):
    # A file scope is now required for any non-web query (see
    # test_file_scope_required.py) — this test's own concern is only that a
    # non-force_web query with a scope still runs the KB pipeline, not web.
    with patch("app.api.api_routes._run_web_search", new=AsyncMock()) as web, patch(
        "app.api.api_routes._get_query_pipeline"
    ) as pipeline:
        pipeline.return_value = lambda *a, **k: {
            "answer": "From your 10-K: total revenue was $391.0B.",
            "confidence": 0.8,
            "sources": [],
        }
        r = _post(client, query="what was total revenue in the 10-K", sources=["10k.pdf"])

    assert r.status_code == 200
    assert "391.0B" in r.json()["answer"]
    web.assert_not_awaited()
