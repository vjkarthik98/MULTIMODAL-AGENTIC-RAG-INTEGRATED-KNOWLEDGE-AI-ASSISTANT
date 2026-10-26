"""/internal/eval/* — the serving app lending its models to the eval container.

What must hold (app/api/eval_internal.py's four gates, plus the contract the
eval side depends on):
  * off unless EVAL_INTERNAL_API_ENABLED, and then 404 — not discoverable;
  * 404 for anything that came through Caddy (X-Forwarded-For present);
  * 404 for any tenant other than EVAL_USER_ID, tenant taken from the token;
  * contexts are scoped exactly like generation_runner._full_contexts;
  * /embed returns exactly one entry per input, null on a failed encode.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth.models import UserPublic, UserRole

EVAL_USER = "eval-tenant-1"


def _user(user_id: str) -> UserPublic:
    return UserPublic(
        user_id=user_id,
        email=f"{user_id}@example.com",
        role=UserRole.USER,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )


def _client(user_id: str = EVAL_USER) -> TestClient:
    from app.api import eval_internal
    from app.auth.dependencies import get_current_user

    app = FastAPI()
    app.include_router(eval_internal.router)
    app.dependency_overrides[get_current_user] = lambda: _user(user_id)
    return TestClient(app)


@pytest.fixture
def retriever():
    r = MagicMock()
    r.search.return_value = [{"text": "full chunk one"}, {"text": "   "}, {"text": "chunk two"}]
    with patch("app.api.eval_internal._get_retriever", return_value=r):
        yield r


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "EVAL_INTERNAL_API_ENABLED", True, raising=False)
    monkeypatch.setenv("EVAL_USER_ID", EVAL_USER)


def test_disabled_flag_hides_route(monkeypatch, retriever):
    from app.core.config import settings

    monkeypatch.setattr(settings, "EVAL_INTERNAL_API_ENABLED", False, raising=False)
    r = _client().post("/internal/eval/contexts", json={"query": "q", "session_id": "s"})
    assert r.status_code == 404
    retriever.search.assert_not_called()


def test_request_through_caddy_is_refused(retriever):
    r = _client().post(
        "/internal/eval/contexts",
        json={"query": "q", "session_id": "s"},
        headers={"X-Forwarded-For": "203.0.113.9"},
    )
    assert r.status_code == 404
    retriever.search.assert_not_called()


def test_other_tenant_is_refused(retriever):
    r = _client(user_id="someone-else").post(
        "/internal/eval/contexts", json={"query": "q", "session_id": "s"}
    )
    assert r.status_code == 404
    retriever.search.assert_not_called()


def test_unset_eval_user_refuses_everyone(monkeypatch, retriever):
    monkeypatch.delenv("EVAL_USER_ID", raising=False)
    r = _client().post("/internal/eval/contexts", json={"query": "q", "session_id": "s"})
    assert r.status_code == 404


def test_contexts_scoped_like_full_contexts(retriever):
    r = _client().post(
        "/internal/eval/contexts",
        json={
            "query": "What was Q4 revenue?",
            "session_id": "eval_ragas_pdf-1",
            "sources": ["apple_10k.pdf"],
            # Body may not pick the tenant: an extra user_id is ignored.
            "user_id": "attacker",
        },
    )
    assert r.status_code == 200
    # Blank chunks dropped, same as _full_contexts.
    assert r.json() == {"contexts": ["full chunk one", "chunk two"]}
    retriever.search.assert_called_once_with(
        query="What was Q4 revenue?",
        session_id="eval_ragas_pdf-1",
        top_k=8,
        user_id=EVAL_USER,
        filters={"sources": ["apple_10k.pdf"]},
    )


def test_contexts_top_k_is_bounded(retriever):
    r = _client().post(
        "/internal/eval/contexts", json={"query": "q", "session_id": "s", "top_k": 500}
    )
    assert r.status_code == 422


def test_retrieval_failure_is_503_not_empty(retriever):
    retriever.search.side_effect = RuntimeError("qdrant down")
    r = _client().post("/internal/eval/contexts", json={"query": "q", "session_id": "s"})
    # An empty 200 would silently grade against the 200-char snippets instead.
    assert r.status_code == 503


def test_embed_is_one_to_one_with_null_on_failure():
    embedder = MagicMock()
    embedder.expected_dim = 3

    def _embed(text):
        if text == "bad":
            raise ValueError("cannot encode")
        return [0.1, 0.2, 0.3]

    embedder.embed_text.side_effect = _embed
    with patch("app.core.model_loader.model_loader.get_embedder", return_value=embedder):
        r = _client().post("/internal/eval/embed", json={"texts": ["good", "bad", ""]})
    assert r.status_code == 200
    body = r.json()
    assert body["dim"] == 3
    assert body["vectors"] == [[0.1, 0.2, 0.3], None, [0.1, 0.2, 0.3]]
    # Empty input embeds the placeholder, never a zero vector.
    assert embedder.embed_text.call_args_list[-1].args[0] == "(empty)"


def test_embed_batch_is_bounded():
    r = _client().post("/internal/eval/embed", json={"texts": ["x"] * 65})
    assert r.status_code == 422
