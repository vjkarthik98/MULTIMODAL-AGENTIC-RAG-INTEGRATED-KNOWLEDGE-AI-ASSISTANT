"""EVAL_REMOTE_MODELS=1 — the eval borrows the app's retriever + encoder.

Pins that the switch actually reroutes both call sites (so the eval container
never loads BGE), and that the remote stand-ins keep the contracts their
callers rely on.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.eval.remote_models import RemoteContextRetriever, RemoteEmbedder


def test_full_context_retriever_goes_remote(monkeypatch):
    from app.eval.runners import generation_runner

    monkeypatch.setenv("EVAL_REMOTE_MODELS", "1")
    with patch("app.core.model_loader.model_loader.get_embedder") as local:
        r = generation_runner._make_full_context_retriever()
    assert isinstance(r, RemoteContextRetriever)
    local.assert_not_called()


def test_remote_contexts_feed_full_contexts_unchanged(monkeypatch):
    from app.eval.runners.generation_runner import _full_contexts

    auth = MagicMock()
    r = RemoteContextRetriever(server_url="http://magik-current:8000", auth=auth)
    with patch(
        "app.eval.remote_models.post_json", return_value={"contexts": ["a", "b"]}
    ) as post:
        out = _full_contexts(r, "q", "eval-user", "sess", sources=["f.pdf"])
    assert out == ["a", "b"]
    url, payload, _auth = post.call_args.args[:3]
    assert url == "http://magik-current:8000/internal/eval/contexts"
    # The tenant is never sent — the server takes it from the token.
    assert payload == {"query": "q", "session_id": "sess", "top_k": 8, "sources": ["f.pdf"]}


def test_remote_embedder_returns_vector_and_learns_dim():
    e = RemoteEmbedder(server_url="http://x", auth=MagicMock())
    with patch(
        "app.eval.remote_models.post_json", return_value={"vectors": [[1.0, 2.0]], "dim": 2}
    ):
        assert e.embed_text("hello") == [1.0, 2.0]
    assert e.expected_dim == 2


def test_remote_embedder_null_vector_raises():
    """So _MagikRagasEmbeddings._embed_one's own fallback decides, exactly as
    for a local encoder failure — never a silently shifted batch."""
    e = RemoteEmbedder(server_url="http://x", auth=MagicMock())
    with patch("app.eval.remote_models.post_json", return_value={"vectors": [None], "dim": 2}):
        with pytest.raises(RuntimeError):
            e.embed_text("bad")


def test_ragas_embeddings_use_remote_when_enabled(monkeypatch):
    pytest.importorskip("ragas")
    from app.eval.metrics.generation import _build_ragas_embeddings

    monkeypatch.setenv("EVAL_REMOTE_MODELS", "1")
    with patch("app.core.model_loader.model_loader.get_embedder") as local:
        emb = _build_ragas_embeddings()
    local.assert_not_called()
    assert isinstance(emb._embedder, RemoteEmbedder)
