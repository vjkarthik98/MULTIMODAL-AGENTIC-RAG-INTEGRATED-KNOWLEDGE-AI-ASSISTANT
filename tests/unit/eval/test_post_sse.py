"""Unit tests for app/eval/http_client.py::post_sse — the SSE client added
for the hallucination-reduction initiative (Phase 2, 2026-08-13) so the eval
harness can exercise /rag/query/stream (the endpoint the UI actually calls),
not just /rag/query. Parses the exact __type__ event protocol
app/api/api_routes.py's event_stream() emits.

No live server — requests.post is mocked to return canned SSE lines.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app.eval.http_client import EvalAuth, post_sse


def _mock_response(status_code: int, lines: list[str]):
    resp = MagicMock()
    resp.status_code = status_code
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status = MagicMock()
    resp.headers = {}
    resp.close = MagicMock()
    return resp


class TestPostSSE:
    def test_plain_text_tokens_concatenate_into_answer(self):
        lines = [
            f"data: {json.dumps('Apple ')}",
            f"data: {json.dumps('reported ')}",
            f"data: {json.dumps('$383,285 million.')}",
            "data: [DONE]",
        ]
        with patch("requests.post", return_value=_mock_response(200, lines)):
            result = post_sse("http://x/rag/query/stream", {}, EvalAuth("u"))
        assert result["answer"] == "Apple reported $383,285 million."
        assert result["refused"] is False

    def test_sources_event_captured(self):
        sources = [{"filename": "apple_10k.pdf", "page": 12}]
        lines = [
            f"data: {json.dumps('Answer text.')}",
            f'data: {{"__type__":"sources","data":{json.dumps(sources)}}}',
            "data: [DONE]",
        ]
        with patch("requests.post", return_value=_mock_response(200, lines)):
            result = post_sse("http://x/rag/query/stream", {}, EvalAuth("u"))
        assert result["sources"] == sources
        assert result["answer"] == "Answer text."

    def test_replace_event_supersedes_streamed_tokens(self):
        lines = [
            f"data: {json.dumps('draft answer')}",
            f'data: {{"__type__":"replace","data":{json.dumps("canonical guarded answer")}}}',
            "data: [DONE]",
        ]
        with patch("requests.post", return_value=_mock_response(200, lines)):
            result = post_sse("http://x/rag/query/stream", {}, EvalAuth("u"))
        assert result["answer"] == "canonical guarded answer"

    def test_refusal_event_empties_answer(self):
        lines = [
            f"data: {json.dumps('some tokens')}",
            'data: {"__type__":"refusal"}',
            "data: [DONE]",
        ]
        with patch("requests.post", return_value=_mock_response(200, lines)):
            result = post_sse("http://x/rag/query/stream", {}, EvalAuth("u"))
        assert result["answer"] == ""
        assert result["refused"] is True

    def test_non_data_lines_ignored(self):
        lines = [
            ": keep-alive comment",
            "",
            f"data: {json.dumps('hello')}",
            "data: [DONE]",
        ]
        with patch("requests.post", return_value=_mock_response(200, lines)):
            result = post_sse("http://x/rag/query/stream", {}, EvalAuth("u"))
        assert result["answer"] == "hello"

    def test_401_retries_once_with_refreshed_token(self):
        auth = EvalAuth("u")
        auth._can_mint = False  # force static-token path, deterministic headers
        auth._static_token = "stale"

        calls = {"n": 0}

        def _fake_post(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _mock_response(401, [])
            return _mock_response(200, [f"data: {json.dumps('ok')}", "data: [DONE]"])

        with patch("requests.post", side_effect=_fake_post):
            result = post_sse("http://x/rag/query/stream", {}, auth)
        assert result["answer"] == "ok"
        assert calls["n"] == 2
