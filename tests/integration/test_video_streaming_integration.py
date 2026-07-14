"""Integration tests for the VIDEO streaming answer path end-to-end:
router -> RAGPipeline.stream() -> citation builder, against the real running
API server and the real (already-ingested) video data. Skipped automatically
when the server isn't up or the known test account has no video data ingested
— so these never hard-fail a plain `pytest tests/` run on a fresh checkout.

These check STRUCTURAL correctness only (the pipeline runs without raising,
returns a non-empty answer, citations have the expected shape, and the router
fix keeps a "beat analyst estimates"-style query grounded in the ingested
video rather than being pushed to pure web search) — never answer-content
accuracy against a gold string. That scoring is out of scope here; see
docs/VIDEO_MODALITY_ACCURACY_REPORT.md for the (separately maintained)
accuracy evaluation harness.

Start the server and have the benchmark video ingested first to run these:
    bash start_server.sh
    pytest tests/integration/test_video_streaming_integration.py -v
"""

import json
import socket
import uuid

import pytest

from app.core.config import settings

_TEST_USER_ID = "36055d60-9099-4f51-81d2-08fe33916356"


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _has_video_data(user_id: str) -> bool:
    try:
        from app.vectorstore.qdrant_store import QdrantVectorStore
        from qdrant_client.http import models as qm
        store = QdrantVectorStore()
        pts, _ = store.client.scroll(
            collection_name="text_collection",
            scroll_filter=qm.Filter(must=[
                qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
                qm.FieldCondition(key="modality", match=qm.MatchValue(value="mp4")),
            ]),
            limit=1, with_payload=False,
        )
        return len(pts) > 0
    except Exception:
        return False


_API_UP = _port_open("127.0.0.1", 8000)
_HAS_DATA = _API_UP and _has_video_data(_TEST_USER_ID)

pytestmark = pytest.mark.skipif(
    not _HAS_DATA,
    reason="API server not up on :8000 or no video data ingested for the known test account",
)


def _mint_token() -> str:
    from app.auth.jwt_handler import issue_tokens
    toks = issue_tokens(_TEST_USER_ID, "video-integration-test@local", "user")
    return toks["access_token"] if isinstance(toks, dict) else toks


def _stream_query(query: str) -> tuple:
    """POST to /rag/query/stream, parse SSE. Returns (answer_text, sources_list)."""
    import requests
    token = _mint_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {
        "query": query,
        "session_id": f"pytest_{uuid.uuid4().hex[:8]}",
        "user_id": _TEST_USER_ID,
        "no_cache": True,
    }
    answer_parts = []
    sources = []
    with requests.post(
        "http://127.0.0.1:8000/rag/query/stream", json=body, headers=headers,
        stream=True, timeout=180,
    ) as r:
        assert r.status_code == 200
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, str):
                answer_parts.append(obj)
            elif isinstance(obj, dict) and obj.get("__type__") == "sources":
                sources = obj.get("data") or []
    return "".join(answer_parts), sources


class TestVideoStreamingStructural:

    def test_video_question_returns_nonempty_answer(self):
        answer, _ = _stream_query("What was Apple's full-year FY2025 annual revenue?")
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0

    def test_video_question_cites_the_ingested_video(self):
        answer, sources = _stream_query("What was Apple's full-year FY2025 annual revenue?")
        assert len(sources) > 0
        modalities = {s.get("modality") for s in sources if isinstance(s, dict)}
        assert "mp4" in modalities or "video" in modalities

    def test_frame_citation_present_with_expected_fields(self):
        _, sources = _stream_query("What was Apple's full-year FY2025 annual revenue?")
        frame_sources = [s for s in sources if isinstance(s, dict) and s.get("is_frame")]
        assert len(frame_sources) > 0
        frame = frame_sources[0]
        assert "frame_timestamp" in frame
        assert "frame_label" in frame

    def test_speaker_citation_present_with_expected_fields(self):
        _, sources = _stream_query("What was Apple's full-year FY2025 annual revenue?")
        spoken_sources = [s for s in sources if isinstance(s, dict) and not s.get("is_frame")]
        assert len(spoken_sources) > 0
        assert "speaker_role" in spoken_sources[0]
        assert "timestamp_start" in spoken_sources[0]

    def test_beat_analyst_estimates_query_stays_grounded_in_video(self):
        # Regression guard for the router fix: this phrasing used to force
        # hybrid/web (triggered by the "analyst" keyword) before ever
        # reaching the ingested earnings-call video. If the router regresses,
        # this assertion (no mp4 source at all) is what would catch it.
        answer, sources = _stream_query(
            "What was Apple's Q4 FY2025 revenue, EPS, and year-over-year "
            "revenue growth, and did these results beat analyst estimates?"
        )
        assert len(answer.strip()) > 0
        modalities = {s.get("modality") for s in sources if isinstance(s, dict)}
        assert "mp4" in modalities or "video" in modalities

    def test_no_broken_asset_path_leaks_to_client(self):
        # asset_path pointing at a file that no longer exists on disk (the
        # ephemeral temp_frames dir is swept on restart) must be nulled, not
        # handed to the client as a dead image URL.
        _, sources = _stream_query("What was Apple's full-year FY2025 annual revenue?")
        for s in sources:
            if isinstance(s, dict) and s.get("asset_path"):
                import os
                assert os.path.exists(s["asset_path"])
