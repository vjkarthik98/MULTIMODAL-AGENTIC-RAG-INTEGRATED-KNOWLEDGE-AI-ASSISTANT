"""Smoke tests for the llama-server LLM backend (Mistral-7B -> Qwen2.5-14B
upgrade). These exercise the REAL running server — skipped automatically when
it isn't up, so they never block a plain `pytest tests/` run on a machine
without the model loaded. Structural checks only (server responds, model
generates SOME non-empty text) — no answer-content/accuracy scoring.

Start the server first to actually run these:
    python start_server.py
    pytest tests/integration/test_llm_server_smoke.py -v
"""

import socket

import pytest

from app.core.config import settings


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


_LLAMA_SERVER_UP = _port_open(settings.LLM_SERVER_HOST, settings.LLM_SERVER_PORT)

pytestmark = pytest.mark.skipif(
    not _LLAMA_SERVER_UP,
    reason=f"llama-server not reachable on {settings.LLM_SERVER_HOST}:{settings.LLM_SERVER_PORT}",
)


class TestLlamaServerUp:

    def test_v1_models_reports_configured_gguf(self):
        import requests
        r = requests.get(
            f"http://{settings.LLM_SERVER_HOST}:{settings.LLM_SERVER_PORT}/v1/models",
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        model_ids = [m["id"] for m in data.get("data", [])]
        assert any(settings.LLM_MODEL_PATH in mid or mid.endswith(".gguf") for mid in model_ids)

    def test_raw_completion_endpoint_responds(self):
        import requests
        r = requests.post(
            f"http://{settings.LLM_SERVER_HOST}:{settings.LLM_SERVER_PORT}/v1/completions",
            json={"prompt": "The capital of France is", "max_tokens": 5, "temperature": 0.0},
            timeout=30,
        )
        assert r.status_code == 200
        text = r.json()["choices"][0]["text"]
        assert isinstance(text, str)
        assert len(text.strip()) > 0


class TestGGUFModelEndToEnd:
    """Exercises the actual client class the rest of the app calls through —
    not just raw HTTP — so a regression in _format_for_model/_clean_output/
    stop-token handling would show up here even if the raw endpoint is fine.
    """

    def setup_method(self):
        from app.llm.gguf_model import GGUFModel
        self.model = GGUFModel()

    def test_generate_returns_nonempty_grounded_answer(self):
        answer = self.model.generate(
            "CONTEXT:\nApple reported Q4 FY2025 revenue of $102.5 billion.\n\n"
            "QUERY:\nWhat was Apple's Q4 FY2025 revenue?\n\n"
            "Answer in one sentence using only the context above.",
            max_tokens=60,
            temperature=0.0,
        )
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0
        # no leaked chat-template control tokens in the cleaned output
        for tok in ("<|im_start|>", "<|im_end|>", "[INST]", "[/INST]"):
            assert tok not in answer

    def test_stream_yields_chunks_that_join_to_nonempty_text(self):
        chunks = list(self.model.stream(
            "CONTEXT:\nApple's Services segment grew 15% year-over-year.\n\n"
            "QUERY:\nHow much did Services grow?\n\n"
            "Answer in one sentence.",
            max_tokens=60,
            temperature=0.0,
        ))
        assert len(chunks) > 0
        joined = "".join(chunks)
        assert len(joined.strip()) > 0
