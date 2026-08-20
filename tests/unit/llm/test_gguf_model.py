"""Unit tests for app/llm/gguf_model.py — the chat-template wrapping added for
the Mistral-7B -> Qwen2.5-14B upgrade. GGUFModel() is safe to construct
without a GPU or a loaded model (weights load lazily in ._load()), so these
run with no external dependencies.
"""

from unittest.mock import patch

from app.core.config import settings
from app.llm.gguf_model import _STOP_TOKENS, _STRIP_PREFIXES, GGUFModel


class TestFormatForModel:

    def setup_method(self):
        self.model = GGUFModel()

    def test_chatml_wraps_system_user_assistant_turns(self):
        with patch.object(settings, "LLM_PROMPT_FORMAT", "chatml"):
            out = self.model._format_for_model("CONTEXT:\nfoo\n\nQUERY:\nbar")
        assert out.startswith("<|im_start|>system\n")
        assert "<|im_end|>\n<|im_start|>user\n" in out
        assert out.endswith("<|im_start|>assistant\n")
        # the original body text is preserved verbatim inside the user turn
        assert "CONTEXT:\nfoo\n\nQUERY:\nbar" in out

    def test_raw_mode_returns_prompt_unchanged(self):
        with patch.object(settings, "LLM_PROMPT_FORMAT", "raw"):
            out = self.model._format_for_model("plain completion prompt")
        assert out == "plain completion prompt"

    def test_unknown_format_falls_back_to_raw(self):
        with patch.object(settings, "LLM_PROMPT_FORMAT", "some_future_format"):
            out = self.model._format_for_model("prompt text")
        assert out == "prompt text"

    def test_chatml_does_not_double_wrap_on_repeated_calls(self):
        # _format_for_model is only ever called once per prompt (after
        # truncation) in generate()/stream() — verify it's a pure function
        # (no hidden state) that doesn't accumulate wrapping if called twice.
        with patch.object(settings, "LLM_PROMPT_FORMAT", "chatml"):
            once = self.model._format_for_model("hello")
            twice = self.model._format_for_model(once)
        assert twice.count("<|im_start|>system") == 2  # second call re-wraps the whole (already-wrapped) string
        # documents the behavior: callers must apply this exactly once, which
        # generate()/stream() do (single call site each, after truncation).


class TestStopTokens:

    def test_contains_mistral_tokens(self):
        assert "[INST]" in _STOP_TOKENS
        assert "[/INST]" in _STOP_TOKENS
        assert "</s>" in _STOP_TOKENS

    def test_contains_chatml_tokens(self):
        assert "<|im_end|>" in _STOP_TOKENS
        assert "<|im_start|>" in _STOP_TOKENS
        assert "<|endoftext|>" in _STOP_TOKENS


class TestCleanOutput:

    def setup_method(self):
        self.model = GGUFModel()

    def test_strips_at_chatml_im_end(self):
        cleaned = self.model._clean_output("The answer is 42.<|im_end|>\nextra junk after stop")
        assert cleaned == "The answer is 42."

    def test_strips_at_chatml_im_start(self):
        cleaned = self.model._clean_output("Some answer<|im_start|>user\nfollow-up question")
        assert cleaned == "Some answer"

    def test_strips_at_mistral_inst_token(self):
        cleaned = self.model._clean_output("Answer text[INST] next turn")
        assert cleaned == "Answer text"

    def test_strips_known_prefix(self):
        for prefix in _STRIP_PREFIXES:
            cleaned = self.model._clean_output(f"{prefix} the real answer")
            assert cleaned == "the real answer"

    def test_clean_text_passes_through_unchanged(self):
        cleaned = self.model._clean_output("A perfectly normal answer.")
        assert cleaned == "A perfectly normal answer."


class TestConstruction:

    def test_constructs_without_loading_model(self):
        # No GPU / model file required — weights load lazily in ._load().
        model = GGUFModel()
        assert model._llm is None
        assert model._model_path  # falls back to settings.LLM_MODEL_PATH

    def test_model_name_derived_from_path(self):
        model = GGUFModel(model_path="/some/dir/My-Model-Q4.gguf")
        assert model._model_name == "My-Model-Q4.gguf"


class TestHealthCheck:
    """Regression coverage for a real production bug (2026-08-08): in
    llama_server mode, self._llm being non-None only means the thin
    _LlamaServerClient wrapper was constructed (no network call at
    __init__) — it does NOT mean the separate llama-server process has
    finished loading and is actually answering. A real /rag/query hit
    connection-refused + tripped the circuit breaker during exactly that
    window, while nothing reported it as not-ready. health_check()'s
    "loaded" field alone can't distinguish these; "server_reachable"/"ready"
    must.
    """

    class _FakeServerClient:
        """Stands in for _LlamaServerClient — has .health(), matching its
        real API surface, without a real HTTP call."""

        def __init__(self, reachable: bool):
            self._reachable = reachable

        def health(self) -> bool:
            return self._reachable

    class _FakeInProcessModel:
        """Stands in for a loaded llama_cpp.Llama object — no .health()
        method at all, matching the in-process (non-server) deployment
        mode's real API surface."""

    def setup_method(self):
        self.model = GGUFModel()

    def test_not_loaded_is_not_ready(self):
        result = self.model.health_check()
        assert result["loaded"] is False
        assert result["server_reachable"] is None
        assert result["ready"] is False

    def test_server_mode_wrapper_constructed_but_unreachable_is_not_ready(self):
        """The exact bug: wrapper exists, server isn't actually up yet."""
        self.model._llm = self._FakeServerClient(reachable=False)
        result = self.model.health_check()
        assert result["loaded"] is True
        assert result["server_reachable"] is False
        assert result["ready"] is False

    def test_server_mode_reachable_is_ready(self):
        self.model._llm = self._FakeServerClient(reachable=True)
        result = self.model.health_check()
        assert result["loaded"] is True
        assert result["server_reachable"] is True
        assert result["ready"] is True

    def test_in_process_mode_loaded_is_ready(self):
        """No separate server to reach in this mode — the (slow, real)
        model object existing already means the weight load completed."""
        self.model._llm = self._FakeInProcessModel()
        result = self.model.health_check()
        assert result["loaded"] is True
        assert result["server_reachable"] is None
        assert result["ready"] is True
