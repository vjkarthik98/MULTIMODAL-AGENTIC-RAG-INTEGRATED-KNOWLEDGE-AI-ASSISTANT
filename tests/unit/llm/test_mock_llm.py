import pytest

from app.llm.mock_llm import MockLLM


class TestMockLLMGenerate:

    def setup_method(self):
        self.llm = MockLLM()

    def test_returns_string(self):
        result = self.llm.generate("What is the capital of France?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_name_keyword_canned_response(self):
        result = self.llm.generate("What is the name mentioned?")
        assert "Answer:" in result

    def test_summarize_keyword_canned_response(self):
        result = self.llm.generate("Please summarize this document.")
        assert "Answer:" in result

    def test_what_keyword_canned_response(self):
        result = self.llm.generate("What is transformer architecture?")
        assert "Answer:" in result

    def test_how_keyword_canned_response(self):
        result = self.llm.generate("How does attention mechanism work?")
        assert "Answer:" in result

    def test_explain_keyword_canned_response(self):
        result = self.llm.generate("Please explain the concept.")
        assert "Answer:" in result

    def test_list_keyword_canned_response(self):
        result = self.llm.generate("List all the items in the document.")
        assert "Answer:" in result

    def test_compare_keyword_canned_response(self):
        result = self.llm.generate("Compare the two approaches.")
        assert "Answer:" in result

    def test_default_response_for_no_keyword_match(self):
        result = self.llm.generate("zxqvb nonce token unique string")
        assert "Answer:" in result

    def test_empty_prompt_returns_default(self):
        result = self.llm.generate("")
        assert isinstance(result, str)

    def test_session_id_accepted(self):
        result = self.llm.generate("What is X?", session_id="test-session")
        assert isinstance(result, str)

    def test_max_tokens_accepted(self):
        result = self.llm.generate("Tell me something.", max_tokens=50)
        assert isinstance(result, str)

    def test_temperature_accepted(self):
        result = self.llm.generate("Query.", temperature=0.0)
        assert isinstance(result, str)

    def test_model_name_attribute(self):
        assert self.llm.model_name == "mock-llm"


class TestMockLLMStream:

    def setup_method(self):
        self.llm = MockLLM()

    def test_returns_iterator(self):
        stream = self.llm.stream("What is X?")
        assert hasattr(stream, "__iter__")

    def test_stream_yields_strings(self):
        tokens = list(self.llm.stream("What is X?"))
        assert len(tokens) > 0
        for tok in tokens:
            assert isinstance(tok, str)

    def test_stream_concatenated_is_non_empty(self):
        result = "".join(self.llm.stream("summarize this")).strip()
        assert len(result) > 0

    def test_stream_session_id_accepted(self):
        tokens = list(self.llm.stream("query", session_id="s1"))
        assert len(tokens) > 0


class TestMockLLMHealthCheck:

    def setup_method(self):
        self.llm = MockLLM()

    def test_returns_dict(self):
        result = self.llm.health_check()
        assert isinstance(result, dict)

    def test_model_key_present(self):
        result = self.llm.health_check()
        assert result["model"] == "mock-llm"

    def test_ready_is_true(self):
        result = self.llm.health_check()
        assert result["ready"] is True

    def test_mock_flag_is_true(self):
        result = self.llm.health_check()
        assert result["mock"] is True
