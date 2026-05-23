import pytest

from app.core.exceptions import (
    AgentError,
    IngestionError,
    LLMError,
    MemoryError,
    RAGError,
    RetrievalError,
    ValidationError,
)


class TestRAGError:

    def test_is_exception(self):
        assert issubclass(RAGError, Exception)

    def test_message_stored(self):
        e = RAGError("something went wrong")
        assert str(e) == "something went wrong"

    def test_default_code(self):
        e = RAGError("msg")
        assert e.code == "RAG_ERROR"

    def test_custom_code(self):
        e = RAGError("msg", code="CUSTOM_CODE")
        assert e.code == "CUSTOM_CODE"

    def test_default_context_empty_dict(self):
        e = RAGError("msg")
        assert e.context == {}

    def test_custom_context(self):
        ctx = {"file": "test.txt", "line": 42}
        e = RAGError("msg", context=ctx)
        assert e.context == ctx

    def test_can_be_raised_and_caught(self):
        with pytest.raises(RAGError, match="test error"):
            raise RAGError("test error")

    def test_none_context_becomes_empty_dict(self):
        e = RAGError("msg", context=None)
        assert e.context == {}


class TestSubclasses:

    def test_ingestion_error_is_rag_error(self):
        assert issubclass(IngestionError, RAGError)

    def test_retrieval_error_is_rag_error(self):
        assert issubclass(RetrievalError, RAGError)

    def test_llm_error_is_rag_error(self):
        assert issubclass(LLMError, RAGError)

    def test_agent_error_is_rag_error(self):
        assert issubclass(AgentError, RAGError)

    def test_validation_error_is_rag_error(self):
        assert issubclass(ValidationError, RAGError)

    def test_memory_error_is_rag_error(self):
        assert issubclass(MemoryError, RAGError)

    def test_ingestion_error_raises_and_caught_as_rag_error(self):
        with pytest.raises(RAGError):
            raise IngestionError("ingest failed", code="INGEST_ERR")

    def test_subclass_preserves_code(self):
        e = LLMError("llm failed", code="LLM_DOWN")
        assert e.code == "LLM_DOWN"
        assert str(e) == "llm failed"

    def test_all_subclasses_accept_context(self):
        for cls in (IngestionError, RetrievalError, LLMError, AgentError, ValidationError, MemoryError):
            e = cls("msg", context={"key": "value"})
            assert e.context == {"key": "value"}

    def test_each_subclass_catchable_individually(self):
        for cls in (IngestionError, RetrievalError, LLMError, AgentError, ValidationError, MemoryError):
            with pytest.raises(cls):
                raise cls("error")
