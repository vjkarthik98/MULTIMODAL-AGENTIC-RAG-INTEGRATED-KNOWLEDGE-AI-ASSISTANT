"""Centralised exception hierarchy for the RAG system."""
from __future__ import annotations


class RAGError(Exception):
    def __init__(self, message: str, code: str = "RAG_ERROR", context: dict | None = None):
        self.code = code
        self.context = context or {}
        super().__init__(message)


class IngestionError(RAGError):
    pass


class RetrievalError(RAGError):
    pass


class LLMError(RAGError):
    pass


class AgentError(RAGError):
    pass


class ValidationError(RAGError):
    pass


class MemoryError(RAGError):
    pass
