from __future__ import annotations

import time
from typing import Any, Iterator, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Canned responses keyed by simple keyword match — good enough for functional testing
_CANNED: list[tuple[str, str]] = [
    ("name",        "Answer: Based on the context, the name mentioned is as provided in your documents.\nConfidence: 0.9\nSources Used: 1"),
    ("summarize",   "Answer: The document covers several key topics as found in the retrieved context.\nConfidence: 0.85\nSources Used: 2"),
    ("what",        "Answer: Based on the retrieved knowledge, the answer is contained within the provided documents.\nConfidence: 0.8\nSources Used: 1"),
    ("how",         "Answer: The process described in the documents involves the steps outlined in the retrieved context.\nConfidence: 0.8\nSources Used: 1"),
    ("explain",     "Answer: The concept is explained in the retrieved documents as follows: it refers to the topic covered in your uploaded files.\nConfidence: 0.85\nSources Used: 2"),
    ("list",        "Answer: The items found in the documents are listed in the retrieved context above.\nConfidence: 0.8\nSources Used: 1"),
    ("compare",     "Answer: Comparing the entities based on retrieved context: both share common characteristics as described in your documents.\nConfidence: 0.75\nSources Used: 2"),
]

_DEFAULT = "Answer: Based on the retrieved context from your documents, I found relevant information that addresses your query.\nConfidence: 0.8\nSources Used: 1"


class MockLLM:
    """Instant stub LLM for development/testing — no model loaded, no GPU needed."""

    model_name = "mock-llm"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        session_id: str = "default",
        **kwargs: Any,
    ) -> str:
        lower = prompt.lower()
        for keyword, response in _CANNED:
            if keyword in lower:
                logger.info(
                    event="mock_llm_generate",
                    matched=keyword,
                    session_id=session_id,
                )
                return response
        logger.info(event="mock_llm_generate", matched="default", session_id=session_id)
        return _DEFAULT

    def stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
        session_id: str = "default",
        **kwargs: Any,
    ) -> Iterator[str]:
        response = self.generate(prompt, max_tokens, temperature, session_id=session_id)
        for word in response.split():
            yield word + " "

    def health_check(self) -> dict:
        return {"model": "mock-llm", "ready": True, "mock": True}
