"""Remote stand-ins for the two models the eval borrows from the serving app.

When EVAL_REMOTE_MODELS=1 (set by deploy/aws/scripts/run_quality_report.sh),
the report process loads NO retrieval models of its own:

  * RemoteContextRetriever replaces the in-process HybridRetriever that
    generation_runner._make_full_context_retriever() used to build.
  * RemoteEmbedder replaces model_loader.get_embedder() inside
    generation._build_ragas_embeddings().

Both call app/api/eval_internal.py on the serving container, which answers from
models it already has resident. The only heavy thing left in the eval process
is the Qwen judge — which is the point: that is what lets the eval container
run under a hard --memory cap without starving the live app.

Each class mirrors exactly the surface its in-process counterpart is used
through, so the callers (`_full_contexts`, `_MagikRagasEmbeddings`) are
unchanged and their existing failure handling still applies.
"""

from __future__ import annotations

import os
from typing import Any

from app.eval.http_client import EvalAuth, post_json

_TIMEOUT_SEC = int(os.getenv("EVAL_REMOTE_TIMEOUT_SEC", "120"))


def remote_models_enabled() -> bool:
    return os.getenv("EVAL_REMOTE_MODELS", "").strip().lower() in ("1", "true", "yes")


def _server_url() -> str:
    return os.getenv("EVAL_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")


def _eval_auth() -> EvalAuth:
    from app.eval.config import EVAL_USER_ID

    return EvalAuth(EVAL_USER_ID)


class RemoteContextRetriever:
    """`.search()` with HybridRetriever's signature, served by /internal/eval/contexts.

    `user_id` is accepted for signature parity and deliberately not sent: the
    server takes the tenant from the token, never from the body.
    """

    def __init__(self, server_url: str | None = None, auth: EvalAuth | None = None) -> None:
        self._url = f"{server_url or _server_url()}/internal/eval/contexts"
        self._auth = auth or _eval_auth()

    def search(
        self,
        query: str,
        session_id: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        user_id: str | None = None,  # noqa: ARG002 - parity with HybridRetriever
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"query": query, "session_id": session_id}
        if top_k:
            payload["top_k"] = int(top_k)
        sources = (filters or {}).get("sources")
        if sources:
            payload["sources"] = list(sources)
        data = post_json(self._url, payload, self._auth, timeout=_TIMEOUT_SEC)
        return [{"text": t} for t in (data.get("contexts") or [])]


class RemoteEmbedder:
    """`.embed_text()` + `.expected_dim`, served by /internal/eval/embed.

    Raises on a null vector instead of inventing one, so
    _MagikRagasEmbeddings._embed_one's existing fallback decides what a failed
    row becomes — the same behaviour as a local encoder failure.
    """

    def __init__(self, server_url: str | None = None, auth: EvalAuth | None = None) -> None:
        from app.core.config import settings

        self._url = f"{server_url or _server_url()}/internal/eval/embed"
        self._auth = auth or _eval_auth()
        self.expected_dim = int(settings.TEXT_EMBEDDING_DIM)

    def embed_text(self, text: str) -> list[float]:
        data = post_json(self._url, {"texts": [text]}, self._auth, timeout=_TIMEOUT_SEC)
        vectors = data.get("vectors") or [None]
        if data.get("dim"):
            self.expected_dim = int(data["dim"])
        vec = vectors[0]
        if not vec:
            raise RuntimeError("remote embedder returned no vector")
        return vec
