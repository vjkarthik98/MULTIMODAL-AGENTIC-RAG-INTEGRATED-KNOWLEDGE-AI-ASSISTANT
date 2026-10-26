"""Internal eval API — the serving process lends its resident models to the
RAGAS / DeepEval report container.

WHY THIS EXISTS. Both report modules need two things the app already has
loaded: full-text grading contexts (a HybridRetriever over BGE-large) and, for
RAGAS answer_relevancy, the same BGE encoder. They used to load their own
copies in a second python process next to the live app — ~3GB of duplicate
weights plus a CUDA context — and then spawn the 4.7GB Qwen judge on top. That
peak is what the host OOM-killer landed on in v1.0.0-rc4 and rc5 (see
app/eval/runners/generation_runner.py::release_full_context_models).

With this router on, deploy/aws/scripts/run_quality_report.sh runs the eval in
its own memory-capped container that loads only the judge, and fetches
contexts/embeddings from here over the private docker network.

NOT A PUBLIC SURFACE. Four independent gates, any one of which refuses:
  1. EVAL_INTERNAL_API_ENABLED must be true (off by default, on via the
     box's committed env file).
  2. Requests that came through Caddy carry X-Forwarded-For — Caddy's
     reverse_proxy always sets it and a client cannot strip it — so any
     request bearing it 404s. The eval container talks to magik-current:8000
     directly on magik-net and never sends one.
  3. A valid JWT (get_current_user), same as every other data route.
  4. Only the EVAL_USER_ID tenant is served, and the tenant comes from the
     token, never the body — so even a valid user token for anyone else
     cannot read through this route.

404 rather than 403 for gates 1, 2 and 4: the route should not be
discoverable from outside.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user
from app.auth.models import UserPublic
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/internal/eval", tags=["internal-eval"])

# Bounds that keep one call cheap: generation_runner._full_contexts asks for 8,
# and ragas embeds one string at a time (see _MagikRagasEmbeddings).
MAX_TOP_K = 8
MAX_EMBED_TEXTS = 64
MAX_TEXT_CHARS = 8000
# Same placeholder generation._build_ragas_embeddings uses, for the same reason:
# a zero vector NaNs ragas' cosine similarity for the whole metric.
_EMPTY_PLACEHOLDER = "(empty)"

_retriever: Any = None
_retriever_lock = threading.Lock()


class ContextsRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(..., min_length=1, max_length=200)
    sources: list[str] | None = Field(default=None, max_length=50)
    top_k: int = Field(default=MAX_TOP_K, ge=1, le=MAX_TOP_K)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_EMBED_TEXTS)


def _eval_user_id() -> str:
    return (os.getenv("EVAL_USER_ID") or "").strip()


def _guard(request: Request, user: UserPublic) -> str:
    """Apply gates 1, 2 and 4 (gate 3 is the Depends). Returns the tenant."""
    if not settings.EVAL_INTERNAL_API_ENABLED:
        raise HTTPException(status_code=404, detail="Not Found")
    if request.headers.get("x-forwarded-for"):
        logger.warning(event="eval_internal_proxied_request_refused", path=request.url.path)
        raise HTTPException(status_code=404, detail="Not Found")
    eval_user = _eval_user_id()
    if not eval_user or user.user_id != eval_user:
        logger.warning(event="eval_internal_wrong_tenant_refused", user_id=user.user_id)
        raise HTTPException(status_code=404, detail="Not Found")
    return eval_user


def _get_retriever() -> Any:
    """Identical construction to generation_runner._make_full_context_retriever
    (bm25 + vector store + BGE, no SigLIP), so contexts served here grade the
    same way the in-process path always has — but over THIS process's
    already-resident singletons, which cost nothing extra to reuse."""
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                from app.core.infra_registry import infra
                from app.core.model_loader import model_loader
                from app.retrieval.hybrid_retriever import HybridRetriever

                _retriever = HybridRetriever(
                    bm25=infra.get_bm25(),
                    vector_store=infra.get_vector_store(),
                    embedder=model_loader.get_embedder(),
                )
    return _retriever


@router.post("/contexts")
def eval_contexts(
    body: ContextsRequest,
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> dict[str, Any]:
    """Full, untruncated retrieved chunks for one query, scoped exactly like
    generation_runner._full_contexts (same `sources` filter, same top_k)."""
    user_id = _guard(request, current_user)
    filters = {"sources": body.sources} if body.sources else None
    try:
        docs = _get_retriever().search(
            query=body.query,
            session_id=body.session_id,
            top_k=body.top_k,
            user_id=user_id,
            filters=filters,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(event="eval_internal_contexts_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="retrieval unavailable") from exc
    contexts = [str(d.get("text") or "") for d in docs if (d.get("text") or "").strip()]
    return {"contexts": contexts}


@router.post("/embed")
def eval_embed(
    body: EmbedRequest,
    request: Request,
    current_user: UserPublic = Depends(get_current_user),
) -> dict[str, Any]:
    """BGE vectors, exactly one per input.

    The 1:1 contract matters: ragas reshapes the result to len(inputs), and
    TextEmbedder silently drops entries it cannot encode. A failed entry comes
    back as null so the caller can decide, rather than shifting every later
    vector by one.
    """
    _guard(request, current_user)
    from app.core.model_loader import model_loader

    embedder = model_loader.get_embedder()
    dim = int(getattr(embedder, "expected_dim", settings.TEXT_EMBEDDING_DIM))
    vectors: list[list[float] | None] = []
    for text in body.texts:
        clean = (text or "")[:MAX_TEXT_CHARS].strip() or _EMPTY_PLACEHOLDER
        try:
            vectors.append([float(x) for x in embedder.embed_text(clean)])
        except Exception as exc:  # noqa: BLE001
            logger.warning(event="eval_internal_embed_failed", error=str(exc))
            vectors.append(None)
    return {"vectors": vectors, "dim": dim}
