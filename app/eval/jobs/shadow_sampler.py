"""Deterministic sampling of live query/answer/context traces into MongoDB.

Phase 31 (Monitoring): the write side of online eval. See
`app/eval/stubs/online_eval.md` for the original design and
`app/eval/jobs/online_eval.py` for the scoring job that reads what this
writes and pushes Prometheus gauges from it.

`sample_and_log()` is called from the hot query-streaming path
(`app/api/api_routes.py::stream_query`) — it must NEVER raise into the
request path and must never add meaningful latency. Every internal step is
defensive (not just relying on the caller's try/except) so a Mongo hiccup on
a scale-to-zero box can't turn into a user-visible failure.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _is_sampled(session_id: str) -> bool:
    rate = settings.ONLINE_EVAL_SAMPLE_RATE
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    # Deterministic on session_id (not random per call) so a whole session's
    # traffic is sampled together — reproducible, matches the Phase 31 stub design.
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < rate


def _retrieval_stats(sources: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Retrieval/rerank-quality signal derived from the SAME sources array
    every caller already builds and passes in — no new data collection, no
    new call into the retriever/reranker.

    `score` on each source is whatever the caller's pipeline stamped as the
    final ranking score (post-rerank `final_score`/`_reranker_raw` — see
    query_pipeline.py's `_build_sources_array` and rag_pipeline.py's
    equivalent), so `top1_score` is a real retrieval-quality proxy: a
    reference-free stand-in for "how confident was the top hit," the same
    role thresholds.yaml's `context_precision` / retrieval `recall@5` play
    offline, but cheap enough to compute on every sampled live request.
    Deliberately NOT BM25-vs-dense contribution split or raw pre-rerank
    candidate-pool stats — those live inside HybridRetriever._search_impl's
    closure and are never returned past `.search()`'s final list, so
    exposing them would mean restructuring the tuned retrieval body just for
    telemetry (against this repo's own monitoring ground rules). What's
    captured here is exactly what the architecture already surfaces.
    """
    scores = [
        float(s["score"])
        for s in (sources or [])
        if isinstance(s, dict) and isinstance(s.get("score"), (int, float))
    ]
    if not scores:
        return {"retrieval_count": len(sources or []), "top1_score": None, "mean_topk_score": None}
    return {
        "retrieval_count": len(sources or []),
        "top1_score": round(max(scores), 6),
        "mean_topk_score": round(sum(scores) / len(scores), 6),
    }


def sample_and_log(
    *,
    session_id: str,
    user_id: str | None,
    query: str,
    answer: str,
    sources: list[dict[str, Any]] | None,
    route: str | None,
    latency_ms: float | None,
) -> None:
    """Best-effort: log a sampled query/answer/context trace for later scoring.

    Guaranteed not to raise — safe to call unguarded, though callers on the
    hot path should still wrap it in try/except as defense in depth.
    """
    try:
        if not session_id or not query or not answer:
            return
        if not _is_sampled(session_id):
            return

        from app.core.infra_registry import infra

        mongo = infra.get_mongo()
        if mongo is None or mongo.db is None:
            return

        # Sources carry a "text"/"snippet" field (see rag_pipeline._build_p248_sources)
        # — a truncated context snippet, not the full chunk. That matches how the
        # offline generation judge already scores (see thresholds.yaml's note on
        # 200-char context-snippet truncation), so it is not a degraded signal.
        contexts = [
            (s.get("text") or s.get("snippet") or "")
            for s in (sources or [])
            if isinstance(s, dict)
        ]
        contexts = [c for c in contexts if c]

        doc = {
            "eval_shadow": True,
            "scored": False,
            "session_id": session_id,
            "user_id": user_id,
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "route": route,
            "latency_ms": latency_ms,
            "sampled_at": time.time(),
            **_retrieval_stats(sources),
        }
        mongo.db[settings.MONGO_EVAL_SHADOW_COLLECTION].insert_one(doc)
    except Exception as exc:
        logger.debug(event="eval_shadow_sample_failed", error=str(exc))
