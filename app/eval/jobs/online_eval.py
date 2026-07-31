"""Online eval (Phase 31): scores sampled live traffic and pushes Prometheus gauges.

Reads unscored documents written by app/eval/jobs/shadow_sampler.py from
MONGO_EVAL_SHADOW_COLLECTION, scores them with REFERENCE-FREE metrics only,
and updates the magik_eval_online_* Prometheus gauges (app/core/metrics.py).

Reference-free scope, deliberately: recall@k / MRR / nDCG / Ragas faithfulness-
with-ground-truth all need a gold reference answer, which live traffic does not
have. Those stay exclusively a CI/offline concern (app/eval/run.py, gated by
thresholds.yaml). This job only computes what CAN be judged without a
reference: lexical faithfulness (answer grounded in its own retrieved
context), lexical answer relevancy, the existing numeric-grounding
hallucination check, latency, and routing-decision distribution.

Uses the pure-Python lexical judge (app/eval/judges/lexical_judge.py), not the
GGUF/Ragas judge — running an extra LLM judge call on live production traffic
on a single-GPU box would compete with real user queries for the same GPU
slot. The lexical judge is the same fallback path the offline harness itself
uses when the GGUF judge is unavailable, so this is a documented degradation,
not an invented one.

Must NEVER raise into its caller (app/main.py's lifespan background task) and
must tolerate Mongo being unavailable or PROMETHEUS_ENABLED being False.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_BATCH_SIZE = 500


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(pct * (len(s) - 1))))
    return s[idx]


def _score_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from app.eval.judges.lexical_judge import lexical_answer_relevancy, lexical_faithfulness
    from app.eval.metrics.hallucination import hallucination_flag_single

    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []
    flagged = 0
    scored = 0
    latencies: list[float] = []
    route_counts: dict[str, int] = {}

    for row in rows:
        query = row.get("query") or ""
        answer = row.get("answer") or ""
        contexts = row.get("contexts") or []
        if not answer:
            continue
        scored += 1

        if contexts:
            faithfulness_scores.append(lexical_faithfulness(answer, contexts))
        if query:
            relevancy_scores.append(lexical_answer_relevancy(answer, query))

        result = hallucination_flag_single(answer, contexts, reference_answer=None)
        if result["flagged"]:
            flagged += 1

        lat = row.get("latency_ms")
        if isinstance(lat, (int, float)):
            latencies.append(float(lat))

        route = row.get("route") or "unknown"
        route_counts[route] = route_counts.get(route, 0) + 1

    return {
        "scored": scored,
        "faithfulness": (
            sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
        ),
        "answer_relevancy": (
            sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None
        ),
        "hallucination_rate": (flagged / scored) if scored else None,
        "latency_p50_ms": _percentile(latencies, 0.50) if latencies else None,
        "latency_p95_ms": _percentile(latencies, 0.95) if latencies else None,
        "route_counts": route_counts,
    }


def _push_gauges(result: dict[str, Any]) -> None:
    from app.core import metrics as m

    m.eval_online_sample_count.set(result["scored"])
    if result["faithfulness"] is not None:
        m.eval_online_faithfulness.set(result["faithfulness"])
    if result["answer_relevancy"] is not None:
        m.eval_online_answer_relevancy.set(result["answer_relevancy"])
    if result["hallucination_rate"] is not None:
        m.eval_online_hallucination_rate.set(result["hallucination_rate"])
    if result["latency_p50_ms"] is not None:
        m.eval_online_latency_p50_ms.set(result["latency_p50_ms"])
    if result["latency_p95_ms"] is not None:
        m.eval_online_latency_p95_ms.set(result["latency_p95_ms"])

    total = sum(result["route_counts"].values()) or 1
    for route, count in result["route_counts"].items():
        m.eval_online_route_share.labels(route=route).set(count / total)


def run_online_eval_once(batch_size: int = _DEFAULT_BATCH_SIZE) -> dict[str, Any]:
    """Score up to `batch_size` unscored shadow docs and push gauges. Never raises."""
    try:
        from app.core.infra_registry import infra

        mongo = infra.get_mongo()
        if mongo is None or mongo.db is None:
            logger.info(event="online_eval_skipped", reason="mongo_unavailable")
            return {"skipped": "mongo_unavailable"}

        coll = mongo.db[settings.MONGO_EVAL_SHADOW_COLLECTION]
        rows = list(coll.find({"scored": False}).limit(batch_size))
        if not rows:
            logger.info(event="online_eval_skipped", reason="no_sampled_traffic")
            return {"skipped": "no_sampled_traffic"}

        result = _score_batch(rows)
        _push_gauges(result)

        ids = [r["_id"] for r in rows]
        coll.update_many({"_id": {"$in": ids}}, {"$set": {"scored": True}})

        logger.info(
            event="online_eval_completed",
            scored=result["scored"],
            faithfulness=result["faithfulness"],
            hallucination_rate=result["hallucination_rate"],
        )
        return result
    except Exception as exc:
        logger.warning(event="online_eval_failed", error=str(exc))
        return {"error": str(exc)}


async def run_online_eval_loop() -> None:
    """Background task: score once shortly after startup, then repeat on an
    interval for as long as the process is alive. A "wake" on the scale-to-zero
    deployment is a fresh process start, so the first iteration satisfies
    "run once per wake"; the loop gives fresher gauges during longer sessions
    instead of requiring a restart to refresh them.
    """
    if not settings.ONLINE_EVAL_ENABLED or settings.ONLINE_EVAL_SAMPLE_RATE <= 0:
        return

    # Let infra warmup (Mongo connection) land before the first attempt.
    await asyncio.sleep(30)
    while True:
        await asyncio.to_thread(run_online_eval_once)
        await asyncio.sleep(settings.ONLINE_EVAL_INTERVAL_SEC)


if __name__ == "__main__":
    # Manual/debug entrypoint: `python -m app.eval.jobs.online_eval`. Scores
    # once and prints the result. NOTE: run this way (a fresh, short-lived
    # process) it does NOT usefully update the live dashboard — Prometheus
    # gauges live in-process and this process exits immediately after. Use
    # this only to sanity-check scoring logic against real sampled data; the
    # dashboard-feeding path is run_online_eval_loop() inside app/main.py's
    # lifespan, sharing the same process as prometheus_client.start_http_server.
    import json

    print(json.dumps(run_online_eval_once(), indent=2, default=str))
