"""End-to-end suite runner.

Calls the live FastAPI /query endpoint — the same HTTP path a real client uses.
Scores all metrics end-to-end: retrieval quality, generation quality, routing
accuracy, latency, and hallucination rate. Requires the server to be running.

If the server is not reachable, the suite records an infra error (exit code 2)
rather than silently passing.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_all_gold
from app.eval.metrics.base import MetricResult, SuiteResult
from app.eval.metrics.generation import compute_generation_metrics
from app.eval.metrics.hallucination import hallucination_rate
from app.eval.metrics.latency import latency_stats
from app.eval.metrics.retrieval import aggregate_retrieval_metrics
from app.eval.metrics.routing import route_accuracy


def _build_base_url(cfg: EvalConfig) -> str:
    from app.core.config import get_settings
    s = get_settings()
    host = getattr(s, "HOST", "127.0.0.1")
    port = getattr(s, "PORT", 8000)
    if host in ("0.0.0.0",):
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def _check_server(base_url: str) -> bool:
    """Return True if the server health endpoint responds."""
    try:
        import requests
        r = requests.get(f"{base_url}/health", timeout=5)
        return r.status_code < 500
    except Exception:
        return False


def run_e2e_suite(cfg: EvalConfig) -> SuiteResult:
    """Run the end-to-end benchmark via the live /query API."""
    t0 = time.time()
    result = SuiteResult(suite="e2e")

    try:
        import requests
    except ImportError:
        result.breached["import_error"] = "requests not installed; pip install requests"
        return result

    base_url = _build_base_url(cfg)
    if not _check_server(base_url):
        result.breached["server_unreachable"] = (
            f"FastAPI server not reachable at {base_url}. "
            "Start the server with: uvicorn app.main:app before running the e2e suite."
        )
        return result

    # Load all modality gold rows with real reference answers
    gold = load_all_gold(
        gold_dir=cfg.gold_dir,
        modalities=None,  # all modalities
        include_todos=False,
    )
    rows: List[Dict[str, Any]] = []
    for modality_rows in gold.values():
        for r in modality_rows:
            ref = r.get("reference_answer", "")
            if ref and ref not in ("TODO", "") and "INJECTION_PROBE" not in ref:
                rows.append(r)

    if not rows:
        result.breached["no_gold_data"] = "No curated gold rows with reference answers."
        return result

    retrieval_results: List[Dict[str, Any]] = []
    generation_rows: List[Dict[str, Any]] = []
    routing_results: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for row in rows:
        query = row["query"]
        payload = {
            "query": query,
            "session_id": f"{cfg.session_prefix}_e2e_{row['id']}",
            "user_id": cfg.user_id,
        }

        q_start = time.time()
        try:
            resp = requests.post(f"{base_url}/query", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            result.breached[f"query_error_{row['id']}"] = str(exc)
            continue

        q_elapsed = time.time() - q_start
        latencies.append(q_elapsed)

        answer = data.get("answer") or data.get("response") or ""
        sources = data.get("sources") or []
        context_texts = [s.get("text") or "" for s in sources if isinstance(s, dict)]
        action = data.get("route") or data.get("action") or ""

        retrieved_ids = []
        for s in sources:
            if not isinstance(s, dict):
                continue
            meta = s.get("metadata") or {}
            cid = s.get("chunk_id") or meta.get("chunk_id") or meta.get("id")
            if cid:
                retrieved_ids.append(str(cid))

        relevant_ids = row.get("relevant_chunk_ids", [])
        if isinstance(relevant_ids, str):
            relevant_ids = [relevant_ids]

        retrieval_results.append({
            "query": query,
            "retrieved_ids": retrieved_ids,
            "retrieved_docs": sources,
            "relevant_ids": relevant_ids,
            "row_id": row["id"],
            "tags": row.get("tags", []),
        })
        generation_rows.append({
            "query": query,
            "answer": answer,
            "contexts": context_texts,
            "reference_answer": row.get("reference_answer"),
            "retrieved_docs": sources,
            "row_id": row["id"],
            "tags": row.get("tags", []),
        })
        web_sources = [s for s in sources if isinstance(s, dict) and s.get("type") == "web"]
        routing_results.append({
            "row_id": row["id"],
            "query": query,
            "actual_route": action,
            "expected_route": row.get("expected_route", ""),
            "sources": sources,
            "web_sources": web_sources,
            "web_source_count": len(web_sources),
            "tags": row.get("tags", []),
        })

    # Retrieval metrics
    if retrieval_results:
        for m in aggregate_retrieval_metrics(retrieval_results).values():
            result.add(m)

    # Generation metrics
    if generation_rows:
        for m in compute_generation_metrics(generation_rows).values():
            result.add(m)
        result.add(hallucination_rate(generation_rows))

    # Routing accuracy
    if routing_results:
        result.add(route_accuracy(routing_results))

    # Latency
    for m in latency_stats(latencies).values():
        result.add(m)

    result.duration_sec = time.time() - t0
    return result
