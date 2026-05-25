"""Retrieval suite runner.

Calls app/retrieval/retriever.py:Retriever.retrieval() directly — the same code production
runs. Scores recall@k, precision@k, MRR, nDCG, context_precision, hit_rate.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_gold
from app.eval.metrics.base import SuiteResult
from app.eval.metrics.latency import latency_stats
from app.eval.metrics.retrieval import aggregate_retrieval_metrics


def run_retrieval_suite(cfg: EvalConfig) -> SuiteResult:
    """Run the retrieval benchmark against the real Retriever.retrieval() call."""
    t0 = time.time()
    result = SuiteResult(suite="retrieval")

    # Import here so infra (Qdrant, BM25) is only loaded when this suite runs
    try:
        from app.retrieval.retriever import Retriever
    except ImportError as e:
        result.breached["import_error"] = str(e)
        return result

    try:
        retriever = Retriever()
    except Exception as e:
        result.breached["retriever_init"] = str(e)
        return result

    gold_rows = load_gold("txt", gold_dir=cfg.gold_dir)
    if not gold_rows:
        result.breached["no_gold_data"] = "No curated text_gold.jsonl rows found. Run build_gold_set --ingest first."
        return result

    eval_results: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for row in gold_rows:
        query = row["query"]
        relevant_ids = row.get("relevant_chunk_ids", [])
        if isinstance(relevant_ids, str):
            relevant_ids = [relevant_ids]

        session_id = f"{cfg.session_prefix}_retrieval_{row['id']}"
        q_start = time.time()
        try:
            retrieved = retriever.retrieval(
                query=query,
                session_id=session_id,
                top_k=cfg.weaken.top_k or 10,
                user_id=cfg.user_id,
                use_mmr=not cfg.weaken.no_mmr,
                use_rrf=not cfg.weaken.no_rrf,
            )
        except Exception as exc:
            result.breached[f"retrieval_error_{row['id']}"] = str(exc)
            continue

        q_elapsed = time.time() - q_start
        latencies.append(q_elapsed)

        # Extract chunk IDs from retrieved results
        retrieved_ids = []
        for doc in retrieved:
            meta = doc.get("metadata") or {}
            cid = doc.get("chunk_id") or meta.get("chunk_id") or meta.get("doc_id") or meta.get("id")
            if cid:
                retrieved_ids.append(str(cid))

        eval_results.append({
            "query": query,
            "retrieved_ids": retrieved_ids,
            "retrieved_docs": retrieved,
            "relevant_ids": relevant_ids,
            "row_id": row["id"],
            "tags": row.get("tags", []),
        })

    # Compute aggregated metrics
    metrics = aggregate_retrieval_metrics(eval_results, k5=5, k10=10)
    for m in metrics.values():
        result.add(m)

    # Latency stats
    for m in latency_stats(latencies, prefix="retrieval").values():
        result.add(m)

    result.duration_sec = time.time() - t0
    return result
