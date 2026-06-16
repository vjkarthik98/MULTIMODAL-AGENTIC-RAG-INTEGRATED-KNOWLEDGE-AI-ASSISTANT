"""Retrieval suite runner.

Calls HybridRetriever.search() — the same code production runs.
Scores recall@k, precision@k, MRR, nDCG, context_precision, hit_rate.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_gold
from app.eval.metrics.base import SuiteResult
from app.eval.metrics.latency import latency_stats
from app.eval.metrics.retrieval import aggregate_retrieval_metrics

_RETRIEVAL_MODALITIES = ["txt", "pdf", "docx", "xlsx"]


def run_retrieval_suite(cfg: EvalConfig) -> SuiteResult:
    """Run the retrieval benchmark against the real Retriever.retrieval() call."""
    t0 = time.time()
    result = SuiteResult(suite="retrieval")

    # Import here so infra (Qdrant, BM25) is only loaded when this suite runs
    try:
        from app.core.infra_registry import infra
        from app.core.model_loader import model_loader
        from app.retrieval.hybrid_retriever import HybridRetriever
    except ImportError as e:
        result.breached["import_error"] = str(e)
        return result

    try:
        retriever = HybridRetriever(
            bm25=infra.get_bm25(),
            vector_store=infra.get_vector_store(),
            embedder=model_loader.get_embedder(),
        )
    except Exception as e:
        result.breached["retriever_init"] = str(e)
        return result

    # Load gold rows from all text-based modalities (not just txt)
    gold_rows: List[Dict[str, Any]] = []
    for mod in _RETRIEVAL_MODALITIES:
        try:
            gold_rows.extend(load_gold(mod, gold_dir=cfg.gold_dir))
        except FileNotFoundError:
            pass  # skip missing gold files

    if not gold_rows:
        result.breached["no_gold_data"] = (
            "No curated gold rows found for modalities: "
            f"{_RETRIEVAL_MODALITIES}. Run build_gold_set --ingest first."
        )
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
            retrieved = retriever.search(
                query=query,
                session_id=session_id,
                top_k=cfg.weaken.top_k or 10,
                user_id=cfg.user_id,
            )
        except Exception as exc:
            result.breached[f"retrieval_error_{row['id']}"] = str(exc)
            continue

        q_elapsed = time.time() - q_start
        latencies.append(q_elapsed)

        # Extract chunk IDs from retrieved results — format: "{source}::chunk_{chunk_id}"
        # to match the gold set format produced by fill_gold_chunk_ids.py
        retrieved_ids = []
        for doc in retrieved:
            meta = doc.get("metadata") or {}
            source = meta.get("source", "")
            cid = meta.get("chunk_id")
            if source and cid is not None:
                retrieved_ids.append(f"{source}::chunk_{cid}")
            elif cid is not None:
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
