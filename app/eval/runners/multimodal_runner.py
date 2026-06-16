"""Multimodal suite runner.

Runs one representative document per modality through the full pipeline,
then asks cross-modal queries to isolate per-modality regressions.
Uses the retrieval + generation metrics for each modality so a regression
in (e.g.) PDF ingestion shows up as a distinct per-modality metric drop.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_all_gold
from app.eval.metrics.base import MetricResult, SuiteResult
from app.eval.metrics.retrieval import aggregate_retrieval_metrics


_MODALITIES = ["txt", "pdf", "docx", "xlsx", "image", "audio", "video"]


def run_multimodal_suite(cfg: EvalConfig) -> SuiteResult:
    """Run per-modality retrieval checks and surface modality-specific regressions."""
    t0 = time.time()
    result = SuiteResult(suite="multimodal")

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

    # Load curated rows per modality
    gold = load_all_gold(
        gold_dir=cfg.gold_dir,
        modalities=_MODALITIES,
        include_todos=False,
    )

    any_results = False
    for modality, rows in gold.items():
        if not rows:
            result.add(MetricResult.empty(
                f"{modality}_recall_at_5",
                f"no curated {modality} gold rows",
            ))
            continue

        eval_results: List[Dict[str, Any]] = []
        for row in rows:
            query = row.get("query", "")
            relevant_ids = row.get("relevant_chunk_ids", [])
            if isinstance(relevant_ids, str):
                relevant_ids = [relevant_ids]

            session_id = f"{cfg.session_prefix}_mm_{modality}_{row['id']}"
            try:
                retrieved = retriever.search(
                    query=query,
                    session_id=session_id,
                    top_k=cfg.weaken.top_k or 10,
                    user_id=cfg.user_id,
                )
            except Exception as exc:
                result.breached[f"{modality}_error_{row['id']}"] = str(exc)
                continue

            retrieved_ids = []
            for doc in retrieved:
                meta = doc.get("metadata") or {}
                source = meta.get("source", "") or meta.get("source_file", "")
                cid = meta.get("chunk_id") or doc.get("chunk_id")
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

        if not eval_results:
            continue

        any_results = True
        metrics = aggregate_retrieval_metrics(eval_results, k5=5, k10=10)
        # Prefix metrics with modality name so they don't clobber each other
        for name, m in metrics.items():
            prefixed = MetricResult(
                name=f"{modality}_{name}",
                value=m.value,
                n=m.n,
                notes=m.notes,
                sub=m.sub,
            )
            result.add(prefixed)

    if not any_results:
        result.breached["no_results"] = (
            "No modality had curated gold rows with real relevant_chunk_ids. "
            "Run build_gold_set --ingest and review TODO rows first."
        )

    result.duration_sec = time.time() - t0
    return result
