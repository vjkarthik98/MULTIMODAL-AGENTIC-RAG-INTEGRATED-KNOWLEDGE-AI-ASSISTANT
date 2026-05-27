"""Generation suite runner.

Calls app/pipeline/query_pipeline.py:query_pipeline() directly — the same path
production uses. Scores faithfulness, answer_relevancy, context_recall, citation_accuracy,
template_leak_rate, and hallucination_rate.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_all_gold
from app.eval.metrics.base import SuiteResult
from app.eval.metrics.generation import compute_generation_metrics
from app.eval.metrics.hallucination import hallucination_rate
from app.eval.metrics.latency import latency_stats


def _load_eval_rows(cfg: EvalConfig) -> List[Dict[str, Any]]:
    """Load all text + PDF + docx gold rows that have real reference answers."""
    gold = load_all_gold(
        gold_dir=cfg.gold_dir,
        modalities=["txt", "pdf", "docx"],
        include_todos=False,
    )
    rows = []
    for modality_rows in gold.values():
        for r in modality_rows:
            ref = r.get("reference_answer", "")
            if ref and ref not in ("TODO", "") and "SEARCH_REQUIRED" not in ref and "INJECTION_PROBE" not in ref:
                rows.append(r)
    return rows


def run_generation_suite(cfg: EvalConfig) -> SuiteResult:
    """Run the generation benchmark against the real query_pipeline()."""
    t0 = time.time()
    result = SuiteResult(suite="generation", judge=cfg.judge_model)

    try:
        from app.pipeline.query_pipeline import query_pipeline
    except ImportError as e:
        result.breached["import_error"] = str(e)
        return result

    gold_rows = _load_eval_rows(cfg)
    if not gold_rows:
        result.breached["no_gold_data"] = (
            "No curated text/pdf/docx gold rows with reference answers. "
            "Run build_gold_set --ingest and review TODO rows first."
        )
        return result

    eval_rows: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for row in gold_rows:
        query = row["query"]
        session_id = f"{cfg.session_prefix}_gen_{row['id']}"

        q_start = time.time()
        try:
            pipeline_result = query_pipeline(
                query=query,
                session_id=session_id,
                user_id=cfg.user_id,
            )
        except Exception as exc:
            result.breached[f"pipeline_error_{row['id']}"] = str(exc)
            continue

        q_elapsed = time.time() - q_start
        latencies.append(q_elapsed)

        answer = pipeline_result.get("answer") or pipeline_result.get("response") or ""
        sources = pipeline_result.get("sources") or []
        context_texts = [s.get("text") or "" for s in sources if isinstance(s, dict)]

        eval_rows.append({
            "query": query,
            "answer": answer,
            "contexts": context_texts,
            "reference_answer": row.get("reference_answer"),
            "retrieved_docs": sources,
            "row_id": row["id"],
            "tags": row.get("tags", []),
        })

    if eval_rows:
        # Generation metrics — prefer_ragas=False when server not running (Phase 26 regression)
        import os as _os
        _prefer_ragas = _os.environ.get("EVAL_PREFER_RAGAS", "true").lower() == "true"
        gen_metrics = compute_generation_metrics(eval_rows, prefer_ragas=_prefer_ragas)
        for m in gen_metrics.values():
            result.add(m)

        # Hallucination rate
        result.add(hallucination_rate(eval_rows))

    # Latency stats
    for m in latency_stats(latencies, prefix="generation").values():
        result.add(m)

    result.duration_sec = time.time() - t0
    return result


def run_hallucination_suite(cfg: EvalConfig) -> SuiteResult:
    """Standalone hallucination suite — runs generation and focuses on ungrounded claims."""
    result = run_generation_suite(cfg)
    result.suite = "hallucination"
    # Keep only hallucination-related metrics
    h_metrics = {k: v for k, v in result.metrics.items()
                 if "halluc" in k or "template" in k or "citation" in k}
    result.metrics = h_metrics
    return result
