"""Generation suite runner.

Calls the live FastAPI server's /rag/query endpoint (HTTP) instead of
importing query_pipeline directly. This prevents double-loading GPU models
when the server is already running — same pattern as GGUFJudge.

Falls back to direct pipeline import if EVAL_SERVER_URL is not reachable.
Scores faithfulness, answer_relevancy, context_recall, citation_accuracy,
template_leak_rate, and hallucination_rate.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_all_gold
from app.eval.metrics.base import SuiteResult
from app.eval.metrics.generation import compute_generation_metrics
from app.eval.metrics.hallucination import compute_finance_fidelity, hallucination_rate
from app.eval.metrics.latency import latency_stats

_SERVER_URL = os.getenv("EVAL_SERVER_URL", "http://127.0.0.1:8000")
_HTTP_TIMEOUT = 300


def _server_available() -> bool:
    """Quick check if the FastAPI server is reachable."""
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(f"{_SERVER_URL}/rag/health")
            return r.status_code == 200
    except Exception:
        return False


def _query_via_server(
    query: str,
    session_id: str,
    user_id: str,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Call /rag/query on the running server. Reuses server's GPU models."""
    headers = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    payload = {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
    }
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        resp = client.post(
            f"{_SERVER_URL}/rag/query",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()


def _query_via_pipeline(
    query: str,
    session_id: str,
    user_id: str,
) -> Dict[str, Any]:
    """Fallback: call pipeline directly (loads models in-process)."""
    from app.pipeline.query_pipeline import query_pipeline
    return query_pipeline(query=query, session_id=session_id, user_id=user_id)


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
    """Run the generation benchmark.

    Prefers HTTP server mode to avoid GPU OOM when server is already running.
    Falls back to direct pipeline if server is not reachable.
    """
    t0 = time.time()
    result = SuiteResult(suite="generation", judge=cfg.judge_model)

    # Decide execution mode
    use_server = _server_available()
    access_token = os.getenv("EVAL_ACCESS_TOKEN", "")

    if use_server:
        print(f"[eval] Server reachable at {_SERVER_URL} — using HTTP mode (no GPU duplication)")
    else:
        print(f"[eval] Server not reachable — falling back to direct pipeline mode")
        try:
            from app.pipeline.query_pipeline import query_pipeline  # noqa: F401
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
            if use_server:
                pipeline_result = _query_via_server(
                    query=query,
                    session_id=session_id,
                    user_id=cfg.user_id,
                    access_token=access_token,
                )
            else:
                pipeline_result = _query_via_pipeline(
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

        fidelity = compute_finance_fidelity(answer, context_texts)
        eval_rows.append({
            "query": query,
            "answer": answer,
            "contexts": context_texts,
            "reference_answer": row.get("reference_answer"),
            "retrieved_docs": sources,
            "finance_fidelity": fidelity,
            "row_id": row["id"],
            "tags": row.get("tags", []),
        })

    if eval_rows:
        _prefer_ragas = os.environ.get("EVAL_PREFER_RAGAS", "true").lower() == "true"
        gen_metrics = compute_generation_metrics(eval_rows, prefer_ragas=_prefer_ragas)
        for m in gen_metrics.values():
            result.add(m)

        result.add(hallucination_rate(eval_rows))

        # Finance numeric fidelity — fraction of cited numbers grounded in context
        fidelity_scores = [r["finance_fidelity"] for r in eval_rows if "finance_fidelity" in r]
        if fidelity_scores:
            from app.eval.metrics.base import MetricResult
            avg_fidelity = sum(fidelity_scores) / len(fidelity_scores)
            result.add(MetricResult(
                name="finance_fidelity",
                value=avg_fidelity,
                n=len(fidelity_scores),
                notes=f"avg over {len(fidelity_scores)} queries (strict 0.5% tol, no scale bridging)",
            ))

    for m in latency_stats(latencies, prefix="generation").values():
        result.add(m)

    result.duration_sec = time.time() - t0
    return result


def run_hallucination_suite(cfg: EvalConfig) -> SuiteResult:
    """Standalone hallucination suite — runs generation and focuses on ungrounded claims."""
    result = run_generation_suite(cfg)
    result.suite = "hallucination"
    h_metrics = {k: v for k, v in result.metrics.items()
                 if "halluc" in k or "template" in k or "citation" in k}
    result.metrics = h_metrics
    return result
