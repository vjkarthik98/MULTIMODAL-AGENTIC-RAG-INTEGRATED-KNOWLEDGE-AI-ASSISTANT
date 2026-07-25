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
from typing import Any

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
    access_token: str | None = None,
    no_cache: bool = True,
) -> dict[str, Any]:
    """Call /rag/query on the running server. Reuses server's GPU models.

    no_cache defaults True: eval must measure the live model, never a stale
    cached answer from a previous run (same session_id+query would hit cache).
    """
    headers = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    payload = {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "no_cache": no_cache,
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
) -> dict[str, Any]:
    """Fallback: call pipeline directly (loads models in-process)."""
    from app.pipeline.query_pipeline import query_pipeline

    return query_pipeline(query=query, session_id=session_id, user_id=user_id)


def _load_eval_rows(cfg: EvalConfig) -> list[dict[str, Any]]:
    """Load gold rows with real reference answers. Defaults to text/pdf/docx;
    a --modality filter narrows to a single modality (any of the 7)."""
    _mods = [cfg.modality] if getattr(cfg, "modality", None) else ["txt", "pdf", "docx"]
    gold = load_all_gold(
        gold_dir=cfg.gold_dir,
        modalities=_mods,
        include_todos=False,
    )
    rows = []
    for modality_rows in gold.values():
        for r in modality_rows:
            ref = r.get("reference_answer", "")
            # Behavioral rows (refusal/adversarial) are scored with their own
            # rubrics, not the standard faithfulness/relevancy/recall metrics —
            # exclude them here so they never get mis-scored as normal answers.
            if r.get("expected_behavior") == "abstain" or r.get("question_type") in (
                "refusal",
                "adversarial",
            ):
                continue
            if (
                ref
                and ref not in ("TODO", "")
                and "SEARCH_REQUIRED" not in ref
                and "INJECTION_PROBE" not in ref
            ):
                rows.append(r)
    return rows


import re as _re

# The verification loop (app/verification/verification_loop.py) appends this hedge
# to answers it can't fully auto-verify. It's a product warning, but graded as an
# ANSWER it reads as uncertainty and unfairly drags faithfulness/correctness down
# (verified: a correct answer scores 1.0 clean, 0.5 with the hedge). Strip it for
# grading — we measure the answer's content, not the product's caution banner.
_HEDGE_RE = _re.compile(
    r"\s*This answer could not be fully verified against the source material\s*[—-]+\s*"
    r"treat the figures? above with caution\.?",
    _re.IGNORECASE,
)


def _strip_verification_hedge(answer: str) -> str:
    return _HEDGE_RE.sub("", answer or "").strip()


def _make_full_context_retriever():
    """In-process HybridRetriever for grading context. The /rag/query API returns
    source text truncated to 200 chars (query_pipeline._build_sources_array), which
    starves the LLM judge — faithfulness/context_recall must be graded against the
    FULL retrieved chunks. Returns None if retrieval infra can't load."""
    try:
        from app.core.infra_registry import infra
        from app.core.model_loader import model_loader
        from app.retrieval.hybrid_retriever import HybridRetriever

        return HybridRetriever(
            bm25=infra.get_bm25(),
            vector_store=infra.get_vector_store(),
            embedder=model_loader.get_embedder(),
        )
    except Exception as exc:
        print(f"[eval] full-context retriever unavailable ({exc}); grading on API snippets")
        return None


def _full_contexts(retriever, query: str, user_id: str, session_id: str) -> list[str]:
    """Full-text chunks for the query (untruncated), for faithful judge grading."""
    if retriever is None:
        return []
    try:
        docs = retriever.search(query=query, session_id=session_id, top_k=8, user_id=user_id)
        return [str(d.get("text") or "") for d in docs if (d.get("text") or "").strip()]
    except Exception:
        return []


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
    full_ctx_retriever = _make_full_context_retriever()

    if use_server:
        print(f"[eval] Server reachable at {_SERVER_URL} — using HTTP mode (no GPU duplication)")
    else:
        print("[eval] Server not reachable — falling back to direct pipeline mode")
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

    eval_rows: list[dict[str, Any]] = []
    latencies: list[float] = []

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
        answer = _strip_verification_hedge(answer)
        sources = pipeline_result.get("sources") or []
        # Grade against FULL retrieved chunks, not the 200-char API source snippets.
        context_texts = _full_contexts(full_ctx_retriever, query, cfg.user_id, session_id)
        if not context_texts:  # fallback: truncated API sources
            context_texts = [s.get("text") or "" for s in sources if isinstance(s, dict)]

        fidelity = compute_finance_fidelity(answer, context_texts)
        eval_rows.append(
            {
                "query": query,
                "answer": answer,
                "contexts": context_texts,
                "reference_answer": row.get("reference_answer"),
                "retrieved_docs": sources,
                "finance_fidelity": fidelity,
                "row_id": row["id"],
                "tags": row.get("tags", []),
            }
        )

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
            result.add(
                MetricResult(
                    name="finance_fidelity",
                    value=avg_fidelity,
                    n=len(fidelity_scores),
                    notes=f"avg over {len(fidelity_scores)} queries (strict 0.5% tol, no scale bridging)",
                )
            )

    for m in latency_stats(latencies, prefix="generation").values():
        result.add(m)

    result.duration_sec = time.time() - t0
    return result


def run_hallucination_suite(cfg: EvalConfig) -> SuiteResult:
    """Standalone hallucination suite — runs generation and focuses on ungrounded claims."""
    result = run_generation_suite(cfg)
    result.suite = "hallucination"
    h_metrics = {
        k: v
        for k, v in result.metrics.items()
        if "halluc" in k or "template" in k or "citation" in k
    }
    result.metrics = h_metrics
    return result
