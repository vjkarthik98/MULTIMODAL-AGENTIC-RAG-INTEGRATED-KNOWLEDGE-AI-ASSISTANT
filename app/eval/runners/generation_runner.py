"""Generation suite runner.

Calls the live FastAPI server's /rag/query endpoint (HTTP) instead of
importing query_pipeline directly. This prevents double-loading GPU models
when the server is already running — same pattern as the eval judge's own
DeepEval wrapper (app/eval/judges/qwen_judge.py).

Falls back to direct pipeline import if EVAL_SERVER_URL is not reachable.
Scores faithfulness, answer_relevancy, context_recall, citation_accuracy,
template_leak_rate, and hallucination_rate.
"""

from __future__ import annotations

import os
import re as _re
import time
from typing import Any

import httpx

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_all_gold
from app.eval.metrics.base import MetricResult, SuiteResult
from app.eval.metrics.generation import compute_generation_metrics
from app.eval.metrics.hallucination import (
    compute_finance_fidelity,
    fabrication_rate,
    hallucination_rate,
    omission_rate,
)
from app.eval.metrics.latency import _pct, latency_stats

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
    auth: Any = None,
    no_cache: bool = True,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Call /rag/query on the running server. Reuses server's GPU models.

    no_cache defaults True: eval must measure the live model, never a stale
    cached answer from a previous run (same session_id+query would hit cache).

    `auth` is an EvalAuth (app/eval/http_client.py), not a raw token string: the
    full suite outlives ACCESS_TOKEN_EXPIRE_MINUTES, so a token captured once at
    suite start expires partway through and every remaining row records a 401
    instead of an answer. EvalAuth re-mints on demand.

    `sources` mirrors the UI's @ picker: /rag/query 400s on an empty scope
    (api_routes.py's FILE SCOPE REQUIRED gate), so every gold row that names a
    real KB file must pass it here or every generation/behavioral call fails
    with "Select a file to scope this query before sending" instead of an
    answer.
    """
    from app.eval.http_client import EvalAuth, post_json

    if auth is None:
        auth = EvalAuth(user_id)

    payload: dict[str, Any] = {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "no_cache": no_cache,
    }
    if sources:
        payload["sources"] = sources
    data = post_json(f"{_SERVER_URL}/rag/query", payload, auth, timeout=int(_HTTP_TIMEOUT))

    # Seed the per-run memo so the e2e sub-suite — which re-queries EVERY one
    # of this suite's rows (105 of its 164) — can reuse this response instead
    # of spending another ~26 minutes re-deriving it. See app/eval/answer_cache.
    # Only this non-streaming path seeds it: the SSE path posts to a different
    # endpoint and returns a differently-shaped (post-rewrap) answer, so its
    # responses are NOT interchangeable with what e2e asks for.
    try:
        from app.eval import answer_cache

        answer_cache.put(
            answer_cache.make_key(query, user_id, sources=sources, force_web=False),
            data,
        )
    except Exception:  # noqa: BLE001 - a memo failure must never fail a row
        pass
    return data


def _query_via_stream_server(
    query: str,
    session_id: str,
    user_id: str,
    auth: Any = None,
    no_cache: bool = True,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Call /rag/query/stream (SSE) on the running server — the endpoint the
    UI actually posts to, running rag_pipeline.stream() incl. the post-
    verification `_conversational_rewrap` tone pass that /rag/query never
    exercises. See app/eval/http_client.py::post_sse() for the SSE parsing
    and thresholds.yaml's "KNOWN COVERAGE GAP" comment for why this exists.

    Same payload/scoping rules as `_query_via_server` (FILE SCOPE REQUIRED
    gate applies here too — api_routes.py:1546).
    """
    from app.eval.http_client import EvalAuth, post_sse

    if auth is None:
        auth = EvalAuth(user_id)

    payload: dict[str, Any] = {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "no_cache": no_cache,
    }
    if sources:
        payload["sources"] = sources
    return post_sse(f"{_SERVER_URL}/rag/query/stream", payload, auth, timeout=int(_HTTP_TIMEOUT))


def _query_via_pipeline(
    query: str,
    session_id: str,
    user_id: str,
) -> dict[str, Any]:
    """Fallback: call pipeline directly (loads models in-process)."""
    from app.pipeline.query_pipeline import query_pipeline

    return query_pipeline(query=query, session_id=session_id, user_id=user_id)


def _load_eval_rows(cfg: EvalConfig) -> list[dict[str, Any]]:
    """Load gold rows with real reference answers. Defaults to ALL 7
    modalities; a --modality filter narrows to a single one.

    COVERAGE FIX (2026-08-13, per-modality quality pass): this defaulted to
    ["txt","pdf","docx"] — only 3 of 7 — which meant the `hallucination:
    gate_enabled: true` gate enabled in the prior initiative's Phase 6 was
    structurally blind to xlsx/image/audio/video. That is exactly where the
    worst scores live (measured per-modality: image grounding_success_rate
    0.000, xlsx 0.417, audio context_recall 0.583 / citation_accuracy_v2
    0.692), so the gate could not have caught any of it. n goes 42 -> 98.
    """
    _ALL_MODALITIES = ["txt", "pdf", "docx", "xlsx", "image", "audio", "video"]
    _mods = [cfg.modality] if getattr(cfg, "modality", None) else _ALL_MODALITIES
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


def release_full_context_models() -> None:
    """Drop the query-phase model stack before the judge phase loads.

    THIS IS THE OOM FIX, and it is only safe because of where it runs. The
    eval is launched with `docker exec` INTO the serving container, so it is a
    second, independent python process with its own `model_loader` singleton —
    dropping references here cannot touch the models the app process is
    serving traffic with. It only frees this process's duplicate copies.

    Why it is needed: `_make_full_context_retriever()` loads BGE-large
    (~1.35GB) plus SigLIP and the SigLIP text encoder (~1.8GB) into the eval
    process, and `model_loader` caches them for the life of that process.
    Neither is in `_EVICTABLE_MODELS` — they are deliberately non-evictable
    core query-path models — so nothing ever released them. They then stayed
    resident while the ~4.7GB Qwen judge worker spawned on top, against a live
    app already holding its own full stack. That peak is what the kernel
    OOM-killer landed on twice: v1.0.0-rc4 (CD run 33134484078, `exit code
    137` at 8m17s) and again in rc5, where it took the self-hosted runner
    offline mid-job.

    The two phases never overlap — every row is queried and its grading
    context collected BEFORE any judging starts — so by the time this is
    called nothing needs these weights again. Calling it is idempotent and
    failure here is never fatal: a process that cannot free memory should
    still be allowed to try to finish its report.
    """
    import gc

    try:
        from app.core.model_loader import model_loader

        for attr in (
            "_text_embedder",
            "_siglip_text_embedder",
            "_siglip_model",
            "_siglip_processor",
            "_siglip_device",
            "_reranker",
        ):
            if hasattr(model_loader, attr):
                setattr(model_loader, attr, None)
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] could not drop query-phase models ({exc}); continuing")

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    print("[eval] released query-phase model stack before judging")


def _full_contexts(
    retriever, query: str, user_id: str, session_id: str, sources: list[str] | None = None
) -> list[str]:
    """Full-text chunks for the query (untruncated), for faithful judge grading.

    Must mirror the file scope the real answer was generated under (`sources`,
    same as `_query_via_server`'s payload) — an unscoped search here can pull
    in chunks from OTHER files in the KB that the LLM never saw, so faithfulness/
    hallucination end up graded against contamination instead of the answer's
    actual evidence. Confirmed live on the audio suite: unscoped grading context
    for a query explicitly scoped to one FOMC press-conference file included an
    unrelated XLSX sheet and a different FOMC recording's Q&A.
    """
    if retriever is None:
        return []
    try:
        filters = {"sources": sources} if sources else None
        docs = retriever.search(
            query=query, session_id=session_id, top_k=8, user_id=user_id, filters=filters
        )
        return [str(d.get("text") or "") for d in docs if (d.get("text") or "").strip()]
    except Exception:
        return []


def _verification_metrics(eval_rows: list[dict]) -> list[MetricResult]:
    """First-ever baseline for thresholds.yaml's `verification.*` section.

    Reuses the VerificationReport already produced by the live VerificationLoop
    pass for each row (see the "verification" key set on eval_rows in
    run_generation_suite) — no duplicate verification call. Rows without a
    report (hybrid_web path, or a server that predates this wiring) are
    excluded, not treated as failures.
    """
    reports = [r["verification"] for r in eval_rows if r.get("verification")]
    if not reports:
        return [
            MetricResult.empty(name, "no VerificationReport on any row")
            for name in (
                "grounding_success_rate",
                "citation_accuracy_v2",
                "retry_success_rate",
                "avg_retry_count",
                "verification_latency_p50",
                "verification_latency_p95",
            )
        ]

    n = len(reports)
    grounded = sum(1 for r in reports if not r.get("unsupported_claims"))
    cited_ok = sum(1 for r in reports if not r.get("bad_citations"))
    retried = [r for r in reports if len(r.get("attempts") or []) > 1]
    retry_successes = sum(1 for r in retried if r.get("verified"))
    retry_counts = [max(len(r.get("attempts") or []) - 1, 0) for r in reports]
    durations_sec = sorted((r.get("total_duration_ms") or 0.0) / 1000.0 for r in reports)

    metrics = [
        MetricResult(
            name="grounding_success_rate",
            value=grounded / n,
            n=n,
            notes="fraction of answers with zero unsupported claims (GroundednessChecker)",
        ),
        MetricResult(
            name="citation_accuracy_v2",
            value=cited_ok / n,
            n=n,
            notes="fraction of answers with zero bad citations (CitationVerifier)",
        ),
        MetricResult(
            name="retry_success_rate",
            value=(retry_successes / len(retried)) if retried else float("nan"),
            n=len(retried),
            notes=f"retried={len(retried)}/{n} | eventually PASS={retry_successes}",
        ),
        MetricResult(
            name="avg_retry_count",
            value=sum(retry_counts) / n,
            n=n,
            notes="mean verification retries per query (cost signal)",
        ),
        MetricResult(
            name="verification_latency_p50",
            value=_pct(durations_sec, 50),
            n=n,
            notes=f"min={durations_sec[0]:.2f}s max={durations_sec[-1]:.2f}s",
        ),
        MetricResult(
            name="verification_latency_p95",
            value=_pct(durations_sec, 95),
            n=n,
            notes="",
        ),
    ]
    return metrics


def run_generation_suite(cfg: EvalConfig) -> SuiteResult:
    """Run the generation benchmark.

    Prefers HTTP server mode to avoid GPU OOM when server is already running.
    Falls back to direct pipeline if server is not reachable.
    """
    t0 = time.time()
    result = SuiteResult(suite="generation", judge=cfg.judge_model)

    # Decide execution mode
    use_server = _server_available()
    from app.eval.http_client import EvalAuth

    eval_auth = EvalAuth(cfg.user_id)
    full_ctx_retriever = _make_full_context_retriever()

    if use_server and getattr(cfg, "live_path", False):
        print(f"[eval] Server reachable at {_SERVER_URL} — using SSE /rag/query/stream (live path)")
    elif use_server:
        print(f"[eval] Server reachable at {_SERVER_URL} — using HTTP mode (no GPU duplication)")
    else:
        if getattr(cfg, "live_path", False):
            print(
                "[eval] WARNING: --live-path requires the server; falling back to direct pipeline (no SSE coverage this run)"
            )
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

        _row_sources = row.get("relevant_doc_ids") or (
            [row["source_file"]] if row.get("source_file") else None
        )

        q_start = time.time()
        try:
            if use_server and getattr(cfg, "live_path", False):
                pipeline_result = _query_via_stream_server(
                    query=query,
                    session_id=session_id,
                    user_id=cfg.user_id,
                    auth=eval_auth,
                    sources=_row_sources,
                )
            elif use_server:
                pipeline_result = _query_via_server(
                    query=query,
                    session_id=session_id,
                    user_id=cfg.user_id,
                    auth=eval_auth,
                    sources=_row_sources,
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
        context_texts = _full_contexts(
            full_ctx_retriever, query, cfg.user_id, session_id, sources=_row_sources
        )
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
                # Phase 32 VerificationReport.to_dict(), when present. HTTP mode
                # (api_routes.py) surfaces it top-level; the direct-pipeline
                # fallback returns query_pipeline()'s raw response dict, which
                # nests it under "metadata" instead — check both.
                "verification": pipeline_result.get("verification")
                or (pipeline_result.get("metadata") or {}).get("verification"),
            }
        )

    if eval_rows:
        gen_metrics = compute_generation_metrics(eval_rows)
        for m in gen_metrics.values():
            result.add(m)

        result.add(hallucination_rate(eval_rows))
        result.add(fabrication_rate(eval_rows))
        result.add(omission_rate(eval_rows))

        for m in _verification_metrics(eval_rows):
            result.add(m)

        # Finance numeric fidelity — fraction of cited numbers grounded in context
        fidelity_scores = [r["finance_fidelity"] for r in eval_rows if "finance_fidelity" in r]
        if fidelity_scores:
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
    """Standalone hallucination suite — runs generation and focuses on ungrounded claims.

    The filter below is also what determines GATING (app/eval/runner.py::
    check_thresholds() gates by suite name, i.e. `result.suite` — set to
    "hallucination" here — not by which top-level thresholds.yaml section a
    metric's threshold happens to live in). fabrication_rate/omission_rate
    (Phase 1) and the verification.* metrics (grounding_success_rate,
    citation_accuracy_v2, retry_success_rate, avg_retry_count,
    verification_latency_p50/p95 — Phase 1/3) must be included here or their
    thresholds.yaml `hallucination: gate_enabled: true` never actually gates
    them: `--suite generation` alone reports these as informational only
    (suite name stays "generation", whose own gate_enabled is deliberately
    still false — see the file's stale-v3-baseline header).
    """
    result = run_generation_suite(cfg)
    result.suite = "hallucination"
    h_metrics = {
        k: v
        for k, v in result.metrics.items()
        if "halluc" in k
        or "template" in k
        or "citation" in k
        or "fabrication" in k
        or "omission" in k
        or "grounding" in k
        or "retry" in k
        or "verification_latency" in k
    }
    result.metrics = h_metrics
    return result
