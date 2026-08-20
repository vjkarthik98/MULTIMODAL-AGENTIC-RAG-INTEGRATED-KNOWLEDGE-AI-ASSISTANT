"""Behavioral suite runner — refusal + adversarial rows across all modalities.

Queries the live pipeline (server if reachable, else direct) for each behavioral
gold row, then scores refusal_accuracy + adversarial_pass with Prometheus.
Reuses generation_runner's query machinery so auth/fallback behaviour is identical.
"""

from __future__ import annotations

import time
from typing import Any

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_all_gold
from app.eval.metrics.base import SuiteResult
from app.eval.metrics.behavioral import compute_behavioral_metrics
from app.eval.metrics.latency import latency_stats
from app.eval.runners.generation_runner import (
    _query_via_pipeline,
    _query_via_server,
    _server_available,
)

_BEHAVIORAL_MODALITIES = ["txt", "pdf", "docx", "xlsx", "image", "audio", "video"]


def _load_behavioral_rows(cfg: EvalConfig) -> list[dict[str, Any]]:
    _mods = [cfg.modality] if getattr(cfg, "modality", None) else _BEHAVIORAL_MODALITIES
    gold = load_all_gold(gold_dir=cfg.gold_dir, modalities=_mods)
    rows = []
    for modality_rows in gold.values():
        for r in modality_rows:
            if r.get("rubric_id") in ("refusal", "adversarial"):
                rows.append(r)
    return rows


def run_behavioral_suite(cfg: EvalConfig) -> SuiteResult:
    t0 = time.time()
    result = SuiteResult(suite="behavioral", judge=cfg.judge_model)

    use_server = _server_available()
    from app.eval.http_client import EvalAuth

    eval_auth = EvalAuth(cfg.user_id)
    gold_rows = _load_behavioral_rows(cfg)
    if not gold_rows:
        result.breached["no_behavioral_data"] = "No refusal/adversarial gold rows found."
        return result

    eval_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    _run_tag = int(time.time())  # unique per run → never collides with a cached prior run
    for row in gold_rows:
        session_id = f"{cfg.session_prefix}_beh_{_run_tag}_{row['id']}"
        q_start = time.time()
        try:
            if use_server:
                _row_sources = row.get("relevant_doc_ids") or (
                    [row["source_file"]] if row.get("source_file") else None
                )
                pr = _query_via_server(
                    row["query"], session_id, cfg.user_id, eval_auth, sources=_row_sources
                )
            else:
                pr = _query_via_pipeline(row["query"], session_id, cfg.user_id)
        except Exception as exc:
            result.breached[f"pipeline_error_{row['id']}"] = str(exc)
            continue
        latencies.append(time.time() - q_start)
        answer = pr.get("answer") or pr.get("response") or ""
        eval_rows.append({**row, "answer": answer})

    if eval_rows:
        for m in compute_behavioral_metrics(eval_rows).values():
            result.add(m)
    for m in latency_stats(latencies, prefix="behavioral").values():
        result.add(m)

    result.duration_sec = time.time() - t0
    return result
