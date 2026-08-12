"""Drift detection (monitoring Phase 3): statistical reference-vs-current
comparison over sampled live traffic.

Uses `scipy.stats.ks_2samp` + a hand-rolled PSI calculation, not Evidently —
Evidently (any version, including the older 0.4.x "classic" API) unconditionally
pulls in a bundled litestar/uvicorn/plotly web-server stack even for pure
batch scoring, ~15 extra packages for what is really a KS-test on 4 numeric
columns. scipy is already an installed transitive dependency in this
environment; this avoids the unnecessary framework per this repo's own rule
against introducing heavy dependencies where a small amount of real code
does the job.

Reads the same MONGO_EVAL_SHADOW_COLLECTION app/eval/jobs/shadow_sampler.py
writes and app/eval/jobs/online_eval.py scores. Unlike online_eval.py, this
job NEVER mutates the collection — it only reads a window, so it can never
interfere with online_eval's `scored` bookkeeping.

Two windows:
  REFERENCE — a static snapshot of past "normal" traffic, written once (or
    periodically re-baselined) to DRIFT_REFERENCE_PATH by build_reference().
    Fixed, not recomputed each run: drift means "different from what used to
    be normal," which only means something if the comparison point is
    stable across runs. If the file does not exist yet, this job skips
    cleanly (see "no_reference_yet" below) — it ships the tooling, not a
    fabricated reference; see this module's own runbook note at the bottom
    for how to build one.
  CURRENT — the most recent DRIFT_WINDOW_SIZE sampled rows, queried fresh
    every run.

Tracked columns: query_length, top1_score, mean_topk_score, latency_ms —
exactly the fields already in the shadow collection (see shadow_sampler.py's
_retrieval_stats()). Deliberately NOT raw embeddings or BM25/dense
contribution splits — those never reach the shadow collection at all (see
shadow_sampler.py's own docstring on why), so comparing them here would mean
inventing data the architecture doesn't actually capture.

Severity is NOT "one column crossed a threshold": INFO means nothing
drifted; WARNING means the fraction of drifted columns crossed
DRIFT_WARNING_THRESHOLD; CRITICAL additionally requires a QUALITY column
(top1_score or mean_topk_score) to have both drifted AND moved in the BAD
direction (current mean below reference) — a retrieval-score distribution
shifting UP is not an incident. Same "prefer correlated signals over a
single statistical test" philosophy as monitoring/alerts/rules.yml's
hallucination-drift alert.

Must NEVER raise into its caller (app/main.py's lifespan background task,
same convention as online_eval.py) and must tolerate Mongo being
unavailable, no reference file existing yet, or too few rows to test.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TRACKED_COLUMNS = ("query_length", "top1_score", "mean_topk_score", "latency_ms")
_QUALITY_COLUMNS = ("top1_score", "mean_topk_score")  # higher = better
_MIN_ROWS_FOR_TEST = 20  # a KS-test below this is noise, not signal


def _row_to_columns(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    query = row.get("query") or ""
    if query:
        out["query_length"] = float(len(query))
    for col in ("top1_score", "mean_topk_score", "latency_ms"):
        val = row.get(col)
        if isinstance(val, (int, float)):
            out[col] = float(val)
    return out


def _extract_series(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {c: [] for c in _TRACKED_COLUMNS}
    for row in rows:
        cols = _row_to_columns(row)
        for c in _TRACKED_COLUMNS:
            if c in cols:
                series[c].append(cols[c])
    return series


def _psi(reference: list[float], current: list[float], buckets: int = 10) -> float | None:
    """Population Stability Index — a secondary, distribution-shape drift
    signal alongside the KS-test p-value. Bucketed on the REFERENCE
    distribution's own quantiles, so PSI is ~0 when current resembles
    reference regardless of the underlying value range. Informational only
    (not part of severity classification) — the KS-test p-value is the
    actual statistical test; PSI is here because it's the industry-standard
    number an operator reading this dashboard will expect to see."""
    if len(reference) < _MIN_ROWS_FOR_TEST or len(current) < _MIN_ROWS_FOR_TEST:
        return None
    import numpy as np

    ref = np.array(reference, dtype=float)
    cur = np.array(current, dtype=float)
    quantiles = np.linspace(0, 100, buckets + 1)
    edges = np.unique(np.percentile(ref, quantiles))
    if len(edges) < 3:
        return None  # reference has near-zero variance — bucketing is meaningless
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_frac = np.clip(ref_counts / max(len(ref), 1), 1e-4, None)
    cur_frac = np.clip(cur_counts / max(len(cur), 1), 1e-4, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def _compare_column(reference: list[float], current: list[float]) -> dict[str, Any]:
    if len(reference) < _MIN_ROWS_FOR_TEST or len(current) < _MIN_ROWS_FOR_TEST:
        return {"tested": False, "reason": "insufficient_rows"}

    from scipy import stats

    ks_result = stats.ks_2samp(reference, current)
    ref_mean = sum(reference) / len(reference)
    cur_mean = sum(current) / len(current)
    return {
        "tested": True,
        "p_value": float(ks_result.pvalue),
        "drifted": bool(ks_result.pvalue < 0.05),
        "psi": _psi(reference, current),
        "reference_mean": ref_mean,
        "current_mean": cur_mean,
        "degraded": cur_mean < ref_mean,
    }


def _classify_severity(comparisons: dict[str, dict[str, Any]]) -> tuple[str, float]:
    tested = {c: r for c, r in comparisons.items() if r.get("tested")}
    if not tested:
        return "info", 0.0

    drifted = [c for c, r in tested.items() if r["drifted"]]
    dataset_score = len(drifted) / len(tested)

    if dataset_score == 0:
        return "info", dataset_score

    quality_degraded = any(
        c in _QUALITY_COLUMNS and tested[c]["drifted"] and tested[c]["degraded"] for c in tested
    )

    if dataset_score >= settings.DRIFT_CRITICAL_THRESHOLD and quality_degraded:
        return "critical", dataset_score
    if dataset_score >= settings.DRIFT_WARNING_THRESHOLD:
        return "warning", dataset_score
    return "info", dataset_score


def _load_reference(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _severity_to_int(severity: str) -> int:
    return {"info": 0, "warning": 1, "critical": 2}.get(severity, 0)


def _push_gauges(
    severity: str,
    dataset_score: float,
    comparisons: dict[str, dict[str, Any]],
    ref_size: int,
    cur_size: int,
) -> None:
    from app.core import metrics as m

    m.drift_severity.set(_severity_to_int(severity))
    m.drift_dataset_score.set(dataset_score)
    m.drift_reference_size.set(ref_size)
    m.drift_current_size.set(cur_size)
    for column, result in comparisons.items():
        if result.get("tested"):
            m.drift_column_pvalue.labels(column=column).set(result["p_value"])


def run_drift_eval_once() -> dict[str, Any]:
    """Compare the current sampled-traffic window against the saved
    reference and push severity/per-column gauges. Never raises."""
    try:
        reference_rows = _load_reference(settings.DRIFT_REFERENCE_PATH)
        if not reference_rows:
            logger.info(event="drift_eval_skipped", reason="no_reference_yet")
            return {"skipped": "no_reference_yet"}

        from app.core.infra_registry import infra

        mongo = infra.get_mongo()
        if mongo is None or mongo.db is None:
            logger.info(event="drift_eval_skipped", reason="mongo_unavailable")
            return {"skipped": "mongo_unavailable"}

        coll = mongo.db[settings.MONGO_EVAL_SHADOW_COLLECTION]
        current_rows = list(coll.find({}).sort("sampled_at", -1).limit(settings.DRIFT_WINDOW_SIZE))
        if not current_rows:
            logger.info(event="drift_eval_skipped", reason="no_sampled_traffic")
            return {"skipped": "no_sampled_traffic"}

        reference_series = _extract_series(reference_rows)
        current_series = _extract_series(current_rows)

        comparisons = {
            col: _compare_column(reference_series[col], current_series[col])
            for col in _TRACKED_COLUMNS
        }
        severity, dataset_score = _classify_severity(comparisons)
        _push_gauges(severity, dataset_score, comparisons, len(reference_rows), len(current_rows))

        result = {
            "severity": severity,
            "dataset_score": dataset_score,
            "reference_size": len(reference_rows),
            "current_size": len(current_rows),
            "columns": comparisons,
        }
        logger.info(
            event="drift_eval_completed",
            severity=severity,
            dataset_score=round(dataset_score, 3),
            reference_size=len(reference_rows),
            current_size=len(current_rows),
        )
        return result
    except Exception as exc:
        logger.warning(event="drift_eval_failed", error=str(exc))
        return {"error": str(exc)}


def build_reference(window_size: int | None = None) -> dict[str, Any]:
    """Operator bootstrap/re-baseline action: snapshot the current sampled-
    traffic window into DRIFT_REFERENCE_PATH as the new reference.

    Run manually (`python -m app.eval.jobs.drift_eval --build-reference`)
    once there's been enough real live traffic (ONLINE_EVAL_SAMPLE_RATE > 0
    for a while) to consider the window "normal." Deliberately not automatic
    — an unattended re-baseline could silently absorb a real regression into
    the new "normal" and make it permanently invisible to this job.
    """
    try:
        from app.core.infra_registry import infra

        mongo = infra.get_mongo()
        if mongo is None or mongo.db is None:
            return {"error": "mongo_unavailable"}

        coll = mongo.db[settings.MONGO_EVAL_SHADOW_COLLECTION]
        size = window_size or settings.DRIFT_WINDOW_SIZE
        rows = list(coll.find({}).sort("sampled_at", -1).limit(size))
        if not rows:
            return {"error": "no_sampled_traffic"}

        path = Path(settings.DRIFT_REFERENCE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for row in rows:
                row = dict(row)
                row.pop("_id", None)  # ObjectId isn't JSON-serializable
                f.write(json.dumps(row, default=str) + "\n")

        logger.info(event="drift_reference_built", rows=len(rows), path=str(path))
        return {"rows_written": len(rows), "path": str(path)}
    except Exception as exc:
        logger.warning(event="drift_reference_build_failed", error=str(exc))
        return {"error": str(exc)}


async def run_drift_eval_loop() -> None:
    """Background task: same run-once-then-repeat convention as
    online_eval.py's run_online_eval_loop(), staggered 60s after it so both
    jobs don't contend for the same Mongo connection at the exact same
    startup instant."""
    if not settings.DRIFT_ENABLED or settings.ONLINE_EVAL_SAMPLE_RATE <= 0:
        return

    await asyncio.sleep(90)
    while True:
        await asyncio.to_thread(run_drift_eval_once)
        await asyncio.sleep(settings.DRIFT_CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    # Manual/debug entrypoint:
    #   python -m app.eval.jobs.drift_eval                  — score once, print result
    #   python -m app.eval.jobs.drift_eval --build-reference — (re)build the reference snapshot
    # Same caveat as online_eval.py's __main__: run this way, a fresh short-
    # lived process, it does NOT update the live dashboard (Prometheus
    # gauges live in-process) — use it to sanity-check the comparison logic
    # against real sampled data, or to bootstrap/re-baseline the reference.
    import sys

    if "--build-reference" in sys.argv:
        print(json.dumps(build_reference(), indent=2, default=str))
    else:
        print(json.dumps(run_drift_eval_once(), indent=2, default=str))
