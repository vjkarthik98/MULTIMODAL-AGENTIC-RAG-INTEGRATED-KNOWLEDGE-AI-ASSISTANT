"""Render a RAGAS / DeepEval report JSON as Prometheus text for the Pushgateway.

deploy/aws/scripts/run_quality_report.sh runs this inside the eval container
after the report is written, copies the output to the host, and PUTs it to
the box's Pushgateway (127.0.0.1:9091) under

    job=magik_quality_report / tool=<ragas|deepeval> / environment=<env>

so each tool's latest run replaces only its own series. The Grafana row
"Offline judge — RAGAS / DeepEval" in monitoring/grafana/dashboards/
rag_quality.json reads these.

Honesty rules are the same as scripts/generate_quality_badges.py's, applied
here as numbers instead of badge text, so a dashboard can never show a score
the badge would refuse to publish without also showing why:
  * magik_quality_judge_ok is 0 when any RAGAS metric came from the lexical
    fallback rather than the Qwen judge.
  * magik_quality_coverage is the graded fraction (RAGAS: faithfulness n over
    n_queries; DeepEval: filled metric-row slots over all slots). Panels colour
    on it, so a 1-row mean is visibly a 1-row mean.
Non-finite values are skipped, never written as NaN, so a metric that did not
score leaves a gap rather than a misleading zero.

Usage:
    python -m app.eval.quality_push --tool ragas --out /app/quality-reports/ragas.prom
    (picks the newest JSON under quality-reports/<tool>/ unless --report is given)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

REPORTS_ROOT = Path(__file__).resolve().parents[2] / "quality-reports"


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _label_value(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", " ").replace('"', '\\"')


def _line(name: str, labels: dict[str, str], value: float) -> str:
    rendered = ",".join(f'{k}="{_label_value(v)}"' for k, v in labels.items())
    return f"{name}{{{rendered}}} {value:.6g}" if rendered else f"{name} {value:.6g}"


def _ragas_series(data: dict) -> tuple[list[tuple[str, float, int | None]], float, bool, int, int]:
    """(metric, value, n) triples plus coverage, judge_ok, rows, errors."""
    metrics = data.get("metrics") or {}
    series: list[tuple[str, float, int | None]] = []
    judge_ok = True
    for name, entry in metrics.items():
        if not isinstance(entry, dict):
            continue
        if "lexical_fallback" in str(entry.get("notes") or ""):
            judge_ok = False
        val = _finite(entry.get("value"))
        if val is not None:
            n = entry.get("n")
            series.append((name, val, n if isinstance(n, int) else None))
    for extra in ("finance_fidelity", "hallucination_rate"):
        val = _finite(data.get(extra))
        if val is not None:
            series.append((extra, val, None))

    rows = int(data.get("n_queries") or 0)
    faith = metrics.get("faithfulness")
    faith_n = faith.get("n") if isinstance(faith, dict) else None
    coverage = (faith_n / rows) if rows and isinstance(faith_n, int) else 0.0
    if not metrics:
        judge_ok = False
    return series, coverage, judge_ok, rows, int(data.get("n_errors") or 0)


def _deepeval_series(
    data: dict,
) -> tuple[list[tuple[str, float, int | None]], float, bool, int, int]:
    summary = data.get("summary") or {}
    rows = len(data.get("per_row") or [])
    series: list[tuple[str, float, int | None]] = []
    filled = 0
    for name, s in summary.items():
        if not isinstance(s, dict):
            continue
        n = int(s.get("n") or 0)
        val = _finite(s.get("mean"))
        if val is not None:
            series.append((name, val, n))
            filled += n
    slots = rows * len(summary)
    coverage = (filled / slots) if slots else 0.0
    # Same bar as the DeepEval badge: a majority of metrics must have scored.
    judge_ok = bool(series) and len(series) * 2 >= len(summary)
    return series, coverage, judge_ok, rows, len(data.get("errors") or [])


def render(data: dict, tool: str, now: float | None = None) -> str:
    if tool == "ragas":
        series, coverage, judge_ok, rows, errors = _ragas_series(data)
    elif tool == "deepeval":
        series, coverage, judge_ok, rows, errors = _deepeval_series(data)
    else:
        raise ValueError(f"unknown tool {tool!r}")

    out = [
        "# HELP magik_quality_metric Offline-judge quality score (RAGAS / DeepEval).",
        "# TYPE magik_quality_metric gauge",
    ]
    for name, value, _ in series:
        out.append(_line("magik_quality_metric", {"metric": name}, value))

    out += [
        "# HELP magik_quality_metric_rows Rows that produced a score for this metric.",
        "# TYPE magik_quality_metric_rows gauge",
    ]
    for name, _, n in series:
        if n is not None:
            out.append(_line("magik_quality_metric_rows", {"metric": name}, float(n)))

    out += [
        "# HELP magik_quality_rows_evaluated Gold rows answered by the server.",
        "# TYPE magik_quality_rows_evaluated gauge",
        _line("magik_quality_rows_evaluated", {}, float(rows)),
        "# HELP magik_quality_row_errors Gold rows that failed before grading.",
        "# TYPE magik_quality_row_errors gauge",
        _line("magik_quality_row_errors", {}, float(errors)),
        "# HELP magik_quality_coverage Graded fraction of the run (0-1).",
        "# TYPE magik_quality_coverage gauge",
        _line("magik_quality_coverage", {}, float(coverage)),
        "# HELP magik_quality_judge_ok 1 if scores came from the Qwen judge, 0 if not.",
        "# TYPE magik_quality_judge_ok gauge",
        _line("magik_quality_judge_ok", {}, 1.0 if judge_ok else 0.0),
        "# HELP magik_quality_last_run_timestamp_seconds Unix time the report was pushed.",
        "# TYPE magik_quality_last_run_timestamp_seconds gauge",
        _line(
            "magik_quality_last_run_timestamp_seconds",
            {},
            float(now if now is not None else time.time()),
        ),
    ]
    return "\n".join(out) + "\n"


def _latest_report(tool: str) -> Path:
    candidates = sorted((REPORTS_ROOT / tool).glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"no {tool} report under {REPORTS_ROOT / tool}")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tool", required=True, choices=["ragas", "deepeval"])
    parser.add_argument("--report", default=None, help="report JSON (default: newest)")
    parser.add_argument("--out", required=True, help="where to write the .prom text")
    args = parser.parse_args()

    path = Path(args.report) if args.report else _latest_report(args.tool)
    data = json.loads(path.read_text())
    Path(args.out).write_text(render(data, args.tool))
    print(f"[quality-push] {args.tool}: {path.name} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
