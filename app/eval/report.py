"""Write eval results to rag_report.{json,md} in app/eval/reports/.

Used by run.py after every eval run.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.eval.config import EvalConfig, load_config
from app.eval.metrics.base import SuiteResult


def write_reports(
    results: Dict[str, SuiteResult],
    cfg: Optional[EvalConfig] = None,
    weaken_spec: Optional[str] = None,
    git_sha: Optional[str] = None,
) -> tuple[Path, Path]:
    """Write rag_report.json and rag_report.md. Returns (json_path, md_path)."""
    cfg = cfg or load_config()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha or _get_git_sha(),
        "weaken_spec": weaken_spec,
        "suites": {name: sr.to_dict() for name, sr in results.items()},
    }

    json_path = cfg.reports_dir / "rag_report.json"
    md_path = cfg.reports_dir / "rag_report.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    _write_markdown(report, results, md_path)

    print(f"\n[REPORT] JSON: {json_path}")
    print(f"[REPORT] MD:   {md_path}")
    return json_path, md_path


def _write_markdown(
    report: Dict[str, Any],
    results: Dict[str, SuiteResult],
    md_path: Path,
) -> None:
    lines = [
        "# RAG Eval Report — Phase 25",
        "",
        f"**Generated:** {report['generated_at']}  ",
        f"**Git SHA:** {report.get('git_sha', 'unknown')}  ",
    ]
    if report.get("weaken_spec"):
        lines += [f"**⚠ Weakened pipeline:** `{report['weaken_spec']}`  ", ""]

    for suite_name, sr in results.items():
        lines += [f"", f"## Suite: `{suite_name}`", ""]
        lines += [f"Duration: {sr.duration_sec:.1f}s", ""]

        if sr.metrics:
            lines += ["| Metric | Value | n | Notes |", "|--------|-------|---|-------|"]
            for name, m in sorted(sr.metrics.items()):
                val = f"{m.value:.4f}" if not math.isnan(m.value) else "nan"
                lines.append(f"| `{name}` | {val} | {m.n} | {m.notes[:60] if m.notes else ''} |")
            lines.append("")

        if sr.breached:
            lines += ["### Breaches / Errors", ""]
            for k, v in sr.breached.items():
                lines.append(f"- `{k}`: {v}")
            lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _get_git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"
