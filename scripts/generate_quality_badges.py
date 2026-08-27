#!/usr/bin/env python
"""Generate shields.io "endpoint" badge JSON from the latest committed report
in each quality-reports/<tool>/ subdirectory, plus a copy-paste Markdown block
for the portfolio site.

Badges are honest by construction: a tool that hasn't been run yet — or whose
latest report carries a NaN/absent score — gets a gray "not yet measured"
badge, never a fabricated number.

Run this after each report you choose to commit. cd.yml's report-quality-metrics
job also runs it on every production promotion, to regenerate badges from the
reports the RAGAS/DeepEval steps just produced; that job publishes the result
rather than committing it, so the "never auto-commit" principle this whole
initiative rests on still holds (see quality-reports/README.md).

shields.io endpoint badges (https://shields.io/badges/endpoint-badge) fetch
this JSON straight from raw.githubusercontent.com at render time — so
quality-badges/ must be committed (it deliberately is NOT gitignored, see
.gitignore's note) and pushed for badges to render on GitHub/the portfolio.

Usage:
    python scripts/generate_quality_badges.py
    python scripts/generate_quality_badges.py --summary-md report.md   # + Markdown table
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "quality-reports"
BADGES_DIR = ROOT / "quality-badges"

GRAY = "lightgrey"
GREEN = "brightgreen"
YELLOW = "yellow"
RED = "red"

NOT_YET = {"schemaVersion": 1, "label": "", "message": "not yet measured", "color": GRAY}


def _latest_json(subdir: str) -> dict | None:
    """Newest report in quality-reports/<subdir>/, by the time it was GENERATED.

    Deliberately not st_mtime: a fresh `actions/checkout` stamps every committed
    file with the checkout time, so on the CI runner that ordering is arbitrary
    and a stale committed report can outrank the one this run just wrote. Both
    writers name their files `<YYYYmmdd-HHMMSS>-<mode>.json` and stamp a
    `generated_at` inside (app/eval/ragas_report.py::_write_report,
    app/eval/deepeval_suite.py::_write_report), so the content is authoritative
    and the filename is a sound tiebreak when it is missing or malformed.
    """
    d = REPORTS_DIR / subdir
    if not d.is_dir():
        return None

    loaded: list[tuple[str, str, dict]] = []
    for path in d.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        generated_at = data.get("generated_at")
        loaded.append((generated_at if isinstance(generated_at, str) else "", path.name, data))

    if not loaded:
        return None
    loaded.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return loaded[0][2]


def _finite(value: object) -> float | None:
    """The value as a float, or None if it cannot honestly be shown on a badge.

    MetricResult.empty() sets value=float("nan") (app/eval/metrics/base.py) and
    json.dump writes that out as a bare `NaN`, which json.loads reads straight
    back. Formatting it would render a red "nan" badge — a measurement claim
    where there is no measurement — so it degrades to "not yet measured".
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _badge(label: str, message: str, color: str) -> dict:
    return {"schemaVersion": 1, "label": label, "message": message, "color": color}


def _score_color(value: float, good: float, ok: float) -> str:
    if value >= good:
        return GREEN
    if value >= ok:
        return YELLOW
    return RED


def ragas_badge() -> dict:
    data = _latest_json("ragas")
    if not isinstance(data, dict):
        return {**NOT_YET, "label": "ragas"}
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return {**NOT_YET, "label": "ragas"}
    entry = metrics.get("faithfulness")
    faith = _finite(entry.get("value")) if isinstance(entry, dict) else None
    if faith is None:
        return {**NOT_YET, "label": "ragas"}
    mode = data.get("mode", "?")
    return _badge("ragas faithfulness", f"{faith:.2f} ({mode})", _score_color(faith, 0.7, 0.4))


def deepeval_badge() -> dict:
    data = _latest_json("deepeval")
    if not isinstance(data, dict):
        return {**NOT_YET, "label": "deepeval"}
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return {**NOT_YET, "label": "deepeval"}
    # summary[name]["mean"] is None whenever every row of that metric errored
    # (app/eval/deepeval_suite.py) — a metric with no successful rows must not
    # drag the average, so drop it rather than counting it as zero.
    means = [
        m
        for s in summary.values()
        if isinstance(s, dict) and (m := _finite(s.get("mean"))) is not None
    ]
    if not means:
        return {**NOT_YET, "label": "deepeval"}
    avg = sum(means) / len(means)
    mode = data.get("mode", "?")
    return _badge("deepeval avg", f"{avg:.2f} ({mode})", _score_color(avg, 0.7, 0.4))


def generic_not_yet(label: str) -> dict:
    # api-contract, performance, browser-performance, security-dast, uptime:
    # each tool's own output format (JUnit XML, k6 summary JSON, Lighthouse
    # JSON, ZAP JSON, Kuma's own status-page badge) is best parsed once
    # there's a real report on disk to inspect — until then, an honest
    # "not yet measured" beats a badge generator that guesses at a schema
    # it has never actually seen.
    return {**NOT_YET, "label": label}


def _summary_markdown(badges: dict[str, dict]) -> str:
    """A GitHub-step-summary table of what the badges now say.

    Built from the badge dicts that were just written, so the table and the
    badges can never disagree — including when a tool reports "not yet
    measured", which is stated as such rather than quietly omitted.
    """
    lines = ["", "| Tool | Result |", "| --- | --- |"]
    for filename, badge in badges.items():
        label = badge.get("label") or filename.removesuffix(".json")
        lines.append(f"| {label} | {badge.get('message', '?')} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate quality-badges/*.json")
    parser.add_argument(
        "--summary-md",
        metavar="PATH",
        default=None,
        help="Also append a Markdown results table to PATH (e.g. $GITHUB_STEP_SUMMARY)",
    )
    args = parser.parse_args()

    BADGES_DIR.mkdir(parents=True, exist_ok=True)

    badges = {
        "ragas.json": ragas_badge(),
        "deepeval.json": deepeval_badge(),
        "api-contract.json": generic_not_yet("api contract"),
        "performance.json": generic_not_yet("k6 load test"),
        "browser-performance.json": generic_not_yet("lighthouse"),
        "security-dast.json": generic_not_yet("zap dast"),
        "uptime.json": generic_not_yet("uptime"),
    }

    for filename, content in badges.items():
        (BADGES_DIR / filename).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")

    if args.summary_md:
        with open(args.summary_md, "a", encoding="utf-8") as fh:
            fh.write(_summary_markdown(badges))

    print(f"Wrote {len(badges)} badge JSON files to {BADGES_DIR}")
    print()
    # The hardcoded slug here used to be "vjkarthik98/multimodal-rag-assistant",
    # which is not this repository — every URL it printed 404'd, so any badge
    # pasted from this output rendered as shields.io's "invalid" placeholder.
    # GITHUB_REPOSITORY is set on every runner; the literal is the local fallback.
    repo = (
        os.getenv("GITHUB_REPOSITORY")
        or "vjkarthik98/MULTIMODAL-AGENTIC-RAG-INTEGRATED-KNOWLEDGE-AI-ASSISTANT"
    )
    branch = os.getenv("QUALITY_BADGES_BRANCH", "main")
    print("Copy-paste for README / portfolio (uses shields.io endpoint badges):")
    print()
    base = (
        "https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/"
        f"{repo}/{branch}/quality-badges"
    )
    for filename in badges:
        name = filename.removesuffix(".json")
        print(f"![{name}]({base}/{filename})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
