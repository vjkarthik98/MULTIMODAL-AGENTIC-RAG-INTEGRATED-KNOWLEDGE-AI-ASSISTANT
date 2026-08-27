# Quality & Performance Reports

Output of the testing & quality reporting initiative — API contract testing,
RAGAS, DeepEval, k6 load/stress/multi-user simulation, Lighthouse browser
performance, OWASP ZAP DAST, and Uptime Kuma. All open-source tooling; see
each subdirectory and the tool's own script/config for details.

**This directory is tracked in git deliberately** — unlike `docs/` (gitignored
except `.gitkeep`; see `.gitignore`'s "Folders tracked as structure only"
section, used for local/regenerable working notes), reports here exist
specifically to be visible on GitHub and linkable from the README and the
portfolio site.

## What gets committed here vs. what doesn't

- **Committed**: a small number of dated snapshot reports (`.md` + `.json`)
  from runs *you* chose to keep — typically live-mode runs, or a local run
  worth citing as a baseline. Committing is always a manual `git add`, never
  automatic.
- **Not committed** (gitignored): raw per-run tool output that's noisy or
  huge (Lighthouse's `.lhci/` folders, k6's raw JSON trace output, ZAP's raw
  scan XML). `quality.yml` (local-mode, every PR) generates these fresh each
  run and uploads them as a GitHub Actions artifact — it never commits
  anything back to the repo (see CLAUDE.md: never commit without explicit
  instruction — that applies doubly to CI).
- **Produced by CI but still not committed**: `cd.yml`'s
  `report-quality-metrics` job runs RAGAS + DeepEval against live production
  after each promotion and attaches `quality-reports/ragas`,
  `quality-reports/deepeval` and `quality-badges/` to the run as the
  `quality-report-<tag>` artifact (90-day retention), with the headline numbers
  in the job summary. Download it, and `git add` the reports worth keeping —
  on `development`, like any other change. An earlier version of that job
  committed and pushed straight to `main`; that both contradicted the rule
  above and could not have worked (`main` is protected by required status
  checks, and the commit was marked `[skip ci]`), so it was removed.

## Subdirectories

| Directory | Tool | Mode |
|---|---|---|
| `api-contract/` | Schemathesis | local (CI) + live (manual) |
| `ragas/` | Ragas (`app/eval/ragas_report.py`) | live by default |
| `deepeval/` | DeepEval (`app/eval/deepeval_suite.py`) | live by default |
| `performance/` | k6 | local (CI) + live (manual) |
| `browser-performance/` | Lighthouse / Lighthouse CI | local (CI) + live (manual) |
| `security-dast/` | OWASP ZAP | local baseline (CI) + live baseline/active (manual) |
| `uptime/` | Uptime Kuma | live only, passive push (see `deploy/aws/lambda/`) |

## Generating the README/portfolio summary

`scripts/generate_quality_badges.py` reads the newest report in each
subdirectory — by the `generated_at` stamp inside it, never file mtime, which
a fresh CI checkout flattens — and writes `quality-badges/*.json` (shields.io
endpoint schema) plus prints a copy-paste Markdown block for the portfolio
site. `--summary-md PATH` appends the same results as a Markdown table, which
is how `cd.yml` fills its job summary.

A tool with no report, an unparseable report, or a `NaN` score (what
`MetricResult.empty()` writes when a metric had nothing to measure) gets the
gray "not yet measured" badge. There is no path through this script that
invents a number.
