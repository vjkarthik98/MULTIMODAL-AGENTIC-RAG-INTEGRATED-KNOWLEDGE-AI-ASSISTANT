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
- **Produced by CI but still not committed**: `quality-report.yml` runs RAGAS +
  DeepEval **on demand** (`workflow_dispatch`) against the **staging** box, and
  attaches `quality-reports/ragas`, `quality-reports/deepeval` and
  `quality-badges/` to the run as artifacts (90-day retention), with the
  headline numbers in the job summary. Download it, and `git add` the reports
  worth keeping — on `development`, like any other change.

  Two pieces of history are worth keeping in view here. An early version of
  that job committed and pushed straight to `main`; that both contradicted the
  rule above and could not have worked (`main` is protected by required status
  checks, and the commit was marked `[skip ci]`), so it was removed. And until
  v1.0.0 the whole thing ran automatically after every promotion, against
  **live production**, on a runner registered on the production box — where
  DeepEval's judge OOM-killed the runner and took it offline mid-release. It is
  manual and staging-only now: reports of this kind gate nothing, so they have
  no business running unattended next to real traffic. See that workflow's
  header for the full post-mortem.

## Subdirectories

| Directory | Tool | Mode |
|---|---|---|
| `api-contract/` | Schemathesis | local (CI) + live (manual) |
| `ragas/` | Ragas (`app/eval/ragas_report.py`) | staging (manual, `quality-report.yml`) |
| `deepeval/` | DeepEval (`app/eval/deepeval_suite.py`) | staging (manual, `quality-report.yml`) |
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
is how `quality-report.yml`'s `badges` job fills its job summary.

A tool with no report, an unparseable report, or a `NaN` score (what
`MetricResult.empty()` writes when a metric had nothing to measure) gets the
gray "not yet measured" badge. There is no path through this script that
invents a number.
