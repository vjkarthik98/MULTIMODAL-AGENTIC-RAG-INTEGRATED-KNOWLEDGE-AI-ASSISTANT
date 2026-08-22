# Contributing to MAGIK

Thanks for taking a look at this project. This guide covers the workflow,
conventions, and checks a change needs to pass — whether you're an external
contributor or this is you, six months from now, having forgotten the
details.

For what the project *is* and how to run it locally, start with the
[README](README.md#getting-started) — this file only covers the contribution
workflow itself.

## Before you start

For anything beyond a trivial fix, open an issue first describing the
problem or proposal. It avoids duplicated work and lets the change get
scoped before code is written — especially relevant here since several
subsystems (guardrails, tenant isolation, the eval gate) have hard
correctness requirements that aren't obvious from the code alone; see the
"Hard rules" section of [CLAUDE.md](CLAUDE.md) for the constraints that apply
to any change.

## Branch workflow

- `main` is protected and release-only. It only receives merges via PR from
  `development`, tagged `v*` on release (which triggers `cd.yml` — a real
  production deploy).
- `development` is the integration branch. Branch off it, open your PR back
  into it.
- Branch naming: `<type>/<short-description>` (e.g. `fix/qdrant-filter-leak`,
  `feat/sec-edgar-tool`) mirrors the commit prefixes below.

## Commit conventions

This repo follows [Conventional Commits](https://www.conventionalcommits.org/)
style, consistent with the existing git history:

```
feat: add financial_calculator tool to agent routing
fix: correct off-by-one in chunk overlap
ci: bump actions/checkout to v5
style: apply black formatting
docs: add deployment runbook for GPU idle-stop Lambda
deps: bump ragas to 0.1.22
refactor: extract shared BM25 tokenizer logic
```

Keep the subject line under ~72 characters; put the "why" in the body if it
isn't obvious from the diff — see [CHANGELOG.md](CHANGELOG.md) for the level
of detail this project expects when a change fixes a real bug (root cause,
not just symptom).

## Before opening a PR

```bash
make lint          # Ruff + black --check + isort --check — matches ci.yml exactly
make typecheck      # mypy over app/ — informational (see note below), but check new code is clean
make test-unit       # fast, no external services or real models
```

Pre-commit hooks are configured (`.pre-commit-config.yaml`) but **not**
installed by `make install-dev` deliberately — `git commit` stays fast and
local; the same checks run in CI on every PR regardless. Run them on demand
with `pre-commit run --all-files` if you want the same feedback before
pushing.

**mypy note:** `app/` carries pre-existing type debt (tracked in
`docs/runbooks/phase29-plan.md`). mypy runs on every PR as a visible,
non-blocking check — new code should still be clean; don't add to the
existing baseline.

If your change touches ingestion, chunking, embedding, or retrieval, also
run the relevant eval suite before opening the PR — a passing test suite
does not catch a retrieval-quality regression:

```bash
make eval-retrieval   # Tier-1 gate — CPU-only, no LLM, this is what CI blocks merges on
```

## What CI actually gates

| Check | Blocks merge? |
|---|---|
| `ci.yml` (lint, mypy*, unit tests) | Lint + unit tests yes, mypy informational |
| `eval-gate.yml` Tier-1 (retrieval regression) | Yes |
| `eval-gate.yml` Tier-2 (full generation + judge) | No — informational, self-hosted GPU runner |
| `security.yml` (secrets, CVEs, SAST, container scan) | Yes on secret detection; others informational today |
| `quality.yml` (API contract, k6 smoke, ZAP baseline, Lighthouse) | No — informational, see `quality-reports/README.md` |

If you're unsure whether a change needs a new test, err toward adding one —
particularly for anything touching `app/guardrails/`, `app/auth/`, or a
per-modality ingestion/chunking/embedding file (each modality's four files
are isolated by design; a test should confirm your change doesn't leak
across that boundary).

## Code style

- Python: Black + isort + Ruff, enforced in CI — don't hand-format, run
  `make format`.
- No new abstractions, feature flags, or "just in case" error handling for
  states that can't occur — see the engineering standards this project holds
  itself to in `.claude/skills/orchestrator/references/engineering-standards.md`.
- Never hardcode a secret, model name, chunk size, or threshold — these flow
  through `app/core/config.py`'s `Settings` class. See the Config section of
  [CLAUDE.md](CLAUDE.md) for what belongs in `.env` versus a code default.

## Reporting a bug vs. reporting a vulnerability

Regular bugs: open a GitHub issue.
Security vulnerabilities (auth bypass, injection bypass, tenant-isolation
break, secret exposure): do **not** open a public issue — see
[SECURITY.md](SECURITY.md) for the private disclosure process.
