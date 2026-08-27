.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format typecheck test test-unit test-auth test-guardrails \
        test-randomized integration eval-retrieval eval-full benchmark security-scan sbom \
        release docker-build docker-run compose-up compose-down clean \
        api-contract ragas-report deepeval-report seed-test-tenants k6-smoke k6-stress \
        k6-multiuser lighthouse zap-baseline

PYTHON ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies only
	$(PYTHON) -m pip install -r requirements.txt

install-dev:  ## Install runtime + lint/type/test tooling (no ML model weights)
	$(PYTHON) -m pip install -e ".[dev]"
	# Deliberately does NOT run `pre-commit install`. Commits are not gated
	# locally — `git commit` must just commit. Lint (ruff/black/isort), types
	# and unit tests are enforced in CI on every PR (.github/workflows/ci.yml),
	# which is the single source of truth for whether a change is clean.
	# .pre-commit-config.yaml is kept for anyone who wants to run the same
	# checks on demand via `make lint` or `pre-commit run --all-files`.

lint:  ## Ruff + black --check + isort --check (matches ci.yml exactly; ui/ is JS/React, not linted here)
	ruff check app/
	black --check app/
	isort --check app/

format:  ## Auto-fix formatting/import order
	ruff check --fix app/
	black app/
	isort app/

typecheck:  ## mypy over app/
	mypy app/

# tests/integration/ RESOLVED: was 27 of 42 files dead (stale APIs, some
# referencing a pre-refactor `src.rag_system.*` package that no longer
# exists) — those 27 are deleted, not just ignored (see docs/runbooks/
# ci-cd.md for the full writeup). The 13 that remain are verified clean:
# `pytest tests/integration/ --ignore=.../test_document_pipeline.py` gives
# 0 failures (12 files run for real; the llama-server-dependent gap/smoke
# tests correctly SKIP rather than hang when no server is up, via the
# shared tests/integration/conftest.py::requires_llama_server marker).
# test_document_pipeline.py stays excluded on its own: `import magic`
# hangs (not just fails) on this Windows dev machine because libmagic isn't
# installed — a real Windows-only environment gap, not a code bug; Linux
# CI/EC2 ship libmagic system-wide and shouldn't hit this.
test:  ## Full non-slow suite (needs live Qdrant/Redis/Mongo — see .env)
	pytest tests/ -m "not slow" --ignore=tests/integration/test_document_pipeline.py -q

test-unit:  ## Fast unit tests only — no external services, no real models (mocked). Scoped to tests/unit/ directly (not tests/ + -m unit) — see ci.yml comment for why.
	# --cov-fail-under=0 matches ci.yml exactly: pyproject.toml's fail_under=70
	# is measured against the FULL app/ package, but tests/unit/ alone only
	# ever covers ~27% of it — without this override, a run that's identical
	# to what CI does still fails locally on the coverage gate, not on an
	# actual test failure. Confirmed live (2026-08-01): this exact mismatch
	# made a genuinely clean local run look red.
	# `unit and not slow`, matching ci.yml: tests/conftest.py auto-marks every
	# file under tests/unit/ as `unit` by directory, so a bare `-m unit` also
	# picks up the two `pytest.mark.slow` files that load the REAL NLI model —
	# contradicting this target's own "no real models (mocked)" contract.
	pytest tests/unit/ -m "unit and not slow" -q --cov=app --cov-report=term-missing --cov-fail-under=0

test-auth:  ## Tenant-isolation / auth test suite. Scoped directly to tests/auth/, not `tests/ -m auth` — same tests/integration/ collection-storm issue as test-unit.
	pytest tests/auth/ -v

test-guardrails:  ## Red-team injection/guardrail suite. Scoped directly to tests/guardrails/ — same reason as test-auth above.
	pytest tests/guardrails/ -v

test-randomized:  ## Audit for hidden test-order dependencies (NOT run in CI by default - see pyproject.toml comment)
	pytest tests/unit/ -m unit -p randomly -q

# Unblocked as of the tests/integration/ cleanup (42 -> 13 files, all verified
# — see the `test` target's comment above and docs/runbooks/ci-cd.md). The
# llama-server-dependent files SKIP cleanly rather than hang when no server is
# running, via tests/integration/conftest.py::requires_llama_server, so this is
# safe to run with or without a live stack. NOT wired into ci.yml: these need
# live Qdrant/Redis/Mongo, which hosted runners don't have.
integration:  ## Integration suite (skips llama-server tests if no server; needs live Qdrant/Redis/Mongo)
	pytest tests/integration/ --ignore=tests/integration/test_document_pipeline.py -v

eval-retrieval:  ## Tier-1 gate: retrieval-only, no LLM, CPU-fine (~3GB RAM). Exit code IS the gate — no separate --gate flag exists.
	$(PYTHON) -m app.eval.run --suite retrieval

# Not a separate harness — deliberately the same Tier-1 run as eval-retrieval,
# re-read through a latency lens. app/eval/run.py already records p50/p95/p99
# per suite into rag_report.json; this surfaces those instead of the quality
# metrics. Writing a second, parallel benchmark runner would mean two code
# paths that could disagree about what "a retrieval call" costs.
benchmark:  ## Latency percentiles (p50/p95/p99) from a real Tier-1 retrieval run
	$(PYTHON) -m app.eval.run --suite retrieval
	@$(PYTHON) -c "import json; \
d = json.load(open('app/eval/reports/rag_report.json')); \
s = d['suites']; \
print(); \
print('LATENCY  (git_sha=%s  generated_at=%s)' % (d.get('git_sha','?'), d.get('generated_at','?'))); \
print('-' * 60); \
[print('  %-34s %8.4f s  (n=%s)' % (k, v['value'], v['n'])) \
 for suite in s.values() for k, v in sorted(suite.get('metrics', {}).items()) if '_sec' in k]"

eval-full:  ## Tier-2: full generation + judge suite — needs a LIVE server (generation calls route via HTTP) + the resident 17GB+ model stack.
	@echo "WARNING: this suite calls the running API's /rag/query over HTTP for"
	@echo "generation (see app/eval/run.py) — start_server.py must already be up."
	@echo "It also needs the full resident model stack. Will not fit on a laptop or"
	@echo "a standard CI runner — run this only on the real GPU box. See docs/runbooks/ci-cd.md."
	$(PYTHON) -m app.eval.run --suite full

# Local mirror of .github/workflows/security.yml, minus the image scan (that
# needs a built image — see `docker-build` + the Trivy step in cd.yml).
# Same flags as CI so a local pass means a CI pass.
security-scan:  ## detect-secrets + Bandit (HIGH blocks) + pip-audit + license check, as CI runs them
	detect-secrets-hook --baseline .secrets.baseline $$(git ls-files)
	bandit -r app/ -x tests,ui,.venv,rag_env -s B104,B108 -lll -q
	-pip-audit -r requirements.txt
	@$(PYTHON) -c "import json,subprocess,sys; \
pkgs = json.loads(subprocess.run(['pip-licenses','--format=json'],capture_output=True,text=True).stdout); \
banned = ('AGPL','SSPL','Server Side Public License'); \
hits = [p for p in pkgs if any(b in (p.get('License') or '') for b in banned)]; \
[print('[BANNED LICENSE] %s %s: %s' % (p['Name'],p['Version'],p['License'])) for p in hits]; \
sys.exit(1 if hits else 0)"
	@echo "Local security scan clean."

# CycloneDX here (Python env), SPDX in cd.yml (container image via Syft) —
# deliberately different scopes, not a mismatch: this answers "what's in my
# venv right now", cd.yml's answers "what shipped in the artifact". Verified
# locally: 400 components, CycloneDX 1.6.
sbom:  ## CycloneDX SBOM of the current Python environment (cd.yml SBOMs the image separately)
	$(PYTHON) -m pip install --quiet --upgrade cyclonedx-bom
	$(PYTHON) -m cyclonedx_py environment -o sbom.local.json
	@echo "Wrote sbom.local.json"

# ── Quality & Performance Reporting ──────────────────────────────────────────
# Local-mode targets — all assume `docker compose up -d api qdrant redis mongo`
# is already running. Live-mode equivalents (against the real deployed URL)
# are manual/on-demand only — see perf/k6/README.md, scripts/schemathesis_live.sh,
# security/zap/README.md, and docs/reports/browser-performance/README.md.
# Never wired into a scheduled workflow — see .github/workflows/quality-live.yml.

api-contract:  ## Schemathesis property-based API contract test (GET-only, safe)
	pytest tests/api_contract/ -v

ragas-report:  ## Standalone RAGAS report (real ragas==0.1.21, local judge) — see app/eval/ragas_report.py
	$(PYTHON) -m app.eval.ragas_report

deepeval-report:  ## DeepEval suite, second-opinion LLM eval, local GGUF judge — see app/eval/deepeval_suite.py
	$(PYTHON) -m app.eval.deepeval_suite

seed-test-tenants:  ## Provision dedicated is_load_test accounts for k6/live-mode tooling (never the public demo login)
	$(PYTHON) -m app.bin.seed_test_tenants --count 10

k6-smoke:  ## k6 sanity check — see perf/k6/README.md for one-time tenant setup
	k6 run perf/k6/smoke.js

k6-stress:  ## k6 ramping load test, local target only
	k6 run perf/k6/stress.js

k6-multiuser:  ## k6 multi-tenant concurrency test — asserts zero cross-tenant leakage under load
	k6 run --vus 8 --iterations 24 perf/k6/multi_user_tenant.js

lighthouse:  ## Browser performance (Core Web Vitals) against the built UI, served locally by lhci
	npm --prefix ui run build
	npx --yes @lhci/cli@0.14 autorun --config=lighthouserc.json

zap-baseline:  ## OWASP ZAP passive baseline DAST scan against the local docker-compose stack
	bash security/zap/run_baseline.sh

# Preflight only — it deliberately does NOT cut the release. Tagging is done by
# .github/workflows/release.yml (Actions tab -> Release -> Run workflow), which
# owns bumping VERSION/pyproject.toml, inserting the CHANGELOG.md section
# (mode=prepare, from development -> PR into main) and then tagging and
# publishing (mode=tag, from main). A `make release` that pushed tags from a
# laptop would be a second, divergent release path. What this DOES do is catch
# the mistakes that make a release messy, before you trigger it.
#
# The changelog check below looks for a line-anchored `## [X.Y.Z]` heading. It
# used to search for the bare substring 'v'+version, which no longer appears
# anywhere: CHANGELOG.md was normalised in v0.32.0 to `## [X.Y.Z] - date` with
# no 'v' inside the brackets, so this check had been failing every release as a
# false negative.
release:  ## Preflight the repo for a release (does not tag — release.yml does that)
	@$(PYTHON) -c "import re,subprocess,sys; \
fail=[]; \
ver=open('VERSION').read().strip(); \
pyproj=re.search(r'(?m)^version\s*=\s*\"([^\"]+)\"', open('pyproject.toml',encoding='utf-8').read()).group(1); \
ver==pyproj or fail.append('VERSION (%s) != pyproject.toml version (%s)' % (ver,pyproj)); \
dirty=subprocess.run(['git','status','--porcelain'],capture_output=True,text=True).stdout.strip(); \
dirty and fail.append('working tree is dirty (%d files) - commit or stash first' % len(dirty.splitlines())); \
tags=subprocess.run(['git','tag'],capture_output=True,text=True).stdout.split(); \
('v'+ver) in tags and fail.append('tag v%s already exists - bump VERSION' % ver); \
re.search(r'(?m)^\#\# \[%s\]' % re.escape(ver), open('CHANGELOG.md',encoding='utf-8').read()) or fail.append('CHANGELOG.md has no [%s] section' % ver); \
[print('  FAIL  '+f) for f in fail]; \
print('  ok    version %s consistent, tree clean, tag free, changelog present' % ver) if not fail else None; \
print(); \
print('Next: GitHub -> Actions -> Release -> Run workflow (mode=tag, version=%s) from main' % ver) if not fail else None; \
sys.exit(1 if fail else 0)"

docker-build:  ## Build the production (CUDA) image explicitly — see Dockerfile stage comments
	docker build --target runtime -t magik:local .

docker-run:  ## Run the production image locally (CPU fallback unless --gpus all is available)
	docker run --rm -p 8000:8000 --env-file .env magik:local

compose-up:  ## Local CPU-only dev stack: API + Qdrant + Redis + Mongo (no models needed for pytest -m unit)
	docker compose up --build

compose-down:  ## Tear down the local dev stack
	docker compose down -v

clean:  ## Remove caches and build artifacts (never touches .hf_cache/ or data/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml build dist *.egg-info
	find . -name "__pycache__" -not -path "./.git/*" -type d -prune -exec rm -rf {} +
