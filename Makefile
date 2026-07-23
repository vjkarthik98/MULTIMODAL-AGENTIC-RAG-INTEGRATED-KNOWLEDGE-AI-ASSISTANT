.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format typecheck test test-unit test-auth test-guardrails \
        eval-retrieval eval-full docker-build docker-run compose-up compose-down clean

PYTHON ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies only
	$(PYTHON) -m pip install -r requirements.txt

install-dev:  ## Install runtime + lint/type/test tooling (no ML model weights)
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

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

test:  ## Full non-slow suite (needs live Qdrant/Redis/Mongo — see .env)
	pytest tests/ -m "not slow" -q

test-unit:  ## Fast unit tests only — no external services, no real models (mocked)
	pytest tests/ -m unit -q --cov=app --cov-report=term-missing

test-auth:  ## Tenant-isolation / auth test suite
	pytest tests/ -m auth -v

test-guardrails:  ## Red-team injection/guardrail suite
	pytest tests/ -m guardrails -v

test-randomized:  ## Audit for hidden test-order dependencies (NOT run in CI by default - see pyproject.toml comment)
	pytest tests/ -m unit -p randomly -q

eval-retrieval:  ## Tier-1 gate: retrieval-only, no LLM, CPU-fine (~3GB RAM). Exit code IS the gate — no separate --gate flag exists.
	$(PYTHON) -m app.eval.run --suite retrieval

eval-full:  ## Tier-2: full generation + judge suite — needs a LIVE server (generation calls route via HTTP) + the resident 17GB+ model stack.
	@echo "WARNING: this suite calls the running API's /rag/query over HTTP for"
	@echo "generation (see app/eval/run.py) — start_server.py must already be up."
	@echo "It also needs the full resident model stack. Will not fit on a laptop or"
	@echo "a standard CI runner — run this only on the real GPU box. See docs/runbooks/ci-cd.md."
	$(PYTHON) -m app.eval.run --suite full

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
