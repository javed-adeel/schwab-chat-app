.DEFAULT_GOAL := help
PY ?= python

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n",$$1,$$2}'

install:  ## install with dev tooling
	$(PY) -m pip install -e ".[dev]"

run:  ## start the web app on http://127.0.0.1:8000
	$(PY) -m newschat

lint:  ## ruff lint + format check
	ruff check .
	ruff format --check .

format:  ## auto-format
	ruff format .
	ruff check . --fix

typecheck:  ## mypy --strict
	mypy src tests

test:  ## unit + integration + eval tests, with coverage gate
	$(PY) -m pytest

eval:  ## retrieval golden-set report with regression thresholds
	$(PY) -m newschat.evaluation.cli --min-hit 0.95 --min-mrr 0.85 --min-gate 1.0

ablation:  ## compare BM25-only / dense-only / hybrid retrieval
	$(PY) -m newschat.evaluation.cli --ablation

check: lint typecheck test eval  ## everything CI runs

docker-build:  ## build the image
	docker build -t newschat .

docker-run:  ## run the image (passes through any provider key set in your shell)
	docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY -e GEMINI_API_KEY -e OPENROUTER_API_KEY -e NEWSCHAT_LLM_PROVIDER newschat

.PHONY: help install run lint format typecheck test eval ablation check docker-build docker-run
