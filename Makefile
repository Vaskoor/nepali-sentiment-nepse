# =============================================================================
# Nepali Sentiment NEPSE — Makefile
# Common dev tasks. Run `make help` for the list.
# =============================================================================
.PHONY: help install install-dev lint format typecheck test pipeline run-phase1 \
        run-phase2 run-phase3 run-phase4 run-phase5 api dashboard docker-build \
        docker-up docker-down clean

PYTHON ?= python3
PIP ?= pip3
PORT_API ?= 8000
PORT_DASH ?= 8501

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\n\nTargets:\n"} \
	      /^[a-zA-Z0-9_/-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Install runtime dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:  ## Install dev/test dependencies
	$(PIP) install -e ".[dev]"

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check src tests scripts api dashboard || true

format:  ## Format with black + ruff
	$(PYTHON) -m black src tests scripts api dashboard || true
	$(PYTHON) -m ruff format src tests scripts api dashboard || true

typecheck:  ## Static type-check with mypy
	$(PYTHON) -m mypy src || true

test:  ## Run pytest unit tests
	$(PYTHON) -m pytest tests -q

pipeline:  ## Run the full local pipeline end-to-end (Phase 1 -> 5)
	$(PYTHON) scripts/phase1_preprocess.py
	$(PYTHON) scripts/phase2_benchmark.py
	$(PYTHON) scripts/phase3_hybrid.py
	$(PYTHON) scripts/phase4_predict.py
	$(PYTHON) scripts/phase5_backtest.py
	@echo "Pipeline complete. See results/"

run-phase1:  ## Phase 1 — preprocess
	$(PYTHON) scripts/phase1_preprocess.py
run-phase2:  ## Phase 2 — Jev benchmark
	$(PYTHON) scripts/phase2_benchmark.py
run-phase3:  ## Phase 3 — hybrid pipeline
	$(PYTHON) scripts/phase3_hybrid.py
run-phase4:  ## Phase 4 — NEPSE prediction
	$(PYTHON) scripts/phase4_predict.py
run-phase5:  ## Phase 5 — backtest
	$(PYTHON) scripts/phase5_backtest.py

api:  ## Run FastAPI service
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port $(PORT_API) --reload

dashboard:  ## Run Streamlit dashboard
	$(PYTHON) -m streamlit run dashboard/app.py --server.port $(PORT_DASH)

docker-build:  ## Build Docker image
	docker build -t nepali-sentiment-nepse:latest .

docker-up:  ## Start stack via docker-compose
	docker compose up -d

docker-down:  ## Stop docker-compose stack
	docker compose down

clean:  ## Remove caches & artifacts
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
