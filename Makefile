.PHONY: help install run-api run-frontend ingest test eval lint clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies from requirements.txt
	pip install -r requirements.txt

run-api: ## Start the FastAPI server
	uvicorn src.main:app --reload --port 8000

run-frontend: ## Start the Streamlit frontend
	streamlit run app.py

ingest: ## Run the ingestion pipeline
	python -m src.ingest

ingest-force: ## Force re-ingest all chunks
	python -m src.ingest --force

create-index: ## Create the Atlas Vector Search index
	python -m src.ingest --create-index

search: ## Run a sample vector search
	python -m src.retrieval "$(query)"

agent: ## Run the RAG agent with a question
	python -m src.graph "$(question)"

test: ## Run the test suite
	python -m pytest tests/ -v --tb=short

eval: ## Run the evaluation harness
	python evals/run_evals.py

eval-json: ## Run evaluation with JSON output
	python evals/run_evals.py --json

eval-trace: ## Run evaluation with LangSmith tracing enabled
	python evals/run_evals.py --trace

lint: ## Run basic linting checks
	python -m pytest tests/ -v --tb=short -x

clean: ## Clean Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/