.DEFAULT_GOAL := help

UV ?= uv

.PHONY: help setup sync test lint format typecheck docs build notebooks figures clean

help:  ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: sync  ## Install all dependencies, including every extra

sync:  ## Sync the uv environment with all extras
	$(UV) sync --all-extras

test:  ## Run the test suite
	$(UV) run pytest

lint:  ## Check lint and formatting
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:  ## Apply lint fixes and formatting
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:  ## Type-check src/ with mypy
	$(UV) run mypy src

docs:  ## Build the documentation with warnings as errors
	$(UV) run mkdocs build --strict

build:  ## Build the wheel and source distribution
	$(UV) build

notebooks:  ## Execute every Jupytext notebook headlessly into results/cache
	@set -e; found=0; mkdir -p results/cache/notebooks; \
	for nb in notebooks/*.py; do \
		[ -e "$$nb" ] || continue; \
		found=1; \
		out="results/cache/notebooks/$$(basename "$$nb" .py).ipynb"; \
		echo "Executing $$nb"; \
		$(UV) run jupytext --to ipynb --execute "$$nb" --output "$$out"; \
	done; \
	if [ "$$found" -eq 0 ]; then echo "No notebooks yet."; fi

figures:  ## Report the figure-gallery status
	@echo "No committed figure gallery yet; figures are produced by notebooks and examples."

clean:  ## Remove local caches and build outputs
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov site dist build
	rm -f .coverage .coverage.* coverage.xml
	find src tests examples notebooks -type d -name __pycache__ -prune -exec rm -rf {} +
