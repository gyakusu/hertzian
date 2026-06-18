# Convenience wrapper around the project's tooling.
#
# The single source of truth for *static analysis* is `.pre-commit-config.yaml`
# (run by `make lint`), which CI executes verbatim. These targets just make the
# common workflows ergonomic. Everything Python goes through `uv` (no raw pip).

.DEFAULT_GOAL := help
SHELL := bash

.PHONY: help setup sync hooks fmt fix lint typecheck lint-rust \
        build gallery test test-py test-rust check ci clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: sync hooks ## One-time dev setup: uv env, git hooks, Rust toolchain
	rustup show

sync: ## Create/refresh the uv-managed virtualenv from uv.lock
	uv sync

hooks: ## Install the git pre-commit hook
	uv run pre-commit install

fmt: ## Auto-format Python (ruff) and Rust (cargo fmt)
	uv run ruff check --fix
	uv run ruff format
	cargo fmt --all

fix: fmt ## Alias for `fmt`

lint: ## Run ALL static analysis exactly as CI does (via pre-commit)
	uv run pre-commit run --all-files --show-diff-on-failure

typecheck: ## Run mypy (strict)
	uv run mypy

lint-rust: ## Run rustfmt --check and clippy (strict, -D warnings)
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings

build: ## Build the native extension into the uv venv (maturin develop)
	uv run maturin develop --uv

gallery: build ## Render the README validation gallery into docs/img (uses matplotlib)
	uv run --with matplotlib python scripts/render_gallery.py
	uv run --with matplotlib python scripts/fit_reduced_law.py
	uv run --with matplotlib python scripts/render_coupling_cross_section.py
	uv run --with matplotlib python scripts/render_pressure_distribution.py

test: test-rust test-py ## Run the full test suite (Rust + Python)

test-rust: ## Run Rust unit tests
	cargo test --all-features --locked

test-py: build ## Build the extension, then run Python tests
	uv run pytest

check: lint test ## Run everything CI runs

ci: check ## Alias for `check`

clean: ## Remove build/test caches and artifacts
	cargo clean
	rm -rf .venv .mypy_cache .ruff_cache .pytest_cache dist wheels
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
