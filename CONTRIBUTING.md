# Contributing

This document covers the development environment and quality gates. The project
is in the scaffolding phase; the solver design lives in the README.

## Toolchain

- **Rust** via `rustup`. The toolchain, including `clippy` and `rustfmt`, is
  pinned in `rust-toolchain.toml` and installed automatically.
- **Python** is managed exclusively by [`uv`](https://docs.astral.sh/uv/). The
  interpreter version is pinned in `.python-version`; dependencies and dev tools
  are pinned in `pyproject.toml` + `uv.lock`.

One-time setup:

```sh
make setup    # uv sync, install the pre-commit git hook, install Rust toolchain
```

## No raw Python (policy)

All Python invocations must go through `uv` — never a bare interpreter or `pip`,
and never legacy environment files. This keeps every environment reproducible
from `uv.lock`.

| Don't                     | Do                          |
| ------------------------- | --------------------------- |
| `python script.py`        | `uv run python script.py`   |
| `pip install foo`         | `uv add foo`                |
| `python -m venv .venv`    | `uv sync`                   |
| `requirements.txt`        | `pyproject.toml` + `uv.lock`|
| `setup.py` / `Pipfile`    | `pyproject.toml`            |
| ad-hoc `black`/`flake8`   | `uv run ruff`               |

Enforced by `scripts/check-no-raw-python.sh` (a POSIX-sh guard) in both
pre-commit and CI. It rejects the legacy files above and rejects raw
`pip`/`python -m pip|venv` calls in scripts, the `Makefile`, and CI workflows.
Documentation may still *mention* these commands.

## Static analysis (single source of truth)

`.pre-commit-config.yaml` is the authoritative definition of every static check.
**CI runs those exact hooks** (`uv run pre-commit run --all-files`), so local and
CI results cannot diverge. The hooks are:

| Hook          | Command                                              | Strictness         |
| ------------- | --------------------------------------------------- | ------------------ |
| ruff (lint)   | `ruff check`                                         | `select = ["ALL"]` |
| ruff (format) | `ruff format`                                        | —                  |
| mypy          | `mypy`                                               | `--strict` + extras|
| cargo fmt     | `cargo fmt --all`                                    | check in CI        |
| cargo clippy  | `cargo clippy --all-targets --all-features`          | `-D warnings`, pedantic + nursery |
| no-raw-python | `scripts/check-no-raw-python.sh`                     | —                  |

Run them all locally before pushing:

```sh
make lint     # == uv run pre-commit run --all-files
make fmt      # auto-fix formatting (ruff + cargo fmt)
```

## Tests

```sh
make test         # cargo test + (maturin develop) + pytest
make test-rust    # Rust unit tests only
make test-py      # build the extension, then pytest
```

## CI jobs

`.github/workflows/ci.yml` runs three jobs on every push/PR:

1. **static-analysis** — `pre-commit run --all-files` (ruff, mypy, clippy, fmt,
   no-raw-python).
2. **rust-test** — `cargo test --locked`.
3. **python-test** — `maturin develop` then `pytest`.

## Commits

Keep commits focused and messages descriptive. Ensure `make lint` and
`make test` pass before pushing.
