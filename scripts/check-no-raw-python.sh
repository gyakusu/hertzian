#!/usr/bin/env sh
# Enforce the "no raw Python" policy: all Python work must go through uv.
#
# This fails when:
#   1. Legacy / non-uv dependency or environment files are committed.
#   2. Tracked execution surfaces (shell scripts, Makefile, CI workflows)
#      invoke `pip` or a bare `python` interpreter instead of `uv` / `uvx`.
#
# It is deliberately POSIX sh (no Python) so it can run anywhere.
set -eu

fail=0

# 1) Forbidden dependency-management / environment files. Use pyproject.toml +
#    uv.lock instead.
forbidden_files="requirements.txt requirements-dev.txt setup.py setup.cfg Pipfile Pipfile.lock poetry.lock environment.yml environment.yaml conda-lock.yml"
for f in $forbidden_files; do
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    echo "ERROR: '$f' is forbidden. Use pyproject.toml + uv (uv.lock)." >&2
    fail=1
  fi
done

# 2) Forbid raw pip / bare python invocations on execution surfaces.
#    Scope is limited to scripts/CI/Makefile so documentation can still
#    *mention* the forbidden commands. `uv`-routed calls are allowed.
pattern='(^|[^[:alnum:]_.-])(pip[[:space:]]+install|python[0-9.]*[[:space:]]+-m[[:space:]]+(pip|venv|virtualenv))'
matches=$(
  git grep -nIE "$pattern" -- \
    '*.sh' 'Makefile' 'makefile' '.github/**' \
    ':(exclude)scripts/check-no-raw-python.sh' \
    2>/dev/null || true
)
# Allow anything routed through uv (uv pip / uv run python / uvx ...).
filtered=$(printf '%s\n' "$matches" | grep -vE 'uvx|uv[[:space:]]+(run[[:space:]]+)?(pip|python)' || true)
if [ -n "$(printf '%s' "$filtered" | tr -d '[:space:]')" ]; then
  echo "ERROR: raw pip/python invocation found; route Python through uv (uv run / uvx / uv pip):" >&2
  printf '%s\n' "$filtered" >&2
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "" >&2
  echo "Policy: this project forbids raw Python. See CONTRIBUTING.md." >&2
  exit 1
fi

echo "no-raw-python: OK"
