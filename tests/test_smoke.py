"""Smoke tests for the packaged extension module.

These only verify that the Rust core builds, installs, and is importable from
Python. Numerical/solver tests are added with the implementation.
"""

from __future__ import annotations

import hertzian


def test_version_is_exposed() -> None:
    """The package re-exports a non-empty version string from the Rust core."""
    assert isinstance(hertzian.__version__, str)
    assert hertzian.__version__
