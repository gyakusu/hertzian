"""Hertzian: FFT-accelerated elastic half-space normal contact solver.

The numerical core is implemented in Rust and exposed through the
``hertzian._core`` extension module. This package is currently scaffolding;
the solver API is added in a later milestone (see the project README).
"""

from __future__ import annotations

from hertzian._core import __version__

__all__ = ["__version__"]
