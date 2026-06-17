"""Hertzian: FFT-accelerated elastic half-space normal contact solver.

The numerical core is implemented in Rust and exposed through the
``hertzian._core`` extension module; this package re-exports its public API.

Two layers are offered (design §8.5):

* **Analytic shortcuts** for the validated Hertz benchmarks --
  :func:`solve_sphere_on_flat`, :func:`solve_sphere_on_sphere`,
  :func:`solve_sphere_on_torus` and :func:`solve_sphere_in_gothic_arch`
  (a ball in a two-arc, ogival bearing groove).
* A **general gap input**, :func:`solve_height_field`, taking an arbitrary
  undeformed-gap height field as a NumPy array.

Each returns a :class:`Solution` carrying the converged pressure field (as a
zero-copy NumPy array) and the derived contact quantities.

Example:
    >>> import hertzian
    >>> sol = hertzian.solve_sphere_on_flat(
    ...     radius=10e-3, load=50.0, e_star=70e9, grid=(256, 256), domain=1.2e-3
    ... )
    >>> sol.diagnostics.converged
    True
"""

from __future__ import annotations

from hertzian._core import (
    Diagnostics,
    Solution,
    __version__,
    solve_height_field,
    solve_sphere_in_gothic_arch,
    solve_sphere_on_flat,
    solve_sphere_on_sphere,
    solve_sphere_on_torus,
)

__all__ = [
    "Diagnostics",
    "Solution",
    "__version__",
    "solve_height_field",
    "solve_sphere_in_gothic_arch",
    "solve_sphere_on_flat",
    "solve_sphere_on_sphere",
    "solve_sphere_on_torus",
]
