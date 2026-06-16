"""Arbitrary-shape and roughness contacts through the height-field binding (P4).

These exercise the general :func:`hertzian.solve_height_field` path with shapes
that are *not* one of the analytic shortcuts:

* a rigid **cone**, validated against Sneddon's closed-form non-Hertzian
  solution (contact radius, approach and load), and
* a smooth sphere with an **added cosine roughness**, validated by the physical
  signatures of asperity contact (a fragmented patch and a raised peak pressure
  at conserved load).

Together they show that any height field — analytic, measured, or a smooth base
plus a roughness — drives the solver, which is the P4 milestone.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

import hertzian

if TYPE_CHECKING:
    from numpy.typing import NDArray

# The free-space DC-FFT solution carries a few percent of grid-discretisation
# error at these resolutions; the cone additionally has an apex curvature
# singularity, so its tolerances are a touch looser than the smooth benchmarks.
LOAD_RTOL = 1e-4
CONE_RADIUS_RTOL = 0.04
CONE_APPROACH_RTOL = 0.04


def _centred_axis(n: int, spacing: float) -> NDArray[np.float64]:
    """Return the origin-centred physical coordinates of a length-``n`` axis."""
    return (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * spacing


def _sneddon_cone(slope: float, load: float, e_star: float) -> tuple[float, float]:
    """Return Sneddon's cone contact radius ``a`` and approach ``delta``.

    For a conical gap ``h(r) = m r`` of surface slope ``m`` pressed by ``load``
    into a half-space of modulus ``e_star``: ``a = sqrt(2 P / (pi E* m))`` and
    ``delta = (pi/2) m a``.
    """
    a = math.sqrt(2.0 * load / (math.pi * e_star * slope))
    delta = 0.5 * math.pi * slope * a
    return a, delta


def _relative_error(actual: float, expected: float) -> float:
    """Return the relative error ``|actual - expected| / |expected|``."""
    return abs(actual - expected) / abs(expected)


def test_cone_matches_sneddon() -> None:
    """A conical height field reproduces Sneddon's closed-form cone contact.

    The cone is the canonical *non*-Hertzian axisymmetric punch: pressure grows
    without bound at the apex, so only the (mesh-convergent) contact radius,
    approach and load are checked, not the singular peak pressure.
    """
    slope, load, e_star = 0.02, 60.0, 100.0e9
    a, delta = _sneddon_cone(slope, load, e_star)

    n = 256
    domain = 6.0 * a
    dx = dy = domain / n
    x = _centred_axis(n, dx)
    r = np.hypot(x[:, np.newaxis], x[np.newaxis, :])
    gap = np.ascontiguousarray(slope * r, dtype=np.float64)

    sol = hertzian.solve_height_field(
        gap=gap, load=load, e_star=e_star, dx=dx, dy=dy, tol=1e-9, max_iter=20000
    )

    assert sol.diagnostics.converged
    assert _relative_error(sol.total_load, load) <= LOAD_RTOL
    assert _relative_error(sol.contact_radius, a) <= CONE_RADIUS_RTOL
    assert _relative_error(sol.approach, delta) <= CONE_APPROACH_RTOL
    # The peak sits at the apex (grid centre) even though its height is mesh-set.
    peak = np.unravel_index(int(np.argmax(sol.pressure)), sol.pressure.shape)
    assert abs(int(peak[0]) - n // 2) <= 1
    assert abs(int(peak[1]) - n // 2) <= 1


def test_added_roughness_fragments_the_contact() -> None:
    """A sphere plus cosine roughness contacts as asperities at conserved load.

    Layering a roughness onto the smooth gap (plain NumPy addition, the Python
    face of ``Gap::plus``) must (i) conserve the applied load, (ii) shrink the
    real contact area below the smooth Hertz disc, and (iii) raise the peak
    pressure as load concentrates on the asperities.
    """
    radius, load, e_star = 10.0e-3, 40.0, 70.0e9
    a = (3.0 * load * radius / (4.0 * e_star)) ** (1.0 / 3.0)
    delta = a * a / radius

    n = 192
    domain = 5.0 * a
    dx = dy = domain / n
    x = _centred_axis(n, dx)
    big_x = x[:, np.newaxis]
    big_y = x[np.newaxis, :]

    smooth = (big_x**2 + big_y**2) / (2.0 * radius)
    wavelength = 1.0 * a
    roughness = (
        0.8
        * delta
        * np.cos(2.0 * math.pi * big_x / wavelength)
        * np.cos(2.0 * math.pi * big_y / wavelength)
    )
    rough_gap = np.ascontiguousarray(smooth + roughness, dtype=np.float64)

    rough = hertzian.solve_height_field(
        gap=rough_gap, load=load, e_star=e_star, dx=dx, dy=dy, tol=1e-9, max_iter=20000
    )
    base = hertzian.solve_sphere_on_flat(
        radius=radius,
        load=load,
        e_star=e_star,
        grid=(n, n),
        domain=domain,
        tol=1e-9,
        max_iter=20000,
    )

    assert rough.diagnostics.converged
    assert _relative_error(rough.total_load, load) <= LOAD_RTOL
    assert rough.contact_area < 0.7 * base.contact_area
    assert rough.max_pressure > 1.5 * base.max_pressure
