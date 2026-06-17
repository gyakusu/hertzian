"""Reproduce the P1/P2 Hertz validation milestones through the Python bindings.

These exercise the full PyO3 surface added in P3 -- the analytic shortcuts, the
general height-field entry point, zero-copy array exchange, and error mapping --
checking results against closed-form or internally-consistent references. So
``pip install`` followed by ``pytest`` reproduces the P1 (circular) and P2
(elliptic) benchmarks from Python, which is the milestone's completion criterion.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

import hertzian

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Relative-error tolerances. The free-space DC-FFT solution carries a few percent
# of grid-discretisation error at these resolutions (cf. the Rust scenario tests),
# so the analytic comparisons are loosened accordingly while the binding's own
# round-trips (height field vs shortcut) must agree to machine precision.
LOAD_RTOL = 1e-4
RADIUS_RTOL = 0.04
PRESSURE_RTOL = 0.06
APPROACH_RTOL = 0.05
DOME_RTOL = 0.05
EXACT_RTOL = 1e-9
MIN_ELLIPTICITY = 1.5
MAX_PEAK_OFFSET_CELLS = 1
# Gothic-arch split tolerances: the two flanks must be near-symmetric and sit
# close to the analytic flank offset y0 (a few percent of grid-discretisation
# headroom on the second is plenty).
FLANK_SYMMETRY_RTOL = 0.02
FLANK_LOCATION_RTOL = 0.10


def _relative_error(actual: float, expected: float) -> float:
    """Return the relative error ``|actual - expected| / |expected|``."""
    return abs(actual - expected) / abs(expected)


def _circular_hertz(radius: float, load: float, e_star: float) -> tuple[float, float, float]:
    """Closed-form circular Hertz contact radius, peak pressure and approach."""
    a = (3.0 * load * radius / (4.0 * e_star)) ** (1.0 / 3.0)
    p0 = 3.0 * load / (2.0 * math.pi * a * a)
    delta = a * a / radius
    return a, p0, delta


def _paraboloid_gap(
    radius: float,
    shape: tuple[int, int],
    spacing: tuple[float, float],
) -> NDArray[np.float64]:
    """Sample a centred paraboloidal gap ``h = (x^2 + y^2) / (2 R)`` onto a grid.

    The centring matches the Rust ``Grid`` so the height-field path lines up cell
    for cell with the sphere-on-flat shortcut.
    """
    nx, ny = shape
    dx, dy = spacing
    x = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * dy
    gap = x[:, np.newaxis] ** 2 / (2.0 * radius) + y[np.newaxis, :] ** 2 / (2.0 * radius)
    return np.ascontiguousarray(gap, dtype=np.float64)


def test_sphere_on_flat_matches_circular_hertz() -> None:
    """P1: sphere on flat reproduces the analytic circular-Hertz a, p0, delta."""
    radius, load, e_star = 10.0e-3, 50.0, 70.0e9
    a, p0, delta = _circular_hertz(radius, load, e_star)
    shape = (128, 128)

    sol = hertzian.solve_sphere_on_flat(
        radius=radius,
        load=load,
        e_star=e_star,
        grid=shape,
        domain=6.0 * a,
        tol=1e-8,
        max_iter=5000,
    )

    assert sol.diagnostics.converged
    assert sol.shape == shape
    assert sol.pressure.shape == shape
    assert sol.pressure.dtype == np.float64
    assert bool(np.all(np.isfinite(sol.pressure)))
    assert _relative_error(sol.total_load, load) <= LOAD_RTOL
    assert _relative_error(sol.contact_radius, a) <= RADIUS_RTOL
    assert _relative_error(sol.max_pressure, p0) <= PRESSURE_RTOL
    assert _relative_error(sol.approach, delta) <= APPROACH_RTOL


def test_sphere_on_sphere_reduces_to_combined_radius() -> None:
    """P1: two equal spheres match a single sphere of the combined radius."""
    radius, load, e_star = 8.0e-3, 30.0, 110.0e9
    combined = 1.0 / (1.0 / radius + 1.0 / radius)  # 1/R = 1/R1 + 1/R2
    a, p0, _ = _circular_hertz(combined, load, e_star)

    sol = hertzian.solve_sphere_on_sphere(
        radius_1=radius,
        radius_2=radius,
        load=load,
        e_star=e_star,
        grid=(128, 128),
        domain=6.0 * a,
        tol=1e-8,
        max_iter=5000,
    )

    assert sol.diagnostics.converged
    assert _relative_error(sol.contact_radius, a) <= RADIUS_RTOL
    assert _relative_error(sol.max_pressure, p0) <= PRESSURE_RTOL


def test_sphere_on_torus_is_elliptic() -> None:
    """P2: a sphere on a torus outer equator gives a Hertzian elliptic contact.

    Without re-deriving the elliptic integrals here, the solution is pinned by
    properties unique to a Hertzian semi-ellipsoid: load conservation, a contact
    elongated circumferentially (``a_x > a_y``), and a peak pressure obeying the
    ellipsoidal relation ``p0 = 3 P / (2 pi a_x a_y)``.
    """
    load = 60.0
    shape = (256, 256)
    sol = hertzian.solve_sphere_on_torus(
        sphere_radius=12.0e-3,
        tube_radius=4.0e-3,
        centre_radius=20.0e-3,
        load=load,
        e_star=100.0e9,
        grid=shape,
        domain=1.2e-3,
        tol=1e-8,
        max_iter=5000,
    )

    assert sol.diagnostics.converged
    assert _relative_error(sol.total_load, load) <= LOAD_RTOL

    a_x, a_y = sol.contact_half_widths
    assert a_x > a_y  # elongated along the gentler circumferential (x) axis
    assert sol.ellipticity > MIN_ELLIPTICITY

    dome_p0 = 3.0 * sol.total_load / (2.0 * math.pi * a_x * a_y)
    assert _relative_error(sol.max_pressure, dome_p0) <= DOME_RTOL

    peak = np.unravel_index(int(np.argmax(sol.pressure)), sol.pressure.shape)
    assert abs(int(peak[0]) - shape[0] // 2) <= MAX_PEAK_OFFSET_CELLS
    assert abs(int(peak[1]) - shape[1] // 2) <= MAX_PEAK_OFFSET_CELLS


def test_sphere_in_gothic_arch_splits_into_two_flank_contacts() -> None:
    """A shimmed Gothic-arch groove makes the ball ride on two flanks.

    The defining behaviour of an ogival ball-bearing groove: with the two arc
    centres offset, the single conformal patch splits into a symmetric pair of
    contacts at ``y = ±y0`` with a contact-free "Gothic point" ridge between
    them, at conserved load. (That each flank is an elliptic Hertz contact
    carrying half the load is pinned analytically in the Rust scenario tests; the
    binding test checks the split, symmetry and ridge it produces.)
    """
    ball, tube, centre_radius, e_star = 4.0e-3, 4.16e-3, 15.0e-3, 100.0e9  # r/Rs = 1.04
    shim, load = 65.0e-6, 800.0
    nx, ny = 84, 720
    domain = (0.66e-3, 5.8e-3)

    sol = hertzian.solve_sphere_in_gothic_arch(
        sphere_radius=ball,
        tube_radius=tube,
        centre_radius=centre_radius,
        centre_offset=shim,
        load=load,
        e_star=e_star,
        grid=(nx, ny),
        domain=domain,
        tol=1e-8,
        max_iter=30000,
    )

    assert sol.diagnostics.converged
    assert _relative_error(sol.total_load, load) <= LOAD_RTOL

    pressure = sol.pressure
    peak = float(pressure.max())
    y = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * (domain[1] / ny)
    y0 = shim * ball / (tube - ball)  # flank offset, amplified by the conformity

    # Two flanks: equal peaks in the two y-halves, located near y = ±y0.
    mid = ny // 2
    upper = float(pressure[:, mid:].max())
    lower = float(pressure[:, :mid].max())
    assert _relative_error(upper, lower) <= FLANK_SYMMETRY_RTOL
    j_peak = int(np.argmax(pressure)) % ny
    assert _relative_error(abs(y[j_peak]), y0) <= FLANK_LOCATION_RTOL

    # The Gothic point carries no load: the central band is contact-free.
    ridge = float(pressure[:, np.abs(y) < 0.3 * y0].max())
    assert ridge <= 0.05 * peak


def test_sphere_in_gothic_arch_half_overlapping_flanks() -> None:
    """A tightened shim makes the two flank contact ellipses overlap by half.

    The companion of the separated arch above: bringing the two arc centres
    closer slides the flank contacts together until their ellipses overlap by
    half — the meridional flank offset ``y0`` equals half the flank's meridional
    semi-axis ``b``, so two ellipses of semi-axis ``b`` whose centres sit ``b``
    apart share exactly half their extent. The single conformal patch is then
    *connected*: the former Gothic point now carries load (the defining contrast
    with the contact-free ridge of the separated arch), yet the two flanks stay
    distinct as a saddle-joined pair. (The precise half-overlap geometry and the
    cross-validation against the dense reference are pinned in the Rust scenario
    tests; this binding test checks the connected, symmetric, saddle-joined pair
    it produces.)
    """
    ball, tube, centre_radius, e_star = 4.0e-3, 4.16e-3, 15.0e-3, 100.0e9  # r/Rs = 1.04
    load = 800.0
    # Tall and narrow: fine across the slim circumferential (x) semi-axis, coarse
    # along the long meridional (y) one the two flanks spread over.
    nx, ny = 64, 260
    domain = (0.6e-3, 4.2e-3)

    # The single arc (no shim) is one full-load elliptic patch; its meridional
    # semi-axis sets the overlap scale. The half-load flank shrinks by the Hertz
    # P^(1/3) load-scaling, so b = (1/2)^(1/3) * a_y(single), and half overlap
    # puts the two flank centres b apart: y0 = b / 2.
    single = hertzian.solve_sphere_in_gothic_arch(
        sphere_radius=ball,
        tube_radius=tube,
        centre_radius=centre_radius,
        centre_offset=0.0,
        load=load,
        e_star=e_star,
        grid=(nx, ny),
        domain=domain,
        tol=1e-9,
        max_iter=20000,
    )
    a_y_single = max(single.contact_half_widths)
    b = (0.5 ** (1.0 / 3.0)) * a_y_single
    y0 = 0.5 * b
    shim = y0 * (tube - ball) / ball

    sol = hertzian.solve_sphere_in_gothic_arch(
        sphere_radius=ball,
        tube_radius=tube,
        centre_radius=centre_radius,
        centre_offset=shim,
        load=load,
        e_star=e_star,
        grid=(nx, ny),
        domain=domain,
        tol=1e-9,
        max_iter=20000,
    )
    assert sol.diagnostics.converged
    assert _relative_error(sol.total_load, load) <= LOAD_RTOL

    pressure = sol.pressure
    peak = float(pressure.max())
    y = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * (domain[1] / ny)
    mid = ny // 2

    # Two symmetric flank peaks, nudged just outboard of ±y0 by the overlap.
    upper = float(pressure[:, mid:].max())
    lower = float(pressure[:, :mid].max())
    assert _relative_error(upper, lower) <= FLANK_SYMMETRY_RTOL
    j_peak = int(np.argmax(pressure)) % ny
    assert 0.5 * y0 < abs(float(y[j_peak])) < 1.5 * y0

    # Connected, not separated: the former Gothic point now carries load (the
    # separated arch leaves it contact-free, cf. the test above), yet stays in a
    # saddle below the flanks, so the two ellipses still read as a distinct pair.
    gothic_point = float(pressure[:, np.abs(y) < 0.1 * y0].max())
    assert gothic_point > 0.3 * peak
    centre_floor = float(pressure[:, mid - 1 : mid + 1].max())
    assert centre_floor < 0.9 * peak

    # The split still lowers the peak below the single full-load arc — just less
    # than full separation, since the overlapping flanks reinforce each other.
    assert peak < single.max_pressure


def test_height_field_matches_sphere_shortcut() -> None:
    """The general height-field path reproduces the sphere-on-flat shortcut.

    Cross-checks the zero-copy gap input against the analytic constructor: handed
    the gap the shortcut samples internally, both must agree to machine precision.
    """
    radius, load, e_star = 10.0e-3, 50.0, 70.0e9
    a, _, _ = _circular_hertz(radius, load, e_star)
    shape = (128, 128)
    domain = 6.0 * a
    dx = dy = domain / shape[0]
    gap = _paraboloid_gap(radius, shape, (dx, dy))

    via_gap = hertzian.solve_height_field(
        gap=gap,
        load=load,
        e_star=e_star,
        dx=dx,
        dy=dy,
        tol=1e-8,
        max_iter=5000,
    )
    via_shortcut = hertzian.solve_sphere_on_flat(
        radius=radius,
        load=load,
        e_star=e_star,
        grid=shape,
        domain=domain,
        tol=1e-8,
        max_iter=5000,
    )

    assert via_gap.diagnostics.converged
    assert via_gap.shape == shape
    assert _relative_error(via_gap.contact_radius, via_shortcut.contact_radius) <= EXACT_RTOL
    assert _relative_error(via_gap.max_pressure, via_shortcut.max_pressure) <= EXACT_RTOL
    assert _relative_error(via_gap.approach, via_shortcut.approach) <= EXACT_RTOL


def test_pressure_array_is_a_fresh_owned_copy() -> None:
    """``pressure`` returns an independent, C-contiguous float64 array each call."""
    sol = hertzian.solve_sphere_on_flat(
        radius=5.0e-3,
        load=20.0,
        e_star=70.0e9,
        grid=(64, 64),
        domain=8.0e-4,
    )

    first = sol.pressure
    assert first.dtype == np.float64
    assert first.flags["C_CONTIGUOUS"]
    assert sol.pressure is not first  # a new array per access

    # Mutating the returned array must not leak back into the solution.
    original = float(first[0, 0])
    first[0, 0] = original + 1.0
    assert float(sol.pressure[0, 0]) == original


def test_repr_is_informative() -> None:
    """The result types expose readable, Python-style reprs."""
    sol = hertzian.solve_sphere_on_flat(
        radius=5.0e-3,
        load=20.0,
        e_star=70.0e9,
        grid=(64, 64),
        domain=8.0e-4,
    )
    assert repr(sol).startswith("Solution(")
    assert "converged=True" in repr(sol)
    assert repr(sol.diagnostics).startswith("Diagnostics(")


def test_negative_radius_raises_value_error() -> None:
    """Invalid physical inputs are rejected as ValueError, not a Rust panic."""
    with pytest.raises(ValueError, match="radius must be positive"):
        hertzian.solve_sphere_on_flat(
            radius=-1.0,
            load=50.0,
            e_star=70.0e9,
            grid=(32, 32),
            domain=1e-3,
        )


def test_gothic_arch_rejects_a_ball_wider_than_the_groove() -> None:
    """A ball at least as wide as the groove has no conformal contact."""
    with pytest.raises(ValueError, match="must be smaller than tube_radius"):
        hertzian.solve_sphere_in_gothic_arch(
            sphere_radius=4.0e-3,
            tube_radius=4.0e-3,  # equal: no clearance, no conformal contact
            centre_radius=15.0e-3,
            centre_offset=50.0e-6,
            load=100.0,
            e_star=100.0e9,
            grid=(32, 64),
            domain=(0.5e-3, 2.0e-3),
        )


def test_periodic_boundary_is_not_implemented() -> None:
    """The reserved periodic boundary raises NotImplementedError (design §3.3)."""
    gap = np.zeros((8, 8), dtype=np.float64)
    with pytest.raises(NotImplementedError):
        hertzian.solve_height_field(
            gap=gap,
            load=1.0,
            e_star=70.0e9,
            dx=1e-4,
            dy=1e-4,
            boundary="periodic",
        )


def test_non_float64_gap_is_rejected() -> None:
    """A non-float64 gap array is rejected at the boundary as TypeError."""
    gap = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(TypeError):
        # Deliberately the wrong dtype: the stub forbids it statically (hence the
        # ignore on the offending argument) and the binding rejects it at runtime.
        hertzian.solve_height_field(
            gap=gap,  # type: ignore[arg-type]
            load=1.0,
            e_star=70.0e9,
            dx=1e-4,
            dy=1e-4,
        )
