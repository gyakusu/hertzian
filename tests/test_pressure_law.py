"""Exercise the reduced flank pressure distribution through the bindings.

:class:`hertzian.FlankPressure` is the Coulomb-friction companion to the reduced
force law: it expands a per-flank load ``Q`` into the closed-form elliptic-Hertz
pressure field ``p(x, y)`` and the spin (drilling) moment ``(3/8) μ Q a E(e)`` a
multibody loop needs but the net force ``F(δ)`` cannot give. These tests pin its
contract: the field integrates to the load, the patch and peak scale as the
Hertzian cube root, the spin moment matches a direct quadrature and the circular
limit, and it composes with :class:`hertzian.GothicArchLaw`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import hertzian

# Relative tolerances: closed-form identities are near machine precision; the
# quadrature cross-check allows for the mesh; the circular limit is exact bar the
# eccentricity solve.
EXACT_RTOL = 1e-12
QUADRATURE_RTOL = 3e-3
CIRCULAR_RTOL = 1e-9

# The README's conformal Gothic flank (strongly elliptic) and its material.
RADIUS_X = 1.6e-3
RADIUS_Y = 26.0e-3
E_STAR = 100.0e9


def _flank() -> hertzian.FlankPressure:
    """Return the reference pressure law calibrated from the conformal flank."""
    return hertzian.FlankPressure.from_elliptic_flank(
        radius_x=RADIUS_X, radius_y=RADIUS_Y, e_star=E_STAR
    )


def test_field_integrates_to_the_flank_load() -> None:
    """The zeroth moment of ``p`` is the load: ``(2/3) π a_x a_y p₀ = Q``."""
    flank = _flank()
    for load in (4.0, 90.0, 600.0):
        a_x, a_y = flank.semi_axes(load)
        integrated = 2.0 / 3.0 * math.pi * a_x * a_y * flank.peak_pressure(load)
        assert math.isclose(integrated, load, rel_tol=EXACT_RTOL)


def test_size_and_peak_scale_as_the_cube_root_of_load() -> None:
    """Semi-axes and peak pressure scale as ``Q^(1/3)``: 8x load gives 2x size."""
    flank = _flank()
    ax1, ay1 = flank.semi_axes(12.0)
    ax8, ay8 = flank.semi_axes(96.0)
    assert math.isclose(ax8 / ax1, 2.0, rel_tol=EXACT_RTOL)
    assert math.isclose(ay8 / ay1, 2.0, rel_tol=EXACT_RTOL)
    assert math.isclose(
        flank.peak_pressure(96.0) / flank.peak_pressure(12.0), 2.0, rel_tol=EXACT_RTOL
    )


def test_pressure_is_the_elliptic_semi_ellipsoid() -> None:
    """``p(x, y) = p0 sqrt(1 - (x/a_x)^2 - (y/a_y)^2)`` inside, zero outside."""
    flank = _flank()
    load = 50.0
    a_x, a_y = flank.semi_axes(load)
    p0 = flank.peak_pressure(load)
    assert math.isclose(flank.pressure_at(load, 0.0, 0.0), p0, rel_tol=EXACT_RTOL)
    x, y = 0.3 * a_x, 0.4 * a_y
    expected = p0 * math.sqrt(1.0 - (x / a_x) ** 2 - (y / a_y) ** 2)
    assert math.isclose(flank.pressure_at(load, x, y), expected, rel_tol=EXACT_RTOL)
    # Outside the ellipse the field vanishes (no adhesion).
    assert flank.pressure_at(load, 1.01 * a_x, 0.0) == 0.0
    assert flank.pressure_at(load, 0.0, 1.01 * a_y) == 0.0


def test_spin_moment_matches_a_direct_quadrature() -> None:
    """The closed form ``(3/8) mu Q a E(e)`` equals ``mu int p rho dA`` over the patch."""
    flank = _flank()
    load, friction = 120.0, 0.1
    a_x, a_y = flank.semi_axes(load)

    # Polar quadrature mapped to the ellipse: x = a_x r cosθ, y = a_y r sinθ.
    n_r, n_theta = 1500, 1500
    r = (np.arange(n_r) + 0.5) / n_r
    theta = 2.0 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    rg, tg = np.meshgrid(r, theta, indexing="ij")
    x = a_x * rg * np.cos(tg)
    y = a_y * rg * np.sin(tg)
    rho = np.hypot(x, y)
    pressure = flank.peak_pressure(load) * np.sqrt(np.clip(1.0 - rg**2, 0.0, None))
    cell = (1.0 / n_r) * (2.0 * np.pi / n_theta)
    numeric = friction * float(np.sum(pressure * rho * a_x * a_y * rg)) * cell

    assert math.isclose(flank.spin_moment(load, friction), numeric, rel_tol=QUADRATURE_RTOL)


def test_spin_moment_reduces_to_the_circular_result() -> None:
    """A circular contact gives the textbook ``3π/16 μ Q a`` (E(0) = π/2)."""
    flank = hertzian.FlankPressure.from_elliptic_flank(
        radius_x=5.0e-3, radius_y=5.0e-3, e_star=E_STAR
    )
    assert abs(flank.eccentricity) <= CIRCULAR_RTOL
    load, friction = 75.0, 0.2
    a, _ = flank.semi_axes(load)
    expected = 3.0 * math.pi / 16.0 * friction * load * a
    assert math.isclose(flank.spin_moment(load, friction), expected, rel_tol=CIRCULAR_RTOL)


def test_spin_moment_and_radius_load_scaling() -> None:
    """``M ∝ Q^{4/3}`` (lever arm ∝ Q^{1/3}), and ``M = μ Q · spin_radius``."""
    flank = _flank()
    friction = 0.15
    assert math.isclose(
        flank.spin_moment(80.0, friction) / flank.spin_moment(10.0, friction),
        8.0 ** (4.0 / 3.0),
        rel_tol=EXACT_RTOL,
    )
    assert math.isclose(flank.spin_radius(80.0) / flank.spin_radius(10.0), 2.0, rel_tol=EXACT_RTOL)
    assert math.isclose(
        flank.spin_moment(80.0, friction),
        friction * 80.0 * flank.spin_radius(80.0),
        rel_tol=EXACT_RTOL,
    )


def test_separated_flank_carries_no_pressure() -> None:
    """A non-positive (lifted-off) load gives zero field, patch and moment."""
    flank = _flank()
    assert flank.pressure_at(0.0, 0.0, 0.0) == 0.0
    assert flank.pressure_at(-1.0, 0.0, 0.0) == 0.0
    assert flank.semi_axes(0.0) == (0.0, 0.0)
    assert flank.peak_pressure(0.0) == 0.0
    assert flank.spin_moment(0.0, 0.3) == 0.0


def test_composes_with_the_force_law_per_flank() -> None:
    """The force law's per-flank loads drive the per-flank pressure distribution."""
    contact_angle = 0.40
    law = hertzian.GothicArchLaw.from_elliptic_flank(
        radius_x=RADIUS_X, radius_y=RADIUS_Y, e_star=E_STAR, contact_angle=contact_angle
    )
    flank = _flank()

    # A symmetric push loads both flanks equally, so their distributions match.
    q_plus, q_minus = law.flank_loads(0.0, 6.0e-6)
    assert math.isclose(q_plus, q_minus, rel_tol=EXACT_RTOL)
    assert math.isclose(
        flank.peak_pressure(q_plus), flank.peak_pressure(q_minus), rel_tol=EXACT_RTOL
    )
    # Past lift-off the lone flank still carries pressure; the lifted one is silent.
    delta_n = 5.0e-6
    delta_t = 2.0 * law.lift_off_transverse(delta_n)
    q_plus, q_minus = law.flank_loads(delta_t, delta_n)
    assert q_minus == 0.0
    assert flank.peak_pressure(q_plus) > 0.0
    assert flank.spin_moment(q_minus, 0.2) == 0.0


def test_from_elliptic_flank_matches_a_hand_built_ellipse() -> None:
    """The two constructors agree: ``from_elliptic_flank`` feeds ``new`` the ellipse."""
    flank = _flank()
    a_x, a_y = flank.semi_axes(64.0)
    rebuilt = hertzian.FlankPressure(semi_axis_x=a_x, semi_axis_y=a_y, reference_load=64.0)
    assert math.isclose(rebuilt.eccentricity, flank.eccentricity, rel_tol=EXACT_RTOL)
    assert math.isclose(rebuilt.peak_pressure(1.0), flank.peak_pressure(1.0), rel_tol=EXACT_RTOL)
    assert math.isclose(rebuilt.spin_radius(1.0), flank.spin_radius(1.0), rel_tol=EXACT_RTOL)


def test_invalid_inputs_raise() -> None:
    """Out-of-range constructor and friction inputs map to ``ValueError``."""
    with pytest.raises(ValueError, match="semi_axis_x"):
        hertzian.FlankPressure(semi_axis_x=-1.0, semi_axis_y=1.0e-3, reference_load=1.0)
    with pytest.raises(ValueError, match="reference_load"):
        hertzian.FlankPressure(semi_axis_x=1.0e-3, semi_axis_y=1.0e-3, reference_load=0.0)
    with pytest.raises(ValueError, match="radius_x"):
        hertzian.FlankPressure.from_elliptic_flank(radius_x=0.0, radius_y=1.0e-3, e_star=E_STAR)
    with pytest.raises(ValueError, match="friction"):
        _flank().spin_moment(50.0, -0.1)
