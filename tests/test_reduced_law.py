"""Exercise the reduced two-flank Gothic-arch force law through the bindings.

The law (:class:`hertzian.GothicArchLaw`) is the lightweight stand-in for the
field solver in a multibody loop. These tests pin its contract: it reproduces a
single Hertz contact in the one-flank limit, is purely normal under a symmetric
push, carries no adhesion, and — the boundary condition it exists to satisfy —
varies ``C¹`` across the two-to-one lift-off transition.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import hertzian

# Relative tolerances: the analytic-limit checks are near machine precision; the
# finite-difference continuity checks allow for the step size.
EXACT_RTOL = 1e-12
FD_RTOL = 1e-4
CONTINUITY_RTOL = 1e-3
ZERO_TOL = 1e-18

# A representative conformal flank (the README's Gothic groove) and contact angle.
RADIUS_X = 1.6e-3
RADIUS_Y = 26.0e-3
E_STAR = 100.0e9
CONTACT_ANGLE = 0.40

# Neighbour-lift coupling: a flank offset small enough (relative to the contact) to
# give a clearly sub-unity coupling — the well-conditioned half-overlap regime.
COUPLING_OFFSET = 5.0e-4
# Two independent, well-separated flanks each carry half the load: η = 2.
SEPARATED_FLANK_COUNT = 2.0


def _law() -> hertzian.GothicArchLaw:
    """Return the reference calibrated law used across the tests."""
    return hertzian.GothicArchLaw.from_elliptic_flank(
        radius_x=RADIUS_X, radius_y=RADIUS_Y, e_star=E_STAR, contact_angle=CONTACT_ANGLE
    )


def _coupled_law() -> hertzian.GothicArchLaw:
    """Return the reference law with the neighbour-lift coupling switched on."""
    return _law().with_flank_coupling(e_star=E_STAR, offset=COUPLING_OFFSET)


def test_from_elliptic_flank_reproduces_the_hertz_load() -> None:
    """``K delta^{3/2}`` reproduces the calibrating elliptic-Hertz load."""
    law = _law()
    for load in (5.0, 180.0):
        delta = (load / law.stiffness) ** (2.0 / 3.0)
        assert math.isclose(law.flank_load(delta), load, rel_tol=EXACT_RTOL)


def test_single_flank_limit_is_a_hertz_contact() -> None:
    """Past lift-off the force is one Hertz contact directed along the flank."""
    law = _law()
    delta_n = 5.0e-6
    delta_t = 2.0 * law.lift_off_transverse(delta_n)

    s_plus, s_minus = law.flank_approaches(delta_t, delta_n)
    assert s_minus < 0.0

    f_t, f_n = law.force(delta_t, delta_n)
    magnitude = law.flank_load(s_plus)
    assert math.isclose(math.hypot(f_t, f_n), magnitude, rel_tol=EXACT_RTOL)
    assert math.isclose(f_t, magnitude * math.sin(law.contact_angle), rel_tol=EXACT_RTOL)
    assert math.isclose(f_n, magnitude * math.cos(law.contact_angle), rel_tol=EXACT_RTOL)


def test_symmetric_push_is_purely_normal() -> None:
    """A straight push (delta_t = 0) loads both flanks equally; no side force."""
    law = _law()
    f_t, f_n = law.force(0.0, 7.0e-6)
    assert abs(f_t) <= ZERO_TOL
    q_plus, q_minus = law.flank_loads(0.0, 7.0e-6)
    assert math.isclose(q_plus, q_minus, rel_tol=EXACT_RTOL)
    assert f_n > 0.0


def test_no_adhesion_when_pulled_out() -> None:
    """A negative normal displacement separates both flanks: zero force."""
    law = _law()
    assert law.force(1.0e-6, -3.0e-6) == (0.0, 0.0)
    assert law.flank_loads(0.0, -1.0e-6) == (0.0, 0.0)


def test_force_is_c1_across_the_two_to_one_transition() -> None:
    """Force *and* tangent stiffness are continuous across the lift-off seam."""
    law = _law()
    delta_n = 5.0e-6
    seam = law.lift_off_transverse(delta_n)
    step = 1.0e-10

    # The step straddles lift-off: the far flank is engaged below, separated above.
    _, s_below = law.flank_approaches(seam - step, delta_n)
    _, s_above = law.flank_approaches(seam + step, delta_n)
    assert s_below > 0.0 > s_above

    f_below = law.force(seam - step, delta_n)
    f_above = law.force(seam + step, delta_n)
    scale = max(abs(f_below[0]), abs(f_below[1]))
    assert abs(f_below[0] - f_above[0]) <= CONTINUITY_RTOL * scale
    assert abs(f_below[1] - f_above[1]) <= CONTINUITY_RTOL * scale

    jac_below = np.asarray(law.jacobian(seam - step, delta_n))
    jac_above = np.asarray(law.jacobian(seam + step, delta_n))
    assert np.allclose(jac_below, jac_above, atol=1e-2 * abs(jac_below[1, 1]))


def test_jacobian_matches_finite_differences() -> None:
    """The analytic tangent stiffness matches a central difference of the force."""
    law = _law()
    delta_t, delta_n = 2.0e-6, 6.0e-6
    step = 1.0e-11
    analytic = np.asarray(law.jacobian(delta_t, delta_n))

    ft_tp, fn_tp = law.force(delta_t + step, delta_n)
    ft_tm, fn_tm = law.force(delta_t - step, delta_n)
    ft_np, fn_np = law.force(delta_t, delta_n + step)
    ft_nm, fn_nm = law.force(delta_t, delta_n - step)
    numeric = np.array(
        [
            [(ft_tp - ft_tm) / (2.0 * step), (ft_np - ft_nm) / (2.0 * step)],
            [(fn_tp - fn_tm) / (2.0 * step), (fn_np - fn_nm) / (2.0 * step)],
        ]
    )
    assert np.allclose(numeric, analytic, rtol=FD_RTOL)
    # Conservative contact: symmetric stiffness.
    assert math.isclose(analytic[0, 1], analytic[1, 0], rel_tol=EXACT_RTOL)


def test_with_flank_coupling_sets_the_cross_compliance() -> None:
    """``with_flank_coupling`` sets κ = 1/(2π E* y0); the default law has κ = 0."""
    assert _law().coupling == 0.0
    expected = 1.0 / (2.0 * math.pi * E_STAR * COUPLING_OFFSET)
    assert math.isclose(_coupled_law().coupling, expected, rel_tol=EXACT_RTOL)


def test_coupling_lowers_the_effective_flank_count_below_two() -> None:
    """A symmetric push loads the coupled pair below two independent flanks."""
    law = _coupled_law()
    delta = 6.0e-6
    q_plus, q_minus = law.coupled_loads(delta, delta)
    eta = (q_plus + q_minus) / (law.stiffness * delta**1.5)
    assert 1.0 < eta < SEPARATED_FLANK_COUNT
    # Two independent flanks (κ = 0) carry exactly η = 2.
    uncoupled = _law().coupled_loads(delta, delta)
    assert math.isclose(
        sum(uncoupled) / (law.stiffness * delta**1.5),
        SEPARATED_FLANK_COUNT,
        rel_tol=EXACT_RTOL,
    )


def test_coupling_leaves_the_single_flank_limit_untouched() -> None:
    """Past lift-off the lone flank lifts nothing: the force is coupling-free."""
    law = _law()
    coupled = _coupled_law()
    delta_n = 5.0e-6
    delta_t = 2.0 * law.lift_off_transverse(delta_n)
    assert law.flank_approaches(delta_t, delta_n)[1] < 0.0
    assert coupled.force(delta_t, delta_n) == law.force(delta_t, delta_n)


def test_coupling_sharpens_the_load_split() -> None:
    """The lift sharpens the split Q_+/Q_- and lowers both flank loads."""
    s_plus, s_minus = 9.0e-6, 4.0e-6
    qp_u, qm_u = _law().coupled_loads(s_plus, s_minus)
    qp_c, qm_c = _coupled_law().coupled_loads(s_plus, s_minus)
    assert qp_c / qm_c > qp_u / qm_u
    assert qp_c < qp_u
    assert qm_c < qm_u


def test_coupled_jacobian_matches_finite_differences() -> None:
    """The coupled tangent stiffness matches a central difference and is symmetric."""
    law = _coupled_law()
    delta_t, delta_n = 2.0e-6, 6.0e-6
    step = 1.0e-11
    analytic = np.asarray(law.jacobian(delta_t, delta_n))

    ft_tp, fn_tp = law.force(delta_t + step, delta_n)
    ft_tm, fn_tm = law.force(delta_t - step, delta_n)
    ft_np, fn_np = law.force(delta_t, delta_n + step)
    ft_nm, fn_nm = law.force(delta_t, delta_n - step)
    numeric = np.array(
        [
            [(ft_tp - ft_tm) / (2.0 * step), (ft_np - ft_nm) / (2.0 * step)],
            [(fn_tp - fn_tm) / (2.0 * step), (fn_np - fn_nm) / (2.0 * step)],
        ]
    )
    assert np.allclose(numeric, analytic, rtol=FD_RTOL)
    assert math.isclose(analytic[0, 1], analytic[1, 0], rel_tol=EXACT_RTOL)


def test_contact_half_angle_is_the_geometric_arcsine() -> None:
    """``alpha = arcsin(y0 / Rs)`` with a zero-offset limit of zero."""
    assert math.isclose(
        hertzian.contact_half_angle(offset=1.0e-3, ball_radius=4.0e-3),
        math.asin(0.25),
        rel_tol=EXACT_RTOL,
    )
    assert hertzian.contact_half_angle(offset=0.0, ball_radius=4.0e-3) == 0.0


def test_invalid_inputs_raise() -> None:
    """Out-of-range constructor and helper inputs map to ``ValueError``."""
    with pytest.raises(ValueError, match="stiffness"):
        hertzian.GothicArchLaw(stiffness=-1.0, contact_angle=0.4)
    with pytest.raises(ValueError, match="contact_angle"):
        hertzian.GothicArchLaw(stiffness=1.0, contact_angle=2.0)
    with pytest.raises(ValueError, match="offset"):
        hertzian.contact_half_angle(offset=5.0e-3, ball_radius=4.0e-3)
    with pytest.raises(ValueError, match="e_star"):
        _law().with_flank_coupling(e_star=-1.0, offset=5.0e-4)
    with pytest.raises(ValueError, match="offset"):
        _law().with_flank_coupling(e_star=100.0e9, offset=0.0)
