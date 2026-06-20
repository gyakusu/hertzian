"""Exercise the automatic calibration pipeline for the reduced Gothic-arch law.

:func:`hertzian.calibrate` turns a physical :class:`hertzian.GrooveSpec` into a
ready-to-use :class:`hertzian.GothicArchLaw` in one call. These tests pin the
geometry reduction, that the inserted coefficients match a hand-built law, and
that the optional field-solver verification confirms the Hertz ``3/2`` law, the
stiffness, the coupled flank count, and a large speed-up.
"""

from __future__ import annotations

import math

import pytest

import hertzian

# The README's conformal Gothic-arch bearing groove.
SPEC = hertzian.GrooveSpec(
    ball_radius=4.0e-3,
    tube_radius=4.16e-3,
    centre_radius=15.0e-3,
    centre_offset=65.0e-6,
    e_star=100.0e9,
)

# A small sweep keeps the verifying solves quick while still pinning the fit.
VERIFY_SAMPLES = 5

EXACT_RTOL = 1e-12
ZERO_TOL = 1e-18
HERTZ_EXPONENT = 1.5
EXPONENT_ABS_TOL = 0.05
MIN_R_SQUARED = 0.999
MAX_K_MISMATCH = 0.03
MAX_LOAD_RESIDUAL = 0.05
MAX_ETA_MISMATCH = 0.1
MIN_SPEEDUP = 100.0


@pytest.fixture(scope="module")
def verified() -> hertzian.Calibration:
    """Calibrate once with the field-solver verification for the whole module."""
    return hertzian.calibrate(SPEC, samples=VERIFY_SAMPLES)


def test_reduction_matches_the_groove_geometry() -> None:
    """The spec reduces to the conformal flank radii, offset and contact angle."""
    radius_x, radius_y = SPEC.flank_radii
    assert math.isclose(
        radius_x, 1.0 / (1.0 / SPEC.ball_radius + 1.0 / SPEC.centre_radius), rel_tol=EXACT_RTOL
    )
    assert math.isclose(
        radius_y, 1.0 / (1.0 / SPEC.ball_radius - 1.0 / SPEC.tube_radius), rel_tol=EXACT_RTOL
    )
    expected_offset = SPEC.centre_offset * SPEC.ball_radius / (SPEC.tube_radius - SPEC.ball_radius)
    assert math.isclose(SPEC.flank_offset, expected_offset, rel_tol=EXACT_RTOL)
    assert math.isclose(
        SPEC.contact_angle, math.asin(SPEC.flank_offset / SPEC.ball_radius), rel_tol=EXACT_RTOL
    )


def test_calibrate_without_verify_builds_the_analytic_law() -> None:
    """Skipping verification inserts the same coefficients a hand-built law has."""
    cal = hertzian.calibrate(SPEC, verify=False)
    assert cal.verification is None

    radius_x, radius_y = SPEC.flank_radii
    reference = hertzian.GothicArchLaw.from_elliptic_flank(
        radius_x=radius_x,
        radius_y=radius_y,
        e_star=SPEC.e_star,
        contact_angle=SPEC.contact_angle,
    ).with_flank_coupling(e_star=SPEC.e_star, offset=SPEC.flank_offset)
    assert math.isclose(cal.law.stiffness, reference.stiffness, rel_tol=EXACT_RTOL)
    assert math.isclose(cal.law.coupling, reference.coupling, rel_tol=EXACT_RTOL)
    assert math.isclose(cal.law.contact_angle, reference.contact_angle, rel_tol=EXACT_RTOL)

    # The law is usable straight away: a symmetric push is purely normal.
    f_t, f_n = cal.law.force(0.0, 6.0e-6)
    assert abs(f_t) <= ZERO_TOL
    assert f_n > 0.0


def test_describe_without_verification_is_a_report() -> None:
    """``describe()`` reports the coefficients and flags the skipped verification."""
    text = hertzian.calibrate(SPEC, verify=False).describe()
    assert "reduced Gothic-arch law calibration" in text
    assert "K     (stiffness)" in text
    assert "skipped" in text


def test_verify_confirms_the_hertz_law_and_stiffness(verified: hertzian.Calibration) -> None:
    """The solver sweep recovers the 3/2 exponent and the calibrated stiffness."""
    v = verified.verification
    assert v is not None
    assert math.isclose(v.fitted_exponent, HERTZ_EXPONENT, abs_tol=EXPONENT_ABS_TOL)
    assert v.r_squared > MIN_R_SQUARED
    assert abs(v.fitted_stiffness / v.analytic_stiffness - 1.0) < MAX_K_MISMATCH
    assert v.max_load_residual < MAX_LOAD_RESIDUAL


def test_verify_confirms_the_coupled_flank_count(verified: hertzian.Calibration) -> None:
    """The coupled law's effective flank count matches the field solver."""
    v = verified.verification
    assert v is not None
    assert abs(v.eta_law - v.eta_solver) / v.eta_solver < MAX_ETA_MISMATCH


def test_verify_confirms_the_speed_up(verified: hertzian.Calibration) -> None:
    """The reduced law evaluates orders of magnitude faster than a field solve."""
    v = verified.verification
    assert v is not None
    assert v.solver_seconds > v.law_seconds
    assert v.speedup > MIN_SPEEDUP


def test_describe_with_verification_reports_the_check(verified: hertzian.Calibration) -> None:
    """The verified report shows the solver check, the flank count and the speed-up."""
    text = verified.describe()
    assert "field solver" in text
    assert "eta (flank count)" in text
    assert "speed-up" in text


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"ball_radius": -1.0}, "ball_radius"),
        ({"e_star": 0.0}, "e_star"),
        ({"centre_offset": 0.0}, "centre_offset"),
        ({"tube_radius": 4.0e-3}, "tube_radius"),
    ],
)
def test_invalid_spec_raises(override: dict[str, float], match: str) -> None:
    """Out-of-range groove inputs map to a ``ValueError`` naming the offender."""
    fields = {
        "ball_radius": 4.0e-3,
        "tube_radius": 4.16e-3,
        "centre_radius": 15.0e-3,
        "centre_offset": 65.0e-6,
        "e_star": 100.0e9,
    }
    fields.update(override)
    with pytest.raises(ValueError, match=match):
        hertzian.GrooveSpec(**fields)
