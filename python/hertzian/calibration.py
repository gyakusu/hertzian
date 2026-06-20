"""Automatic calibration pipeline for the reduced Gothic-arch contact law.

The Rust core (:class:`hertzian.GothicArchLaw`) is a *pure function*
``F(delta_t, delta_n) -> (F_t, F_n)`` parameterised by three coefficients: the
per-flank Hertz stiffness ``K``, the contact half-angle ``alpha``, and the
neighbour-lift cross-compliance ``kappa``. This module is the Python shell that
turns a *physical* groove -- a ball in a Gothic-arch race, given by its radii,
shim offset and modulus -- into those coefficients and inserts them, so a usable
law is one call away:

    >>> import hertzian
    >>> spec = hertzian.GrooveSpec(
    ...     ball_radius=4.0e-3,
    ...     tube_radius=4.16e-3,
    ...     centre_radius=15.0e-3,
    ...     centre_offset=65.0e-6,
    ...     e_star=100.0e9,
    ... )
    >>> cal = hertzian.calibrate(spec)  # geometry -> coefficients (+ solver check)
    >>> f_t, f_n = cal.law.force(1.0e-6, 6.0e-6)
    >>> print(cal.describe())  # coefficients, accuracy, speed

By default :func:`calibrate` also *verifies* the reduction against the very
FFT+BCCG field solver it distils: a short single-arc load sweep confirms the
Hertzian ``3/2`` exponent and the stiffness ``K`` (and times the speed-up), and
one two-flank solve checks the coupled effective flank count ``eta``. Pass
``verify=False`` to skip the solver and get the analytic coefficients instantly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hertzian._core import (
    GothicArchLaw,
    contact_half_angle,
    solve_sphere_in_gothic_arch,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "Calibration",
    "FlankReduction",
    "GrooveSpec",
    "SolverVerification",
    "calibrate",
]

# The Hertzian load exponent the calibration both assumes and checks for.
_HERTZ_EXPONENT = 1.5

# Grid sizing for the verification solves. The approach is an integral quantity
# that converges on coarse grids, and anisotropic spacing (one cell size per
# semi-axis) keeps the grid small even for a ~10:1 elongated conformal contact.
_CELLS_PER_SEMI = 8.0
_MARGIN = 2.5
_MIN_GRID = 24

# Default single-arc load sweep for the fit (newtons): a handful of points across
# a decade is enough to pin the exponent and stiffness without a heavy solve count.
_DEFAULT_SAMPLES = 6
_MIN_SAMPLES = 2
_FIT_LOAD_MIN = 20.0
_FIT_LOAD_MAX = 400.0

# Field-solver settings for the verification solves.
_SOLVE_TOL = 1.0e-8
_SOLVE_MAX_ITER = 20_000

# Repetitions for timing one reduced-law force evaluation. Indicative, not a
# benchmark: enough to average out per-call noise.
_FORCE_TIMING_REPS = 100_000
_TIMING_DELTA_T = 1.0e-6
_TIMING_DELTA_N = 6.0e-6

# Display scales for describe().
_MS_PER_S = 1.0e3
_NS_PER_S = 1.0e9
_PERCENT = 100.0


@dataclass(frozen=True)
class GrooveSpec:
    """A physical ball-in-Gothic-arch-groove contact, in SI units.

    The inputs a bearing engineer already has; :func:`calibrate` reduces them to
    the flank coefficients of :class:`hertzian.GothicArchLaw`.

    Attributes:
        ball_radius: Ball radius ``R_s`` (m).
        tube_radius: Groove (tube) radius ``r`` (m); must exceed ``ball_radius``
            for a conformal contact.
        centre_radius: Reference centre-circle radius ``R_0`` (m).
        centre_offset: Arc-centre shim (m), the small offset of each groove arc
            from the centre line. Must be positive: a zero offset is a single-arc
            groove, not a two-flank Gothic law.
        e_star: Effective contact modulus ``E*`` (Pa).
    """

    ball_radius: float
    tube_radius: float
    centre_radius: float
    centre_offset: float
    e_star: float

    def __post_init__(self) -> None:
        """Validate the geometry, mirroring the Rust constructors' preconditions."""
        values = {
            "ball_radius": self.ball_radius,
            "tube_radius": self.tube_radius,
            "centre_radius": self.centre_radius,
            "centre_offset": self.centre_offset,
            "e_star": self.e_star,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0.0:
                msg = f"GrooveSpec.{name} must be positive and finite, got {value!r}"
                raise ValueError(msg)
        if self.ball_radius >= self.tube_radius:
            msg = (
                "GrooveSpec needs ball_radius < tube_radius for a conformal groove "
                f"contact, got ball_radius={self.ball_radius!r}, tube_radius={self.tube_radius!r}"
            )
            raise ValueError(msg)
        if self.flank_offset >= self.ball_radius:
            msg = (
                "GrooveSpec centre_offset is too large: the amplified flank offset "
                f"y0={self.flank_offset!r} m must stay below ball_radius={self.ball_radius!r} m"
            )
            raise ValueError(msg)

    @property
    def flank_radii(self) -> tuple[float, float]:
        """The ``(circumferential R_x, meridional R_y)`` relative radii (m)."""
        radius_x = 1.0 / (1.0 / self.ball_radius + 1.0 / self.centre_radius)
        radius_y = 1.0 / (1.0 / self.ball_radius - 1.0 / self.tube_radius)
        return radius_x, radius_y

    @property
    def flank_offset(self) -> float:
        """The meridional flank offset ``y0`` (m): the shim amplified by ``R_s/(r-R_s)``."""
        return self.centre_offset * self.ball_radius / (self.tube_radius - self.ball_radius)

    @property
    def contact_angle(self) -> float:
        """The geometric contact half-angle ``alpha = arcsin(y0 / R_s)`` (rad)."""
        return contact_half_angle(offset=self.flank_offset, ball_radius=self.ball_radius)


@dataclass(frozen=True)
class FlankReduction:
    """The per-flank elliptic-Hertz geometry a :class:`GrooveSpec` reduces to.

    Attributes:
        radius_x: Circumferential (convex) relative radius ``R_x`` (m).
        radius_y: Meridional (conformal) relative radius ``R_y`` (m).
        flank_offset: Meridional flank offset ``y0`` (m); the flanks sit at ``+/- y0``.
        contact_angle: Geometric contact half-angle ``alpha`` (rad).
    """

    radius_x: float
    radius_y: float
    flank_offset: float
    contact_angle: float


@dataclass(frozen=True)
class SolverVerification:
    """Field-solver cross-check of the calibrated coefficients.

    Produced by :func:`calibrate` with ``verify=True``: a short single-arc load
    sweep refits the stiffness and confirms the Hertzian exponent, one two-flank
    solve checks the coupled effective flank count, and both are timed against a
    reduced-law evaluation.

    Attributes:
        loads: The applied loads of the single-arc sweep (N).
        approaches: The solver approach at each load (m).
        solver_loads: The solver's integrated total load at each point (N) -- the
            ground truth the fit regresses.
        fitted_exponent: Free exponent of ``P = K delta^m`` (Hertz: 1.5).
        fitted_stiffness: Stiffness ``K`` from the fit at the fixed 3/2 slope.
        analytic_stiffness: The law's analytic ``K`` (the coefficient in use).
        r_squared: Coefficient of determination of the 3/2-slope fit.
        max_load_residual: Max relative error of the law's load vs the solver.
        eta_solver: Two-flank effective flank count from the solver.
        eta_law: The coupled law's effective flank count at the same approach.
        solver_seconds: Mean wall time of one field solve (s).
        law_seconds: Mean wall time of one reduced-law force evaluation (s).
        speedup: ``solver_seconds / law_seconds``.
    """

    loads: tuple[float, ...]
    approaches: tuple[float, ...]
    solver_loads: tuple[float, ...]
    fitted_exponent: float
    fitted_stiffness: float
    analytic_stiffness: float
    r_squared: float
    max_load_residual: float
    eta_solver: float
    eta_law: float
    solver_seconds: float
    law_seconds: float
    speedup: float


@dataclass(frozen=True)
class Calibration:
    """A calibrated reduced law and the record of how it was obtained.

    Returned by :func:`calibrate`. ``law`` is ready for the multibody inner loop;
    :meth:`describe` renders the coefficients and (if run) the verification.

    Attributes:
        spec: The physical groove the law was calibrated from.
        reduction: The per-flank geometry the spec reduced to.
        law: The calibrated, ready-to-use reduced force law.
        verification: The field-solver cross-check, or ``None`` if skipped.
    """

    spec: GrooveSpec
    reduction: FlankReduction
    law: GothicArchLaw
    verification: SolverVerification | None

    def describe(self) -> str:
        """Return a human-readable report of the coefficients and verification."""
        return _describe(self)


def calibrate(
    spec: GrooveSpec,
    *,
    verify: bool = True,
    samples: int = _DEFAULT_SAMPLES,
    tol: float = _SOLVE_TOL,
    max_iter: int = _SOLVE_MAX_ITER,
) -> Calibration:
    """Reduce a groove to the coefficients of a :class:`hertzian.GothicArchLaw`.

    Builds the law analytically from the geometry (instant, no solve) and, unless
    ``verify=False``, cross-checks it against the field solver it distils.

    Args:
        spec: The physical groove and material.
        verify: Run the field-solver verification sweep (default ``True``).
        samples: Number of single-arc loads in the verification sweep (``>= 2``).
        tol: Field-solver convergence tolerance for the verification solves.
        max_iter: Field-solver iteration cap for the verification solves.

    Returns:
        A :class:`Calibration` holding the law and, if requested, the verification.
    """
    radius_x, radius_y = spec.flank_radii
    reduction = FlankReduction(
        radius_x=radius_x,
        radius_y=radius_y,
        flank_offset=spec.flank_offset,
        contact_angle=spec.contact_angle,
    )
    # Insert the derived coefficients into the Rust pure-function foundation: the
    # analytic stiffness K and shape from the flank, the contact angle, and the
    # neighbour-lift kappa from the flank offset (exact in both the separated and
    # the lifted-off limits, so always safe to enable).
    law = GothicArchLaw.from_elliptic_flank(
        radius_x=radius_x,
        radius_y=radius_y,
        e_star=spec.e_star,
        contact_angle=reduction.contact_angle,
    ).with_flank_coupling(e_star=spec.e_star, offset=reduction.flank_offset)
    verification = (
        _verify(spec, law, samples=samples, tol=tol, max_iter=max_iter) if verify else None
    )
    return Calibration(spec=spec, reduction=reduction, law=law, verification=verification)


def _verify(
    spec: GrooveSpec,
    law: GothicArchLaw,
    *,
    samples: int,
    tol: float,
    max_iter: int,
) -> SolverVerification:
    """Run the field-solver sweep and assemble the verification record."""
    if samples < _MIN_SAMPLES:
        msg = f"calibrate needs samples >= {_MIN_SAMPLES} to fit the law, got {samples}"
        raise ValueError(msg)

    loads = np.geomspace(_FIT_LOAD_MIN, _FIT_LOAD_MAX, samples)
    approaches = np.empty(samples)
    solver_loads = np.empty(samples)
    solver_time = 0.0
    for i, load in enumerate(loads):
        start = time.perf_counter()
        delta, total = _single_arc_solve(spec, law, float(load), tol=tol, max_iter=max_iter)
        solver_time += time.perf_counter() - start
        approaches[i] = delta
        solver_loads[i] = total

    exponent, fitted_stiffness, r_squared = _fit_power_law(approaches, solver_loads)
    analytic_stiffness = law.stiffness
    predicted = analytic_stiffness * approaches**_HERTZ_EXPONENT
    max_residual = float(np.max(np.abs(predicted - solver_loads) / solver_loads))

    solver_seconds = solver_time / samples
    law_seconds = _time_force(law)
    speedup = solver_seconds / law_seconds if law_seconds > 0.0 else math.inf

    eta_solver, eta_law = _check_effective_flank_count(spec, law, loads, tol=tol, max_iter=max_iter)

    return SolverVerification(
        loads=tuple(float(v) for v in loads),
        approaches=tuple(float(v) for v in approaches),
        solver_loads=tuple(float(v) for v in solver_loads),
        fitted_exponent=exponent,
        fitted_stiffness=fitted_stiffness,
        analytic_stiffness=analytic_stiffness,
        r_squared=r_squared,
        max_load_residual=max_residual,
        eta_solver=eta_solver,
        eta_law=eta_law,
        solver_seconds=solver_seconds,
        law_seconds=law_seconds,
        speedup=speedup,
    )


def _check_effective_flank_count(
    spec: GrooveSpec,
    law: GothicArchLaw,
    loads: NDArray[np.float64],
    *,
    tol: float,
    max_iter: int,
) -> tuple[float, float]:
    """Compare the solver and coupled-law effective flank count at the spec offset.

    At a representative mid-sweep load, a two-flank solve gives the effective flank
    count ``eta = P / (K delta^{3/2})``; the coupled law predicts the same ``eta``
    from its loads at that approach. Their agreement validates the coupling ``kappa``.
    """
    load = float(np.sqrt(loads[0] * loads[-1]))
    delta, total = _two_flank_solve(spec, law, load, tol=tol, max_iter=max_iter)
    reference = law.stiffness * delta**_HERTZ_EXPONENT
    q_plus, q_minus = law.coupled_loads(delta, delta)
    return total / reference, (q_plus + q_minus) / reference


def _single_arc_solve(
    spec: GrooveSpec,
    law: GothicArchLaw,
    load: float,
    *,
    tol: float,
    max_iter: int,
) -> tuple[float, float]:
    """Solve the single-arc (one-flank) groove; return ``(approach, total_load)``."""
    semi_x, semi_y = law.flank_pressure(load).semi_axes
    nx, ny, width_x, width_y = _grid_for(semi_x, semi_y, flank_offset=0.0)
    sol = solve_sphere_in_gothic_arch(
        sphere_radius=spec.ball_radius,
        tube_radius=spec.tube_radius,
        centre_radius=spec.centre_radius,
        centre_offset=0.0,
        load=load,
        e_star=spec.e_star,
        grid=(nx, ny),
        domain=(width_x, width_y),
        tol=tol,
        max_iter=max_iter,
    )
    return sol.approach, sol.total_load


def _two_flank_solve(
    spec: GrooveSpec,
    law: GothicArchLaw,
    load: float,
    *,
    tol: float,
    max_iter: int,
) -> tuple[float, float]:
    """Solve the two-flank groove at the spec offset; return ``(approach, total_load)``."""
    semi_x, semi_y = law.flank_pressure(load / 2.0).semi_axes
    nx, ny, width_x, width_y = _grid_for(semi_x, semi_y, flank_offset=spec.flank_offset)
    sol = solve_sphere_in_gothic_arch(
        sphere_radius=spec.ball_radius,
        tube_radius=spec.tube_radius,
        centre_radius=spec.centre_radius,
        centre_offset=spec.centre_offset,
        load=load,
        e_star=spec.e_star,
        grid=(nx, ny),
        domain=(width_x, width_y),
        tol=tol,
        max_iter=max_iter,
    )
    return sol.approach, sol.total_load


def _grid_for(
    semi_x: float, semi_y: float, *, flank_offset: float
) -> tuple[int, int, float, float]:
    """Size an anisotropic centred grid for a contact of semi-axes ``(semi_x, semi_y)``.

    One cell size per semi-axis keeps the grid small for an elongated contact; the
    meridional extent also spans the two flanks at ``+/- flank_offset``. Returns
    ``(nx, ny, width_x, width_y)``.
    """
    dx = semi_x / _CELLS_PER_SEMI
    dy = semi_y / _CELLS_PER_SEMI
    nx = _even(2.0 * _MARGIN * semi_x / dx)
    ny = _even(2.0 * (flank_offset + _MARGIN * semi_y) / dy)
    return nx, ny, nx * dx, ny * dy


def _even(value: float) -> int:
    """Round ``value`` up to the next even integer, at least ``_MIN_GRID``."""
    n = max(math.ceil(value), _MIN_GRID)
    return n + (n & 1)


def _fit_power_law(
    delta: NDArray[np.float64], load: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Fit ``P = K delta^m``; return ``(free exponent m, K at m=3/2, R^2)``."""
    log_d = np.log(delta)
    log_p = np.log(load)
    exponent = float(np.polyfit(log_d, log_p, 1)[0])
    log_k = float(np.mean(log_p - _HERTZ_EXPONENT * log_d))
    predicted = log_k + _HERTZ_EXPONENT * log_d
    ss_res = float(np.sum((log_p - predicted) ** 2))
    ss_tot = float(np.sum((log_p - np.mean(log_p)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return exponent, math.exp(log_k), r_squared


def _time_force(law: GothicArchLaw) -> float:
    """Return the mean wall time of one reduced-law force evaluation (s)."""
    start = time.perf_counter()
    for _ in range(_FORCE_TIMING_REPS):
        law.force(_TIMING_DELTA_T, _TIMING_DELTA_N)
    return (time.perf_counter() - start) / _FORCE_TIMING_REPS


def _describe(cal: Calibration) -> str:
    """Render the calibration report (see :meth:`Calibration.describe`)."""
    spec = cal.spec
    red = cal.reduction
    law = cal.law
    ratio = spec.tube_radius / spec.ball_radius
    lines = [
        "hertzian: reduced Gothic-arch law calibration",
        "=" * 48,
        "groove (input):",
        f"  R_s   (ball)      = {spec.ball_radius:.4e} m",
        f"  r     (tube)      = {spec.tube_radius:.4e} m   (r/R_s = {ratio:.3f})",
        f"  R_0   (centre)    = {spec.centre_radius:.4e} m",
        f"  shim  (offset)    = {spec.centre_offset:.4e} m",
        f"  E*    (modulus)   = {spec.e_star:.4e} Pa",
        "reduced flank (derived):",
        f"  R_x  (circumf.)   = {red.radius_x:.4e} m",
        f"  R_y  (meridional) = {red.radius_y:.4e} m",
        f"  y0   (flank off.) = {red.flank_offset:.4e} m",
        f"  alpha (contact)   = {math.degrees(red.contact_angle):.2f} deg",
        "coefficients (inserted into the Rust law):",
        f"  K     (stiffness) = {law.stiffness:.6g} N/m^1.5",
        f"  kappa (coupling)  = {law.coupling:.6g} m/N",
    ]
    if cal.verification is None:
        lines.append("verification: skipped (verify=False)")
        return "\n".join(lines)

    v = cal.verification
    k_ratio = v.fitted_stiffness / v.analytic_stiffness
    eta_delta = abs(v.eta_law - v.eta_solver) / v.eta_solver
    lines += [
        "verification (FFT+BCCG field solver):",
        f"  exponent m        = {v.fitted_exponent:.4f}   (Hertz 1.5)",
        f"  fit R^2           = {v.r_squared:.6f}",
        f"  K (solver fit)    = {v.fitted_stiffness:.6g} N/m^1.5  (x{k_ratio:.3f} vs analytic)",
        f"  max load residual = {_PERCENT * v.max_load_residual:.2f} %",
        f"  eta (flank count) = {v.eta_solver:.3f} solver / {v.eta_law:.3f} law"
        f"  ({_PERCENT * eta_delta:.1f} %)",
        "speed:",
        f"  field solve       = {v.solver_seconds * _MS_PER_S:.2f} ms",
        f"  reduced force()   = {v.law_seconds * _NS_PER_S:.1f} ns",
        f"  speed-up          = {v.speedup:.3g} x",
    ]
    return "\n".join(lines)
