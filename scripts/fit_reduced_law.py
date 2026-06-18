"""Calibrate and verify the reduced two-flank Gothic-arch contact law.

The FFT + BCCG core solves the *field* problem — a full pressure distribution per
configuration — which is far too heavy for a multibody inner loop. This script
distils it into the lightweight algebraic force law :class:`hertzian.GothicArchLaw`
``F(delta_t, delta_n) -> (F_t, F_n)``, **fitting the law to the shape** from a
sweep of full solves and checking the one boundary condition that matters: the
force varies **C¹** as the contact passes from two flanks to one, collapsing onto
the single-groove Hertz law ``F = k delta^{3/2}``.

It runs the solver ~100 times to produce four checks, drawn into one figure:

* **(A) Calibration.** A single-arc load sweep recovers the Hertz exponent ``3/2``
  and the per-flank stiffness ``K = P / delta^{3/2}`` (a free-slope regression
  lands on 1.500); a separated two-flank sweep lands on the ``2 K`` line, so two
  flanks superpose.
* **(B) The force law.** The calibrated ``F(delta_t, delta_n)`` swept transversely
  through the lift-off, with the single-flank Hertz asymptote — the deliverable.
* **(C) The C¹ "kiss".** The unloading flank follows the universal
  ``Q_-/Q_-(0) = (1 - xi)^{3/2}``, meeting zero with zero slope; solver markers
  (an asymmetric-well experiment) land on it. The ``3/2`` exponent *is* the C¹.
* **(D) Coupling.** Sweeping the shim from merged to separated, the effective flank
  count ``eta = P / (K delta^{3/2})`` runs from ~1 (single arc) toward 2 (two
  flanks). The uncoupled superposition is frozen at 2; the neighbour-lift law (each
  flank lifts the half-space under the other by ``u ~ Q/(pi E* . 2 y0)``) tracks the
  solver down into the half-overlap regime, closing almost all of the shortfall
  below 2 that the single-``K`` model would otherwise fold into its residual.

Run it (matplotlib is a render-only dependency, kept out of the locked env):

    uv run --with matplotlib python scripts/fit_reduced_law.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import hertzian

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Output path for the rendered panel (repo-root/docs/img).
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"

# Unit scales: SI in, engineering units out.
MM = 1.0e3
UM = 1.0e6
EPS = 2.220_446_049_250_313e-16

# A muted reference colour for analytic curves; warm tones for solver data.
REFERENCE_COLOUR = "#26c6da"
SOLVER_COLOUR = "#ef6c00"
LAW_COLOUR = "#6a1b9a"

# Plot the unloading markers a little past lift-off (xi = 1) to show they stay zero.
MARKER_XI_MAX = 1.2

# The applied example: the README's conformal Gothic-arch bearing groove.
BALL = 4.0e-3
TUBE = 4.16e-3  # r/Rs = 1.04, a textbook bearing conformity
CENTRE_RADIUS = 15.0e-3
E_STAR = 100.0e9


# --------------------------------------------------------------------------- #
# Analytic references (independent re-implementation of the Rust closed forms).
# --------------------------------------------------------------------------- #
def gothic_radii(ball: float, tube: float, centre_radius: float) -> tuple[float, float]:
    """Return the ``(circumferential, meridional)`` relative radii of a groove."""
    radius_x = 1.0 / (1.0 / ball + 1.0 / centre_radius)
    radius_y = 1.0 / (1.0 / ball - 1.0 / tube)
    return radius_x, radius_y


def _complete_elliptic_integrals(modulus: float) -> tuple[float, float]:
    """Return ``(K, E)`` for modulus ``k`` via the arithmetic-geometric mean."""
    a, b = 1.0, math.sqrt(max(1.0 - modulus * modulus, 0.0))
    summation = 0.5 * modulus * modulus
    two_pow = 1.0
    for _ in range(60):
        a_next = 0.5 * (a + b)
        b_next = math.sqrt(a * b)
        c_next = 0.5 * (a - b)
        summation += two_pow * c_next * c_next
        two_pow *= 2.0
        a, b = a_next, b_next
        if abs(c_next) <= EPS * a_next:
            break
    big_k = math.pi / (2.0 * a)
    return big_k, big_k * (1.0 - summation)


def _curvature_ratio(eccentricity: float) -> float:
    """Return the principal-radius ratio implied by a contact eccentricity."""
    big_k, big_e = _complete_elliptic_integrals(eccentricity)
    return (big_e / (1.0 - eccentricity * eccentricity) - big_k) / (big_k - big_e)


def elliptic_hertz(
    radius_x: float, radius_y: float, load: float, e_star: float
) -> tuple[float, ...]:
    """Return ``(semi_axis_x, semi_axis_y, approach)`` for elliptic Hertz."""
    radius_major, radius_minor = max(radius_x, radius_y), min(radius_x, radius_y)
    ratio = radius_major / radius_minor
    low, high = 0.0, 1.0 - 1.0e-12
    for _ in range(100):
        mid = 0.5 * (low + high)
        if _curvature_ratio(mid) < ratio:
            low = mid
        else:
            high = mid
    eccentricity = 0.5 * (low + high)
    big_k, big_e = _complete_elliptic_integrals(eccentricity)
    e_sq = eccentricity * eccentricity
    semi_major = (3.0 * load * radius_major * (big_k - big_e) / (math.pi * e_sq * e_star)) ** (
        1.0 / 3.0
    )
    semi_minor = semi_major * math.sqrt(1.0 - e_sq)
    approach = semi_major * semi_major * big_k * e_sq / (2.0 * radius_major * (big_k - big_e))
    if radius_x >= radius_y:
        return semi_major, semi_minor, approach
    return semi_minor, semi_major, approach


# --------------------------------------------------------------------------- #
# Solver sweeps.
# --------------------------------------------------------------------------- #
# Resolution and free-space margin for the load-deflection sweeps. The approach is
# an integral quantity that converges on coarse grids, and *anisotropic* spacing
# (one cell size per semi-axis) keeps the grid small even for the ~10:1 elongated
# conformal contact — without it, resolving the short axis would span the long one
# with many hundreds of cells.
CELLS_PER_SEMI = 8.0
MARGIN = 2.5
SOLVE_TOL = 1e-8
SOLVE_MAX_ITER = 40000


def _even(value: float) -> int:
    """Return ``value`` rounded up to the next even integer (>= 24)."""
    n = max(math.ceil(value), 24)
    return n + (n & 1)


def single_arc_point(load: float) -> tuple[float, float]:
    """Solve the single-arc (one-flank) groove; return ``(approach, total_load)``.

    The anisotropic grid is sized to the analytic elliptic contact at this load, so
    the discretisation error stays roughly constant across the sweep — the clean
    power-law data the calibration regresses.
    """
    radius_x, radius_y = gothic_radii(BALL, TUBE, CENTRE_RADIUS)
    ax_a, ay_a, _ = elliptic_hertz(radius_x, radius_y, load, E_STAR)
    dx, dy = ax_a / CELLS_PER_SEMI, ay_a / CELLS_PER_SEMI
    nx = _even(2.0 * MARGIN * ax_a / dx)
    ny = _even(2.0 * MARGIN * ay_a / dy)
    sol = hertzian.solve_sphere_in_gothic_arch(
        sphere_radius=BALL,
        tube_radius=TUBE,
        centre_radius=CENTRE_RADIUS,
        centre_offset=0.0,
        load=load,
        e_star=E_STAR,
        grid=(nx, ny),
        domain=(nx * dx, ny * dy),
        tol=SOLVE_TOL,
        max_iter=SOLVE_MAX_ITER,
    )
    return sol.approach, sol.total_load


def two_flank_point(load: float, separation_in_b: float) -> tuple[float, float]:
    """Solve the two-flank groove; return ``(approach, total_load)``.

    ``separation_in_b`` sets the flank offset in units of the meridional flank
    semi-axis ``b`` at half load, so the same separation reads the same across the
    load sweep. A few ``b`` is well separated; a fraction of a ``b`` is the overlap.
    """
    radius_x, radius_y = gothic_radii(BALL, TUBE, CENTRE_RADIUS)
    ax_a, ay_a, _ = elliptic_hertz(radius_x, radius_y, load / 2.0, E_STAR)
    y0 = separation_in_b * ay_a
    offset = y0 * (TUBE - BALL) / BALL
    dx, dy = ax_a / CELLS_PER_SEMI, ay_a / CELLS_PER_SEMI
    nx = _even(2.0 * MARGIN * ax_a / dx)
    ny = _even(2.0 * (y0 + MARGIN * ay_a) / dy)
    sol = hertzian.solve_sphere_in_gothic_arch(
        sphere_radius=BALL,
        tube_radius=TUBE,
        centre_radius=CENTRE_RADIUS,
        centre_offset=offset,
        load=load,
        e_star=E_STAR,
        grid=(nx, ny),
        domain=(nx * dx, ny * dy),
        tol=SOLVE_TOL,
        max_iter=SOLVE_MAX_ITER,
    )
    return sol.approach, sol.total_load


def asymmetric_unloading(load: float, samples: int) -> tuple[NDArray[np.float64], ...]:
    """Drive one flank to lift-off and watch its load vanish (the solver C¹ check).

    Two well-separated flanks are loaded at a fixed total ``load``; a lateral drive
    ``u`` lowers one well floor and raises the other (the half-space stand-in for
    displacing the ball toward one flank). As ``u`` grows the far flank unloads
    until it lifts off, dropping the contact from two flanks to one. Returns the
    normalised drive ``xi = u / u*`` and the two per-flank loads normalised by their
    symmetric value, with ``u*`` the lift-off drive.
    """
    radius_x, radius_y = gothic_radii(BALL, TUBE, CENTRE_RADIUS)
    ax_a, ay_a, approach = elliptic_hertz(radius_x, radius_y, load / 2.0, E_STAR)
    y0 = 2.0 * ay_a  # well separated, so each flank reads as its own patch
    dx, dy = ax_a / CELLS_PER_SEMI, ay_a / CELLS_PER_SEMI
    nx = _even(2.0 * MARGIN * ax_a / dx)
    ny = _even(2.0 * (y0 + MARGIN * ay_a) / dy)
    x = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * dy
    cell = dx * dy
    mid = ny // 2

    def split(drive: float) -> tuple[float, float]:
        well_plus = (y - y0) ** 2 / (2.0 * radius_y) - drive
        well_minus = (y + y0) ** 2 / (2.0 * radius_y) + drive
        gap = x[:, None] ** 2 / (2.0 * radius_x) + np.minimum(well_plus, well_minus)[None, :]
        sol = hertzian.solve_height_field(
            gap=np.ascontiguousarray(gap),
            load=load,
            e_star=E_STAR,
            dx=dx,
            dy=dy,
            tol=SOLVE_TOL,
            max_iter=SOLVE_MAX_ITER,
        )
        pressure = np.asarray(sol.pressure)
        return float(pressure[:, mid:].sum() * cell), float(pressure[:, :mid].sum() * cell)

    drives = np.linspace(0.0, 1.0 * approach, samples)
    plus0, minus0 = split(0.0)
    plus = np.empty(samples)
    minus = np.empty(samples)
    for i, drive in enumerate(drives):
        plus[i], minus[i] = split(float(drive))

    # Lift-off drive u*: where the far flank's load first reaches zero.
    lifted = np.flatnonzero(minus <= 1e-6 * minus0)
    u_star = float(drives[lifted[0]]) if lifted.size else float(drives[-1])
    xi = drives / u_star
    return xi, plus / plus0, minus / minus0


def coupling_curve(
    load: float, separations_in_b: NDArray[np.float64], stiffness: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the solver and coupled-law effective flank counts per separation.

    Sweeping the flank separation from a fraction of ``b`` (overlapping, one nearly
    merged arc, ``eta -> 1``) out to several ``b`` (two separated flanks,
    ``eta -> 2``) traces the geometry-driven two-to-one transition. The solver value
    ``eta = P/(K delta^{3/2})`` is returned alongside the reduced law's neighbour-lift
    prediction at the same approach: each flank lifts the half-space under the other
    by ``u ~ Q/(pi E* . 2 y0)``, so the coupled flank loads carry a sub-2 ``eta`` that
    tracks the solver where the uncoupled single-``K`` superposition stays pinned at 2.
    """
    radius_x, radius_y = gothic_radii(BALL, TUBE, CENTRE_RADIUS)
    semi_minor_meridional = elliptic_hertz(radius_x, radius_y, load / 2.0, E_STAR)[1]
    eta_solver = np.empty(separations_in_b.size)
    eta_law = np.empty(separations_in_b.size)
    for i, separation in enumerate(separations_in_b):
        approach, total = two_flank_point(load, float(separation))
        eta_solver[i] = total / (stiffness * approach**1.5)
        # The coupled law at the solver's own approach, calibrated to the same flank
        # (the contact angle is irrelevant to the symmetric flank-load split here).
        y0 = float(separation) * semi_minor_meridional
        law = hertzian.GothicArchLaw.from_elliptic_flank(
            radius_x=radius_x, radius_y=radius_y, e_star=E_STAR, contact_angle=0.1
        ).with_flank_coupling(e_star=E_STAR, offset=y0)
        q_plus, q_minus = law.coupled_loads(approach, approach)
        eta_law[i] = (q_plus + q_minus) / (stiffness * approach**1.5)
    return eta_solver, eta_law


# --------------------------------------------------------------------------- #
# Regression.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Calibration:
    """The fitted per-flank law and the data it was fit to."""

    exponent: float
    stiffness: float
    r_squared: float
    single_delta: NDArray[np.float64]
    single_load: NDArray[np.float64]
    two_delta: NDArray[np.float64]
    two_load: NDArray[np.float64]


def fit_power_law(
    delta: NDArray[np.float64], load: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Fit ``P = K delta^m``; return ``(free exponent m, stiffness K at m=3/2, R²)``."""
    log_d = np.log(delta)
    log_p = np.log(load)
    exponent = float(np.polyfit(log_d, log_p, 1)[0])
    log_k = float(np.mean(log_p - 1.5 * log_d))
    predicted = log_k + 1.5 * log_d
    ss_res = float(np.sum((log_p - predicted) ** 2))
    ss_tot = float(np.sum((log_p - np.mean(log_p)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return exponent, math.exp(log_k), r_squared


def calibrate() -> Calibration:
    """Run the calibration sweeps and fit the per-flank Hertz law."""
    loads = np.geomspace(15.0, 600.0, 12)
    single = np.array([single_arc_point(float(p)) for p in loads])
    two = np.array([two_flank_point(float(p), 3.0) for p in loads])  # 3 b: well separated
    exponent, stiffness, r_squared = fit_power_law(single[:, 0], single[:, 1])
    return Calibration(
        exponent=exponent,
        stiffness=stiffness,
        r_squared=r_squared,
        single_delta=single[:, 0],
        single_load=single[:, 1],
        two_delta=two[:, 0],
        two_load=two[:, 1],
    )


# --------------------------------------------------------------------------- #
# Figure.
# --------------------------------------------------------------------------- #
def _panel_calibration(ax: plt.Axes, cal: Calibration) -> None:
    """Draw the log-log calibration: solver points on the K and 2K Hertz lines."""
    delta_line = np.geomspace(
        min(cal.single_delta.min(), cal.two_delta.min()),
        max(cal.single_delta.max(), cal.two_delta.max()),
        100,
    )
    ax.plot(
        delta_line * UM,
        cal.stiffness * delta_line**1.5,
        color=REFERENCE_COLOUR,
        lw=2.0,
        label=r"$K\,\delta^{3/2}$ (one flank)",
    )
    ax.plot(
        delta_line * UM,
        2.0 * cal.stiffness * delta_line**1.5,
        color=REFERENCE_COLOUR,
        lw=2.0,
        ls="--",
        label=r"$2K\,\delta^{3/2}$ (two flanks)",
    )
    ax.scatter(
        cal.single_delta * UM,
        cal.single_load,
        s=26,
        c=SOLVER_COLOUR,
        zorder=3,
        label="solver, 1 arc",
    )
    ax.scatter(
        cal.two_delta * UM,
        cal.two_load,
        s=26,
        marker="s",
        c=LAW_COLOUR,
        zorder=3,
        label="solver, 2 flanks",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"approach $\delta$ (µm)")
    ax.set_ylabel("total load $P$ (N)")
    ax.set_title(
        f"(A) Calibration — exponent {cal.exponent:.3f}, "
        f"$K$={cal.stiffness:.3g}, $R^2$={cal.r_squared:.5f}",
        fontweight="bold",
        fontsize=10,
    )
    ax.grid(visible=True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def _panel_force(ax: plt.Axes, law: hertzian.GothicArchLaw) -> None:
    """Draw the calibrated force vector swept transversely through lift-off."""
    delta_n = 6.0e-6
    seam = law.lift_off_transverse(delta_n)
    delta_t = np.linspace(0.0, 1.6 * seam, 400)
    forces = np.array([law.force(float(t), delta_n) for t in delta_t])
    f_t, f_n = forces[:, 0], forces[:, 1]
    magnitude = np.hypot(f_t, f_n)

    ax.axvspan(seam / seam, 1.6, color="0.92", label="one flank")
    ax.plot(delta_t / seam, magnitude, color="black", lw=2.4, label=r"$|F|$")
    ax.plot(delta_t / seam, f_n, color=LAW_COLOUR, lw=1.8, label=r"$F_n$ (normal)")
    ax.plot(delta_t / seam, f_t, color=SOLVER_COLOUR, lw=1.8, label=r"$F_t$ (transverse)")

    # The single-flank Hertz asymptote: past lift-off |F| is exactly K s_+^{3/2}.
    sin, cos = math.sin(law.contact_angle), math.cos(law.contact_angle)
    s_plus = delta_n * cos + delta_t * sin
    ax.plot(
        delta_t / seam,
        law.stiffness * s_plus**1.5,
        color=REFERENCE_COLOUR,
        lw=1.6,
        ls=":",
        label=r"single Hertz $K s_+^{3/2}$",
    )
    ax.axvline(1.0, color="0.4", lw=1.0, ls="--")
    ax.annotate(
        "2 → 1 flank\n(C¹: no kink)",
        xy=(1.0, float(np.interp(1.0, delta_t / seam, magnitude))),
        xytext=(1.05, 0.45 * float(magnitude.max())),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "0.4"},
    )
    ax.set_xlabel(r"transverse displacement $\delta_t / \delta_t^*$")
    ax.set_ylabel("contact force (N)")
    ax.set_title(
        r"(B) Reduced force law $F(\delta_t,\delta_n)$ at fixed $\delta_n$",
        fontweight="bold",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.6)
    ax.set_ylim(bottom=0.0)
    ax.grid(visible=True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def _panel_kiss(ax: plt.Axes, xi: NDArray[np.float64], minus: NDArray[np.float64]) -> None:
    """Draw the universal unloading curve and the solver markers landing on it."""
    xi_line = np.linspace(0.0, 1.0, 200)
    ax.plot(
        xi_line,
        (1.0 - xi_line) ** 1.5,
        color=REFERENCE_COLOUR,
        lw=2.2,
        label=r"law  $(1-\xi)^{3/2}$",
    )
    ax.plot([1.0, 1.6], [0.0, 0.0], color=REFERENCE_COLOUR, lw=2.2)
    inside = xi <= MARKER_XI_MAX
    ax.scatter(
        xi[inside], minus[inside], s=30, c=SOLVER_COLOUR, zorder=3, label="solver (far flank)"
    )
    ax.axvline(1.0, color="0.4", lw=1.0, ls="--")
    ax.annotate(
        "tangent to zero\n⇒ $C^1$, not $C^2$",
        xy=(1.0, 0.0),
        xytext=(0.62, 0.42),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "0.4"},
    )
    ax.set_xlabel(r"normalised transverse drive $\xi = u / u^*$")
    ax.set_ylabel(r"far-flank load $Q_- / Q_-(0)$")
    ax.set_title(
        "(C) The unloading flank kisses zero — the $C^1$ handover",
        fontweight="bold",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.4)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(visible=True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def _panel_coupling(
    ax: plt.Axes,
    offsets_over_b: NDArray[np.float64],
    eta_solver: NDArray[np.float64],
    eta_law: NDArray[np.float64],
) -> None:
    """Draw the effective flank count: solver points and the coupled-law curve."""
    ax.axhline(1.0, color="0.6", lw=1.0, ls=":", label="1 flank (single arc)")
    ax.axhline(2.0, color="0.6", lw=1.0, ls="--", label="2 flanks (uncoupled)")
    ax.plot(
        offsets_over_b,
        eta_law,
        color=LAW_COLOUR,
        lw=2.0,
        label=r"coupled law  $u\sim Q/(\pi E^* \cdot 2y_0)$",
    )
    ax.scatter(
        offsets_over_b,
        eta_solver,
        s=28,
        c=SOLVER_COLOUR,
        zorder=3,
        label=r"solver  $\eta = P/(K\delta^{3/2})$",
    )
    ax.axvspan(0.4, 0.5, color="0.92", label="half overlap")
    ax.set_xlabel(r"flank separation  $y_0 / b$")
    ax.set_ylabel(r"effective flank count $\eta$")
    ax.set_title(
        "(D) Geometry-driven 2 → 1: the lift tracks η below 2",
        fontweight="bold",
        fontsize=10,
    )
    ax.set_ylim(0.8, 2.2)
    ax.grid(visible=True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def main() -> None:
    """Run every sweep, fit the law, print a summary and render the figure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})
    print("calibrating the reduced Gothic-arch law against the field solver ...")

    cal = calibrate()
    radius_x, radius_y = gothic_radii(BALL, TUBE, CENTRE_RADIUS)
    # Contact angle from the gallery shim (geometry only; it sets the force split).
    y0 = 65.0e-6 * BALL / (TUBE - BALL)
    alpha = hertzian.contact_half_angle(offset=y0, ball_radius=BALL)
    law = hertzian.GothicArchLaw.from_elliptic_flank(
        radius_x=radius_x, radius_y=radius_y, e_star=E_STAR, contact_angle=alpha
    )

    print(f"  fitted exponent  m = {cal.exponent:.4f}   (Hertz: 1.5)")
    print(f"  per-flank K       = {cal.stiffness:.6g} N/m^1.5")
    ratio = cal.stiffness / law.stiffness
    print(f"  analytic K        = {law.stiffness:.6g} N/m^1.5  (ratio {ratio:.4f})")
    print(f"  fit R^2           = {cal.r_squared:.6f}")
    print(f"  contact angle a   = {math.degrees(alpha):.2f} deg")

    xi, _plus, minus = asymmetric_unloading(120.0, 11)
    residual = float(np.max(np.abs(minus[xi <= 1.0] - (1.0 - xi[xi <= 1.0]) ** 1.5)))
    print(f"  unloading vs (1-xi)^1.5 : max residual {residual:.3f}")

    separations_over_b = np.linspace(0.4, 6.0, 12)
    eta_solver, eta_law = coupling_curve(120.0, separations_over_b, cal.stiffness)
    print(
        f"  effective flank count eta: {eta_solver.min():.2f} (merged) -> "
        f"{eta_solver.max():.2f} (separated)"
    )
    coupling_residual = float(np.max(np.abs(eta_law - eta_solver) / eta_solver))
    print(f"  coupled-law eta vs solver: max residual {100.0 * coupling_residual:.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0))
    _panel_calibration(axes[0, 0], cal)
    _panel_force(axes[0, 1], law)
    _panel_kiss(axes[1, 0], xi, minus)
    _panel_coupling(axes[1, 1], separations_over_b, eta_solver, eta_law)
    fig.suptitle(
        "hertzian — a reduced, C¹ two-flank contact law fit to the field solver",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "reduced_law.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024.0
    print(f"  wrote {path.relative_to(OUT_DIR.parent.parent)}  ({size_kb:.0f} KiB)")
    print("done.")


if __name__ == "__main__":
    main()
