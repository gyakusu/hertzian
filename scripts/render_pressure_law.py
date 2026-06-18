"""Verify the reduced flank pressure distribution and its Coulomb spin moment.

The reduced force law :class:`hertzian.GothicArchLaw` gives each flank's *resultant*
load ``Q`` — enough for a frictionless normal contact. **Coulomb friction needs the
distribution**: the local traction is bounded by ``mu p(x, y)`` and the spin
(drilling) moment is ``mu int p rho dA``, neither recoverable from the net force.
This script checks the lightweight companion :class:`hertzian.FlankPressure` — the
elliptic-Hertz field ``p(x, y) = p0 sqrt(1 - (x/a_x)^2 - (y/a_y)^2)`` with ``a, p0``
both proportional to ``Q^(1/3)`` — against the FFT + BCCG field solver, in four
panels:

* **(A) The field.** The reduced two-flank pressure field (two semi-ellipsoids at
  ``y = ±y0``, scaled to the solver's own per-flank loads) under the field solver's
  contact outline — the lightweight field fills the same footprint.
* **(B) The profile.** The meridional cut ``p(0, y)``: the reduced semi-ellipsoids
  land on the solver's pressure samples (the distribution, not just the peak).
* **(C) Load scaling.** Semi-axes ``a_x, a_y`` and peak ``p₀`` swept over load on a
  single-arch contact, solver points on the reduced ``Q^{1/3}`` lines.
* **(D) The friction payoff.** The spin moment ``M = (3/8) mu Q a E(e)`` vs the field
  solver's first moment ``mu int p rho dA`` across the load sweep — the reduced law
  lands on the solver, while a circular-radius stand-in (a naive friction model) is
  off by tens of percent because the conformal patch is strongly elliptic.

Run it (matplotlib is a render-only dependency, kept out of the locked env):

    uv run --with matplotlib python scripts/render_pressure_law.py
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
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

# Output path for the rendered panel (repo-root/docs/img).
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"

# Unit scales: SI in, engineering units out (metres -> mm/µm, pascals -> GPa).
MM = 1.0e3
UM = 1.0e6
GPA = 1.0e-9
EPS = 2.220_446_049_250_313e-16

# Warm tones for the solver (exact), a muted reference for analytics, purple for
# the reduced law — the gallery palette.
SOLVER_COLOUR = "#ef6c00"
REFERENCE_COLOUR = "#26c6da"
LAW_COLOUR = "#6a1b9a"
CIRCULAR_COLOUR = "#9e9e9e"

# The applied example: the README's conformal Gothic-arch bearing groove.
BALL = 4.0e-3
TUBE = 4.16e-3  # r/Rs = 1.04, a textbook bearing conformity
CENTRE_RADIUS = 15.0e-3
E_STAR = 100.0e9

# Load for the two-flank field panels, and the flank separation (in meridional
# semi-axes b): one cleanly separated (each its own ellipse), one at the half
# overlap (centres b apart, one connected patch with a saddle between the peaks).
FIELD_LOAD = 120.0
SEPARATION_IN_B = 2.0
OVERLAP_IN_B = 0.5

# A representative friction coefficient for the spin-moment panel.
FRICTION = 0.12

# Solver resolution: cells per (short) semi-axis and free-space margin. The
# approach, semi-axes and pressure moments are integral quantities that converge
# on coarse anisotropic grids (one cell size per semi-axis keeps the ~6:1 patch
# small), so this is kept modest — the script runs the solver a couple of dozen
# times.
CELLS_PER_SEMI = 10.0
MARGIN = 2.5
SOLVE_TOL = 1.0e-9
SOLVE_MAX_ITER = 40000

# A cell is "in contact" above this fraction of the peak (the contact outline).
CONTACT_FLOOR = 1.0e-3


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
) -> tuple[float, float]:
    """Return the ``(semi_axis_x, semi_axis_y)`` of an elliptic Hertz contact."""
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
    if radius_x >= radius_y:
        return semi_major, semi_minor
    return semi_minor, semi_major


def _even(value: float) -> int:
    """Return ``value`` rounded up to the next even integer (>= 24)."""
    n = max(math.ceil(value), 24)
    return n + (n & 1)


# --------------------------------------------------------------------------- #
# Solver sweeps.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArchPoint:
    """One single-arch solve: the patch the reduced law is checked against."""

    load: float
    semi_axis_x: float
    semi_axis_y: float
    peak_pressure: float
    spin_moment: float  # field first moment integral(p rho) dA, per unit friction
    contact_radius: float


def single_arch_point(load: float) -> ArchPoint:
    """Solve a single-arch (one elliptic flank) contact and read off its moments.

    The patch carries the full load as one elliptic Hertz contact, so it is the
    cleanest cross-check of the reduced distribution: its semi-axes, peak, and the
    first moment of pressure about its centroid.
    """
    radius_x, radius_y = gothic_radii(BALL, TUBE, CENTRE_RADIUS)
    ax_a, ay_a = elliptic_hertz(radius_x, radius_y, load, E_STAR)
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
    pressure = np.asarray(sol.pressure)
    x = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * dy
    cell = dx * dy
    # First moment of pressure about the (origin-centred) patch centroid.
    rho = np.hypot(x[:, None], y[None, :])
    spin_moment = float(np.sum(pressure * rho) * cell)
    a_x, a_y = sol.contact_half_widths
    return ArchPoint(
        load=sol.total_load,
        semi_axis_x=a_x,
        semi_axis_y=a_y,
        peak_pressure=float(sol.max_pressure),
        spin_moment=spin_moment,
        contact_radius=float(sol.contact_radius),
    )


@dataclass(frozen=True)
class FieldCase:
    """One separated two-flank solve: the field and meridional cut for panels A/B."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    pressure: NDArray[np.float64]
    cut: NDArray[np.float64]
    y0: float
    load_plus: float
    load_minus: float


def separated_field(load: float, separation_in_b: float) -> FieldCase:
    """Solve a well-separated two-flank groove; return the field and the x=0 cut."""
    radius_x, radius_y = gothic_radii(BALL, TUBE, CENTRE_RADIUS)
    ax_a, ay_a = elliptic_hertz(radius_x, radius_y, load / 2.0, E_STAR)
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
    pressure = np.asarray(sol.pressure)
    x = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * dy
    cell = dx * dy
    mid = ny // 2
    load_plus = float(pressure[:, mid:].sum() * cell)
    load_minus = float(pressure[:, :mid].sum() * cell)
    return FieldCase(
        x=x,
        y=y,
        pressure=pressure,
        cut=pressure[nx // 2, :],
        y0=y0,
        load_plus=load_plus,
        load_minus=load_minus,
    )


def reduced_two_flank_field(flank: hertzian.FlankPressure, case: FieldCase) -> NDArray[np.float64]:
    """Build the reduced two-flank pressure field on the case's ``(x, y)`` grid (Pa).

    Two elliptic-Hertz semi-ellipsoids centred at ``y = ±y0`` and scaled to the
    per-flank loads, assembled by the pointwise *maximum* (not the sum): the rule
    that gives the connected overlap its saddle, identical to the sum where the
    patches are separated. The dual of the Gothic gap being the pointwise minimum.
    """
    field = np.zeros((case.x.size, case.y.size))
    for load, centre in ((case.load_plus, case.y0), (case.load_minus, -case.y0)):
        if load <= 0.0:
            continue
        a_x, a_y = flank.semi_axes(load)
        p0 = flank.peak_pressure(load)
        radial = (case.x[:, None] / a_x) ** 2 + ((case.y[None, :] - centre) / a_y) ** 2
        field = np.maximum(field, p0 * np.sqrt(np.clip(1.0 - radial, 0.0, None)))
    return field


# --------------------------------------------------------------------------- #
# Panels.
# --------------------------------------------------------------------------- #
def _panel_field(ax: Axes, case: FieldCase, flank: hertzian.FlankPressure) -> None:
    """Draw the reduced two-flank field with the solver's contact outline on top."""
    reduced = reduced_two_flank_field(flank, case)
    # y (meridional, long axis) horizontal; x (circumferential, short) vertical. The
    # reduced field is (nx, ny) = (x, y), exactly pcolormesh(y_horiz, x_vert, C). The
    # patch is ~10:1 elongated, so the axes are scaled independently (noted below).
    mesh = ax.pcolormesh(case.y * MM, case.x * MM, reduced * GPA, cmap="magma", shading="auto")
    plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, label="reduced $p$ (GPa)")
    # Solver contact outline (the in-contact boundary) and the reduced contact
    # ellipses: the lightweight field fills the same footprint the solver finds.
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    for load, centre in ((case.load_plus, case.y0), (case.load_minus, -case.y0)):
        a_x, a_y = flank.semi_axes(load)
        ax.plot(
            (centre + a_y * np.cos(theta)) * MM,
            a_x * np.sin(theta) * MM,
            ls="--",
            lw=1.3,
            color=REFERENCE_COLOUR,
        )
        ax.plot(centre * MM, 0.0, "o", ms=4.0, mfc="white", mec="black", zorder=4)
    floor = CONTACT_FLOOR * case.pressure.max()
    ax.contour(
        case.y * MM,
        case.x * MM,
        case.pressure,
        levels=[floor],
        colors=SOLVER_COLOUR,
        linewidths=1.6,
    )
    ax.plot([], [], ls="--", lw=1.3, color=REFERENCE_COLOUR, label="reduced contact ellipse")
    ax.plot([], [], color=SOLVER_COLOUR, lw=1.6, label="solver contact edge")
    ax.set_xlabel("y (mm) — meridional  ·  axes scaled independently")
    ax.set_ylabel("x (mm) — circumferential")
    ax.set_title(
        "(A) Reduced two-flank field fills the solver footprint",
        fontweight="bold",
        fontsize=10,
    )
    ax.legend(frameon=True, fontsize=8, loc="upper right", framealpha=0.9)


def _panel_profile(ax: Axes, case: FieldCase, flank: hertzian.FlankPressure) -> None:
    """Draw the meridional cut p(0, y): reduced semi-ellipsoids vs solver samples."""
    y_line = np.linspace(case.y.min(), case.y.max(), 600)
    reduced_cut = np.zeros_like(y_line)
    for load, centre in ((case.load_plus, case.y0), (case.load_minus, -case.y0)):
        _, a_y = flank.semi_axes(load)
        p0 = flank.peak_pressure(load)
        reduced_cut += p0 * np.sqrt(np.clip(1.0 - ((y_line - centre) / a_y) ** 2, 0.0, None))
    ax.plot(
        y_line * MM,
        reduced_cut * GPA,
        color=LAW_COLOUR,
        lw=2.2,
        label=r"reduced $p(0,y)$",
    )
    ax.scatter(
        case.y * MM, case.cut * GPA, s=12, c=SOLVER_COLOUR, alpha=0.8, zorder=3, label="solver"
    )
    for centre in (case.y0, -case.y0):
        ax.axvline(centre * MM, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel("y (mm) — meridional")
    ax.set_ylabel("pressure (GPa)")
    ax.set_title(
        r"(B) Meridional cut — the distribution, not just the peak",
        fontweight="bold",
        fontsize=10,
    )
    ax.set_ylim(bottom=0.0)
    ax.grid(visible=True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def _panel_scaling(ax: Axes, arch: list[ArchPoint], flank: hertzian.FlankPressure) -> None:
    """Draw the Q^{1/3} load scaling of the semi-axes and peak pressure."""
    loads = np.array([p.load for p in arch])
    line = np.geomspace(loads.min(), loads.max(), 100)
    ax_red = np.array([flank.semi_axes(p)[0] for p in line])
    ay_red = np.array([flank.semi_axes(p)[1] for p in line])
    ax.plot(
        line, ay_red * MM, color=REFERENCE_COLOUR, lw=2.0, label=r"reduced $a_y\propto Q^{1/3}$"
    )
    ax.plot(line, ax_red * MM, color=REFERENCE_COLOUR, lw=2.0, ls="--", label=r"reduced $a_x$")
    ax.scatter(
        loads,
        [p.semi_axis_y * MM for p in arch],
        s=26,
        c=SOLVER_COLOUR,
        zorder=3,
        label="solver $a_y$",
    )
    ax.scatter(
        loads,
        [p.semi_axis_x * MM for p in arch],
        s=26,
        marker="s",
        c=LAW_COLOUR,
        zorder=3,
        label="solver $a_x$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("flank load $Q$ (N)")
    ax.set_ylabel("contact semi-axis (mm)")
    ax.set_title("(C) Patch size scales as $Q^{1/3}$", fontweight="bold", fontsize=10)
    ax.grid(visible=True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def _panel_spin(ax: Axes, arch: list[ArchPoint], flank: hertzian.FlankPressure) -> None:
    """Draw the spin (drilling) moment: reduced closed form vs field, vs circular."""
    loads = np.array([p.load for p in arch])
    line = np.geomspace(loads.min(), loads.max(), 100)
    reduced = np.array([flank.spin_moment(p, FRICTION) for p in line])
    ax.plot(
        line,
        reduced,
        color=LAW_COLOUR,
        lw=2.2,
        label=r"reduced $\frac{3}{8}\mu Q\,a\,E(e)$",
    )
    ax.scatter(
        loads,
        [FRICTION * p.spin_moment for p in arch],
        s=30,
        c=SOLVER_COLOUR,
        zorder=3,
        label=r"solver $\mu\!\int\! p\,\rho\,dA$",
    )
    # The naive circular stand-in: a friction model using an equal-area disc radius.
    circular = np.array([FRICTION * p.load * 3.0 * math.pi / 16.0 * p.contact_radius for p in arch])
    ax.plot(
        loads,
        circular,
        color=CIRCULAR_COLOUR,
        lw=1.8,
        ls=":",
        label=r"circular stand-in $\frac{3\pi}{16}\mu Q\,a_{eq}$",
    )
    shortfall = float(np.mean(1.0 - circular / np.array([FRICTION * p.spin_moment for p in arch])))
    ax.annotate(
        f"circular model\nlow by {100.0 * shortfall:.0f}%\n($e={flank.eccentricity:.3f}$)",
        xy=(loads[len(loads) // 2], circular[len(loads) // 2]),
        xytext=(0.42, 0.18),
        textcoords="axes fraction",
        fontsize=8,
        color="0.35",
        arrowprops={"arrowstyle": "->", "color": "0.5"},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("flank load $Q$ (N)")
    ax.set_ylabel(rf"spin moment ($\mu={FRICTION}$) (N·m)")
    ax.set_title("(D) The friction payoff — the spin moment", fontweight="bold", fontsize=10)
    ax.grid(visible=True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def _panel_overlap_field(ax: Axes, case: FieldCase, flank: hertzian.FlankPressure) -> None:
    """Draw the connected half-overlap field (reduced pointwise max) + solver edge."""
    reduced = reduced_two_flank_field(flank, case)
    mesh = ax.pcolormesh(case.y * MM, case.x * MM, reduced * GPA, cmap="magma", shading="auto")
    plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, label="reduced $p$ (GPa)")
    floor = CONTACT_FLOOR * case.pressure.max()
    ax.contour(
        case.y * MM,
        case.x * MM,
        case.pressure,
        levels=[floor],
        colors=SOLVER_COLOUR,
        linewidths=1.6,
    )
    for centre in (case.y0, -case.y0):
        ax.plot(centre * MM, 0.0, "o", ms=4.0, mfc="white", mec="black", zorder=4)
    ax.plot([], [], color=SOLVER_COLOUR, lw=1.6, label="solver contact edge")
    ax.set_xlabel("y (mm) — meridional  ·  axes scaled independently")
    ax.set_ylabel("x (mm) — circumferential")
    ax.set_title(
        "(A) Half overlap — one connected patch (reduced = pointwise max)",
        fontweight="bold",
        fontsize=10,
    )
    ax.legend(frameon=True, fontsize=8, loc="upper right", framealpha=0.9)


def _panel_overlap_cut(ax: Axes, case: FieldCase, flank: hertzian.FlankPressure) -> None:
    """Draw the overlap meridional cut: solver vs reduced max vs the wrong sum."""
    y_line = np.linspace(case.y.min(), case.y.max(), 600)
    red_max = np.array(
        [
            flank.two_flank_pressure_at(case.load_plus, case.load_minus, case.y0, 0.0, t)
            for t in y_line
        ]
    )
    red_sum = np.array(
        [
            flank.pressure_at(case.load_plus, 0.0, t - case.y0)
            + flank.pressure_at(case.load_minus, 0.0, t + case.y0)
            for t in y_line
        ]
    )
    ax.plot(
        y_line * MM,
        red_sum * GPA,
        color=CIRCULAR_COLOUR,
        lw=1.6,
        ls=":",
        label="naive sum (spurious hump)",
    )
    ax.plot(
        y_line * MM,
        red_max * GPA,
        color=LAW_COLOUR,
        lw=2.2,
        label="reduced max (connected saddle)",
    )
    ax.scatter(
        case.y * MM, case.cut * GPA, s=12, c=SOLVER_COLOUR, alpha=0.8, zorder=3, label="solver"
    )
    for centre in (case.y0, -case.y0):
        ax.axvline(centre * MM, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel("y (mm) — meridional")
    ax.set_ylabel("pressure (GPa)")
    ax.set_title(
        "(B) The overlap rule — max gives the saddle, sum overshoots",
        fontweight="bold",
        fontsize=10,
    )
    ax.set_ylim(bottom=0.0)
    ax.grid(visible=True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


# --------------------------------------------------------------------------- #
# Figure.
# --------------------------------------------------------------------------- #
def main() -> None:
    """Run the sweeps, check the reduced law, and render the four-panel figure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})
    print("verifying the reduced flank pressure distribution against the field solver ...")

    radius_x, radius_y = gothic_radii(BALL, TUBE, CENTRE_RADIUS)
    flank = hertzian.FlankPressure.from_elliptic_flank(
        radius_x=radius_x, radius_y=radius_y, e_star=E_STAR
    )
    print(f"  flank eccentricity e = {flank.eccentricity:.4f}")

    case = separated_field(FIELD_LOAD, SEPARATION_IN_B)
    print(
        f"  two-flank field: y0={case.y0 * MM:.3f} mm  "
        f"Q+={case.load_plus:.2f} N  Q-={case.load_minus:.2f} N"
    )

    loads = np.geomspace(20.0, 500.0, 9)
    arch = [single_arch_point(float(p)) for p in loads]

    # Report the agreement the figure shows.
    peak_err = max(
        abs(flank.peak_pressure(p.load) - p.peak_pressure) / p.peak_pressure for p in arch
    )
    spin_err = max(
        abs(flank.spin_moment(p.load, 1.0) - p.spin_moment) / p.spin_moment for p in arch
    )
    circ_low = float(
        np.mean(
            [1.0 - (p.load * 3.0 * math.pi / 16.0 * p.contact_radius) / p.spin_moment for p in arch]
        )
    )
    print(f"  reduced peak vs solver:        max residual {100.0 * peak_err:.1f}%")
    print(f"  reduced spin moment vs solver: max residual {100.0 * spin_err:.1f}%")
    print(f"  circular stand-in undershoots solver spin moment by {100.0 * circ_low:.0f}%")

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
    _panel_field(axes[0, 0], case, flank)
    _panel_profile(axes[0, 1], case, flank)
    _panel_scaling(axes[1, 0], arch, flank)
    _panel_spin(axes[1, 1], arch, flank)
    fig.suptitle(
        "hertzian — a reduced flank pressure distribution for Coulomb friction, "
        "verified against the field solver",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    path = OUT_DIR / "pressure_law.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024.0
    print(f"  wrote {path.relative_to(OUT_DIR.parent.parent)}  ({size_kb:.0f} KiB)")

    # The overlap: the same flank pressure, but the shim tightened so the two
    # patches merge into one connected contact (the pressure dual of gothic_overlap).
    overlap = separated_field(FIELD_LOAD, OVERLAP_IN_B)
    y = overlap.y
    span = np.abs(y) <= overlap.y0 + flank.semi_axes(overlap.load_plus)[1]
    peak_field = float(overlap.cut.max())
    centre_field = float(overlap.cut[np.argmin(np.abs(y))])

    def _max_cut(t: float) -> float:
        return flank.two_flank_pressure_at(
            overlap.load_plus, overlap.load_minus, overlap.y0, 0.0, t
        )

    rms_max = math.sqrt(
        float(
            np.mean(
                [(_max_cut(t) - p) ** 2 for t, p in zip(y[span], overlap.cut[span], strict=True)]
            )
        )
    )
    print(
        f"  half-overlap: y0={overlap.y0 * MM:.3f} mm  "
        f"saddle/peak solver={centre_field / peak_field:.2f}  "
        f"reduced-max profile RMS/peak={rms_max / peak_field:.2f}"
    )

    fig2, axes2 = plt.subplots(1, 2, figsize=(13.0, 5.2))
    _panel_overlap_field(axes2[0], overlap, flank)
    _panel_overlap_cut(axes2[1], overlap, flank)
    fig2.suptitle(
        "hertzian — the overlap: two flank pressures merge into one connected patch "
        "(pointwise max, the dual of the gap's min)",
        fontsize=12,
        fontweight="bold",
    )
    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path2 = OUT_DIR / "pressure_overlap.png"
    fig2.savefig(path2, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(
        f"  wrote {path2.relative_to(OUT_DIR.parent.parent)}  "
        f"({path2.stat().st_size / 1024.0:.0f} KiB)"
    )
    print("done.")


if __name__ == "__main__":
    main()
