"""Render the *asymmetric* per-flank pressure cap — a 2:1 two-torus with interfering flanks.

The companion :mod:`render_pressure_distribution` validates the per-flank Coulomb
cap for a ball pushed *straight* into a Gothic-arch groove, where the two flanks
share the load equally and the two pressure crests are identical (1:1). That is
the easy, symmetric case. Coulomb friction, though, is engaged precisely when the
ball is *also* dragged across the groove — a transverse drive — and then the load
shifts onto one flank and the two crests pull apart.

This script analyses that regime *while the flanks interfere*. It keeps the shape
fixed (the same two-torus groove) at the **half-overlap** shim ``y0 = b/2`` — the
canonical flank-interference configuration, where the two flank footprints cross
into a single *connected* patch (the former Gothic point now carries load). It then
drives the groove off-centre until the two-torus pressure peaks stand in a **2:1**
ratio, and compares the lightweight cap against the exact field solver there.

The shape is untouched: the gap is still the pointwise minimum of two flank wells
at ``y = ±y0`` (``GothicArchProfile``), with ``y0 = b/2`` so the flanks overlap. The
*drive* is what changes — an off-centre push, built here as a meridional floor
offset ``df`` between the two wells (the height-field dual of a transverse ball
displacement: the nearer flank is pressed ``df`` deeper). The field solver carries
the full flank-to-flank interference (the loaded flank lifts the half-space under
its neighbour, and the two footprints reinforce through a connected saddle).

Because each flank is still an elliptic-Hertz patch carrying its own load ``Q``, the
peak obeys the cube-root cap ``p0 = cp Q^{1/3}``, so a **2:1 peak ratio is an 8:1
load split**. The whole-groove cap is then the *envelope* (pointwise maximum) of the
two footprints, **not** their sum: where the footprints overlap, the naive sum
double-counts the seam, while the envelope drops it and tracks the connected saddle.

Four panels, drawn from a handful of full solves:

* **(A) Reaching 2:1 off-centre.** Sweeping the off-centre drive at the half-overlap
  shim, the two-torus peak ratio climbs from 1:1 to 2:1, and the lightweight coupled
  law tracks the solver — a 2:1 peak is an 8:1 load split, by the cube-root cap.
* **(B) The 2:1 interfering cut — exact vs lightweight.** The meridional cut: the
  solver (exact), the lightweight envelope ``max(p+, p-)`` and the naive *sum*. The
  two crests stand 2:1, joined by a *loaded* saddle (the flanks interfere into one
  connected patch); the sum double-counts the overlap into a seam spike, while the
  envelope tracks the solver. The shaded band is the Coulomb cap ``mu p``.
* **(C) The 2-D asymmetric Coulomb cap.** ``mu p(x, y)`` over the two *overlapping*,
  unequal flank ellipses, sampled straight off the cap with ``groove.pressure_mesh(nx,
  ny)`` — the per-cell ``(centre, normal load)`` lattice a discrete Coulomb solver
  meshes the contact into — with the solver's connected contact outline.
* **(D) Interference across the shim.** Closing the shim from separated into overlap
  at the off-centre drive, the connected saddle rises; the naive sum's seam balloons
  with the overlap while the envelope follows the solver.

Run it (matplotlib is a render-only dependency, kept out of the locked env):

    uv run --with matplotlib python scripts/render_pressure_distribution_asymmetric.py
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

# Unit scales: SI in, engineering units out.
MM = 1.0e3
GPA = 1.0e-9
EPS = 2.220_446_049_250_313e-16

# Warm tones for the solver (exact), a muted reference for analytics, purple for the
# reduced law / its pressure cap, red for the naive sum — the gallery palette.
SOLVER_COLOUR = "#ef6c00"
REFERENCE_COLOUR = "#26c6da"
LAW_COLOUR = "#6a1b9a"
SUM_COLOUR = "#c62828"

# The applied example: the README's conformal Gothic-arch bearing groove.
BALL = 4.0e-3
TUBE = 4.16e-3  # r/Rs = 1.04, a textbook bearing conformity
CENTRE_RADIUS = 15.0e-3
E_STAR = 100.0e9
LOAD = 800.0

# The target asymmetry: a 2:1 ratio of the two pressure crests. By the cube-root cap
# p0 = cp Q^{1/3} this is an 8:1 split of the flank loads.
TARGET_PEAK_RATIO = 2.0

# An illustrative dry-steel-ish friction coefficient for the Coulomb cap mu p.
MU = 0.12

# Solver resolution: cells per semi-axis, free-space margin, tolerances. The grid is
# *anisotropic* (one cell size per semi-axis) so the ~10:1 elongated conformal contact
# stays small. Coarser cells drive the off-centre search; the rendered panels resolve
# the heavier flank on a finer mesh.
CELLS_PER_SEMI = 16.0
CELLS_PER_SEMI_FINE = 20.0
MARGIN = 2.6
SOLVE_TOL = 1.0e-9
SOLVE_MAX_ITER = 60000

# A flank is "in contact" where the solver pressure clears this fraction of the peak.
CONTACT_FLOOR = 1.0e-2

# Bisection steps to land the off-centre drive on the target peak ratio.
BISECTION_STEPS = 18

# A genuine two-torus cut has two crests; below this the flanks have merged to one.
TWO_CRESTS = 2


# --------------------------------------------------------------------------- #
# Analytic reference (independent re-implementation of the Rust closed forms).
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
) -> tuple[float, float, float]:
    """Return ``(semi_axis_x, semi_axis_y, peak_pressure)`` for elliptic Hertz."""
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
    p0 = 3.0 * load / (2.0 * math.pi * semi_major * semi_minor)
    if radius_x >= radius_y:
        return semi_major, semi_minor, p0
    return semi_minor, semi_major, p0


RADIUS_X, RADIUS_Y = gothic_radii(BALL, TUBE, CENTRE_RADIUS)
# The calibrated reduced law (the contact angle does not enter the flank-load split
# or the per-flank pressure; it only orients the force vector).
LAW = hertzian.GothicArchLaw.from_elliptic_flank(
    radius_x=RADIUS_X, radius_y=RADIUS_Y, e_star=E_STAR, contact_angle=0.1
)
# The overlap scale b: the meridional semi-axis of one isolated half-load flank. The
# half overlap is the shim that sets the flank offset y0 = b/2 — the canonical
# flank-interference configuration, where the two footprints cross into one patch.
B = elliptic_hertz(RADIUS_X, RADIUS_Y, LOAD / 2.0, E_STAR)[1]
OVERLAP_Y0 = 0.5 * B


def _even(value: float) -> int:
    """Return ``value`` rounded up to the next even integer (>= 24)."""
    n = max(math.ceil(value), 24)
    return n + (n & 1)


@dataclass(frozen=True)
class Crests:
    """The two flank crests of a meridional cut and the saddle between them."""

    heavy: float
    heavy_y: float
    light: float
    light_y: float
    saddle: float
    saddle_y: float

    @property
    def ratio(self) -> float:
        """The crest ratio (deep/dragged-into flank over the shallow one)."""
        return self.heavy / max(self.light, EPS)


def find_crests(y: NDArray[np.float64], cut: NDArray[np.float64]) -> Crests | None:
    """Find the two largest local maxima of ``cut`` and the saddle between them.

    Proper local-maxima detection is essential in the overlap: a simple "max of each
    y-half" would catch the dominant flank's *tail* spilling across the groove centre
    rather than the shallow flank's own crest, reading a far-too-small ratio.
    """
    maxima = sorted(
        (
            (float(cut[j]), int(j))
            for j in range(1, cut.size - 1)
            if cut[j] > cut[j - 1] and cut[j] >= cut[j + 1] and cut[j] > 0.0
        ),
        reverse=True,
    )
    if len(maxima) < TWO_CRESTS:
        return None
    (heavy, hj), (light, lj) = maxima[0], maxima[1]
    lo, hi = sorted((hj, lj))
    sj = lo + int(np.argmin(cut[lo : hi + 1]))
    return Crests(heavy, float(y[hj]), light, float(y[lj]), float(cut[sj]), float(y[sj]))


@dataclass(frozen=True)
class GrooveSolve:
    """One off-centre half-overlap groove solve and the quantities read off it."""

    y0: float
    floor_offset: float
    approach: float
    dx: float
    dy: float
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    pressure: NDArray[np.float64]
    cut: NDArray[np.float64]
    crests: Crests | None


def solve_groove(
    y0: float, floor_offset: float, load: float = LOAD, cells: float = CELLS_PER_SEMI
) -> GrooveSolve:
    """Solve an off-centre groove driven by a meridional well-floor offset ``df``.

    The shape is the unchanged two-torus gap — the pointwise minimum of two flank
    wells at ``y = ±y0`` — but the lower well is lifted by ``floor_offset``, the
    height-field dual of a transverse ball displacement pressing the upper flank
    that much deeper. At ``y0 = b/2`` the two footprints overlap into one connected
    patch, so the flanks interfere.
    """
    half_curv_x = 0.5 / RADIUS_X
    half_curv_y = 0.5 / RADIUS_Y
    ax_heavy, ay_heavy, _ = elliptic_hertz(RADIUS_X, RADIUS_Y, load, E_STAR)
    dx, dy = ax_heavy / cells, ay_heavy / cells
    nx = _even(2.0 * MARGIN * ax_heavy / dx)
    ny = _even(2.0 * (y0 + MARGIN * ay_heavy) / dy)
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    well_upper = half_curv_y * (y[None, :] - y0) ** 2
    well_lower = half_curv_y * (y[None, :] + y0) ** 2 + floor_offset
    gap = half_curv_x * x[:, None] ** 2 + np.minimum(well_upper, well_lower)
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
    cut = pressure[nx // 2, :]
    return GrooveSolve(
        y0=y0,
        floor_offset=floor_offset,
        approach=sol.approach,
        dx=dx,
        dy=dy,
        x=x,
        y=y,
        pressure=pressure,
        cut=cut,
        crests=find_crests(y, cut),
    )


# A single crest means the shallow flank has lifted off (over-driven), so the bisection
# should read that as "past the target ratio" and pull the drive back down.
_OVERDRIVEN_RATIO = 99.0


def _peak_ratio(solve: GrooveSolve) -> float:
    """The two-crest ratio; large when only one crest survives (shallow flank lifted)."""
    return solve.crests.ratio if solve.crests else _OVERDRIVEN_RATIO


def floor_offset_for_peak_ratio(y0: float, target: float, cells: float = CELLS_PER_SEMI) -> float:
    """Bisect the well-floor offset until the field crest ratio hits ``target``."""
    low, high = 0.0, 10.0e-6
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (low + high)
        if _peak_ratio(solve_groove(y0, mid, cells=cells)) < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def lightweight_loads(solve: GrooveSolve) -> tuple[float, float]:
    """Return the lightweight coupled flank loads ``(Q_+, Q_-)`` for a solve's drive.

    The off-centre drive is ``(s_+, s_-) = (delta, delta - df)`` in flank-approach
    space — the solver's rigid approach into the deeper flank, and that minus the
    floor offset into the shallower one — so the coupled two-flank solve yields the
    loads with no field integral (which the overlap would contaminate anyway).
    """
    law = LAW.with_flank_coupling(e_star=E_STAR, offset=solve.y0)
    return law.coupled_loads(solve.approach, solve.approach - solve.floor_offset)


def lightweight_groove(solve: GrooveSolve) -> hertzian.GrooveContactPressure:
    """Build the lightweight envelope cap from the drive the field solver saw."""
    law = LAW.with_flank_coupling(e_star=E_STAR, offset=solve.y0)
    q_plus, q_minus = lightweight_loads(solve)
    return law.groove_pressure(q_plus, q_minus, offset=solve.y0)


# --------------------------------------------------------------------------- #
# (A) Reaching 2:1 off-centre: peak ratio vs the drive, solver vs lightweight.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DriveSweep:
    """An off-centre drive sweep at the half-overlap shim: peak ratio and split."""

    drive: NDArray[np.float64]
    ratio_solver: NDArray[np.float64]
    ratio_law: NDArray[np.float64]
    split_err: float


def sweep_drive(target_offset: float) -> DriveSweep:
    """Sweep the off-centre drive to the 2:1 point and read the crest ratio vs split."""
    offsets = np.linspace(0.0, target_offset, 7)
    solves = [solve_groove(OVERLAP_Y0, float(df)) for df in offsets]
    delta0 = solves[0].approach
    ratio_solver = np.array([_peak_ratio(s) for s in solves])
    # The lightweight predicted peak ratio: (Q_+/Q_-)^{1/3} from the coupled loads.
    law_loads = [lightweight_loads(s) for s in solves]
    ratio_law = np.array([(qp / max(qm, EPS)) ** (1.0 / 3.0) for qp, qm in law_loads])
    valid = ratio_solver > 1.0 + 1.0e-6
    split_err = (
        float(np.max(np.abs(ratio_law[valid] - ratio_solver[valid]) / ratio_solver[valid]))
        if valid.any()
        else 0.0
    )
    return DriveSweep(
        drive=offsets / delta0,
        ratio_solver=ratio_solver,
        ratio_law=ratio_law,
        split_err=split_err,
    )


def _panel_drive(ax: Axes, sweep: DriveSweep) -> None:
    """Draw the peak ratio vs off-centre drive: solver vs lightweight, up to 2:1."""
    ax.axhline(
        TARGET_PEAK_RATIO,
        color="0.6",
        lw=1.0,
        ls=":",
        label=rf"target $p_+ : p_- = {TARGET_PEAK_RATIO:.0f} : 1$",
    )
    ax.plot(
        sweep.drive,
        sweep.ratio_law,
        color=LAW_COLOUR,
        lw=2.0,
        label=r"lightweight $(Q_+/Q_-)^{1/3}$",
    )
    ax.scatter(
        sweep.drive, sweep.ratio_solver, s=34, c=SOLVER_COLOUR, zorder=3, label="solver $p_+/p_-$"
    )
    ax.annotate(
        "8:1 load\n2:1 peak",
        xy=(sweep.drive[-1], sweep.ratio_solver[-1]),
        xytext=(0.28, 0.55),
        textcoords="axes fraction",
        fontsize=8.5,
        ha="left",
        arrowprops={"arrowstyle": "->", "color": "0.4", "lw": 1.0},
    )
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=1.0)
    ax.set_xlabel(r"off-centre drive $df / \delta_0$ (transverse push)")
    ax.set_ylabel(r"peak ratio $p_+ / p_-$")
    ax.set_title(
        "(A) Driven off-centre to 2:1 — the lightweight law tracks the split",
        fontweight="bold",
        fontsize=9.0,
    )
    ax.grid(visible=True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


# --------------------------------------------------------------------------- #
# (B) The 2:1 interfering cut: solver vs envelope vs naive sum.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OverlapCut:
    """The 2:1 half-overlap solve plus its envelope and naive-sum reconstructions."""

    solve: GrooveSolve
    groove: hertzian.GrooveContactPressure
    env_cut: NDArray[np.float64]
    sum_cut: NDArray[np.float64]
    env_err: float
    sum_saddle_over: float


def overlap_cut(target_offset: float) -> OverlapCut:
    """Solve the 2:1 half-overlap groove and reconstruct its meridional cut."""
    solve = solve_groove(OVERLAP_Y0, target_offset, cells=CELLS_PER_SEMI_FINE)
    groove = lightweight_groove(solve)
    cap_plus, cap_minus = groove.flanks
    y0 = solve.y0
    env = np.array([groove.pressure_at(0.0, float(yy)) for yy in solve.y])
    sum_cut = np.array(
        [
            cap_plus.pressure_at(0.0, float(yy) - y0) + cap_minus.pressure_at(0.0, float(yy) + y0)
            for yy in solve.y
        ]
    )
    peak = solve.cut.max()
    # The naive sum's double-count is largest at the connected saddle; gauge it there.
    crests = solve.crests
    if crests is not None:
        sj = int(np.argmin(np.abs(solve.y - crests.saddle_y)))
        sum_saddle_over = float((sum_cut[sj] - solve.cut[sj]) / max(solve.cut[sj], EPS))
    else:
        sum_saddle_over = 0.0
    return OverlapCut(
        solve=solve,
        groove=groove,
        env_cut=env,
        sum_cut=sum_cut,
        env_err=float((env.max() - peak) / peak),
        sum_saddle_over=sum_saddle_over,
    )


def _panel_overlap_cut(ax: Axes, cut: OverlapCut) -> None:
    """Draw the 2:1 interfering cut: solver vs envelope (saddle) vs naive sum (spike)."""
    solve = cut.solve
    y_mm = solve.y * MM
    ax.fill_between(
        y_mm,
        MU * cut.env_cut * GPA,
        color=LAW_COLOUR,
        alpha=0.18,
        label=rf"Coulomb cap $\mu p$  ($\mu={MU}$)",
    )
    ax.plot(
        y_mm, cut.sum_cut * GPA, color=SUM_COLOUR, lw=1.8, ls="--", label="naive sum (double-count)"
    )
    ax.plot(y_mm, cut.env_cut * GPA, color=LAW_COLOUR, lw=2.0, label=r"envelope $\max(p_+, p_-)$")
    ax.scatter(
        y_mm, solve.cut * GPA, s=10, c=SOLVER_COLOUR, alpha=0.85, zorder=3, label="solver (exact)"
    )
    crests = solve.crests
    if crests is not None:
        for cy, cp, tag in (
            (crests.heavy_y, crests.heavy, "$p_+$"),
            (crests.light_y, crests.light, "$p_-$"),
        ):
            ax.annotate(
                f"{tag}={cp * GPA:.2f}",
                xy=(cy * MM, cp * GPA),
                xytext=(cy * MM, cp * GPA + 0.16),
                fontsize=8.5,
                ha="center",
                color="0.25",
            )
        ax.annotate(
            "loaded saddle\n(flanks interfere)",
            xy=(crests.saddle_y * MM, crests.saddle * GPA),
            xytext=(0.04, 0.60),
            textcoords="axes fraction",
            fontsize=8.0,
            ha="left",
            color="0.3",
            arrowprops={"arrowstyle": "->", "color": "0.45", "lw": 1.0},
        )
    span = solve.y0 + 2.0 * B
    ax.set_xlim(-span * MM, span * MM)
    ax.set_ylim(bottom=0.0, top=1.10 * cut.sum_cut.max() * GPA)
    ax.set_xlabel("y (mm) — across the groove")
    ax.set_ylabel("pressure (GPa)")
    ratio = solve.crests.ratio if solve.crests else float("nan")
    ax.set_title(
        f"(B) The 2:1 interfering cut — envelope {cut.env_err * 100:+.1f}%, "
        f"sum +{cut.sum_saddle_over * 100:.0f}% at the saddle  ($p_+:p_-={ratio:.2f}:1$)",
        fontweight="bold",
        fontsize=8.5,
    )
    ax.grid(visible=True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


# --------------------------------------------------------------------------- #
# (C) The 2-D asymmetric Coulomb traction cap (overlapping flanks).
# --------------------------------------------------------------------------- #
def _panel_traction_cap(ax: Axes, cut: OverlapCut) -> None:
    """Draw the 2-D Coulomb cap mu p(x, y) over the two overlapping flank ellipses."""
    solve = cut.solve
    upper, lower = cut.groove.flanks
    # The cap field comes straight off pressure_mesh: it tiles the connected footprint
    # into nx*ny cells and hands back the cell centres (mesh.x, mesh.y) and the
    # envelope pressure there — the lattice a discrete Coulomb solver meshes the
    # contact into, no hand-rolled grid loop over pressure_at.
    mesh = cut.groove.pressure_mesh(nx=161, ny=321)
    gx, gy = np.meshgrid(mesh.x, mesh.y, indexing="ij")
    field = ax.pcolormesh(gx * MM, gy * MM, MU * mesh.pressure * GPA, cmap="magma", shading="auto")
    plt.colorbar(field, ax=ax, label=r"$\mu\,p$ (GPa)", fraction=0.046, pad=0.04)

    # The solver's connected contact outline (where its pressure clears the floor).
    ax.contour(
        solve.x[:, None] * MM + 0.0 * solve.y[None, :],
        0.0 * solve.x[:, None] + solve.y[None, :] * MM,
        solve.pressure,
        levels=[CONTACT_FLOOR * solve.pressure.max()],
        colors=[REFERENCE_COLOUR],
        linewidths=1.6,
        linestyles="--",
    )
    ax.plot([], [], color=REFERENCE_COLOUR, lw=1.6, ls="--", label="solver contact edge")
    ax.annotate(
        rf"$\mu Q_+ = {MU * upper.load:.0f}$ N",
        xy=(0.0, solve.y0 * MM),
        fontsize=8.0,
        ha="center",
        va="center",
        color="white",
    )
    ax.annotate(
        rf"$\mu Q_- = {MU * lower.load:.0f}$ N",
        xy=(0.0, -solve.y0 * MM),
        fontsize=8.0,
        ha="center",
        va="center",
        color="white",
    )
    # The conformal contact is ~10:1 elongated, so the axes carry different scales.
    ax.set_xlabel("x (mm) — circumferential (note: scale ≠ y)")
    ax.set_ylabel("y (mm) — meridional")
    ax.set_title(
        r"(C) 2-D Coulomb cap $\mu\,p(x,y)$ — overlapping, the dragged flank dominant",
        fontweight="bold",
        fontsize=8.5,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")


# --------------------------------------------------------------------------- #
# (D) Interference across the shim: the saddle the sum gets wrong.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ShimSweep:
    """A fixed-drive offset sweep: the connected saddle vs naive sum and envelope."""

    offsets: NDArray[np.float64]
    solver_saddle: NDArray[np.float64]
    env_saddle: NDArray[np.float64]
    sum_saddle: NDArray[np.float64]
    overlap_edge: float


def sweep_shim(floor_offset: float) -> ShimSweep:
    """Sweep the flank offset at a fixed off-centre drive; read the saddle three ways."""
    fractions = np.linspace(0.45, 1.6, 8)
    offsets = fractions * B
    solver_saddle = np.empty(offsets.size)
    env_saddle = np.empty(offsets.size)
    sum_saddle = np.empty(offsets.size)
    for i, y0 in enumerate(offsets):
        solve = solve_groove(float(y0), floor_offset)
        groove = lightweight_groove(solve)
        cap_plus, cap_minus = groove.flanks
        crests = solve.crests
        y_saddle = crests.saddle_y if crests else 0.0
        solver_saddle[i] = crests.saddle if crests else float(solve.cut[solve.cut.size // 2])
        env_saddle[i] = groove.pressure_at(0.0, y_saddle)
        sum_saddle[i] = cap_plus.pressure_at(0.0, y_saddle - float(y0)) + cap_minus.pressure_at(
            0.0, y_saddle + float(y0)
        )
    # The overlap edge: where the half-load footprints just touch (2 y0 = 2 b -> y0 = b).
    return ShimSweep(
        offsets=fractions,
        solver_saddle=solver_saddle,
        env_saddle=env_saddle,
        sum_saddle=sum_saddle,
        overlap_edge=1.0,
    )


def _panel_shim(ax: Axes, sweep: ShimSweep) -> None:
    """Draw the connected-saddle pressure vs shim: solver vs envelope vs naive sum."""
    ax.axvspan(sweep.offsets.min(), sweep.overlap_edge, color="0.92", label="overlap (interfering)")
    ax.plot(
        sweep.offsets,
        sweep.sum_saddle * GPA,
        color=SUM_COLOUR,
        lw=2.0,
        ls="--",
        label="naive sum at saddle",
    )
    ax.plot(
        sweep.offsets, sweep.env_saddle * GPA, color=LAW_COLOUR, lw=2.0, label="envelope at saddle"
    )
    ax.scatter(
        sweep.offsets,
        sweep.solver_saddle * GPA,
        s=28,
        c=SOLVER_COLOUR,
        zorder=3,
        label="solver saddle",
    )
    ax.set_xlabel(r"flank offset $y_0 / b$")
    ax.set_ylabel("saddle pressure (GPa)")
    ax.set_title(
        "(D) Closing the shim into overlap — the sum double-counts the connected saddle",
        fontweight="bold",
        fontsize=8.5,
    )
    ax.grid(visible=True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


# --------------------------------------------------------------------------- #
# Figure.
# --------------------------------------------------------------------------- #
def main() -> None:
    """Run the sweeps, print residuals and render the four-panel figure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})
    print("analysing the asymmetric 2:1 per-flank cap with interfering flanks ...")

    target_offset = floor_offset_for_peak_ratio(OVERLAP_Y0, TARGET_PEAK_RATIO)
    print(
        f"  half-overlap shim y0 = b/2 = {OVERLAP_Y0 * MM:.3f} mm; "
        f"2:1 drive: floor offset df = {target_offset * 1.0e6:.3f} um"
    )

    sweep = sweep_drive(target_offset)
    print(
        f"  lightweight peak-ratio prediction tracks the solver to <= {sweep.split_err * 100:.1f}%"
    )

    cut = overlap_cut(target_offset)
    crests = cut.solve.crests
    if crests is None:
        message = "the 2:1 cut must resolve two distinct crests"
        raise RuntimeError(message)
    print(
        f"  2:1 cut: p+ = {crests.heavy * GPA:.3f} GPa @ {crests.heavy_y * MM:+.2f} mm, "
        f"p- = {crests.light * GPA:.3f} GPa @ {crests.light_y * MM:+.2f} mm "
        f"(ratio {crests.ratio:.3f})"
    )
    print(
        f"  connected saddle {crests.saddle * GPA:.3f} GPa "
        f"({crests.saddle / crests.light * 100:.0f}% of the light crest) -> flanks interfere"
    )
    print(
        f"  envelope {cut.env_err * 100:+.1f}% vs solver peak; "
        f"naive sum +{cut.sum_saddle_over * 100:.0f}% at the saddle"
    )
    upper, lower = cut.groove.flanks
    print(
        f"  per-flank caps integrate to mu Q: mu Q+ = {MU * upper.load:.1f} N, "
        f"mu Q- = {MU * lower.load:.1f} N"
    )

    shim = sweep_shim(target_offset)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4))
    _panel_drive(axes[0, 0], sweep)
    _panel_overlap_cut(axes[0, 1], cut)
    _panel_traction_cap(axes[1, 0], cut)
    _panel_shim(axes[1, 1], shim)
    fig.suptitle(
        "hertzian — the asymmetric per-flank Coulomb cap: a 2:1 two-torus with interfering flanks",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    path = OUT_DIR / "flank_pressure_asymmetric.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(
        f"  wrote {path.relative_to(OUT_DIR.parent.parent)} "
        f"({path.stat().st_size / 1024.0:.0f} KiB)"
    )
    print("done.")


if __name__ == "__main__":
    main()
