"""Render the *asymmetric* per-flank pressure cap — a 2:1 two-torus.

The companion :mod:`render_pressure_distribution` validates the per-flank Coulomb
cap for a ball pushed *straight* into a Gothic-arch groove, where the two flanks
share the load equally and the two pressure crests are identical (1:1). That is
the easy case. Coulomb friction, though, is engaged precisely when the ball is
*also* dragged across the groove — a transverse drive — and then the load shifts
onto one flank and the two crests pull apart. This script analyses that regime,
keeping the shape fixed (the same two-torus flank-interference groove) but driving
it off-centre until the two-torus pressure peaks stand in a **2:1** ratio, and
compares the lightweight cap against the exact field solver there.

The shape is untouched: the gap is still the pointwise minimum of two flank wells
at ``y = ±y0`` (``GothicArchProfile``). The *drive* is what changes — an off-centre
push, built here as a meridional floor offset ``df`` between the two wells (the
height-field dual of a transverse ball displacement: the nearer flank is pressed
``df`` deeper). The field solver then carries the full flank-to-flank interference
(the loaded flank lifts the half-space under its neighbour). Because each flank is
still an elliptic-Hertz patch carrying its own load ``Q``, the peak obeys the same
cube-root cap ``p0 = cp Q^{1/3}``, so a **2:1 peak ratio is an 8:1 load split** —
the deeper flank takes ~8x the load of the shallower one.

Four panels, drawn from a handful of full solves:

* **(A) The split that makes 2:1.** Sweeping the off-centre drive, the two field
  crests ``p+`` and ``p-`` both ride the single cube-root line ``cp Q^{1/3}`` —
  each flank an elliptic-Hertz patch at its own load. The 2:1 peak pair sits at the
  8:1 load split, the operating point the rest of the figure dissects.
* **(B) The 2:1 cut — exact vs lightweight.** The meridional cut at the 2:1 drive:
  the solver (exact) against the lightweight envelope ``max(p+, p-)``, the two
  crests a clean 2:1. The shaded band is the Coulomb cap ``mu p`` a friction model
  rides under — now lopsided, the dragged flank carrying the traction.
* **(C) The 2-D asymmetric Coulomb cap.** ``mu p(x, y)`` over the two unequal flank
  ellipses, with the solver's contact outline: a large dominant patch and a small
  neighbour, each integrating to ``mu Q`` of full-sliding friction.
* **(D) The split the law predicts.** Driving off-centre, the two-torus peak ratio
  climbs from 1:1 to the 2:1 target, and the lightweight coupled law tracks the
  solver the whole way — a 2:1 peak is an 8:1 load split, by the cube-root cap.

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
# reduced law / its pressure cap — the same palette the rest of the gallery uses.
SOLVER_COLOUR = "#ef6c00"
REFERENCE_COLOUR = "#26c6da"
LAW_COLOUR = "#6a1b9a"

# The applied example: the README's conformal Gothic-arch bearing groove.
BALL = 4.0e-3
TUBE = 4.16e-3  # r/Rs = 1.04, a textbook bearing conformity
CENTRE_RADIUS = 15.0e-3
E_STAR = 100.0e9
LOAD = 800.0

# The separated, gallery groove (65 um shim -> y0 ~ 1.6 mm): two resolved flanks.
SEPARATED_SHIM = 65.0e-6

# The target asymmetry: a 2:1 ratio of the two pressure crests. By the cube-root cap
# p0 = cp Q^{1/3} this is an 8:1 split of the flank loads.
TARGET_PEAK_RATIO = 2.0

# An illustrative dry-steel-ish friction coefficient for the Coulomb cap mu p.
MU = 0.12

# Solver resolution: cells per semi-axis, free-space margin, tolerances. The grid is
# *anisotropic* (one cell size per semi-axis) so the ~10:1 elongated conformal contact
# stays small. Coarser cells drive the off-centre search; the rendered panels resolve
# the heavier flank on a finer mesh.
CELLS_PER_SEMI = 12.0
CELLS_PER_SEMI_FINE = 18.0
MARGIN = 2.5
SOLVE_TOL = 1.0e-9
SOLVE_MAX_ITER = 60000

# A flank is "in contact" where the solver pressure clears this fraction of the peak.
CONTACT_FLOOR = 1.0e-2


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
# Pressure scaling coefficient: p0 = CP * Q^{1/3} (the unit-load elliptic-Hertz peak).
CP = elliptic_hertz(RADIUS_X, RADIUS_Y, 1.0, E_STAR)[2]
# The overlap scale b: the meridional semi-axis of one isolated half-load flank.
B = elliptic_hertz(RADIUS_X, RADIUS_Y, LOAD / 2.0, E_STAR)[1]
# The separated flank offset of the gallery groove (65 um shim).
SEPARATED_Y0 = SEPARATED_SHIM * BALL / (TUBE - BALL)

# Bisection steps to land the off-centre drive on the target peak ratio. A coarse
# bracket on [0, 12] um halves to sub-nm in this many steps — far below the mesh.
BISECTION_STEPS = 18


def _even(value: float) -> int:
    """Return ``value`` rounded up to the next even integer (>= 24)."""
    n = max(math.ceil(value), 24)
    return n + (n & 1)


@dataclass(frozen=True)
class AsymmetricSolve:
    """One off-centre groove solve and the per-flank quantities read off it."""

    y0: float
    floor_offset: float
    approach: float
    dx: float
    dy: float
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    pressure: NDArray[np.float64]
    cut: NDArray[np.float64]
    peak_upper: float
    peak_lower: float
    load_upper: float
    load_lower: float

    @property
    def peak_ratio(self) -> float:
        """The ratio of the deeper (upper) crest to the shallower (lower) one."""
        return self.peak_upper / max(self.peak_lower, EPS)

    @property
    def load_ratio(self) -> float:
        """The ratio of the upper flank load to the lower flank load."""
        return self.load_upper / max(self.load_lower, EPS)


def solve_asymmetric(
    y0: float, floor_offset: float, load: float = LOAD, cells: float = CELLS_PER_SEMI
) -> AsymmetricSolve:
    """Solve an off-centre groove driven by a meridional well-floor offset ``df``.

    The shape is the unchanged two-torus gap — the pointwise minimum of two flank
    wells at ``y = ±y0`` — but the lower well is lifted by ``floor_offset``, the
    height-field dual of a transverse ball displacement: the upper flank is pressed
    that much deeper and so carries the larger load. Returns the solve plus the two
    flank peaks (upper/lower meridional half) and loads (pressure integrated over
    each half).
    """
    half_curv_x = 0.5 / RADIUS_X
    half_curv_y = 0.5 / RADIUS_Y
    # Size the grid by the heavier flank, which at full load bounds both footprints.
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
    mid = ny // 2
    cell = dx * dy
    return AsymmetricSolve(
        y0=y0,
        floor_offset=floor_offset,
        approach=sol.approach,
        dx=dx,
        dy=dy,
        x=x,
        y=y,
        pressure=pressure,
        cut=cut,
        peak_upper=float(cut[mid:].max()),
        peak_lower=float(cut[:mid].max()),
        load_upper=float(pressure[:, mid:].sum() * cell),
        load_lower=float(pressure[:, :mid].sum() * cell),
    )


def floor_offset_for_peak_ratio(y0: float, target: float, cells: float = CELLS_PER_SEMI) -> float:
    """Bisect the well-floor offset until the field peak ratio hits ``target``."""
    low, high = 0.0, 12.0e-6
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (low + high)
        if solve_asymmetric(y0, mid, cells=cells).peak_ratio < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def lightweight_loads(solve: AsymmetricSolve) -> tuple[float, float]:
    """Return the lightweight coupled flank loads ``(Q_+, Q_-)`` for a solve's drive.

    The off-centre drive is ``(s_+, s_-) = (delta, delta - df)`` in flank-approach
    space — the solver's rigid approach into the deeper flank, and that minus the
    floor offset into the shallower one — so the coupled two-flank solve yields the
    loads with no field integral needed.
    """
    law = LAW.with_flank_coupling(e_star=E_STAR, offset=solve.y0)
    return law.coupled_loads(solve.approach, solve.approach - solve.floor_offset)


def lightweight_groove(solve: AsymmetricSolve) -> hertzian.GrooveContactPressure:
    """Build the lightweight envelope cap from the drive the field solver saw.

    The off-centre drive is ``(s_+, s_-) = (delta, delta - df)`` in flank-approach
    space — the solver's rigid approach into the deeper flank, and that minus the
    floor offset into the shallower one — so the coupled two-flank solve yields the
    loads ``(Q_+, Q_-)`` with no field integral needed.
    """
    law = LAW.with_flank_coupling(e_star=E_STAR, offset=solve.y0)
    q_plus, q_minus = lightweight_loads(solve)
    return law.groove_pressure(q_plus, q_minus, offset=solve.y0)


# --------------------------------------------------------------------------- #
# (A) The split that makes 2:1: both crests on the cube-root line.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SplitSweep:
    """An off-centre drive sweep: per-flank loads, field crests and the split."""

    drive: NDArray[np.float64]
    load_upper: NDArray[np.float64]
    load_lower: NDArray[np.float64]
    peak_upper: NDArray[np.float64]
    peak_lower: NDArray[np.float64]
    ratio_solver: NDArray[np.float64]
    ratio_law: NDArray[np.float64]
    cap_ratio_mean: float
    cap_ratio_spread: float


def sweep_split(target_offset: float) -> SplitSweep:
    """Sweep the off-centre drive to the 2:1 point and read crests, loads and split.

    Shared by panels (A) and (D): for each drive it records the two field crests and
    loads, and the lightweight coupled prediction of the peak ratio (the cube root of
    the predicted load split). The drive is reported relative to the centred approach
    ``delta_0`` (the symmetric ``df = 0`` solve), so ``0`` is a straight push.
    """
    offsets = np.linspace(0.0, target_offset, 7)
    solves = [solve_asymmetric(SEPARATED_Y0, float(df)) for df in offsets]
    delta0 = solves[0].approach
    peak_upper = np.array([s.peak_upper for s in solves])
    peak_lower = np.array([s.peak_lower for s in solves])
    load_upper = np.array([s.load_upper for s in solves])
    load_lower = np.array([s.load_lower for s in solves])
    ratio_solver = peak_upper / peak_lower
    # The lightweight law's predicted peak ratio: (Q_+/Q_-)^{1/3} from the coupled
    # loads at the same drive — the cube-root cap turning the load split into peaks.
    law_loads = [lightweight_loads(s) for s in solves]
    ratio_law = np.array([(qp / qm) ** (1.0 / 3.0) for qp, qm in law_loads])
    # Each crest should sit on cp Q^{1/3}; gauge the scatter of both families.
    predicted = CP * np.concatenate([load_upper, load_lower]) ** (1.0 / 3.0)
    cap_ratios = np.concatenate([peak_upper, peak_lower]) / predicted
    return SplitSweep(
        drive=offsets / delta0,
        load_upper=load_upper,
        load_lower=load_lower,
        peak_upper=peak_upper,
        peak_lower=peak_lower,
        ratio_solver=ratio_solver,
        ratio_law=ratio_law,
        cap_ratio_mean=float(np.mean(cap_ratios)),
        cap_ratio_spread=float(np.max(cap_ratios) - np.min(cap_ratios)),
    )


def _panel_split(ax: Axes, sweep: SplitSweep) -> None:
    """Draw both crest families landing on the single cube-root cap line."""
    loads = np.concatenate([sweep.load_lower, sweep.load_upper])
    line = np.geomspace(loads.min(), loads.max(), 100)
    ax.plot(
        line,
        CP * line ** (1.0 / 3.0) * GPA,
        color=REFERENCE_COLOUR,
        lw=2.0,
        label=r"$c_p\,Q^{1/3}$ (one flank, load $Q$)",
    )
    ax.scatter(
        sweep.load_upper,
        sweep.peak_upper * GPA,
        s=34,
        marker="^",
        c=SOLVER_COLOUR,
        zorder=3,
        label="solver, deep flank $p_+$",
    )
    ax.scatter(
        sweep.load_lower,
        sweep.peak_lower * GPA,
        s=34,
        marker="v",
        c=LAW_COLOUR,
        zorder=3,
        label="solver, shallow flank $p_-$",
    )
    # The 2:1 operating point: deep/shallow crests at an 8:1 load split.
    q_hi, q_lo = sweep.load_upper[-1], sweep.load_lower[-1]
    p_hi, p_lo = sweep.peak_upper[-1] * GPA, sweep.peak_lower[-1] * GPA
    ax.plot([q_lo, q_hi], [p_lo, p_hi], color="0.4", lw=1.0, ls=":", zorder=2)
    ax.annotate(
        f"$p_+ : p_- = {p_hi / p_lo:.2f} : 1$\n($Q_+ : Q_- = {q_hi / q_lo:.1f} : 1$)",
        xy=(q_hi, p_hi),
        xytext=(0.32, 0.74),
        textcoords="axes fraction",
        fontsize=8.5,
        ha="left",
        arrowprops={"arrowstyle": "->", "color": "0.4", "lw": 1.0},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("flank load $Q$ (N)")
    ax.set_ylabel("peak pressure $p_0$ (GPa)")
    ax.set_title(
        f"(A) Both crests ride $c_p Q^{{1/3}}$ — solver/line {sweep.cap_ratio_mean:.4f}",
        fontweight="bold",
        fontsize=9.5,
    )
    ax.grid(visible=True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


# --------------------------------------------------------------------------- #
# (B) The 2:1 cut: the exact solver vs the lightweight envelope.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TwoToOneCut:
    """The 2:1 separated solve plus its lightweight envelope reconstruction."""

    solve: AsymmetricSolve
    env_cut: NDArray[np.float64]
    groove: hertzian.GrooveContactPressure
    env_err: float


def two_to_one_cut(target_offset: float) -> TwoToOneCut:
    """Solve the 2:1 separated groove and reconstruct its lightweight cut."""
    solve = solve_asymmetric(SEPARATED_Y0, target_offset, cells=CELLS_PER_SEMI_FINE)
    groove = lightweight_groove(solve)
    env = np.array([groove.pressure_at(0.0, float(yy)) for yy in solve.y])
    peak = solve.cut.max()
    return TwoToOneCut(
        solve=solve,
        env_cut=env,
        groove=groove,
        env_err=float((env.max() - peak) / peak),
    )


def _panel_two_to_one(ax: Axes, cut: TwoToOneCut) -> None:
    """Draw the 2:1 meridional cut: solver (exact) vs envelope, two crests 2:1."""
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
        y_mm,
        cut.env_cut * GPA,
        color=LAW_COLOUR,
        lw=2.0,
        label=r"lightweight envelope $\max(p_+, p_-)$",
    )
    ax.scatter(
        y_mm, solve.cut * GPA, s=10, c=SOLVER_COLOUR, alpha=0.8, zorder=3, label="solver (exact)"
    )
    p_hi, p_lo = solve.peak_upper * GPA, solve.peak_lower * GPA
    for sign, peak, tag in ((+1.0, p_hi, "$p_+$"), (-1.0, p_lo, "$p_-$")):
        ax.annotate(
            f"{tag} = {peak:.2f} GPa",
            xy=(sign * solve.y0 * MM, peak),
            xytext=(sign * solve.y0 * MM, peak + 0.18),
            fontsize=8.5,
            ha="center",
            color="0.25",
        )
    ax.set_xlim(-(solve.y0 + 2.0 * B) * MM, (solve.y0 + 2.0 * B) * MM)
    ax.set_ylim(bottom=0.0, top=1.18 * p_hi)
    ax.set_xlabel("y (mm) — across the groove")
    ax.set_ylabel("pressure (GPa)")
    ax.set_title(
        f"(B) The 2:1 cut — envelope {cut.env_err * 100:+.1f}% vs solver, "
        f"$p_+ : p_- = {solve.peak_ratio:.2f} : 1$",
        fontweight="bold",
        fontsize=9.0,
    )
    ax.grid(visible=True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


# --------------------------------------------------------------------------- #
# (C) The 2-D asymmetric Coulomb traction cap.
# --------------------------------------------------------------------------- #
def _panel_traction_cap(ax: Axes, cut: TwoToOneCut) -> None:
    """Draw the 2-D Coulomb cap mu p(x, y) over the two unequal flank ellipses."""
    solve = cut.solve
    upper, lower = cut.groove.flanks
    axp, ayp = upper.semi_axes
    span_x = 1.4 * axp
    span_y = solve.y0 + 1.6 * ayp
    xs = np.linspace(-span_x, span_x, 161)
    ys = np.linspace(-span_y, span_y, 321)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    field = np.array([[cut.groove.pressure_at(float(xx), float(yy)) for yy in ys] for xx in xs])
    mesh = ax.pcolormesh(gx * MM, gy * MM, MU * field * GPA, cmap="magma", shading="auto")
    plt.colorbar(mesh, ax=ax, label=r"$\mu\,p$ (GPa)", fraction=0.046, pad=0.04)

    # The solver's contact outline (where its pressure clears the contact floor).
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
        rf"$\mu Q_+ = {MU * upper.load:.1f}$ N",
        xy=(0.0, solve.y0 * MM),
        fontsize=8.0,
        ha="center",
        va="center",
        color="white",
    )
    ax.annotate(
        rf"$\mu Q_- = {MU * lower.load:.1f}$ N",
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
        r"(C) 2-D Coulomb cap $\mu\,p(x,y)$ — the dragged flank dominates",
        fontweight="bold",
        fontsize=9.0,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")


# --------------------------------------------------------------------------- #
# (D) The split the lightweight law predicts: peak ratio vs the off-centre drive.
# --------------------------------------------------------------------------- #
def _panel_ratio(ax: Axes, sweep: SplitSweep) -> None:
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
        sweep.drive,
        sweep.ratio_solver,
        s=34,
        c=SOLVER_COLOUR,
        zorder=3,
        label="solver $p_+/p_-$",
    )
    ax.annotate(
        "8:1 load\n2:1 peak",
        xy=(sweep.drive[-1], sweep.ratio_solver[-1]),
        xytext=(0.30, 0.52),
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
        "(D) The lightweight law tracks the asymmetric split to 2:1",
        fontweight="bold",
        fontsize=9.5,
    )
    ax.grid(visible=True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


# --------------------------------------------------------------------------- #
# Figure.
# --------------------------------------------------------------------------- #
def main() -> None:
    """Run the sweeps, print residuals and render the four-panel figure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})
    print("analysing the asymmetric 2:1 per-flank pressure cap ...")

    target_offset = floor_offset_for_peak_ratio(SEPARATED_Y0, TARGET_PEAK_RATIO)
    print(
        f"  2:1 drive: well-floor offset df = {target_offset * 1.0e6:.3f} um "
        f"(y0 = {SEPARATED_Y0 * MM:.3f} mm)"
    )

    sweep = sweep_split(target_offset)
    print(
        f"  both crests on cp Q^{{1/3}}: solver/line {sweep.cap_ratio_mean:.4f} "
        f"(spread {sweep.cap_ratio_spread:.4f})"
    )
    split_err = float(np.max(np.abs(sweep.ratio_law - sweep.ratio_solver) / sweep.ratio_solver))
    print(f"  lightweight peak-ratio prediction tracks the solver to <= {split_err * 100:.1f}%")

    cut = two_to_one_cut(target_offset)
    solve = cut.solve
    print(
        f"  2:1 cut: p+ = {solve.peak_upper * GPA:.3f} GPa, "
        f"p- = {solve.peak_lower * GPA:.3f} GPa (ratio {solve.peak_ratio:.3f}); "
        f"Q+ = {solve.load_upper:.1f} N, Q- = {solve.load_lower:.1f} N "
        f"(split {solve.load_ratio:.2f}); envelope {cut.env_err * 100:+.1f}% vs solver"
    )
    upper, lower = cut.groove.flanks
    print(
        f"  per-flank caps integrate to mu Q: mu Q+ = {MU * upper.load:.2f} N, "
        f"mu Q- = {MU * lower.load:.2f} N"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4))
    _panel_split(axes[0, 0], sweep)
    _panel_two_to_one(axes[0, 1], cut)
    _panel_traction_cap(axes[1, 0], cut)
    _panel_ratio(axes[1, 1], sweep)
    fig.suptitle(
        "hertzian — the asymmetric per-flank pressure cap: a 2:1 two-torus",
        fontsize=13,
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
