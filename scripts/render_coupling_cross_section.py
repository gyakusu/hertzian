"""Render the meridional cross-section of the first-order elastic flank coupling.

The existing gallery shows the Gothic-arch contact *from above* — the pressure
field on the contact interface (``gothic.png``, ``gothic_overlap.png``). This
script adds the missing **side view**: the meridional cross-section of the
originally-assumed geometry — a ball (sphere) cradled by **two tori** (two
tube-radius arcs) — with the **contact-force vectors** drawn on it, comparing the
**exact** field solver against the **approximate** reduced law.

It draws the three regimes the coupling section turns on, left to right:

* **(a) single arc — one ellipse.** No shim: the two arcs coincide, the contact
  is a single elliptic Hertz patch carrying the full load straight up. The
  effective flank count is ``eta = 1``.
* **(b) half overlap — the in-between.** The shim opens the groove until the two
  flank ellipses overlap by half (``y0 = b/2``). The contacts close in, their
  elastic fields overlap, and each flank lifts the half-space under the other:
  ``eta`` is pulled well below the naive ``2``, and the first-order coupled law
  recovers most of that drop.
* **(c) separated flanks — left/right symmetric.** A larger shim sets two clearly
  separated, symmetric flank contacts (the gallery's 65 µm groove). The coupling
  has faded, ``eta -> ~2``, and the coupled and uncoupled laws agree with the
  solver.

Top row: the side cross-section — sphere + two tori, the contact patch the solver
finds, and the per-flank load vectors ``Q_+ n_+`` and ``Q_- n_-`` summing to the
applied load ``P`` (the tube radius is drawn schematically enlarged so the two
arcs separate from the near-conformal ball; the contact angle ``alpha`` and the
contact positions ``+/- y0`` are exact). Bottom row: the meridional pressure cut
at ``x = 0`` — the solver (exact) against the analytic Hertz reference
(approximate) — with the effective flank count ``eta`` reported three ways
(exact solver, coupled law, naive uncoupled = 2).

Run it (matplotlib is a render-only dependency, kept out of the locked env):

    uv run --with matplotlib python scripts/render_coupling_cross_section.py
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

# Unit scales: SI in, engineering units out (metres -> mm, pascals -> GPa).
MM = 1.0e3
GPA = 1.0e-9
UM = 1.0e6
EPS = 2.220_446_049_250_313e-16

# Warm tones for the solver (exact), a muted reference for analytics, purple for
# the reduced law — the same palette the rest of the gallery uses.
SOLVER_COLOUR = "#ef6c00"
REFERENCE_COLOUR = "#26c6da"
LAW_COLOUR = "#6a1b9a"
BALL_FACE = "#eceff4"
BALL_EDGE = "#90a4ae"
# The two tori get two tints so the overlaid arcs read as a pair; the single arch
# uses the neutral one.
TORUS_PLUS = "#3949ab"
TORUS_MINUS = "#00897b"
TORUS_SINGLE = "#5c6bc0"

# The applied example: the README's conformal Gothic-arch bearing groove.
BALL = 4.0e-3
TUBE = 4.16e-3  # r/Rs = 1.04, a textbook bearing conformity
CENTRE_RADIUS = 15.0e-3
E_STAR = 100.0e9
LOAD = 800.0

# The shim of the separated, gallery groove (65 um -> y0 ~ 1.6 mm, alpha ~ 24 deg).
SEPARATED_SHIM = 65.0e-6

# Solver resolution: cells per (short) semi-axis, free-space margin, tolerances.
# The approach and effective flank count are integral quantities that converge on
# coarse grids (this resolution reproduces the same eta to three digits as a far
# finer one), so it is kept modest — the script runs the solver a handful of times.
CELLS_PER_SEMI = 12.0
MARGIN = 3.0
SOLVE_TOL = 1.0e-8
SOLVE_MAX_ITER = 40000

# A flank is "in contact" where its cut pressure clears this fraction of the peak;
# used to stroke the contact patch the solver finds on the ball surface.
CONTACT_FLOOR = 1.0e-3

# Schematic enlargement of the tube radius. The true groove is near-conformal
# (r/Rs = 1.04), so the two arcs hug the ball within microns and would be
# indistinguishable; drawing them at this multiple of the ball radius separates
# them for the eye. Only the curvature is schematic — the contact angle and the
# contact offsets +/- y0 are placed at their exact values.
TUBE_DRAW_FACTOR = 1.6

# Side-view window (mm): the ball reads as a circle, the groove sits below it, and
# the applied-load arrow drives down through the top — tight enough that the load
# vectors are not dwarfed by the (8 mm) ball.
CROSS_HALF_WIDTH = 4.6
CROSS_TOP = 4.6
CROSS_BOTTOM = -5.0

# The applied load is drawn as an arrow this many mm long; per-flank reactions are
# scaled to the same mm-per-newton so their vertical parts visibly sum to it.
FORCE_SPAN_MM = 3.1

# The naive (uncoupled) superposition always reads two independent flanks.
UNCOUPLED_FLANK_COUNT = 2.0


# --------------------------------------------------------------------------- #
# Analytic references (independent re-implementation of the Rust closed forms).
# --------------------------------------------------------------------------- #
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


def hertz_elliptic(
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


def gothic_radii(ball: float, tube: float, centre_radius: float) -> tuple[float, float]:
    """Return the ``(circumferential, meridional)`` relative radii of a groove."""
    radius_x = 1.0 / (1.0 / ball + 1.0 / centre_radius)
    radius_y = 1.0 / (1.0 / ball - 1.0 / tube)
    return radius_x, radius_y


# --------------------------------------------------------------------------- #
# Solved-case bundle.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CaseResult:
    """One solved regime, with everything the two rows need to draw it."""

    label: str
    offset: float
    y0: float
    alpha: float
    split: bool
    approach: float
    peak_pressure: float
    eta_solver: float
    eta_coupled: float
    y: NDArray[np.float64]
    cut: NDArray[np.float64]
    p0_reference: float
    ay_reference: float
    p0_single: float
    view: float


def _even(value: float) -> int:
    """Return ``value`` rounded up to the next even integer (>= 24)."""
    n = max(math.ceil(value), 24)
    return n + (n & 1)


def solve_case(label: str, offset: float, stiffness: float) -> CaseResult:
    """Solve one groove and bundle the cross-section + cut + coupling diagnostics.

    ``offset`` is the arc-centre shim (zero is the single arch). ``stiffness`` is
    the per-flank Hertz ``K`` used to read off the effective flank count
    ``eta = P / (K delta^{3/2})`` and the coupled-law prediction at the solver's
    own approach.
    """
    radius_x, radius_y = gothic_radii(BALL, TUBE, CENTRE_RADIUS)
    split = offset > 0.0
    # Analytic semi-axes: per-flank (half load) when split, single arch (full load)
    # otherwise — the reference each row is drawn against.
    half_load = LOAD / 2.0 if split else LOAD
    ax_a, ay_ref, p0_reference = hertz_elliptic(radius_x, radius_y, half_load, E_STAR)
    _, ay_single, p0_single = hertz_elliptic(radius_x, radius_y, LOAD, E_STAR)

    y0 = offset * BALL / (TUBE - BALL)
    alpha = math.asin(y0 / BALL) if split else 0.0

    spacing = ax_a / CELLS_PER_SEMI
    half_x = MARGIN * ax_a
    half_y = (y0 + ay_ref + MARGIN * ax_a) if split else (ay_single + MARGIN * ax_a)
    nx = _even(2.0 * half_x / spacing)
    ny = _even(2.0 * half_y / spacing)
    sol = hertzian.solve_sphere_in_gothic_arch(
        sphere_radius=BALL,
        tube_radius=TUBE,
        centre_radius=CENTRE_RADIUS,
        centre_offset=offset,
        load=LOAD,
        e_star=E_STAR,
        grid=(nx, ny),
        domain=(nx * spacing, ny * spacing),
        tol=SOLVE_TOL,
        max_iter=SOLVE_MAX_ITER,
    )
    approach = sol.approach
    pressure = np.asarray(sol.pressure)
    y = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * spacing
    cut = pressure[nx // 2, :]

    eta_solver = sol.total_load / (stiffness * approach**1.5)
    eta_coupled = _coupled_flank_count(radius_x, radius_y, y0, approach, stiffness)
    view = (y0 + ay_ref + 1.5 * ax_a) * MM if split else (ay_single + 1.5 * ax_a) * MM

    return CaseResult(
        label=label,
        offset=offset,
        y0=y0,
        alpha=alpha,
        split=split,
        approach=approach,
        peak_pressure=float(sol.max_pressure),
        eta_solver=eta_solver,
        eta_coupled=eta_coupled,
        y=y,
        cut=cut,
        p0_reference=p0_reference,
        ay_reference=ay_ref,
        p0_single=p0_single,
        view=view,
    )


def _coupled_flank_count(
    radius_x: float,
    radius_y: float,
    y0: float,
    approach: float,
    stiffness: float,
) -> float:
    """Return the reduced coupled law's ``eta`` at the solver's own approach.

    The neighbour-lift law each flank sees the other's Boussinesq far field, so
    ``Q_+ + Q_-`` drops below the uncoupled ``2 K delta^{3/2}``. The single arch
    (``y0 = 0``) has no neighbour to lift it, so its ``eta`` is one by construction.
    """
    if y0 <= 0.0:
        return 1.0
    law = hertzian.GothicArchLaw.from_elliptic_flank(
        radius_x=radius_x, radius_y=radius_y, e_star=E_STAR, contact_angle=0.1
    ).with_flank_coupling(e_star=E_STAR, offset=y0)
    q_plus, q_minus = law.coupled_loads(approach, approach)
    return (q_plus + q_minus) / (stiffness * approach**1.5)


# --------------------------------------------------------------------------- #
# Row 1 — the side cross-section: sphere + two tori + load vectors.
# --------------------------------------------------------------------------- #
def _draw_ball(ax: Axes) -> None:
    """Draw the sphere cross-section, a circle of the ball radius."""
    ax.add_patch(
        plt.Circle(
            (0.0, 0.0),
            BALL * MM,
            facecolor=BALL_FACE,
            edgecolor=BALL_EDGE,
            lw=1.4,
            zorder=1,
        )
    )


def _torus_centre(contact_sign: float, alpha: float, tube_draw: float) -> tuple[float, float]:
    """Return the drawn centre (mm) of the arc the ball contacts at ``contact_sign*y0``.

    Internal tangency puts the groove arc's centre of curvature a distance
    ``r - Rs`` from the ball centre, along the inward contact normal — which, for
    the ogival (crossed) Gothic arch, sits on the *opposite* side from the contact.
    """
    reach = (tube_draw - BALL) * MM
    return (-contact_sign * reach * math.sin(alpha), reach * math.cos(alpha))


def _draw_tori(ax: Axes, case: CaseResult, tube_draw: float) -> None:
    """Draw the one or two tube-radius circles (tori) that form the groove."""
    radius = tube_draw * MM
    if case.split:
        signs_and_colours = ((1.0, TORUS_PLUS), (-1.0, TORUS_MINUS))
    else:
        signs_and_colours = ((0.0, TORUS_SINGLE),)
    for sign, colour in signs_and_colours:
        cx, cy = _torus_centre(sign, case.alpha, tube_draw)
        ax.add_patch(
            plt.Circle((cx, cy), radius, fill=False, lw=1.6, ec=colour, alpha=0.55, zorder=2)
        )
    label = "two tori (tube $r$)" if case.split else "one torus (tube $r$)"
    ax.text(
        -CROSS_HALF_WIDTH + 0.2,
        CROSS_TOP - 0.5,
        label,
        fontsize=9,
        color=TORUS_PLUS if case.split else TORUS_SINGLE,
    )


def _ball_surface_point(y_mm: float) -> tuple[float, float]:
    """Return the lower ball-surface point (mm) at meridional position ``y_mm``."""
    radius = BALL * MM
    return (y_mm, -math.sqrt(max(radius * radius - y_mm * y_mm, 0.0)))


def _draw_contact_patch(ax: Axes, case: CaseResult) -> None:
    """Stroke the contact patch the solver finds, on the ball surface.

    The arc is drawn only where the cut pressure is in contact, so a separated
    two-flank contact shows as two arcs (with the non-contact Gothic point between
    them) while the overlapping one shows as a single connected arc.
    """
    radius = BALL * MM
    y_mm = case.y * MM
    z = -np.sqrt(np.clip(radius * radius - y_mm * y_mm, 0.0, None))
    in_contact = case.cut > CONTACT_FLOOR * case.cut.max()
    z_contact = np.where(in_contact, z, np.nan)
    ax.plot(y_mm, z_contact, color=SOLVER_COLOUR, lw=5.0, zorder=4, solid_capstyle="round")
    # Mark the contact centre(s) at +/- y0 so the offsets read off the ball.
    for sign in (1.0, -1.0) if case.split else (0.0,):
        ax.plot(*_ball_surface_point(sign * case.y0 * MM), "o", ms=4.0, color="white", zorder=5)


def _draw_load_vectors(ax: Axes, case: CaseResult, force_scale: float) -> None:
    """Draw the applied load and the per-flank reaction vectors on the ball."""
    # Applied load P: one arrow pressing the ball down into the groove from above.
    tail = CROSS_TOP - 0.2
    ax.annotate(
        "",
        xy=(0.0, tail - LOAD * force_scale),
        xytext=(0.0, tail),
        arrowprops={"arrowstyle": "-|>", "color": "0.25", "lw": 2.8},
        zorder=5,
    )
    ax.text(0.22, tail - 0.5 * LOAD * force_scale, "$P$ (applied)", fontsize=11, color="0.2")

    # Per-flank reactions Q_+/- along the (exact) flank normals, summing to P.
    sin_a, cos_a = math.sin(case.alpha), math.cos(case.alpha)
    flank_load = LOAD / (2.0 * cos_a) if case.split else LOAD
    length = flank_load * force_scale
    contact_signs = (1.0, -1.0) if case.split else (0.0,)
    for sign in contact_signs:
        base = _ball_surface_point(sign * case.y0 * MM)
        if case.split:  # a faint vertical guide so the tilt alpha reads off it
            ax.plot([base[0], base[0]], [base[1], base[1] + length], color="0.7", lw=0.8, ls=":")
        # Reaction on the ball: up and toward the axis, the inward flank normal.
        ax.annotate(
            "",
            xy=(base[0] - sign * sin_a * length, base[1] + cos_a * length),
            xytext=base,
            arrowprops={"arrowstyle": "-|>", "color": SOLVER_COLOUR, "lw": 2.8},
            zorder=6,
        )
    _annotate_flank_force(ax, case, flank_load, length)


def _annotate_flank_force(ax: Axes, case: CaseResult, flank_load: float, length: float) -> None:
    """Label one reaction vector with its magnitude and the contact angle."""
    if case.split:
        base = _ball_surface_point(case.y0 * MM)
        tip = (base[0] - math.sin(case.alpha) * length, base[1] + math.cos(case.alpha) * length)
        ax.text(
            tip[0] + 0.15,
            tip[1] + 0.05,
            rf"$Q_\pm\!=\!{flank_load:.0f}\,$N",
            fontsize=9,
            color=SOLVER_COLOUR,
            ha="left",
        )
        ax.annotate(
            rf"$\alpha={math.degrees(case.alpha):.0f}^\circ$",
            xy=(base[0], base[1] + 0.45 * length),
            fontsize=10,
            color="0.3",
            ha="center",
        )
    else:
        base = _ball_surface_point(0.0)
        ax.text(
            0.2,
            base[1] + 0.5 * length,
            rf"$Q={flank_load:.0f}\,$N",
            fontsize=9,
            color=SOLVER_COLOUR,
        )


def draw_cross_section(ax: Axes, case: CaseResult) -> None:
    """Draw the full side cross-section for one regime (the top row)."""
    tube_draw = TUBE_DRAW_FACTOR * BALL
    _draw_ball(ax)
    _draw_tori(ax, case, tube_draw)
    _draw_contact_patch(ax, case)
    _draw_load_vectors(ax, case, FORCE_SPAN_MM / LOAD)

    ax.set_xlim(-CROSS_HALF_WIDTH, CROSS_HALF_WIDTH)
    ax.set_ylim(CROSS_BOTTOM, CROSS_TOP)
    ax.set_aspect("equal")
    ax.set_xlabel("y (mm) — across the groove (meridional)")
    ax.set_ylabel("z (mm) — depth")
    ax.set_title(case.label, fontweight="bold", fontsize=11)
    ax.grid(visible=True, alpha=0.15)


# --------------------------------------------------------------------------- #
# Row 2 — the meridional pressure cut: exact solver vs approximate Hertz.
# --------------------------------------------------------------------------- #
def _draw_reference_flanks(ax: Axes, case: CaseResult) -> None:
    """Draw the analytic Hertz semi-ellipse(s) the cut is validated against."""
    centres = (case.y0, -case.y0) if case.split else (0.0,)
    for centre in centres:
        s = np.linspace(centre - case.ay_reference, centre + case.ay_reference, 200)
        profile = case.p0_reference * np.sqrt(
            np.clip(1.0 - ((s - centre) / case.ay_reference) ** 2, 0.0, None)
        )
        ax.plot(s * MM, profile * GPA, color=REFERENCE_COLOUR, lw=2.0, ls="--")
    flank_label = "analytic flanks (Hertz, P/2)" if case.split else "analytic Hertz (P)"
    ax.plot([], [], color=REFERENCE_COLOUR, lw=2.0, ls="--", label=flank_label)


def draw_pressure_cut(ax: Axes, case: CaseResult) -> None:
    """Draw the meridional pressure cut: solver (exact) vs Hertz (approximate)."""
    if case.split:
        overlap = case.ay_reference - case.y0
        if overlap > 0.0:
            ax.axvspan(-overlap * MM, overlap * MM, color="0.9", label="flank overlap (½ each)")
    _draw_reference_flanks(ax, case)
    ax.scatter(
        case.y * MM,
        case.cut * GPA,
        s=9,
        c=SOLVER_COLOUR,
        alpha=0.75,
        zorder=3,
        label="solver (exact)",
    )
    if case.split:
        ax.axhline(
            case.p0_single * GPA, ls=":", c="0.6", lw=1.2, label="single-arc peak (Hertz, P)"
        )
    ax.set_xlim(-case.view, case.view)
    ax.set_ylim(0.0, 1.18 * case.p0_single * GPA)
    ax.set_xlabel("y (mm) — across the groove")
    ax.set_ylabel("pressure (GPa)")
    ax.set_title(_eta_caption(case), fontsize=10, color="0.15")
    ax.grid(visible=True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def _eta_caption(case: CaseResult) -> str:
    """Return the effective-flank-count caption (exact / coupled / uncoupled)."""
    if not case.split:
        return rf"$\eta=1$ (one ellipse) · $\delta={case.approach * UM:.2f}\,\mu$m"
    return (
        rf"$\eta$: {case.eta_solver:.2f} exact · {case.eta_coupled:.2f} coupled · "
        rf"{UNCOUPLED_FLANK_COUNT:.0f} uncoupled"
    )


# --------------------------------------------------------------------------- #
# Figure.
# --------------------------------------------------------------------------- #
def main() -> None:
    """Solve the three regimes and render the two-row cross-section figure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})
    print("solving the Gothic-arch coupling regimes for the cross-section ...")

    radius_x, radius_y = gothic_radii(BALL, TUBE, CENTRE_RADIUS)
    stiffness = hertzian.GothicArchLaw.from_elliptic_flank(
        radius_x=radius_x, radius_y=radius_y, e_star=E_STAR, contact_angle=0.1
    ).stiffness
    _, ay_half, _ = hertz_elliptic(radius_x, radius_y, LOAD / 2.0, E_STAR)
    half_overlap_shim = 0.5 * ay_half * (TUBE - BALL) / BALL

    cases = [
        solve_case("(a) single arc — one ellipse", 0.0, stiffness),
        solve_case("(b) half overlap — the in-between", half_overlap_shim, stiffness),
        solve_case("(c) separated flanks — symmetric", SEPARATED_SHIM, stiffness),
    ]
    for case in cases:
        print(
            f"  {case.label}: y0={case.y0 * MM:.3f} mm  alpha={math.degrees(case.alpha):.1f} deg  "
            f"delta={case.approach * UM:.2f} um  peak={case.peak_pressure * GPA:.3f} GPa  "
            f"eta={case.eta_solver:.3f} (coupled {case.eta_coupled:.3f})"
        )

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.6))
    for column, case in enumerate(cases):
        draw_cross_section(axes[0, column], case)
        draw_pressure_cut(axes[1, column], case)

    fig.suptitle(
        "hertzian — first-order flank coupling, side cross-section: sphere + two tori, "
        "load vectors, exact vs approximate",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path = OUT_DIR / "coupling_cross_section.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024.0
    print(f"  wrote {path.relative_to(OUT_DIR.parent.parent)}  ({size_kb:.0f} KiB)")
    print("done.")


if __name__ == "__main__":
    main()
