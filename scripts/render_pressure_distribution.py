"""Render and verify the per-flank pressure footprint — the Coulomb-friction cap.

The reduced law (:class:`hertzian.GothicArchLaw`) gives a multibody integrator the
*resultant* ``F(delta)``; a Coulomb friction model needs more — the *distribution*
``p(x, y)`` the tangential traction is capped by, ``|tau| <= mu p``. Each flank is an
elliptic-Hertz contact carrying its (coupled) load ``Q``, so its pressure is the
half-ellipsoid ``p = p0 sqrt(1 - (x/ax)^2 - (y/ay)^2)`` with ``p0 = 3Q/(2 pi ax ay)``.
By Hertz's cube-root load scaling the whole footprint follows from a once-calibrated
reference (``a ~ Q^{1/3}``, ``p0 = cp Q^{1/3}``), so ``law.flank_pressure(Q)`` builds it
in a couple of ``cbrt``s — no eccentricity solve in the inner loop.

Four panels, drawn from a handful of full solves:

* **(A) Peak-pressure calibration.** A single-arch load sweep lands the solver peak on
  the ``p0 = cp P^{1/3}`` line; separated two-flank peaks land on ``cp (P/2)^{1/3}`` —
  each flank a half-load Hertz patch. Pins the cube-root pressure scaling.
* **(B) The cap vs the solver (separated).** The reconstructed per-flank half-ellipsoids
  (from the coupled loads) land on the solver's meridional cut; the shaded area is the
  Coulomb cap ``mu p`` a friction model rides under.
* **(C) The 2-D Coulomb traction cap.** ``mu p(x, y)`` over the two flank ellipses, with
  the solver's contact outline — the bound a tangential model integrates to ``mu Q``.
* **(D) Validity.** Reconstructed vs solver peak as the shim closes: exact where the two
  footprints are *resolved* (``y0 >~ ay``), with naive superposition over-counting the
  seam once they merge — the single-patch coalescence left to the next stage.

Run it (matplotlib is a render-only dependency, kept out of the locked env):

    uv run --with matplotlib python scripts/render_pressure_distribution.py
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

# The separated, gallery groove (65 um shim -> y0 ~ 1.6 mm, alpha ~ 24 deg).
SEPARATED_SHIM = 65.0e-6

# An illustrative dry-steel-ish friction coefficient for the Coulomb cap mu p.
MU = 0.12

# Solver resolution: cells per semi-axis, free-space margin, tolerances. The grid is
# *anisotropic* (one cell size per semi-axis) so the ~10:1 elongated conformal contact
# stays small; the peak sits where the field is locally flat, so it is well resolved
# even on a modest mesh. The separated B/C solve is run a touch finer for a smooth
# contact outline.
CELLS_PER_SEMI = 12.0
CELLS_PER_SEMI_FINE = 18.0
MARGIN = 2.5
SOLVE_TOL = 1.0e-9
SOLVE_MAX_ITER = 40000

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


def _even(value: float) -> int:
    """Return ``value`` rounded up to the next even integer (>= 24)."""
    n = max(math.ceil(value), 24)
    return n + (n & 1)


def solve_groove(
    offset: float, load: float, cells: float = CELLS_PER_SEMI
) -> tuple[object, float, float, float]:
    """Solve one groove; return ``(solution, y0, dx, dy)`` (anisotropic spacing)."""
    split = offset > 0.0
    half = load / 2.0 if split else load
    ax_a, ay_a, _ = elliptic_hertz(RADIUS_X, RADIUS_Y, half, E_STAR)
    y0 = offset * BALL / (TUBE - BALL)
    dx, dy = ax_a / cells, ay_a / cells
    half_x = MARGIN * ax_a
    half_y = (y0 + MARGIN * ay_a) if split else (MARGIN * ay_a)
    nx, ny = _even(2.0 * half_x / dx), _even(2.0 * half_y / dy)
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
    return sol, y0, dx, dy


# --------------------------------------------------------------------------- #
# (A) Peak-pressure calibration p0 = cp Q^{1/3}.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PeakCalibration:
    """The single-arch and separated two-flank peak-pressure sweeps."""

    single_load: NDArray[np.float64]
    single_peak: NDArray[np.float64]
    two_load: NDArray[np.float64]
    two_peak: NDArray[np.float64]
    ratio_mean: float
    ratio_spread: float


def calibrate_peak() -> PeakCalibration:
    """Sweep the load and read the solver peak against ``cp Q^{1/3}``."""
    loads = np.geomspace(25.0, 800.0, 8)
    single_peak = np.array([solve_groove(0.0, float(p))[0].max_pressure for p in loads])
    two_peak = np.array([solve_groove(SEPARATED_SHIM, float(p))[0].max_pressure for p in loads])
    ratios = single_peak / (CP * loads ** (1.0 / 3.0))
    return PeakCalibration(
        single_load=loads,
        single_peak=single_peak,
        two_load=loads,
        two_peak=two_peak,
        ratio_mean=float(np.mean(ratios)),
        ratio_spread=float(np.max(ratios) - np.min(ratios)),
    )


def _panel_calibration(ax: Axes, cal: PeakCalibration) -> None:
    """Draw the log-log peak-pressure calibration."""
    line = np.geomspace(cal.single_load.min(), cal.single_load.max(), 100)
    ax.plot(
        line,
        CP * line ** (1.0 / 3.0) * GPA,
        color=REFERENCE_COLOUR,
        lw=2.0,
        label=r"$c_p\,Q^{1/3}$ (one flank, load $Q$)",
    )
    ax.plot(
        line,
        CP * (line / 2.0) ** (1.0 / 3.0) * GPA,
        color=REFERENCE_COLOUR,
        lw=2.0,
        ls="--",
        label=r"$c_p\,(P/2)^{1/3}$ (each of two flanks)",
    )
    ax.scatter(
        cal.single_load,
        cal.single_peak * GPA,
        s=28,
        c=SOLVER_COLOUR,
        zorder=3,
        label="solver, 1 arch",
    )
    ax.scatter(
        cal.two_load,
        cal.two_peak * GPA,
        s=28,
        marker="s",
        c=LAW_COLOUR,
        zorder=3,
        label="solver, 2 flanks",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total load $P$ (N)")
    ax.set_ylabel("peak pressure $p_0$ (GPa)")
    ax.set_title(
        f"(A) Peak scaling $p_0 = c_p Q^{{1/3}}$ — solver/line {cal.ratio_mean:.4f}",
        fontweight="bold",
        fontsize=10,
    )
    ax.grid(visible=True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


# --------------------------------------------------------------------------- #
# (B, C) The separated cap vs the solver, and the 2-D Coulomb traction cap.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeparatedCap:
    """One separated solve plus the reconstructed per-flank caps."""

    y0: float
    y: NDArray[np.float64]
    cut: NDArray[np.float64]
    recon_cut: NDArray[np.float64]
    peak_ratio: float
    cut_residual: float
    cap_plus: hertzian.FlankPressure
    cap_minus: hertzian.FlankPressure
    q_plus: float
    pressure: NDArray[np.float64]
    x_grid: NDArray[np.float64]
    y_grid: NDArray[np.float64]


def separated_cap() -> SeparatedCap:
    """Solve the separated groove and reconstruct its per-flank pressure caps."""
    sol, y0, dx, dy = solve_groove(SEPARATED_SHIM, LOAD, cells=CELLS_PER_SEMI_FINE)
    delta = sol.approach
    pressure = np.asarray(sol.pressure)
    nx, ny = pressure.shape
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    cut = pressure[nx // 2, :]

    law = LAW.with_flank_coupling(e_star=E_STAR, offset=y0)
    q_plus, q_minus = law.coupled_loads(delta, delta)
    cap_plus = law.flank_pressure(q_plus)
    cap_minus = law.flank_pressure(q_minus)
    recon = np.array(
        [
            cap_plus.pressure_at(0.0, float(yy) - y0) + cap_minus.pressure_at(0.0, float(yy) + y0)
            for yy in y
        ]
    )
    mask = cut > 0.02 * cut.max()
    residual = float(np.sqrt(np.mean((recon - cut)[mask] ** 2)) / cut.max())
    return SeparatedCap(
        y0=y0,
        y=y,
        cut=cut,
        recon_cut=recon,
        peak_ratio=float(recon.max() / cut.max()),
        cut_residual=residual,
        cap_plus=cap_plus,
        cap_minus=cap_minus,
        q_plus=q_plus,
        pressure=pressure,
        x_grid=x,
        y_grid=y,
    )


def _panel_cut(ax: Axes, cap: SeparatedCap) -> None:
    """Draw the separated meridional cut: solver vs reconstructed cap (shaded mu p)."""
    y_mm = cap.y * MM
    ax.fill_between(
        y_mm,
        MU * cap.recon_cut * GPA,
        color=LAW_COLOUR,
        alpha=0.18,
        label=rf"Coulomb cap $\mu p$  ($\mu={MU}$)",
    )
    ax.plot(
        y_mm,
        cap.recon_cut * GPA,
        color=LAW_COLOUR,
        lw=2.0,
        label=r"reduced cap $p$ (two half-ellipsoids)",
    )
    ax.scatter(
        y_mm, cap.cut * GPA, s=10, c=SOLVER_COLOUR, alpha=0.8, zorder=3, label="solver (exact)"
    )
    ax.set_xlim(
        -(cap.y0 + 2.2 * cap.cap_plus.semi_axes[1]) * MM,
        (cap.y0 + 2.2 * cap.cap_plus.semi_axes[1]) * MM,
    )
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("y (mm) — across the groove")
    ax.set_ylabel("pressure (GPa)")
    title = (
        f"(B) Per-flank cap vs solver — peak {cap.peak_ratio:.3f}, "
        f"cut {cap.cut_residual * 100:.1f}%"
    )
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.grid(visible=True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def _panel_traction_cap(ax: Axes, cap: SeparatedCap) -> None:
    """Draw the 2-D Coulomb traction cap mu p(x, y) with the solver contact outline."""
    axp, ayp = cap.cap_plus.semi_axes
    span_x = 1.4 * axp
    span_y = cap.y0 + 1.6 * ayp
    xs = np.linspace(-span_x, span_x, 160)
    ys = np.linspace(-span_y, span_y, 320)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    p0 = cap.cap_plus.peak_pressure
    field = p0 * np.sqrt(np.clip(1.0 - (gx / axp) ** 2 - ((gy - cap.y0) / ayp) ** 2, 0.0, None))
    field += p0 * np.sqrt(np.clip(1.0 - (gx / axp) ** 2 - ((gy + cap.y0) / ayp) ** 2, 0.0, None))
    mesh = ax.pcolormesh(gx * MM, gy * MM, MU * field * GPA, cmap="magma", shading="auto")
    plt.colorbar(mesh, ax=ax, label=r"$\mu\,p$ (GPa)", fraction=0.046, pad=0.04)

    # The solver's contact outline (where its pressure clears the contact floor).
    ax.contour(
        cap.x_grid[:, None] * MM + 0 * cap.y_grid[None, :],
        0 * cap.x_grid[:, None] + cap.y_grid[None, :] * MM,
        cap.pressure,
        levels=[CONTACT_FLOOR * cap.pressure.max()],
        colors=[REFERENCE_COLOUR],
        linewidths=1.6,
        linestyles="--",
    )
    ax.plot([], [], color=REFERENCE_COLOUR, lw=1.6, ls="--", label="solver contact edge")
    # The conformal contact is ~10:1 elongated, so the axes carry different scales
    # (equal aspect would draw the footprint as two unreadable slivers).
    ax.set_xlabel("x (mm) — circumferential (note: scale ≠ y)")
    ax.set_ylabel("y (mm) — meridional")
    ax.set_title(r"(C) 2-D Coulomb traction cap $\mu\,p(x,y)$", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


# --------------------------------------------------------------------------- #
# (D) Validity: reconstructed vs solver peak as the shim closes.
# --------------------------------------------------------------------------- #
def validity_curve(
    separations_in_b: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(solver_peak, reconstructed_peak)`` in GPa per flank separation."""
    semi_minor = elliptic_hertz(RADIUS_X, RADIUS_Y, LOAD / 2.0, E_STAR)[1]
    solver_peak = np.empty(separations_in_b.size)
    recon_peak = np.empty(separations_in_b.size)
    for i, sep in enumerate(separations_in_b):
        y0 = float(sep) * semi_minor
        offset = y0 * (TUBE - BALL) / BALL
        sol, _, _, dy = solve_groove(offset, LOAD)
        pressure = np.asarray(sol.pressure)
        nx, ny = pressure.shape
        y = (np.arange(ny) - (ny - 1) / 2.0) * dy
        solver_peak[i] = pressure[nx // 2, :].max()
        law = LAW.with_flank_coupling(e_star=E_STAR, offset=y0)
        q_plus, q_minus = law.coupled_loads(sol.approach, sol.approach)
        cap_p, cap_m = law.flank_pressure(q_plus), law.flank_pressure(q_minus)
        recon = np.array(
            [
                cap_p.pressure_at(0.0, float(yy) - y0) + cap_m.pressure_at(0.0, float(yy) + y0)
                for yy in y
            ]
        )
        recon_peak[i] = recon.max()
    return solver_peak, recon_peak


def _panel_validity(
    ax: Axes,
    separations: NDArray[np.float64],
    solver_peak: NDArray[np.float64],
    recon_peak: NDArray[np.float64],
) -> None:
    """Draw the reconstructed vs solver peak vs separation, marking the resolved band."""
    ax.axvspan(0.0, 1.0, color="0.92", label=r"overlap (merge, next stage)")
    ax.plot(
        separations, recon_peak * GPA, color=LAW_COLOUR, lw=2.0, label="reduced cap (superposed)"
    )
    ax.scatter(separations, solver_peak * GPA, s=28, c=SOLVER_COLOUR, zorder=3, label="solver peak")
    ax.set_xlabel(r"flank separation $y_0 / b$")
    ax.set_ylabel("peak pressure (GPa)")
    ax.set_title(
        "(D) Exact where flanks are resolved; superposition over-counts the merge",
        fontweight="bold",
        fontsize=9.5,
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
    print("verifying the per-flank pressure cap against the field solver ...")

    cal = calibrate_peak()
    print(
        f"  peak scaling p0 = cp Q^{{1/3}}: solver/line {cal.ratio_mean:.4f} "
        f"(spread {cal.ratio_spread:.4f})"
    )

    cap = separated_cap()
    print(
        f"  separated cap: peak ratio {cap.peak_ratio:.4f}  cut rel-L2 "
        f"{cap.cut_residual * 100:.2f}%"
    )
    print(
        f"  footprint integrates to Q = {cap.cap_plus.load:.3f} N "
        f"(coupled Q_+ = {cap.q_plus:.3f} N)  ->  full-sliding friction mu Q = "
        f"{MU * cap.cap_plus.load:.2f} N"
    )

    separations = np.linspace(0.4, 2.4, 9)
    solver_peak, recon_peak = validity_curve(separations)
    resolved = separations >= 1.0
    resolved_err = float(
        np.max(np.abs(recon_peak[resolved] - solver_peak[resolved]) / solver_peak[resolved])
    )
    print(f"  resolved-regime peak error (y0/b >= 1): {resolved_err * 100:.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4))
    _panel_calibration(axes[0, 0], cal)
    _panel_cut(axes[0, 1], cap)
    _panel_traction_cap(axes[1, 0], cap)
    _panel_validity(axes[1, 1], separations, solver_peak, recon_peak)
    fig.suptitle(
        "hertzian — the per-flank pressure footprint: a lightweight Coulomb-friction cap",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    path = OUT_DIR / "flank_pressure.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(
        f"  wrote {path.relative_to(OUT_DIR.parent.parent)} "
        f"({path.stat().st_size / 1024.0:.0f} KiB)"
    )
    print("done.")


if __name__ == "__main__":
    main()
