"""Render the validation gallery embedded in the README.

Solves the four contact problems the core currently handles and draws, for each,
the converged pressure field beside the analytic reference it is validated
against:

* **circular Hertz** -- a sphere on a flat (P1);
* **elliptic Hertz** -- a sphere on a torus outer equator (P2);
* **Sneddon's cone** -- a non-Hertzian singular-apex punch (P4); and
* **rough contact** -- a sphere plus cosine roughness, which fragments the patch
  into asperities (P4).

The analytic Hertz / Sneddon closed forms are re-implemented here (independently
of the Rust core) so each panel shows the solver landing on its reference rather
than on itself. Figures are written to ``docs/img/`` and embedded by the README.

Run it (matplotlib is a render-only dependency, deliberately kept out of the
locked environment, exactly like the Tamaas cross-validation):

    uv run --with matplotlib python scripts/render_gallery.py
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
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

# Output directory for the rendered panels (repo-root/docs/img).
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"

# Unit scales: SI in, engineering units out (metres -> mm, pascals -> MPa).
MM = 1.0e3
MPA = 1.0e-6

# Perceptually-uniform map for pressure; a muted reference colour for analytics.
PRESSURE_CMAP = "inferno"
REFERENCE_COLOUR = "#26c6da"
EPS = 2.220_446_049_250_313e-16

# Radial-profile view extent, in units of the contact radius `a`.
R_OVER_A_MAX = 1.5


# --------------------------------------------------------------------------- #
# Analytic references (independent re-implementation of the Rust closed forms).
# --------------------------------------------------------------------------- #
def hertz_circular(radius: float, load: float, e_star: float) -> tuple[float, float, float]:
    """Return ``(contact_radius, peak_pressure, approach)`` for circular Hertz."""
    a = (3.0 * load * radius / (4.0 * e_star)) ** (1.0 / 3.0)
    p0 = 3.0 * load / (2.0 * math.pi * a * a)
    return a, p0, a * a / radius


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


def sneddon_cone(slope: float, load: float, e_star: float) -> tuple[float, float]:
    """Return ``(contact_radius, approach)`` for Sneddon's rigid cone."""
    a = math.sqrt(2.0 * load / (math.pi * e_star * slope))
    return a, 0.5 * math.pi * slope * a


# --------------------------------------------------------------------------- #
# Solved-problem bundle.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Panel:
    """A solved field on a centred grid, ready to draw (lengths in metres)."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    pressure: NDArray[np.float64]
    title: str


def _centred_axis(n: int, spacing: float) -> NDArray[np.float64]:
    """Return the origin-centred physical coordinates of a length-``n`` axis."""
    return (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * spacing


def draw_field(
    ax: Axes,
    panel: Panel,
    *,
    overlay: Callable[[Axes], None] | None = None,
    vmax: float | None = None,
    title: str | None = None,
) -> None:
    """Render a pressure field as a heatmap with mm axes and an MPa colour bar."""
    extent = (
        float(panel.x[0] * MM),
        float(panel.x[-1] * MM),
        float(panel.y[0] * MM),
        float(panel.y[-1] * MM),
    )
    image = ax.imshow(
        panel.pressure.T * MPA,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=PRESSURE_CMAP,
        vmin=0.0,
        vmax=None if vmax is None else vmax * MPA,
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title or panel.title, fontweight="bold", fontsize=11)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("pressure (MPa)", fontsize=9)
    if overlay is not None:
        overlay(ax)


# --------------------------------------------------------------------------- #
# Individual scenarios -> Panel.
# --------------------------------------------------------------------------- #
def solve_circular() -> tuple[Panel, dict[str, float]]:
    """Circular Hertz: a sphere on a flat (P1)."""
    radius, load, e_star, n = 10.0e-3, 50.0, 70.0e9, 256
    a, p0, _ = hertz_circular(radius, load, e_star)
    domain = 5.0 * a
    sol = hertzian.solve_sphere_on_flat(
        radius=radius,
        load=load,
        e_star=e_star,
        grid=(n, n),
        domain=domain,
        tol=1e-9,
        max_iter=20000,
    )
    axis = _centred_axis(n, domain / n)
    panel = Panel(
        x=axis,
        y=axis,
        pressure=np.asarray(sol.pressure),
        title="Circular Hertz - sphere on flat",
    )
    return panel, {"a": a, "p0": p0, "radius": radius, "domain": domain, "n": float(n)}


def solve_elliptic() -> tuple[Panel, dict[str, float]]:
    """Elliptic Hertz: a sphere on a torus outer equator (P2)."""
    sphere_r, tube_r, centre_r, load, e_star = 12.0e-3, 4.0e-3, 20.0e-3, 60.0, 100.0e9
    radius_x = 1.0 / (1.0 / sphere_r + 1.0 / (centre_r + tube_r))
    radius_y = 1.0 / (1.0 / sphere_r + 1.0 / tube_r)
    ax_a, ay_a, p0 = hertz_elliptic(radius_x, radius_y, load, e_star)
    spacing = ay_a / 24.0
    nx = math.ceil(2.0 * 3.0 * ax_a / spacing)
    ny = math.ceil(2.0 * 3.0 * ay_a / spacing)
    nx += nx & 1
    ny += ny & 1
    sol = hertzian.solve_sphere_on_torus(
        sphere_radius=sphere_r,
        tube_radius=tube_r,
        centre_radius=centre_r,
        load=load,
        e_star=e_star,
        grid=(nx, ny),
        domain=(nx * spacing, ny * spacing),
        tol=1e-8,
        max_iter=20000,
    )
    panel = Panel(
        x=_centred_axis(nx, spacing),
        y=_centred_axis(ny, spacing),
        pressure=np.asarray(sol.pressure),
        title="Elliptic Hertz - sphere on torus",
    )
    return panel, {"ax": ax_a, "ay": ay_a, "p0": p0}


def solve_cone() -> tuple[Panel, dict[str, float]]:
    """Sneddon's cone: a non-Hertzian singular-apex punch (P4)."""
    slope, load, e_star, n = 0.02, 60.0, 100.0e9, 288
    a, _approach = sneddon_cone(slope, load, e_star)
    domain = 6.0 * a
    spacing = domain / n
    axis = _centred_axis(n, spacing)
    radial = np.hypot(axis[:, None], axis[None, :])
    gap = np.ascontiguousarray(slope * radial)
    sol = hertzian.solve_height_field(
        gap=gap, load=load, e_star=e_star, dx=spacing, dy=spacing, tol=1e-9, max_iter=20000
    )
    panel = Panel(
        x=axis,
        y=axis,
        pressure=np.asarray(sol.pressure),
        title="Sneddon cone - non-Hertzian apex",
    )
    return panel, {"a": a, "slope": slope, "e_star": e_star, "n": float(n), "domain": domain}


def solve_rough() -> tuple[Panel, Panel, dict[str, float]]:
    """Rough contact: a sphere plus cosine roughness, fragmented (P4)."""
    radius, load, e_star, n = 10.0e-3, 40.0, 70.0e9, 224
    a, _, delta = hertz_circular(radius, load, e_star)
    domain = 5.0 * a
    spacing = domain / n
    axis = _centred_axis(n, spacing)
    big_x, big_y = axis[:, None], axis[None, :]
    smooth = (big_x**2 + big_y**2) / (2.0 * radius)
    wavelength = 0.9 * a
    wave_x = np.cos(2.0 * math.pi * big_x / wavelength)
    wave_y = np.cos(2.0 * math.pi * big_y / wavelength)
    roughness = 0.8 * delta * wave_x * wave_y
    rough = hertzian.solve_height_field(
        gap=np.ascontiguousarray(smooth + roughness),
        load=load,
        e_star=e_star,
        dx=spacing,
        dy=spacing,
        tol=1e-9,
        max_iter=20000,
    )
    base = hertzian.solve_sphere_on_flat(
        radius=radius,
        load=load,
        e_star=e_star,
        grid=(n, n),
        domain=domain,
        tol=1e-9,
        max_iter=20000,
    )
    area_ratio = rough.contact_area / base.contact_area
    peak_ratio = rough.max_pressure / base.max_pressure
    smooth_panel = Panel(
        x=axis,
        y=axis,
        pressure=np.asarray(base.pressure),
        title="Smooth sphere (baseline)",
    )
    rough_panel = Panel(
        x=axis,
        y=axis,
        pressure=np.asarray(rough.pressure),
        title=f"Sphere + roughness  (area x{area_ratio:.2f}, peak x{peak_ratio:.1f})",
    )
    return smooth_panel, rough_panel, {"peak": float(rough.max_pressure)}


# --------------------------------------------------------------------------- #
# Figures.
# --------------------------------------------------------------------------- #
def _radial_overlay(a: float) -> Callable[[Axes], None]:
    """Return an overlay drawing the analytic contact circle of radius ``a``."""

    def overlay(ax: Axes) -> None:
        circle = plt.Circle((0.0, 0.0), a * MM, fill=False, ls="--", lw=1.4, ec=REFERENCE_COLOUR)
        ax.add_patch(circle)

    return overlay


def _ellipse_overlay(ax_a: float, ay_a: float) -> Callable[[Axes], None]:
    """Return an overlay drawing the analytic contact ellipse."""

    def overlay(ax: Axes) -> None:
        theta = np.linspace(0.0, 2.0 * math.pi, 256)
        ax.plot(
            ax_a * MM * np.cos(theta),
            ay_a * MM * np.sin(theta),
            ls="--",
            lw=1.4,
            color=REFERENCE_COLOUR,
        )

    return overlay


def figure_circular() -> None:
    """Circular Hertz: pressure field + radial profile vs the Hertz ellipsoid."""
    panel, meta = solve_circular()
    fig, (ax_field, ax_prof) = plt.subplots(1, 2, figsize=(9.4, 4.2))

    a, p0 = meta["a"], meta["p0"]
    draw_field(ax_field, panel, overlay=_radial_overlay(a), vmax=p0)

    xx, yy = np.meshgrid(panel.x, panel.y, indexing="ij")
    r = np.hypot(xx, yy).ravel() / a
    p = panel.pressure.ravel() * MPA
    mask = p > 1e-3 * p0 * MPA
    ax_prof.scatter(
        r[mask][::7], p[mask][::7], s=4, c="0.5", alpha=0.35, label="solver (all cells)"
    )
    r_line = np.linspace(0.0, 1.0, 200)
    ax_prof.plot(
        r_line,
        p0 * MPA * np.sqrt(1.0 - r_line**2),
        color=REFERENCE_COLOUR,
        lw=2.2,
        label=r"analytic  $p_0\sqrt{1-(r/a)^2}$",
    )
    ax_prof.set_xlabel("r / a")
    ax_prof.set_ylabel("pressure (MPa)")
    ax_prof.set_title("Radial profile vs analytic Hertz", fontweight="bold", fontsize=11)
    ax_prof.set_xlim(0.0, R_OVER_A_MAX)
    ax_prof.grid(visible=True, alpha=0.3)
    ax_prof.legend(frameon=False, fontsize=9)

    _save(fig, "circular.png")


def figure_elliptic() -> None:
    """Elliptic Hertz: pressure field with the analytic ellipse + axis cuts."""
    panel, meta = solve_elliptic()
    ax_a, ay_a, p0 = meta["ax"], meta["ay"], meta["p0"]
    fig, (ax_field, ax_prof) = plt.subplots(1, 2, figsize=(9.4, 4.2))

    draw_field(ax_field, panel, overlay=_ellipse_overlay(ax_a, ay_a), vmax=p0)

    j0 = panel.pressure.shape[1] // 2
    i0 = panel.pressure.shape[0] // 2
    ax_prof.scatter(
        panel.x * MM,
        panel.pressure[:, j0] * MPA,
        s=8,
        c="#ef6c00",
        alpha=0.6,
        label="solver, cut at y=0",
    )
    ax_prof.scatter(
        panel.y * MM,
        panel.pressure[i0, :] * MPA,
        s=8,
        c="#6a1b9a",
        alpha=0.6,
        label="solver, cut at x=0",
    )
    s_major = np.linspace(-ax_a, ax_a, 200)
    s_minor = np.linspace(-ay_a, ay_a, 200)
    ax_prof.plot(
        s_major * MM, p0 * MPA * np.sqrt(1.0 - (s_major / ax_a) ** 2), color="#ef6c00", lw=2.0
    )
    ax_prof.plot(
        s_minor * MM, p0 * MPA * np.sqrt(1.0 - (s_minor / ay_a) ** 2), color="#6a1b9a", lw=2.0
    )
    ax_prof.set_xlabel("position along axis (mm)")
    ax_prof.set_ylabel("pressure (MPa)")
    ax_prof.set_title("Principal-axis cuts vs analytic", fontweight="bold", fontsize=11)
    ax_prof.grid(visible=True, alpha=0.3)
    ax_prof.legend(frameon=False, fontsize=9)

    _save(fig, "elliptic.png")


def figure_cone() -> None:
    """Sneddon's cone: pressure field + radial profile showing the apex peak."""
    panel, meta = solve_cone()
    a, slope, e_star = meta["a"], meta["slope"], meta["e_star"]
    fig, (ax_field, ax_prof) = plt.subplots(1, 2, figsize=(9.4, 4.2))

    mean_pressure = 0.5 * e_star * slope
    draw_field(ax_field, panel, overlay=_radial_overlay(a), vmax=4.0 * mean_pressure)

    xx, yy = np.meshgrid(panel.x, panel.y, indexing="ij")
    r = np.hypot(xx, yy).ravel() / a
    p = panel.pressure.ravel() * MPA
    mask = (p > 1e-3 * mean_pressure * MPA) & (r < R_OVER_A_MAX)
    ax_prof.scatter(r[mask][::5], p[mask][::5], s=4, c="0.5", alpha=0.3, label="solver (all cells)")
    r_line = np.linspace(1e-3, 0.999, 400)
    ax_prof.plot(
        r_line,
        mean_pressure * MPA * np.arccosh(1.0 / r_line),
        color=REFERENCE_COLOUR,
        lw=2.2,
        label=r"analytic  $\frac{E^*m}{2}\,\mathrm{arccosh}(a/r)$",
    )
    ax_prof.axhline(mean_pressure * MPA, ls=":", c="0.4", lw=1.2, label="mean pressure  E*m/2")
    ax_prof.set_xlabel("r / a")
    ax_prof.set_ylabel("pressure (MPa)")
    ax_prof.set_title("Apex singularity vs Sneddon", fontweight="bold", fontsize=11)
    ax_prof.set_xlim(0.0, R_OVER_A_MAX)
    ax_prof.set_ylim(0.0, 4.0 * mean_pressure * MPA)
    ax_prof.grid(visible=True, alpha=0.3)
    ax_prof.legend(frameon=False, fontsize=9, loc="upper right")

    _save(fig, "cone.png")


def figure_roughness() -> None:
    """Rough contact: smooth baseline next to the fragmented rough patch."""
    smooth_panel, rough_panel, meta = solve_rough()
    vmax = meta["peak"]
    fig, (ax_smooth, ax_rough) = plt.subplots(1, 2, figsize=(9.4, 4.2))
    draw_field(ax_smooth, smooth_panel, vmax=vmax)
    draw_field(ax_rough, rough_panel, vmax=vmax)
    _save(fig, "roughness.png")


def figure_hero() -> None:
    """One banner row: the pressure field of each solved problem."""
    circular, _ = solve_circular()
    elliptic, ell_meta = solve_elliptic()
    cone, cone_meta = solve_cone()
    _, rough, _ = solve_rough()

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8))
    draw_field(axes[0], circular, title="Circular Hertz")
    draw_field(
        axes[1],
        elliptic,
        overlay=_ellipse_overlay(ell_meta["ax"], ell_meta["ay"]),
        title="Elliptic Hertz",
    )
    draw_field(
        axes[2],
        cone,
        vmax=4.0 * 0.5 * cone_meta["e_star"] * cone_meta["slope"],
        title="Sneddon cone",
    )
    draw_field(axes[3], rough, title="Sphere + roughness")
    fig.suptitle(
        "hertzian - validated contact pressure fields (FFT + BCCG, free-space DC-FFT)",
        fontsize=13,
        fontweight="bold",
    )
    _save(fig, "hero.png")


def _save(fig: plt.Figure, name: str) -> None:
    """Tighten layout and write ``fig`` to the output directory as a PNG."""
    fig.tight_layout()
    path = OUT_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024.0
    print(f"  wrote {path.relative_to(OUT_DIR.parent.parent)}  ({size_kb:.0f} KiB)")


def main() -> None:
    """Render every gallery figure into ``docs/img/``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})
    print(f"rendering gallery into {OUT_DIR} ...")
    figure_hero()
    figure_circular()
    figure_elliptic()
    figure_cone()
    figure_roughness()
    print("done.")


if __name__ == "__main__":
    main()
