"""Cross-validation against Tamaas, an independent contact-mechanics code (P4).

Rough and arbitrary-shape contacts have no closed-form reference, so the P4
milestone validates the solver against `Tamaas <https://tamaas.readthedocs.io>`_
(EPFL): a mature, independently developed boundary-element contact solver.
Tamaas is normally periodic, but its **non-periodic** (free-space) Boussinesq
operator matches hertzian's boundary condition exactly, so the two solve the
*same* problem on the *same* grid and must agree.

Tamaas is an optional dependency; the module skips when it is not installed.
Install it for a run with, e.g., ``uv run --with tamaas pytest``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

import hertzian

if TYPE_CHECKING:
    from numpy.typing import NDArray

tamaas = pytest.importorskip("tamaas")

# Both codes use the same free-space Boussinesq kernel and a Polonsky-Keer
# iteration, so on an identical grid they converge to the same discrete solution:
# agreement is limited only by the two solver tolerances, far below any physical
# tolerance. The bounds below leave generous headroom over that floor.
LOAD_RTOL = 1e-3
PEAK_RTOL = 5e-3
FIELD_RTOL = 1e-2


def _tamaas_free_space(
    gap: NDArray[np.float64], load: float, e_star: float, spacing: float
) -> NDArray[np.float64]:
    """Solve the same contact in Tamaas with its non-periodic operator.

    Mirrors :func:`hertzian.solve_height_field`: a rigid profile of the given
    undeformed ``gap`` pressed onto an elastic half-space of modulus ``e_star``
    on an isotropic grid of the given ``spacing``, under a prescribed total
    ``load``. Returns the contact pressure field on the same ``(nx, ny)`` grid.
    ``nu = 0`` is chosen so Tamaas's contact modulus ``E / (1 - nu^2)`` equals
    ``e_star``.
    """
    tamaas.set_log_level(tamaas.LogLevel.error)
    nx, ny = gap.shape
    size_x, size_y = nx * spacing, ny * spacing
    model = tamaas.ModelFactory.createModel(tamaas.model_type.basic_2d, [size_x, size_y], [nx, ny])
    model.E = e_star
    model.nu = 0.0
    tamaas.ModelFactory.registerNonPeriodic(model, "free_space")
    # Tamaas heights are the surface elevation (apex highest), i.e. the negation
    # of hertzian's non-negative gap.
    solver = tamaas.PolonskyKeerRey(
        model,
        np.ascontiguousarray(-gap),
        1e-12,
        tamaas.PolonskyKeerRey.pressure,
        tamaas.PolonskyKeerRey.pressure,
    )
    solver.setIntegralOperator("free_space")
    solver.max_iter = 30000
    solver.solve(load / (size_x * size_y))  # pressure-controlled: mean pressure
    return np.asarray(model.traction, dtype=np.float64).reshape(nx, ny)


def _agree(name: str, gap: NDArray[np.float64], load: float, e_star: float, spacing: float) -> None:
    """Assert hertzian and Tamaas agree on the contact for ``gap``."""
    ours = hertzian.solve_height_field(
        gap=gap, load=load, e_star=e_star, dx=spacing, dy=spacing, tol=1e-10, max_iter=30000
    )
    theirs = _tamaas_free_space(gap, load, e_star, spacing)
    cell_area = spacing * spacing

    our_pressure = ours.pressure
    peak = float(theirs.max())
    assert ours.diagnostics.converged, name
    assert abs(float(our_pressure.sum()) * cell_area - load) / load <= LOAD_RTOL, name
    assert abs(float(our_pressure.max()) - peak) / peak <= PEAK_RTOL, name
    assert float(np.abs(our_pressure - theirs).max()) / peak <= FIELD_RTOL, name


def _sphere_gap(radius: float, n: int, spacing: float) -> NDArray[np.float64]:
    """Sample a centred paraboloidal sphere gap ``h = (x^2 + y^2) / (2 R)``."""
    x = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * spacing
    gap = (x[:, np.newaxis] ** 2 + x[np.newaxis, :] ** 2) / (2.0 * radius)
    return np.ascontiguousarray(gap, dtype=np.float64)


def test_sphere_matches_tamaas() -> None:
    """A smooth Hertz sphere agrees with Tamaas to the solver tolerance."""
    radius, load, e_star = 10.0e-3, 50.0, 70.0e9
    a = (3.0 * load * radius / (4.0 * e_star)) ** (1.0 / 3.0)
    n = 128
    spacing = 6.0 * a / n
    _agree("sphere", _sphere_gap(radius, n, spacing), load, e_star, spacing)


def test_rough_sphere_matches_tamaas() -> None:
    """A rough (sphere + cosine waviness) contact agrees with Tamaas.

    This is the case with no analytic reference — exactly where an independent
    code earns its keep — yet the two solvers still match on the fragmented
    contact.
    """
    radius, load, e_star = 10.0e-3, 50.0, 70.0e9
    a = (3.0 * load * radius / (4.0 * e_star)) ** (1.0 / 3.0)
    delta = a * a / radius
    n = 128
    spacing = 6.0 * a / n
    x = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * spacing
    big_x, big_y = x[:, np.newaxis], x[np.newaxis, :]
    wavelength = 1.2 * a
    gap = _sphere_gap(radius, n, spacing) + 0.3 * delta * np.cos(
        2.0 * math.pi * big_x / wavelength
    ) * np.cos(2.0 * math.pi * big_y / wavelength)
    _agree("rough", np.ascontiguousarray(gap), load, e_star, spacing)
