# Type stubs for the native ``hertzian._core`` extension module.
#
# Hand-written to mirror the PyO3 bindings in ``src/python.rs``. Keep the two in
# sync: every ``#[pyfunction]`` / ``#[pyclass]`` exported by ``register`` has a
# declaration here so ``mypy --strict`` and IDEs see the real types.

from typing import final

import numpy as np
from numpy.typing import NDArray

__version__: str

@final
class Diagnostics:
    @property
    def iterations(self) -> int: ...
    @property
    def residual(self) -> float: ...
    @property
    def converged(self) -> bool: ...

@final
class Solution:
    @property
    def pressure(self) -> NDArray[np.float64]: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def approach(self) -> float: ...
    @property
    def total_load(self) -> float: ...
    @property
    def contact_area(self) -> float: ...
    @property
    def contact_radius(self) -> float: ...
    @property
    def max_pressure(self) -> float: ...
    @property
    def contact_half_widths(self) -> tuple[float, float]: ...
    @property
    def ellipticity(self) -> float: ...
    @property
    def diagnostics(self) -> Diagnostics: ...

def solve_sphere_on_flat(
    *,
    radius: float,
    load: float,
    e_star: float,
    grid: tuple[int, int],
    domain: float | tuple[float, float],
    tol: float = ...,
    max_iter: int = ...,
) -> Solution: ...
def solve_sphere_on_sphere(
    *,
    radius_1: float,
    radius_2: float,
    load: float,
    e_star: float,
    grid: tuple[int, int],
    domain: float | tuple[float, float],
    tol: float = ...,
    max_iter: int = ...,
) -> Solution: ...
def solve_sphere_on_torus(
    *,
    sphere_radius: float,
    tube_radius: float,
    centre_radius: float,
    load: float,
    e_star: float,
    grid: tuple[int, int],
    domain: float | tuple[float, float],
    tol: float = ...,
    max_iter: int = ...,
) -> Solution: ...
def solve_sphere_in_gothic_arch(
    *,
    sphere_radius: float,
    tube_radius: float,
    centre_radius: float,
    centre_offset: float,
    load: float,
    e_star: float,
    grid: tuple[int, int],
    domain: float | tuple[float, float],
    tol: float = ...,
    max_iter: int = ...,
) -> Solution: ...
def solve_height_field(
    *,
    gap: NDArray[np.float64],
    load: float,
    e_star: float,
    dx: float,
    dy: float,
    tol: float = ...,
    max_iter: int = ...,
    boundary: str = ...,
) -> Solution: ...
