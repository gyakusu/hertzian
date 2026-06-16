//! `PyO3` bindings: the `hertzian._core` extension module (design §8).
//!
//! This is a deliberately thin marshalling layer over `contact-core`. It
//! converts `NumPy` arrays and Python scalars into the core types, runs the
//! solver with the GIL released, and hands results back to Python:
//!
//! - **Zero-copy boundaries (§8.2).** Inputs are *borrowed* from `NumPy` via
//!   [`PyReadonlyArray2`] (no copy for validation); results move into `NumPy` via
//!   [`IntoPyArray::into_pyarray`] (one allocation, no return copy).
//! - **GIL release (§8.3).** The numerical core touches no Python objects, so
//!   the heavy solve runs inside [`Python::detach`]. A Python-borrowed view is
//!   not `Send` and must not cross that boundary, so the single input gap is
//!   copied to an owned `Array2` *before* detaching (the one boundary copy the
//!   design sanctions).
//! - **Errors (§8.2).** Invalid shapes/values are rejected at the boundary as
//!   `ValueError`/`TypeError`, never as Rust panics surfacing as
//!   `PanicException`.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::geometry::Torus;
use crate::grid::Grid;
use crate::material::Material;
use crate::scenarios::{solve_sampled_gap, sphere_on_flat, sphere_on_sphere, sphere_on_torus};
use crate::solution::{Diagnostics as CoreDiagnostics, Solution as CoreSolution};
use crate::solver::Config;

/// Registers the classes and functions on the `_core` module.
#[allow(
    clippy::redundant_pub_crate,
    reason = "called from the crate root; `pub` here would instead trip `unreachable_pub`"
)]
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Solution>()?;
    module.add_class::<Diagnostics>()?;
    module.add_function(wrap_pyfunction!(solve_sphere_on_flat, module)?)?;
    module.add_function(wrap_pyfunction!(solve_sphere_on_sphere, module)?)?;
    module.add_function(wrap_pyfunction!(solve_sphere_on_torus, module)?)?;
    module.add_function(wrap_pyfunction!(solve_height_field, module)?)?;
    Ok(())
}

/// Convergence diagnostics from the iterative solver (read-only).
#[pyclass(name = "Diagnostics", module = "hertzian._core", frozen)]
struct Diagnostics {
    /// Number of iterations performed.
    #[pyo3(get)]
    iterations: usize,
    /// Final relative pressure-update residual.
    #[pyo3(get)]
    residual: f64,
    /// Whether the tolerance was met before the iteration cap.
    #[pyo3(get)]
    converged: bool,
}

impl From<CoreDiagnostics> for Diagnostics {
    fn from(core: CoreDiagnostics) -> Self {
        Self {
            iterations: core.iterations,
            residual: core.residual,
            converged: core.converged,
        }
    }
}

#[pymethods]
impl Diagnostics {
    fn __repr__(&self) -> String {
        format!(
            "Diagnostics(iterations={}, residual={:.3e}, converged={})",
            self.iterations,
            self.residual,
            py_bool(self.converged),
        )
    }
}

/// A solved contact: the converged pressure field and its derived quantities.
///
/// Returned by every solver entry point. Immutable from Python; every accessor
/// is a read-only property.
#[pyclass(name = "Solution", module = "hertzian._core", frozen)]
struct Solution {
    inner: CoreSolution,
}

impl Solution {
    const fn wrap(inner: CoreSolution) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl Solution {
    /// The converged contact pressure field `p(x, y)` (pascals).
    ///
    /// A fresh, C-contiguous `(nx, ny)` ``float64`` array; axis 0 is `x`, axis 1
    /// is `y`. The buffer is moved into `NumPy` without a return copy.
    #[getter]
    fn pressure<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.pressure().to_owned().into_pyarray(py)
    }

    /// Grid shape `(nx, ny)` of the pressure field.
    #[getter]
    const fn shape(&self) -> (usize, usize) {
        self.inner.grid().dims()
    }

    /// Rigid-body approach `delta` (metres).
    #[getter]
    const fn approach(&self) -> f64 {
        self.inner.approach()
    }

    /// Integrated total normal load `sum(p) * cell_area` (newtons).
    #[getter]
    fn total_load(&self) -> f64 {
        self.inner.total_load()
    }

    /// Total contact area (square metres).
    #[getter]
    fn contact_area(&self) -> f64 {
        self.inner.contact_area()
    }

    /// Equivalent circular contact radius `sqrt(area / pi)` (metres).
    ///
    /// For an elliptic contact this is the geometric mean `sqrt(a_x a_y)`.
    #[getter]
    fn contact_radius(&self) -> f64 {
        self.inner.contact_radius()
    }

    /// Peak contact pressure (pascals).
    #[getter]
    fn max_pressure(&self) -> f64 {
        self.inner.max_pressure()
    }

    /// Measured contact semi-axes `(a_x, a_y)` along the grid axes (metres).
    #[getter]
    fn contact_half_widths(&self) -> (f64, f64) {
        self.inner.contact_half_widths()
    }

    /// Measured ellipticity `max(a_x, a_y) / min(a_x, a_y) >= 1`.
    #[getter]
    fn ellipticity(&self) -> f64 {
        self.inner.ellipticity()
    }

    /// Solver convergence diagnostics.
    #[getter]
    fn diagnostics(&self) -> Diagnostics {
        self.inner.diagnostics().into()
    }

    fn __repr__(&self) -> String {
        let diagnostics = self.inner.diagnostics();
        format!(
            "Solution(converged={}, iterations={}, contact_radius={:.6e}, \
             max_pressure={:.6e}, approach={:.6e})",
            py_bool(diagnostics.converged),
            diagnostics.iterations,
            self.inner.contact_radius(),
            self.inner.max_pressure(),
            self.inner.approach(),
        )
    }
}

/// Solve a sphere of radius `radius` pressed onto a flat (circular Hertz, P1).
#[pyfunction]
#[pyo3(signature = (*, radius, load, e_star, grid, domain, tol = 1.0e-8, max_iter = 10_000))]
#[allow(
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    reason = "PyO3 extracts every keyword argument by value; the rich solver API needs many"
)]
fn solve_sphere_on_flat(
    py: Python<'_>,
    radius: f64,
    load: f64,
    e_star: f64,
    grid: (usize, usize),
    domain: Bound<'_, PyAny>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Solution> {
    require_positive(radius, "radius")?;
    let (material, config) = prepare(load, e_star, tol, max_iter)?;
    let grid = build_grid(grid, &domain)?;
    let solution = py.detach(move || sphere_on_flat(radius, load, material, grid, config));
    Ok(Solution::wrap(solution))
}

/// Solve two spheres of radii `radius_1`, `radius_2` in contact (circular Hertz).
#[pyfunction]
#[pyo3(signature = (*, radius_1, radius_2, load, e_star, grid, domain, tol = 1.0e-8, max_iter = 10_000))]
#[allow(
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    reason = "PyO3 extracts every keyword argument by value; the rich solver API needs many"
)]
fn solve_sphere_on_sphere(
    py: Python<'_>,
    radius_1: f64,
    radius_2: f64,
    load: f64,
    e_star: f64,
    grid: (usize, usize),
    domain: Bound<'_, PyAny>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Solution> {
    require_positive(radius_1, "radius_1")?;
    require_positive(radius_2, "radius_2")?;
    let (material, config) = prepare(load, e_star, tol, max_iter)?;
    let grid = build_grid(grid, &domain)?;
    let solution =
        py.detach(move || sphere_on_sphere(radius_1, radius_2, load, material, grid, config));
    Ok(Solution::wrap(solution))
}

/// Solve a sphere pressed onto a torus outer equator (elliptic Hertz, P2).
///
/// The torus is described by its tube radius `tube_radius` (`r`) and
/// centre-circle radius `centre_radius` (`R0`); the convex–convex contact runs
/// long circumferentially (along `x`).
#[pyfunction]
#[pyo3(signature = (*, sphere_radius, tube_radius, centre_radius, load, e_star, grid, domain, tol = 1.0e-8, max_iter = 10_000))]
#[allow(
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    reason = "PyO3 extracts every keyword argument by value; the rich solver API needs many"
)]
fn solve_sphere_on_torus(
    py: Python<'_>,
    sphere_radius: f64,
    tube_radius: f64,
    centre_radius: f64,
    load: f64,
    e_star: f64,
    grid: (usize, usize),
    domain: Bound<'_, PyAny>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Solution> {
    require_positive(sphere_radius, "sphere_radius")?;
    require_positive(tube_radius, "tube_radius")?;
    require_positive(centre_radius, "centre_radius")?;
    let (material, config) = prepare(load, e_star, tol, max_iter)?;
    let grid = build_grid(grid, &domain)?;
    let torus = Torus::new(tube_radius, centre_radius);
    let solution =
        py.detach(move || sphere_on_torus(sphere_radius, torus, load, material, grid, config));
    Ok(Solution::wrap(solution))
}

/// Solve the contact for an arbitrary gap height field `h(x, y)` (design §8.5).
///
/// `gap` is a 2-D ``float64`` array of the undeformed surface separation on a
/// uniform grid of spacings `dx`, `dy`; axis 0 is `x`, axis 1 is `y`. Only the
/// free-space boundary is implemented in v1; `boundary="periodic"` is reserved
/// for a later milestone and raises `NotImplementedError`.
#[pyfunction]
#[pyo3(signature = (*, gap, load, e_star, dx, dy, tol = 1.0e-8, max_iter = 10_000, boundary = "free"))]
#[allow(
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    reason = "PyO3 extracts every argument by value (the gap is borrowed, then copied once)"
)]
fn solve_height_field(
    py: Python<'_>,
    gap: PyReadonlyArray2<'_, f64>,
    load: f64,
    e_star: f64,
    dx: f64,
    dy: f64,
    tol: f64,
    max_iter: usize,
    boundary: &str,
) -> PyResult<Solution> {
    require_boundary(boundary)?;
    require_positive(dx, "dx")?;
    require_positive(dy, "dy")?;
    let (material, config) = prepare(load, e_star, tol, max_iter)?;

    let view = gap.as_array();
    let (nx, ny) = view.dim();
    require_dim(nx, "gap rows (nx)")?;
    require_dim(ny, "gap columns (ny)")?;
    let grid = Grid::new(nx, ny, dx, dy);

    // One owned copy at the boundary: a `NumPy`-borrowed view is not `Send` and
    // must not cross `detach`. `to_owned` also standardises the layout, so a
    // non-contiguous input is accepted (design §8.3).
    let gap_owned: Array2<f64> = view.to_owned();
    let solution = py.detach(move || solve_sampled_gap(gap_owned, material, load, grid, config));
    Ok(Solution::wrap(solution))
}

// --- shared validation / construction helpers -----------------------------

// Validates the loading and solver settings common to every entry point and
// builds the core configuration objects.
fn prepare(load: f64, e_star: f64, tol: f64, max_iter: usize) -> PyResult<(Material, Config)> {
    require_positive(load, "load")?;
    require_positive(e_star, "e_star")?;
    require_positive(tol, "tol")?;
    require_dim(max_iter, "max_iter")?;
    Ok((
        Material::from_e_star(e_star),
        Config {
            tolerance: tol,
            max_iterations: max_iter,
        },
    ))
}

// Builds a centred grid from a `(nx, ny)` point count and a physical domain
// extent, given either as a single width (square) or a `(width_x, width_y)`
// pair. The spacings are `width / count` per axis.
#[allow(
    clippy::cast_precision_loss,
    reason = "grid point counts are tiny relative to f64's 53-bit integer range"
)]
fn build_grid(grid: (usize, usize), domain: &Bound<'_, PyAny>) -> PyResult<Grid> {
    let (nx, ny) = grid;
    require_dim(nx, "grid[0] (nx)")?;
    require_dim(ny, "grid[1] (ny)")?;
    let (width_x, width_y) = extract_domain(domain)?;
    require_positive(width_x, "domain width along x")?;
    require_positive(width_y, "domain width along y")?;
    let dx = width_x / nx as f64;
    let dy = width_y / ny as f64;
    require_positive(dx, "grid spacing dx")?;
    require_positive(dy, "grid spacing dy")?;
    Ok(Grid::new(nx, ny, dx, dy))
}

// Accepts `domain` as a single float (square) or a `(width_x, width_y)` pair.
fn extract_domain(domain: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    if let Ok(width) = domain.extract::<f64>() {
        return Ok((width, width));
    }
    domain.extract::<(f64, f64)>().map_err(|_| {
        PyTypeError::new_err("domain must be a float or a (width_x, width_y) tuple of two floats")
    })
}

// Only the free-space boundary is implemented in v1 (§3.3).
fn require_boundary(boundary: &str) -> PyResult<()> {
    match boundary {
        "free" => Ok(()),
        "periodic" => Err(PyNotImplementedError::new_err(
            "periodic boundaries are a future milestone (design §3.3); use boundary=\"free\"",
        )),
        other => Err(PyValueError::new_err(format!(
            "boundary must be \"free\", got {other:?}"
        ))),
    }
}

fn require_positive(value: f64, name: &str) -> PyResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be positive and finite, got {value}"
        )))
    }
}

fn require_dim(value: usize, name: &str) -> PyResult<()> {
    if value > 0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be a positive integer, got {value}"
        )))
    }
}

// Renders a bool the Python way for `__repr__` (`True`/`False`).
const fn py_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}
