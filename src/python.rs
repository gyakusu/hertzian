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

use core::f64::consts::FRAC_PI_2;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::geometry::{GothicArchGroove, Torus};
use crate::grid::Grid;
use crate::material::Material;
use crate::reduced::{contact_half_angle as core_contact_half_angle, GothicArchLaw as CoreLaw};
use crate::scenarios::{
    solve_sampled_gap, sphere_in_gothic_arch, sphere_on_flat, sphere_on_sphere, sphere_on_torus,
};
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
    module.add_class::<GothicArchLaw>()?;
    module.add_function(wrap_pyfunction!(solve_sphere_on_flat, module)?)?;
    module.add_function(wrap_pyfunction!(solve_sphere_on_sphere, module)?)?;
    module.add_function(wrap_pyfunction!(solve_sphere_on_torus, module)?)?;
    module.add_function(wrap_pyfunction!(solve_sphere_in_gothic_arch, module)?)?;
    module.add_function(wrap_pyfunction!(solve_height_field, module)?)?;
    module.add_function(wrap_pyfunction!(contact_half_angle, module)?)?;
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

/// A reduced, closed-form two-flank contact law for a Gothic-arch groove.
///
/// The lightweight stand-in for the field solver in a multibody inner loop: a
/// `force(delta_t, delta_n) -> (F_t, F_n)` map built from one flank's Hertz
/// stiffness and the contact half-angle (see `hertzian.reduced` design notes). It
/// reduces to a single Hertz contact when one flank lifts off, and varies `C¹`
/// across that two-to-one transition because the Hertzian `3/2` exponent makes a
/// flank engage with zero load *and* zero stiffness. [`GothicArchLaw.jacobian`]
/// returns the analytic tangent stiffness for implicit integrators.
#[pyclass(name = "GothicArchLaw", module = "hertzian._core", frozen)]
struct GothicArchLaw {
    inner: CoreLaw,
}

#[pymethods]
impl GothicArchLaw {
    /// Build from a per-flank stiffness `K` (N·m^−3/2) and half-angle `α` (rad).
    #[new]
    #[pyo3(signature = (*, stiffness, contact_angle))]
    fn new(stiffness: f64, contact_angle: f64) -> PyResult<Self> {
        require_positive(stiffness, "stiffness")?;
        require_contact_angle(contact_angle)?;
        Ok(Self {
            inner: CoreLaw::new(stiffness, contact_angle),
        })
    }

    /// Calibrate the stiffness from one flank's elliptic-Hertz contact.
    ///
    /// `radius_x`, `radius_y` are the flank's principal relative radii and
    /// `e_star` the equivalent modulus; `contact_angle` is the geometric flank
    /// half-angle `α` (see [`contact_half_angle`]).
    #[staticmethod]
    #[pyo3(signature = (*, radius_x, radius_y, e_star, contact_angle))]
    fn from_elliptic_flank(
        radius_x: f64,
        radius_y: f64,
        e_star: f64,
        contact_angle: f64,
    ) -> PyResult<Self> {
        require_positive(radius_x, "radius_x")?;
        require_positive(radius_y, "radius_y")?;
        require_positive(e_star, "e_star")?;
        require_contact_angle(contact_angle)?;
        Ok(Self {
            inner: CoreLaw::from_elliptic_flank(radius_x, radius_y, e_star, contact_angle),
        })
    }

    /// Enable the neighbour-lift coupling from the modulus `e_star` and offset `y0`.
    ///
    /// Returns a copy with the cross-compliance `κ = 1 / (2 π E* y0)` set — the
    /// Boussinesq far-field lift `u ≈ Q/(π E* · 2 y0)` one flank raises under the
    /// other. Composes with the calibrating constructor:
    /// `GothicArchLaw.from_elliptic_flank(...).with_flank_coupling(e_star=.., offset=..)`.
    #[pyo3(signature = (*, e_star, offset))]
    fn with_flank_coupling(&self, e_star: f64, offset: f64) -> PyResult<Self> {
        require_positive(e_star, "e_star")?;
        require_positive(offset, "offset")?;
        Ok(Self {
            inner: self.inner.with_flank_coupling(e_star, offset),
        })
    }

    /// The per-flank Hertz stiffness `K` (N·m^−3/2).
    #[getter]
    const fn stiffness(&self) -> f64 {
        self.inner.stiffness()
    }

    /// The neighbour-lift cross-compliance `κ` (m·N⁻¹); `0` when uncoupled.
    #[getter]
    const fn coupling(&self) -> f64 {
        self.inner.coupling()
    }

    /// The contact half-angle `α` (radians).
    #[getter]
    const fn contact_angle(&self) -> f64 {
        self.inner.contact_angle()
    }

    /// One flank's Hertz load `Q = K⌊s⌋₊^{3/2}` for an approach `s` (no adhesion).
    fn flank_load(&self, approach: f64) -> f64 {
        self.inner.flank_load(approach)
    }

    /// The two flank approaches `(s_+, s_-)` for a displacement `(delta_t, delta_n)`.
    fn flank_approaches(&self, delta_t: f64, delta_n: f64) -> (f64, f64) {
        self.inner.flank_approaches(delta_t, delta_n)
    }

    /// The two flank loads `(Q_+, Q_-)` for prescribed flank approaches `(s_+, s_-)`.
    ///
    /// The self-consistent solution of `Q_± = K⌊s_± − κ Q_∓⌋₊^{3/2}` — two
    /// independent Hertz loads when uncoupled, the coupled pair when
    /// [`with_flank_coupling`](Self::with_flank_coupling) is on.
    fn coupled_loads(&self, s_plus: f64, s_minus: f64) -> (f64, f64) {
        self.inner.coupled_loads(s_plus, s_minus)
    }

    /// The two flank loads `(Q_+, Q_-)` for a displacement `(delta_t, delta_n)`.
    fn flank_loads(&self, delta_t: f64, delta_n: f64) -> (f64, f64) {
        self.inner.flank_loads(delta_t, delta_n)
    }

    /// The net contact force `(F_t, F_n)` for a displacement `(delta_t, delta_n)`.
    fn force(&self, delta_t: f64, delta_n: f64) -> (f64, f64) {
        self.inner.force(delta_t, delta_n)
    }

    /// The analytic tangent stiffness `dF/dδ` as `((∂F_t/∂δ_t, ∂F_t/∂δ_n), …)`.
    fn jacobian(&self, delta_t: f64, delta_n: f64) -> ((f64, f64), (f64, f64)) {
        let j = self.inner.jacobian(delta_t, delta_n);
        ((j[0][0], j[0][1]), (j[1][0], j[1][1]))
    }

    /// The transverse displacement `δ_t* = δ_n cot α` at which a flank lifts off.
    fn lift_off_transverse(&self, delta_n: f64) -> f64 {
        self.inner.lift_off_transverse(delta_n)
    }

    fn __repr__(&self) -> String {
        format!(
            "GothicArchLaw(stiffness={:.6e}, contact_angle={:.6e}, coupling={:.6e})",
            self.inner.stiffness(),
            self.inner.contact_angle(),
            self.inner.coupling(),
        )
    }
}

/// Geometric contact half-angle `α = arcsin(offset / ball_radius)` (radians).
///
/// The angle the flank contact normal makes with the groove axis when the flank
/// contact sits a meridional distance `offset` from the axis on a ball of radius
/// `ball_radius`. Use it to orient [`GothicArchLaw`]'s flank normals.
#[pyfunction]
#[pyo3(signature = (*, offset, ball_radius))]
fn contact_half_angle(offset: f64, ball_radius: f64) -> PyResult<f64> {
    require_positive(ball_radius, "ball_radius")?;
    require_non_negative(offset, "offset")?;
    if offset >= ball_radius {
        return Err(PyValueError::new_err(format!(
            "offset ({offset}) must be smaller than ball_radius ({ball_radius})"
        )));
    }
    Ok(core_contact_half_angle(offset, ball_radius))
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

/// Solve a sphere pressed into a Gothic-arch (ogival) groove.
///
/// The concave counterpart of [`solve_sphere_on_torus`]: the ball sits inside a
/// conformal groove built from two arcs of tube radius `tube_radius` (`r`) whose
/// centre circles are displaced by `±centre_offset` from a reference circle of
/// radius `centre_radius` (`R0`), on which the ball centre sits. With a large
/// offset the ball rides on two well-separated flanks and the contact splits into
/// a pair of elliptic patches; a smaller offset brings those patches into a
/// partial overlap (a single connected contact); `centre_offset = 0` recovers a
/// single conformal elliptic contact.
#[pyfunction]
#[pyo3(signature = (*, sphere_radius, tube_radius, centre_radius, centre_offset, load, e_star, grid, domain, tol = 1.0e-8, max_iter = 10_000))]
#[allow(
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    reason = "PyO3 extracts every keyword argument by value; the rich solver API needs many"
)]
fn solve_sphere_in_gothic_arch(
    py: Python<'_>,
    sphere_radius: f64,
    tube_radius: f64,
    centre_radius: f64,
    centre_offset: f64,
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
    require_non_negative(centre_offset, "centre_offset")?;
    if sphere_radius >= tube_radius {
        return Err(PyValueError::new_err(format!(
            "sphere_radius ({sphere_radius}) must be smaller than tube_radius ({tube_radius}) for a conformal groove contact"
        )));
    }
    let (material, config) = prepare(load, e_star, tol, max_iter)?;
    let grid = build_grid(grid, &domain)?;
    let groove = GothicArchGroove::new(tube_radius, centre_radius, centre_offset);
    let solution = py
        .detach(move || sphere_in_gothic_arch(sphere_radius, groove, load, material, grid, config));
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

fn require_non_negative(value: f64, name: &str) -> PyResult<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be non-negative and finite, got {value}"
        )))
    }
}

// The flank contact half-angle must be a real groove angle in (0, pi/2).
fn require_contact_angle(value: f64) -> PyResult<()> {
    if value.is_finite() && value > 0.0 && value < FRAC_PI_2 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "contact_angle must lie in (0, pi/2), got {value}"
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
