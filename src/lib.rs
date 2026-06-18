//! Hertzian — FFT-accelerated elastic half-space normal contact solver.
//!
//! Two elastic bodies in normal, frictionless contact are each approximated as
//! an elastic half-space and reduced to a single equivalent half-space problem
//! on a shared uniform interface grid. The pressure -> displacement relation is
//! a convolution `u = K * p`, evaluated in `O(N log N)` by a zero-padded
//! free-space DC-FFT, and the non-penetration / non-adhesion constraints are
//! solved with a Polonsky–Keer bound-constrained conjugate gradient scheme.
//!
//! This crate implements the circular Hertz milestone (sphere on flat / sphere
//! on sphere) and the elliptic one (a sphere on a torus outer equator), each
//! validated against its analytic solution.

// Two-dimensional contact mechanics is written in the field's standard notation:
// single-letter symbols (x, y, a, b, p, u, g) and `_x`/`_y` axis suffixes are
// ubiquitous and intentional, and numerical formulae are kept in their textbook
// form rather than refactored into `mul_add` chains (a micro-optimisation
// deferred until profiling, per the measure-then-optimise convention). The
// corresponding style lints are relaxed crate-wide.
#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::suboptimal_flops
)]

use pyo3::prelude::*;

pub mod fft;
pub mod geometry;
pub mod grid;
pub mod influence;
pub mod kernel;
pub mod material;
pub mod pressure;
pub mod problem;
mod python;
pub mod reduced;
pub mod reference;
pub mod scenarios;
pub mod solution;
pub mod solver;
pub mod validation;

pub use geometry::{
    Cone, Gap, GothicArchGroove, GothicArchProfile, HeightField, Paraboloid, Sum, Torus, Waviness,
};
pub use grid::Grid;
pub use influence::{DirectSum, FreeSpaceBoussinesq, InfluenceOperator};
pub use material::Material;
pub use pressure::FlankPressure;
pub use problem::{Control, Problem};
pub use reduced::{contact_half_angle, GothicArchLaw};
pub use reference::DenseReference;
pub use solution::{Diagnostics, Solution};
pub use solver::{Bccg, Config, Solver};
pub use validation::{HertzCircular, HertzElliptic, SneddonCone};

/// Crate version string, surfaced to Python as `hertzian._core.__version__`.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The native extension module backing the `hertzian` Python package.
///
/// Exposes the package version, the `Solution` / `Diagnostics` result types,
/// and the solver entry points (`solve_sphere_on_flat`,
/// `solve_sphere_on_sphere`, `solve_sphere_on_torus`, `solve_height_field`).
/// The Python-facing layer lives in the `python` module; this just assembles it.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", VERSION)?;
    python::register(module)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_is_non_empty() {
        assert!(!VERSION.is_empty(), "CARGO_PKG_VERSION must be set");
    }
}
