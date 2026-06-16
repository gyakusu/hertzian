//! Hertzian — FFT-accelerated elastic half-space normal contact solver.
//!
//! Two elastic bodies in normal, frictionless contact are each approximated as
//! an elastic half-space and reduced to a single equivalent half-space problem
//! on a shared uniform interface grid. The pressure -> displacement relation is
//! a convolution `u = K * p`, evaluated in `O(N log N)` by a zero-padded
//! free-space DC-FFT, and the non-penetration / non-adhesion constraints are
//! solved with a Polonsky–Keer bound-constrained conjugate gradient scheme.
//!
//! This crate currently implements the first milestone: circular Hertz contact
//! (sphere on flat / sphere on sphere), validated against the analytic solution.

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
pub mod problem;
pub mod scenarios;
pub mod solution;
pub mod solver;
pub mod validation;

pub use geometry::{Gap, Paraboloid};
pub use grid::Grid;
pub use influence::{DirectSum, FreeSpaceBoussinesq, InfluenceOperator};
pub use material::Material;
pub use problem::{Control, Problem};
pub use solution::{Diagnostics, Solution};
pub use solver::{Bccg, Config, Solver};
pub use validation::HertzCircular;

/// Crate version string, surfaced to Python as `hertzian._core.__version__`.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The native extension module backing the `hertzian` Python package.
///
/// For now it only exposes the package version so the build/packaging pipeline
/// can be verified end to end. Solver entry points are added later.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", VERSION)?;
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
