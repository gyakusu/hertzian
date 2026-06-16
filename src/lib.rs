//! Hertzian — FFT-accelerated elastic half-space normal contact solver.
//!
//! This crate currently contains only project scaffolding. The numerical core
//! (influence functions, zero-padded DC-FFT convolution, and the constrained
//! conjugate-gradient / BCCG solver) is implemented in a later milestone; see
//! `README.md` for the design and roadmap.

use pyo3::prelude::*;

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
