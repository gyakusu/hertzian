//! Result of a solved contact problem.

use core::f64::consts::PI;

use ndarray::{Array2, ArrayView2};

/// Convergence diagnostics from the iterative solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diagnostics {
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final relative pressure-update residual.
    pub residual: f64,
    /// Whether the tolerance was met before the iteration cap.
    pub converged: bool,
}

/// The solved contact state.
#[derive(Debug, Clone)]
pub struct Solution {
    pressure: Array2<f64>,
    approach: f64,
    total_load: f64,
    contact_area: f64,
    diagnostics: Diagnostics,
}

impl Solution {
    /// Assembles a solution from solver output.
    #[must_use]
    pub const fn new(
        pressure: Array2<f64>,
        approach: f64,
        total_load: f64,
        contact_area: f64,
        diagnostics: Diagnostics,
    ) -> Self {
        Self {
            pressure,
            approach,
            total_load,
            contact_area,
            diagnostics,
        }
    }

    /// A view of the converged pressure field.
    #[must_use]
    pub fn pressure(&self) -> ArrayView2<'_, f64> {
        self.pressure.view()
    }

    /// Rigid-body approach `delta`.
    #[must_use]
    pub const fn approach(&self) -> f64 {
        self.approach
    }

    /// Integrated total normal load.
    #[must_use]
    pub const fn total_load(&self) -> f64 {
        self.total_load
    }

    /// Total contact area (cells in contact times the cell area).
    #[must_use]
    pub const fn contact_area(&self) -> f64 {
        self.contact_area
    }

    /// Equivalent circular contact radius, `sqrt(area / pi)`.
    #[must_use]
    pub fn contact_radius(&self) -> f64 {
        (self.contact_area / PI).sqrt()
    }

    /// Peak contact pressure.
    #[must_use]
    pub fn max_pressure(&self) -> f64 {
        self.pressure.iter().fold(0.0_f64, |m, &v| m.max(v))
    }

    /// Solver diagnostics.
    #[must_use]
    pub const fn diagnostics(&self) -> Diagnostics {
        self.diagnostics
    }
}
