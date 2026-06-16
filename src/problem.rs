//! Problem definition: interface gap and loading.

use ndarray::{Array2, ArrayView2};

use crate::grid::Grid;

/// How the contact is driven.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Control {
    /// Prescribed total normal load `P` (newtons); the approach is solved for.
    Load(f64),
}

/// A single contact interface to solve.
///
/// Couples the discretised gap to the loading. The equivalent modulus lives in
/// the [`InfluenceOperator`](crate::InfluenceOperator), which must be built on
/// the same grid.
#[derive(Debug, Clone)]
pub struct Problem {
    grid: Grid,
    gap: Array2<f64>,
    control: Control,
}

impl Problem {
    /// Builds a problem from a grid, a sampled gap, and a loading mode.
    ///
    /// # Panics
    /// Panics if `gap`'s shape does not match `grid`.
    #[must_use]
    pub fn new(grid: Grid, gap: Array2<f64>, control: Control) -> Self {
        assert_eq!(gap.dim(), grid.dims(), "gap shape must match the grid");
        Self { grid, gap, control }
    }

    /// The interface grid.
    #[must_use]
    pub const fn grid(&self) -> &Grid {
        &self.grid
    }

    /// A view of the sampled undeformed gap.
    #[must_use]
    pub fn gap(&self) -> ArrayView2<'_, f64> {
        self.gap.view()
    }

    /// The loading mode.
    #[must_use]
    pub const fn control(&self) -> Control {
        self.control
    }
}
