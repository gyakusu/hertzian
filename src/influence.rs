//! The pressure -> displacement linear operator `K * p`.
//!
//! The solver only ever needs to *apply* this operator (it never forms the dense
//! influence matrix), so everything kernel-specific lives behind the
//! [`InfluenceOperator`] trait. Swapping free-space for periodic or layered
//! kernels later means adding an impl, not touching the solver.

#![allow(
    clippy::cast_possible_wrap,
    reason = "grid indices are tiny relative to isize's range"
)]

use ndarray::{Array2, ArrayView2};

use crate::fft::DcFft;
use crate::grid::Grid;
use crate::kernel::influence_coefficient;

/// A linear operator mapping a pressure field to surface normal displacement.
pub trait InfluenceOperator {
    /// Returns `u = K * p` sampled on the grid.
    #[must_use]
    fn apply(&self, pressure: ArrayView2<'_, f64>) -> Array2<f64>;

    /// The grid this operator is defined on.
    #[must_use]
    fn grid(&self) -> &Grid;
}

/// Free-space Boussinesq operator evaluated by zero-padded DC-FFT.
pub struct FreeSpaceBoussinesq {
    grid: Grid,
    dcfft: DcFft,
}

impl FreeSpaceBoussinesq {
    /// Builds the operator for `grid` and equivalent modulus `e_star`.
    #[must_use]
    pub fn new(grid: Grid, e_star: f64) -> Self {
        let dcfft = DcFft::new(&grid, e_star);
        Self { grid, dcfft }
    }
}

impl InfluenceOperator for FreeSpaceBoussinesq {
    fn apply(&self, pressure: ArrayView2<'_, f64>) -> Array2<f64> {
        self.dcfft.apply(pressure)
    }

    fn grid(&self) -> &Grid {
        &self.grid
    }
}

/// Reference `O(N^2)` direct-summation operator.
///
/// Evaluates `u_i = sum_j K_{i-j} p_j` explicitly. Used to validate the FFT
/// convolver on small grids; not intended for production-size problems.
pub struct DirectSum {
    grid: Grid,
    e_star: f64,
}

impl DirectSum {
    /// Builds the operator for `grid` and equivalent modulus `e_star`.
    #[must_use]
    pub const fn new(grid: Grid, e_star: f64) -> Self {
        Self { grid, e_star }
    }
}

impl InfluenceOperator for DirectSum {
    fn apply(&self, pressure: ArrayView2<'_, f64>) -> Array2<f64> {
        let (nx, ny) = self.grid.dims();
        Array2::from_shape_fn((nx, ny), |(i, j)| -> f64 {
            pressure
                .indexed_iter()
                .map(|((ip, jp), &p)| {
                    let di = i as isize - ip as isize;
                    let dj = j as isize - jp as isize;
                    influence_coefficient(&self.grid, di, dj, self.e_star) * p
                })
                .sum()
        })
    }

    fn grid(&self) -> &Grid {
        &self.grid
    }
}

#[cfg(test)]
mod tests {
    use super::{DirectSum, FreeSpaceBoussinesq, InfluenceOperator};
    use crate::grid::Grid;

    #[test]
    fn fft_convolution_matches_direct_sum() {
        // A deliberately non-square grid exercises the axis bookkeeping.
        let grid = Grid::new(12, 16, 0.5e-3, 0.4e-3);
        let e_star = 70.0e9;
        let fft_op = FreeSpaceBoussinesq::new(grid.clone(), e_star);
        let direct = DirectSum::new(grid.clone(), e_star);

        // A smooth, compactly supported pressure bump well inside the domain.
        let sigma = 1.2e-3;
        let pressure = grid.sample(|x, y| (-(x * x + y * y) / (2.0 * sigma * sigma)).exp());

        let u_fft = fft_op.apply(pressure.view());
        let u_direct = direct.apply(pressure.view());

        let scale = u_direct.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        let max_err = (&u_fft - &u_direct)
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(
            max_err <= 1e-10 * scale,
            "FFT convolution must match direct sum to machine precision \
             (max_err = {max_err:e}, scale = {scale:e})",
        );
    }
}
