//! Result of a solved contact problem.

// The contact-cell count is cast to f64 to form means; the count is tiny
// relative to f64's 53-bit integer range, so the cast is exact.
#![allow(
    clippy::cast_precision_loss,
    reason = "contact-cell counts are exact in f64"
)]

use core::f64::consts::PI;

use ndarray::{Array2, ArrayView2};

use crate::grid::Grid;

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
///
/// Holds the converged pressure field together with the grid it lives on, so
/// that geometric quantities (contact area, semi-axes, ellipticity) are derived
/// consistently rather than stored redundantly.
#[derive(Debug, Clone)]
pub struct Solution {
    grid: Grid,
    pressure: Array2<f64>,
    approach: f64,
    diagnostics: Diagnostics,
}

impl Solution {
    /// Assembles a solution from solver output on `grid`.
    ///
    /// # Panics
    /// Panics if `pressure`'s shape does not match `grid`.
    #[must_use]
    pub fn new(grid: Grid, pressure: Array2<f64>, approach: f64, diagnostics: Diagnostics) -> Self {
        assert_eq!(
            pressure.dim(),
            grid.dims(),
            "pressure shape must match the grid",
        );
        Self {
            grid,
            pressure,
            approach,
            diagnostics,
        }
    }

    /// The interface grid the solution is defined on.
    #[must_use]
    pub const fn grid(&self) -> &Grid {
        &self.grid
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

    /// Integrated total normal load, `sum(p) * cell_area`.
    #[must_use]
    pub fn total_load(&self) -> f64 {
        self.pressure.sum() * self.grid.cell_area()
    }

    /// Total contact area (cells in contact times the cell area).
    #[must_use]
    pub fn contact_area(&self) -> f64 {
        let cells = self.pressure.iter().filter(|&&p| p > 0.0).count();
        cells as f64 * self.grid.cell_area()
    }

    /// Equivalent circular contact radius, `sqrt(area / pi)`.
    ///
    /// For an elliptic contact this is the geometric mean `sqrt(a b)` of the
    /// semi-axes; for a circular contact it is the contact radius.
    #[must_use]
    pub fn contact_radius(&self) -> f64 {
        (self.contact_area() / PI).sqrt()
    }

    /// Peak contact pressure.
    #[must_use]
    pub fn max_pressure(&self) -> f64 {
        self.pressure.iter().fold(0.0_f64, |m, &v| m.max(v))
    }

    /// Measured contact semi-axes `(a_x, a_y)` along the grid axes.
    ///
    /// Estimated from the second moments of the contact region (cells with
    /// `p > 0`) about its centroid: a uniformly filled ellipse has
    /// `<x^2> = a_x^2 / 4`, so `a_x = 2 sqrt(<x^2>)` and likewise for `a_y`. The
    /// estimate is grid-convergent and needs no axis alignment beyond the grid.
    /// Returns `(0, 0)` when there is no contact.
    #[must_use]
    pub fn contact_half_widths(&self) -> (f64, f64) {
        let in_contact = || self.pressure.indexed_iter().filter(|&(_, &p)| p > 0.0);

        let count = in_contact().count();
        if count == 0 {
            return (0.0, 0.0);
        }
        let inv = 1.0 / count as f64;

        let (sum_x, sum_y) = in_contact().fold((0.0_f64, 0.0_f64), |(sx, sy), ((i, j), _)| {
            (sx + self.grid.x(i), sy + self.grid.y(j))
        });
        let (centre_x, centre_y) = (sum_x * inv, sum_y * inv);

        let (mxx, myy) = in_contact().fold((0.0_f64, 0.0_f64), |(ax, ay), ((i, j), _)| {
            let dx = self.grid.x(i) - centre_x;
            let dy = self.grid.y(j) - centre_y;
            (ax + dx * dx, ay + dy * dy)
        });

        (2.0 * (mxx * inv).sqrt(), 2.0 * (myy * inv).sqrt())
    }

    /// Measured ellipticity `max(a_x, a_y) / min(a_x, a_y) >= 1`.
    ///
    /// Returns `1` for a degenerate (empty) contact.
    #[must_use]
    pub fn ellipticity(&self) -> f64 {
        let (a_x, a_y) = self.contact_half_widths();
        if a_x <= 0.0 || a_y <= 0.0 {
            1.0
        } else {
            a_x.max(a_y) / a_x.min(a_y)
        }
    }

    /// Solver diagnostics.
    #[must_use]
    pub const fn diagnostics(&self) -> Diagnostics {
        self.diagnostics
    }
}

#[cfg(test)]
mod tests {
    use super::{Diagnostics, Solution};
    use crate::grid::Grid;

    fn diagnostics() -> Diagnostics {
        Diagnostics {
            iterations: 1,
            residual: 0.0,
            converged: true,
        }
    }

    #[test]
    fn half_widths_recover_a_filled_ellipse() {
        // Paint a uniform elliptic contact and check the second-moment estimate
        // recovers its semi-axes (a_x along x, a_y along y).
        let grid = Grid::square(201, 1.0e-4);
        let a_x = 6.0e-3;
        let a_y = 3.0e-3;
        let pressure = grid.sample(|x, y| {
            let rx = x / a_x;
            let ry = y / a_y;
            if rx * rx + ry * ry <= 1.0 {
                1.0
            } else {
                0.0
            }
        });
        let solution = Solution::new(grid, pressure, 0.0, diagnostics());

        let (measured_x, measured_y) = solution.contact_half_widths();
        assert!(
            (measured_x - a_x).abs() <= 0.02 * a_x,
            "a_x: measured={measured_x:e} expected={a_x:e}",
        );
        assert!(
            (measured_y - a_y).abs() <= 0.02 * a_y,
            "a_y: measured={measured_y:e} expected={a_y:e}",
        );
        assert!((solution.ellipticity() - a_x / a_y).abs() <= 0.03 * (a_x / a_y));
    }

    #[test]
    fn empty_contact_is_degenerate() {
        let grid = Grid::square(8, 1.0e-3);
        let pressure = grid.sample(|_, _| 0.0);
        let solution = Solution::new(grid, pressure, 0.0, diagnostics());

        assert_eq!(solution.contact_half_widths(), (0.0, 0.0));
        assert!((solution.ellipticity() - 1.0).abs() <= 1e-15);
        assert!(solution.total_load().abs() <= 1e-15);
        assert!(solution.contact_area().abs() <= 1e-15);
    }
}
