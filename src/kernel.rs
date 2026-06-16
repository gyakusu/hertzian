//! Boussinesq influence coefficients for a uniformly loaded rectangular cell.
//!
//! Discretising the interface into uniform cells, each carrying a
//! piecewise-constant pressure, the surface normal deflection becomes a discrete
//! convolution `u_i = sum_j K_{i-j} p_j`. The coefficient `K` depends only on
//! the integer offset `i - j` (translation invariance on a uniform grid), which
//! is exactly what enables the FFT acceleration.
//!
//! The per-cell coefficient is the classic closed form for the deflection due to
//! uniform pressure on a rectangle (Love, 1929; Johnson, *Contact Mechanics*,
//! 1985), divided by `pi E*`.

use core::f64::consts::PI;

use crate::grid::Grid;

/// Influence coefficient `K_{di,dj}` for an integer cell offset.
///
/// Returns the surface normal deflection at a cell offset `(di, dj)` from a
/// source cell carrying unit uniform pressure, including the `1/(pi E*)` factor,
/// so that `u = sum K p` is in metres for pressure in pascals.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "grid offsets are tiny relative to f64's 53-bit integer range"
)]
pub fn influence_coefficient(grid: &Grid, di: isize, dj: isize, e_star: f64) -> f64 {
    let a = grid.dx() * 0.5;
    let b = grid.dy() * 0.5;
    let x = di as f64 * grid.dx();
    let y = dj as f64 * grid.dy();
    rectangle_coefficient(x, y, a, b) / (PI * e_star)
}

/// Geometric part `C(x, y)` of the deflection from a uniformly loaded rectangle.
///
/// `(x, y)` is the observation point relative to the cell centre and `(a, b)`
/// are the cell half-widths. The full influence coefficient is
/// `C(x, y) / (pi E*)`. On a uniform grid the four log arguments are always
/// strictly positive (offsets are integer multiples of the spacing, never a
/// half-cell), so the expression is finite, including the self term.
fn rectangle_coefficient(x: f64, y: f64, a: f64, b: f64) -> f64 {
    let xp = x + a;
    let xm = x - a;
    let yp = y + b;
    let ym = y - b;
    let h_pp = xp.hypot(yp);
    let h_pm = xp.hypot(ym);
    let h_mp = xm.hypot(yp);
    let h_mm = xm.hypot(ym);
    xp * ((yp + h_pp) / (ym + h_pm)).ln()
        + yp * ((xp + h_pp) / (xm + h_mp)).ln()
        + xm * ((ym + h_mm) / (yp + h_mp)).ln()
        + ym * ((xm + h_mm) / (xp + h_pm)).ln()
}

#[cfg(test)]
mod tests {
    use super::influence_coefficient;
    use crate::grid::Grid;

    #[test]
    fn coefficient_is_symmetric_positive_and_decaying() {
        let grid = Grid::square(8, 1.0e-3);
        let e_star = 1.0e9;
        let c00 = influence_coefficient(&grid, 0, 0, e_star);
        assert!(c00 > 0.0, "self term must be positive");

        for (di, dj) in [(1isize, 0isize), (2, 3), (0, 2), (3, 1)] {
            let base = influence_coefficient(&grid, di, dj, e_star);
            assert!(base > 0.0);
            for (sx, sy) in [(-1isize, 1isize), (1, -1), (-1, -1)] {
                let flipped = influence_coefficient(&grid, di * sx, dj * sy, e_star);
                assert!(
                    (flipped - base).abs() <= 1e-12 * base,
                    "kernel must be symmetric under offset sign flips",
                );
            }
        }

        assert!(
            influence_coefficient(&grid, 6, 0, e_star) < c00,
            "deflection must decay with distance",
        );
    }
}
