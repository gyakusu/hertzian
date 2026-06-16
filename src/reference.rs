//! Independent dense reference solver for cross-validation.
//!
//! A deliberately *different* solver from the production FFT + Polonsky–Keer
//! BCCG path. It forms the dense influence matrix explicitly and solves the
//! bound-constrained normal-contact complementarity problem by **projected
//! Gauss–Seidel** (pointwise SOR relaxation with a non-negativity projection),
//! wrapping a **bisection on the rigid approach** to meet a prescribed load.
//!
//! Rough and arbitrary-shape contacts have no closed-form reference, so the
//! agreement between this solver and the BCCG one — built on the *same*
//! influence coefficients but solved by an unrelated iteration — cross-checks
//! the iterative solver itself, the role an external code (Tamaas) or an FEM
//! discretisation plays across implementations.
//!
//! It is `O(N^2)` in both memory and per-sweep work, so it is intended only for
//! the small grids used in validation, never for production-size problems.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    reason = "grid sizes are tiny relative to the f64/isize integer ranges"
)]

use ndarray::{Array1, Array2, Zip};

use crate::grid::Grid;
use crate::kernel::influence_coefficient;
use crate::problem::{Control, Problem};
use crate::solution::{Diagnostics, Solution};
use crate::solver::Config;

/// Over-relaxation factor for the projected SOR sweeps. Any value in `(0, 2)`
/// converges for the symmetric positive-definite influence matrix; `1.5`
/// accelerates the plain Gauss–Seidel iteration without risking divergence.
const RELAXATION: f64 = 1.5;

/// Bisection steps on the approach once the target load is bracketed. ~50
/// halvings drive the bracket far below any tolerance of interest.
const BISECTION_STEPS: usize = 60;

/// Maximum doublings used to bracket the load from above before bisection.
const MAX_BRACKET_DOUBLINGS: usize = 80;

/// Dense influence-matrix reference solver (projected Gauss–Seidel + bisection).
///
/// Built once per grid and modulus; [`DenseReference::solve`] then runs the
/// load-controlled iteration for a given problem on that grid.
pub struct DenseReference {
    grid: Grid,
    /// Dense `N x N` influence matrix, `N = nx * ny`, in row-major flat order
    /// `k = i * ny + j`. Symmetric positive definite.
    matrix: Array2<f64>,
    /// The (constant) self-influence on the diagonal, used as the SOR pivot.
    self_term: f64,
}

impl DenseReference {
    /// Builds the dense reference operator for `grid` and modulus `e_star`.
    ///
    /// Forms the full `N x N` influence matrix (`N = nx * ny`), so it is only
    /// practical for the small grids used in cross-validation.
    #[must_use]
    pub fn new(grid: Grid, e_star: f64) -> Self {
        let (nx, ny) = grid.dims();
        let n = nx * ny;
        let decode = |k: usize| (k / ny, k % ny);
        let matrix = Array2::from_shape_fn((n, n), |(k, l)| {
            let (ik, jk) = decode(k);
            let (il, jl) = decode(l);
            influence_coefficient(
                &grid,
                ik as isize - il as isize,
                jk as isize - jl as isize,
                e_star,
            )
        });
        let self_term = influence_coefficient(&grid, 0, 0, e_star);
        Self {
            grid,
            matrix,
            self_term,
        }
    }

    /// Solves the load-controlled contact for `problem` on this grid.
    ///
    /// Brackets and bisects the rigid approach so the integrated load matches the
    /// prescribed total, solving the inner displacement-controlled LCP by
    /// projected Gauss–Seidel at each trial approach.
    ///
    /// # Panics
    /// Panics if `problem`'s grid does not match the operator's grid.
    #[must_use]
    pub fn solve(&self, problem: &Problem, config: Config) -> Solution {
        assert_eq!(
            problem.grid().dims(),
            self.grid.dims(),
            "reference operator grid must match the problem grid",
        );
        let Control::Load(target_load) = problem.control();
        let cell_area = self.grid.cell_area();

        // Flatten the gap in the matrix's row-major order.
        let gap: Array1<f64> = problem.gap().iter().copied().collect();
        let gap_min = gap.iter().copied().fold(f64::INFINITY, f64::min);
        let gap_max = gap.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // A single warm-started pressure vector reused across every trial: the
        // approach changes little between bisection steps, so each subsequent
        // relaxation converges in a handful of sweeps.
        let mut pressure = Array1::<f64>::zeros(gap.len());
        let mut sweeps = 0;
        let load_at = |approach: f64, pressure: &mut Array1<f64>, sweeps: &mut usize| {
            *sweeps += self.relax(&gap, approach, pressure, config);
            pressure.sum() * cell_area
        };

        // The approach below the lowest gap point carries no load; bracket the
        // target from above by doubling a span seeded at the gap's relief.
        let approach_lo = gap_min;
        let mut span = (gap_max - gap_min)
            .max(self.self_term * target_load)
            .max(1e-12);
        let mut approach_hi = gap_min + span;
        let mut bracketed = false;
        for _ in 0..MAX_BRACKET_DOUBLINGS {
            if load_at(approach_hi, &mut pressure, &mut sweeps) >= target_load {
                bracketed = true;
                break;
            }
            span *= 2.0;
            approach_hi = gap_min + span;
        }

        let mut lo = approach_lo;
        let mut hi = approach_hi;
        for _ in 0..BISECTION_STEPS {
            let mid = 0.5 * (lo + hi);
            if load_at(mid, &mut pressure, &mut sweeps) < target_load {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let approach = 0.5 * (lo + hi);
        let final_load = load_at(approach, &mut pressure, &mut sweeps);

        let residual = (final_load - target_load).abs() / target_load;
        let converged = bracketed && residual <= config.tolerance.max(1e-6);
        let field = Array2::from_shape_vec(self.grid.dims(), pressure.to_vec())
            .expect("pressure length equals nx * ny by construction");

        Solution::new(
            self.grid.clone(),
            field,
            approach,
            Diagnostics {
                iterations: sweeps,
                residual,
                converged,
            },
        )
    }

    // Projected Gauss–Seidel (SOR) for the displacement-controlled LCP at a
    // fixed `approach`: find p >= 0 with residual `(K p)_k + gap_k - approach`
    // zero where p_k > 0 and non-negative where p_k = 0. Warm-starts from the
    // incoming `pressure`. Returns the number of sweeps performed.
    //
    // The residual `r = K p + gap - approach` is maintained incrementally: each
    // accepted pressure change updates `r` by a rank-one term (a column of the
    // symmetric `K`), so a sweep that touches few points is far cheaper than the
    // dense `K p` it would otherwise recompute.
    fn relax(
        &self,
        gap: &Array1<f64>,
        approach: f64,
        pressure: &mut Array1<f64>,
        config: Config,
    ) -> usize {
        let mut residual = self.matrix.dot(pressure);
        residual += gap;
        residual -= approach;

        for sweep in 1..=config.max_iterations {
            let mut max_change = 0.0_f64;
            for k in 0..pressure.len() {
                let updated = (pressure[k] - RELAXATION * residual[k] / self.self_term).max(0.0);
                let change = updated - pressure[k];
                if change != 0.0 {
                    pressure[k] = updated;
                    Zip::from(&mut residual)
                        .and(self.matrix.row(k))
                        .for_each(|r, &k_lk| *r += k_lk * change);
                    max_change = max_change.max(change.abs());
                }
            }
            let scale = pressure.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            if scale == 0.0 || max_change <= config.tolerance * scale {
                return sweep;
            }
        }
        config.max_iterations
    }
}

#[cfg(test)]
mod tests {
    use super::DenseReference;
    use crate::geometry::{Gap, Paraboloid};
    use crate::grid::Grid;
    use crate::problem::{Control, Problem};
    use crate::solver::Config;
    use crate::validation::HertzCircular;

    #[allow(
        clippy::cast_precision_loss,
        reason = "grid sizes are tiny relative to f64's integer range"
    )]
    fn centred_grid(n: usize, half_width: f64) -> Grid {
        Grid::square(n, 2.0 * half_width / n as f64)
    }

    #[test]
    fn dense_reference_reproduces_circular_hertz() {
        // The independent solver must itself land on the analytic Hertz contact,
        // confirming it is a trustworthy yardstick for the BCCG solver.
        let radius = 10.0e-3;
        let load = 50.0;
        let e_star = 70.0e9;
        let hertz = HertzCircular::new(radius, load, e_star);

        let grid = centred_grid(48, 3.0 * hertz.contact_radius());
        let gap = Paraboloid::sphere(radius).sample(&grid);
        let problem = Problem::new(grid.clone(), gap, Control::Load(load));
        let reference = DenseReference::new(grid, e_star);
        let solution = reference.solve(&problem, Config::default());

        assert!(
            solution.diagnostics().converged,
            "reference did not converge"
        );
        let rel = |a: f64, b: f64| (a - b).abs() / b;
        assert!(rel(solution.total_load(), load) <= 1e-3, "load");
        assert!(
            rel(solution.contact_radius(), hertz.contact_radius()) <= 0.04,
            "contact radius: got {:e} want {:e}",
            solution.contact_radius(),
            hertz.contact_radius(),
        );
        assert!(
            rel(solution.max_pressure(), hertz.max_pressure()) <= 0.06,
            "peak pressure: got {:e} want {:e}",
            solution.max_pressure(),
            hertz.max_pressure(),
        );
    }
}
