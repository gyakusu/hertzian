//! Bound-constrained conjugate-gradient contact solver (Polonsky–Keer, 1999).
//!
//! Solves the frictionless normal-contact complementarity problem
//! (`p >= 0`, `gap >= 0`, `p * gap = 0`) under a prescribed total load. Each
//! iteration applies the influence operator once or twice (the only
//! `O(N log N)` cost) and projects onto the non-negativity constraint; the rigid
//! approach `delta` is recovered as the mean in-contact value of `u + h`.
//!
//! The iteration is the canonical Polonsky–Keer single loop: the search
//! direction is conjugate over the current contact set and reverts to steepest
//! descent for points as they enter contact (their stored direction is zero).
//! Per the functional-core / imperative-shell convention the public `solve` is
//! pure; the loop below is the localised, deterministic numerical kernel.

#![allow(
    clippy::cast_precision_loss,
    reason = "contact-point counts are tiny relative to f64's integer range"
)]

use ndarray::{Array2, Zip};

use crate::influence::InfluenceOperator;
use crate::problem::{Control, Problem};
use crate::solution::{Diagnostics, Solution};

/// Iterative-solver configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Relative tolerance on the inter-iteration pressure change.
    pub tolerance: f64,
    /// Hard cap on the number of iterations.
    pub max_iterations: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tolerance: 1.0e-8,
            max_iterations: 10_000,
        }
    }
}

/// A frictionless normal-contact solver.
pub trait Solver {
    /// Solves `problem`, using `operator` for the elastic response.
    #[must_use]
    fn solve(&self, problem: &Problem, operator: &dyn InfluenceOperator) -> Solution;
}

/// Polonsky–Keer bound-constrained conjugate gradient.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Bccg {
    config: Config,
}

impl Bccg {
    /// Creates a solver with the given configuration.
    #[must_use]
    pub const fn new(config: Config) -> Self {
        Self { config }
    }

    /// The solver configuration.
    #[must_use]
    pub const fn config(&self) -> Config {
        self.config
    }
}

impl Solver for Bccg {
    fn solve(&self, problem: &Problem, operator: &dyn InfluenceOperator) -> Solution {
        solve_load_controlled(problem, operator, self.config)
    }
}

fn solve_load_controlled(
    problem: &Problem,
    operator: &dyn InfluenceOperator,
    config: Config,
) -> Solution {
    let grid = problem.grid();
    assert_eq!(
        grid.dims(),
        operator.grid().dims(),
        "operator grid must match the problem grid",
    );
    let Control::Load(target_load) = problem.control();
    let gap = problem.gap();
    let cell_area = grid.cell_area();

    let mut pressure =
        Array2::from_elem(grid.dims(), target_load / (grid.len() as f64 * cell_area));
    let mut search = Array2::<f64>::zeros(grid.dims());
    let mut g_norm_prev = 1.0_f64;
    let mut approach = 0.0_f64;
    let mut residual = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;

    while iterations < config.max_iterations {
        iterations += 1;

        // Elastic deflection and raw gap `u + h`; the active set is `p > 0`.
        let displacement = operator.apply(pressure.view());
        let raw_gap = &displacement + &gap;
        let active = pressure.mapv(|p| p > 0.0);
        let n_active = active.iter().filter(|&&a| a).count();
        if n_active == 0 {
            break;
        }

        // Reduced gap `g = (u + h) - delta`, with `delta` the mean in-contact value.
        approach = masked_mean(&raw_gap, &active, n_active);
        let reduced = raw_gap.mapv(|v| v - approach);
        let g_norm = masked_sum_sq(&reduced, &active);

        // Conjugate search direction (steepest for points entering contact).
        let beta = if g_norm_prev > 0.0 {
            g_norm / g_norm_prev
        } else {
            0.0
        };
        Zip::from(&mut search)
            .and(&reduced)
            .and(&active)
            .for_each(|t, &g, &a| {
                *t = if a { g + beta * *t } else { 0.0 };
            });
        g_norm_prev = g_norm;

        // Step length from the operator action on the search direction.
        let projected = operator.apply(search.view());
        let mean_projected = masked_mean(&projected, &active, n_active);
        let numerator = masked_dot(&reduced, &search, &active);
        let denominator =
            Zip::from(&projected)
                .and(&search)
                .and(&active)
                .fold(0.0_f64, |acc, &q, &t, &a| {
                    if a {
                        acc + (q - mean_projected) * t
                    } else {
                        acc
                    }
                });
        let step = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        // Projected update, then force penetrating inactive points into contact.
        let previous = pressure.clone();
        Zip::from(&mut pressure).and(&search).for_each(|p, &t| {
            *p = (*p - step * t).max(0.0);
        });
        let reactivated = reactivate(&mut pressure, &reduced, step);

        // Enforce the total-load constraint.
        let current_load = pressure.sum() * cell_area;
        if current_load > 0.0 {
            let scale = target_load / current_load;
            pressure.mapv_inplace(|p| p * scale);
        }

        residual = relative_change(&pressure, &previous);
        if residual < config.tolerance && reactivated == 0 {
            converged = true;
            break;
        }
    }

    let total_load = pressure.sum() * cell_area;
    let contact_cells = pressure.iter().filter(|&&p| p > 0.0).count();
    let contact_area = contact_cells as f64 * cell_area;
    Solution::new(
        pressure,
        approach,
        total_load,
        contact_area,
        Diagnostics {
            iterations,
            residual,
            converged,
        },
    )
}

// Forces inactive but penetrating points (`p <= 0`, reduced gap `< 0`) into
// contact, returning how many were reactivated. A non-zero count means the
// active set changed, so convergence is not yet declared.
fn reactivate(pressure: &mut Array2<f64>, reduced: &Array2<f64>, step: f64) -> usize {
    let mut count = 0;
    Zip::from(pressure).and(reduced).for_each(|p, &g| {
        if *p <= 0.0 && g < 0.0 {
            *p = -step * g;
            count += 1;
        }
    });
    count
}

// Mean of `field` over the active set.
fn masked_mean(field: &Array2<f64>, active: &Array2<bool>, n_active: usize) -> f64 {
    let sum = Zip::from(field)
        .and(active)
        .fold(0.0_f64, |acc, &v, &a| if a { acc + v } else { acc });
    sum / n_active as f64
}

// Sum of squares of `field` over the active set.
fn masked_sum_sq(field: &Array2<f64>, active: &Array2<bool>) -> f64 {
    Zip::from(field)
        .and(active)
        .fold(0.0_f64, |acc, &v, &a| if a { acc + v * v } else { acc })
}

// Inner product of `lhs` and `rhs` over the active set.
fn masked_dot(lhs: &Array2<f64>, rhs: &Array2<f64>, active: &Array2<bool>) -> f64 {
    Zip::from(lhs)
        .and(rhs)
        .and(active)
        .fold(0.0_f64, |acc, &x, &y, &a| if a { acc + x * y } else { acc })
}

// Relative L1 change between successive pressure fields.
fn relative_change(current: &Array2<f64>, previous: &Array2<f64>) -> f64 {
    let change = Zip::from(current)
        .and(previous)
        .fold(0.0_f64, |acc, &c, &p| acc + (c - p).abs());
    let magnitude: f64 = current.iter().map(|v| v.abs()).sum();
    if magnitude > 0.0 {
        change / magnitude
    } else {
        0.0
    }
}
