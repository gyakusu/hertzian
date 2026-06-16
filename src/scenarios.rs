//! High-level scenario constructors (the "analytic shortcut" API layer).
//!
//! These mirror the intended Python entry points (design §8.5): build the gap,
//! the free-space DC-FFT operator, and the problem for a named geometry, then
//! solve in one call.

use crate::geometry::{Gap, Paraboloid};
use crate::grid::Grid;
use crate::influence::FreeSpaceBoussinesq;
use crate::material::Material;
use crate::problem::{Control, Problem};
use crate::solution::Solution;
use crate::solver::{Bccg, Config, Solver};
use crate::validation::HertzCircular;

/// Solves the contact for an arbitrary gap on a prepared grid.
#[must_use]
pub fn solve_gap(
    gap: &dyn Gap,
    material: Material,
    load: f64,
    grid: Grid,
    config: Config,
) -> Solution {
    let sampled = gap.sample(&grid);
    let operator = FreeSpaceBoussinesq::new(grid.clone(), material.e_star());
    let problem = Problem::new(grid, sampled, Control::Load(load));
    Bccg::new(config).solve(&problem, &operator)
}

/// Solves a sphere of radius `radius` pressed onto a flat.
#[must_use]
pub fn sphere_on_flat(
    radius: f64,
    load: f64,
    material: Material,
    grid: Grid,
    config: Config,
) -> Solution {
    solve_gap(&Paraboloid::sphere(radius), material, load, grid, config)
}

/// Solves two spheres of radii `radius_1`, `radius_2` in contact.
///
/// The pair reduces to an equivalent single sphere of the combined radius
/// `1/R = 1/R1 + 1/R2` pressed onto a flat.
#[must_use]
pub fn sphere_on_sphere(
    radius_1: f64,
    radius_2: f64,
    load: f64,
    material: Material,
    grid: Grid,
    config: Config,
) -> Solution {
    let radius = HertzCircular::combined_radius(radius_1, radius_2);
    solve_gap(&Paraboloid::sphere(radius), material, load, grid, config)
}

#[cfg(test)]
mod tests {
    use super::{sphere_on_flat, sphere_on_sphere};
    use crate::grid::Grid;
    use crate::material::Material;
    use crate::solver::Config;
    use crate::validation::HertzCircular;

    #[allow(
        clippy::cast_precision_loss,
        reason = "grid sizes are tiny relative to f64's integer range"
    )]
    fn centred_grid(n: usize, half_width: f64) -> Grid {
        Grid::square(n, 2.0 * half_width / n as f64)
    }

    fn assert_relative(actual: f64, expected: f64, tolerance: f64, what: &str) {
        let rel_err = (actual - expected).abs() / expected;
        assert!(
            rel_err <= tolerance,
            "{what}: actual={actual:e} expected={expected:e} rel_err={rel_err:e} (> {tolerance:e})",
        );
    }

    #[test]
    fn sphere_on_flat_matches_hertz() {
        let radius = 10.0e-3;
        let load = 50.0;
        let material = Material::from_e_star(70.0e9);
        let hertz = HertzCircular::new(radius, load, material.e_star());

        // Domain a few contact radii wide so the free-space boundary is clean.
        let grid = centred_grid(128, 3.0 * hertz.contact_radius());
        let config = Config {
            tolerance: 1.0e-8,
            max_iterations: 5_000,
        };
        let solution = sphere_on_flat(radius, load, material, grid, config);

        assert!(solution.diagnostics().converged, "solver did not converge");
        assert_relative(solution.total_load(), load, 1.0e-6, "total load");
        assert_relative(
            solution.contact_radius(),
            hertz.contact_radius(),
            0.03,
            "contact radius",
        );
        assert_relative(
            solution.max_pressure(),
            hertz.max_pressure(),
            0.05,
            "peak pressure",
        );
        assert_relative(solution.approach(), hertz.approach(), 0.04, "approach");
    }

    #[test]
    fn sphere_on_sphere_reduces_to_combined_radius() {
        let radius = 8.0e-3;
        let load = 30.0;
        let material = Material::from_e_star(110.0e9);
        let combined = HertzCircular::combined_radius(radius, radius);
        let hertz = HertzCircular::new(combined, load, material.e_star());

        let grid = centred_grid(128, 3.0 * hertz.contact_radius());
        let solution = sphere_on_sphere(radius, radius, load, material, grid, Config::default());

        assert!(solution.diagnostics().converged, "solver did not converge");
        assert_relative(
            solution.contact_radius(),
            hertz.contact_radius(),
            0.03,
            "contact radius",
        );
        assert_relative(
            solution.max_pressure(),
            hertz.max_pressure(),
            0.05,
            "peak pressure",
        );
    }
}
