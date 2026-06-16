//! High-level scenario constructors (the "analytic shortcut" API layer).
//!
//! These mirror the intended Python entry points (design §8.5): build the gap,
//! the free-space DC-FFT operator, and the problem for a named geometry, then
//! solve in one call.

use ndarray::Array2;

use crate::geometry::{Cone, Gap, Paraboloid, Torus};
use crate::grid::Grid;
use crate::influence::FreeSpaceBoussinesq;
use crate::material::Material;
use crate::problem::{Control, Problem};
use crate::solution::Solution;
use crate::solver::{Bccg, Config, Solver};
use crate::validation::HertzCircular;

/// Solves the contact for a gap already sampled on `grid`.
///
/// The free-space DC-FFT path shared by every scenario: build the operator on
/// `grid`, assemble the load-controlled problem, and run the BCCG solver. The
/// analytic-shortcut constructors below all funnel through here after sampling
/// their gap, and the Python height-field entry point reuses it directly with a
/// caller-supplied gap array (design §8.5).
///
/// # Panics
/// Panics if `gap`'s shape does not match `grid` (see [`Problem::new`]).
#[must_use]
pub fn solve_sampled_gap(
    gap: Array2<f64>,
    material: Material,
    load: f64,
    grid: Grid,
    config: Config,
) -> Solution {
    let operator = FreeSpaceBoussinesq::new(grid.clone(), material.e_star());
    let problem = Problem::new(grid, gap, Control::Load(load));
    Bccg::new(config).solve(&problem, &operator)
}

/// Solves the contact for an arbitrary gap on a prepared grid.
#[must_use]
pub fn solve_gap(
    gap: &dyn Gap,
    material: Material,
    load: f64,
    grid: Grid,
    config: Config,
) -> Solution {
    solve_sampled_gap(gap.sample(&grid), material, load, grid, config)
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

/// Solves a sphere of radius `sphere_radius` pressed onto a torus outer equator.
///
/// The convex–convex contact (design §5.2) is elliptic: the circumferential
/// direction (`x`) is gentler than the meridional one (`y`), so the contact runs
/// long along `x`. The torus and sphere reduce to a single paraboloidal gap with
/// distinct effective radii (see [`Torus::against_sphere`]).
#[must_use]
pub fn sphere_on_torus(
    sphere_radius: f64,
    torus: Torus,
    load: f64,
    material: Material,
    grid: Grid,
    config: Config,
) -> Solution {
    solve_gap(
        &torus.against_sphere(sphere_radius),
        material,
        load,
        grid,
        config,
    )
}

/// Solves a rigid cone of surface slope `slope` pressed onto a flat.
///
/// The non-Hertzian arbitrary-shape benchmark: the conical gap `h = m r` is fed
/// through the same path as any other shape and validated against Sneddon's
/// closed form (see [`SneddonCone`](crate::validation::SneddonCone)).
#[must_use]
pub fn cone_on_flat(
    slope: f64,
    load: f64,
    material: Material,
    grid: Grid,
    config: Config,
) -> Solution {
    solve_gap(&Cone::new(slope), material, load, grid, config)
}

#[cfg(test)]
mod tests {
    use super::{
        cone_on_flat, solve_gap, solve_sampled_gap, sphere_on_flat, sphere_on_sphere,
        sphere_on_torus,
    };
    use crate::geometry::{Gap, Paraboloid, Torus, Waviness};
    use crate::grid::Grid;
    use crate::material::Material;
    use crate::problem::{Control, Problem};
    use crate::reference::DenseReference;
    use crate::solver::Config;
    use crate::validation::{HertzCircular, HertzElliptic, SneddonCone};

    #[allow(
        clippy::cast_precision_loss,
        reason = "grid sizes are tiny relative to f64's integer range"
    )]
    fn centred_grid(n: usize, half_width: f64) -> Grid {
        Grid::square(n, 2.0 * half_width / n as f64)
    }

    // An origin-centred grid sized to an elliptic contact: isotropic spacing
    // resolving the minor semi-axis, a domain `margin` semi-axes wide on each
    // side so the free-space boundary is clean, and an even point count per axis.
    fn elliptic_grid(reference: &HertzElliptic) -> Grid {
        let margin = 3.0;
        let spacing = reference.semi_minor() / 24.0;
        let nx = even_ceil(2.0 * margin * reference.semi_axis_x() / spacing);
        let ny = even_ceil(2.0 * margin * reference.semi_axis_y() / spacing);
        Grid::new(nx, ny, spacing, spacing)
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the argument is a small positive grid-point count"
    )]
    fn even_ceil(value: f64) -> usize {
        let n = value.ceil() as usize;
        n + (n & 1)
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
    fn presampled_gap_matches_the_sphere_shortcut() {
        // The height-field path (used by the Python `solve_height_field`
        // binding) must reproduce the analytic shortcut when handed the same
        // gap the shortcut would sample internally.
        let radius = 10.0e-3;
        let load = 50.0;
        let material = Material::from_e_star(70.0e9);
        let hertz = HertzCircular::new(radius, load, material.e_star());
        let grid = centred_grid(128, 3.0 * hertz.contact_radius());
        let config = Config {
            tolerance: 1.0e-8,
            max_iterations: 5_000,
        };

        let gap = Paraboloid::sphere(radius).sample(&grid);
        let presampled = solve_sampled_gap(gap, material, load, grid.clone(), config);
        let shortcut = sphere_on_flat(radius, load, material, grid, config);

        assert!(
            presampled.diagnostics().converged,
            "solver did not converge"
        );
        assert_relative(
            presampled.contact_radius(),
            shortcut.contact_radius(),
            1.0e-12,
            "contact radius",
        );
        assert_relative(
            presampled.max_pressure(),
            shortcut.max_pressure(),
            1.0e-12,
            "peak pressure",
        );
        assert_relative(
            presampled.approach(),
            shortcut.approach(),
            1.0e-12,
            "approach",
        );
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

    #[test]
    fn sphere_on_torus_matches_elliptic_hertz() {
        // Sphere on a torus outer equator: the P2 elliptic-contact benchmark.
        let torus = Torus::new(4.0e-3, 20.0e-3);
        let sphere_radius = 12.0e-3;
        let load = 60.0;
        let material = Material::from_e_star(100.0e9);

        // The reference is built from the same effective relative radii the
        // scenario derives, then computed independently via elliptic integrals.
        let gap = torus.against_sphere(sphere_radius);
        let reference = HertzElliptic::new(gap.radius_x(), gap.radius_y(), load, material.e_star());
        assert!(
            reference.ellipticity() > 1.5,
            "test geometry should be clearly elliptic (got {:.3})",
            reference.ellipticity(),
        );

        let grid = elliptic_grid(&reference);
        let config = Config {
            tolerance: 1.0e-8,
            max_iterations: 5_000,
        };
        let solution = sphere_on_torus(sphere_radius, torus, load, material, grid, config);

        assert!(solution.diagnostics().converged, "solver did not converge");
        assert_relative(solution.total_load(), load, 1.0e-6, "total load");

        // Orientation: the contact is elongated circumferentially (along x).
        let (a_x, a_y) = solution.contact_half_widths();
        assert!(
            a_x > a_y,
            "contact must run long along x (circumferential): a_x={a_x:e} a_y={a_y:e}",
        );

        // The solver agrees with the independently derived elliptic reference to
        // well under 1% at this resolution; tolerances leave headroom for the
        // grid-discretisation error of the second-moment semi-axis estimate.
        assert_relative(a_x, reference.semi_axis_x(), 0.02, "semi-axis x");
        assert_relative(a_y, reference.semi_axis_y(), 0.02, "semi-axis y");
        assert_relative(
            solution.ellipticity(),
            reference.ellipticity(),
            0.02,
            "ellipticity",
        );
        assert_relative(
            solution.max_pressure(),
            reference.max_pressure(),
            0.02,
            "peak pressure",
        );
        assert_relative(solution.approach(), reference.approach(), 0.01, "approach");
    }

    #[test]
    fn cone_on_flat_matches_sneddon() {
        // P4 arbitrary-shape benchmark: a rigid cone fed through the height-field
        // path reproduces Sneddon's closed-form contact radius, approach and
        // load. The apex pressure singularity is mesh-dependent, so peak pressure
        // is deliberately not compared.
        let slope = 0.02;
        let load = 60.0;
        let material = Material::from_e_star(100.0e9);
        let reference = SneddonCone::new(slope, load, material.e_star());

        // A fine grid spanning a few contact radii: the apex and the contact
        // edge both need resolution for the area-based radius to converge.
        let grid = centred_grid(320, 3.0 * reference.contact_radius());
        let config = Config {
            tolerance: 1.0e-8,
            max_iterations: 5_000,
        };
        let solution = cone_on_flat(slope, load, material, grid, config);

        assert!(solution.diagnostics().converged, "solver did not converge");
        assert_relative(solution.total_load(), load, 1.0e-6, "total load");
        assert_relative(
            solution.contact_radius(),
            reference.contact_radius(),
            0.03,
            "contact radius",
        );
        assert_relative(solution.approach(), reference.approach(), 0.03, "approach");
    }

    #[test]
    fn rough_sphere_cross_validates_against_the_dense_reference() {
        // P4 cross-validation: a sphere with added cosine roughness has no closed
        // form, so the production FFT + BCCG solution is checked against the
        // independent dense projected-Gauss–Seidel reference on the same grid.
        // Both use identical influence coefficients, so the only difference is
        // the iterative scheme — agreement validates the solver itself.
        let radius = 10.0e-3;
        let load = 40.0;
        let material = Material::from_e_star(70.0e9);
        let hertz = HertzCircular::new(radius, load, material.e_star());

        // A small grid keeps the O(N^2) dense solve cheap; the roughness
        // wavelength is a fraction of the smooth contact so several asperities
        // fall inside the patch.
        let grid = centred_grid(40, 2.5 * hertz.contact_radius());
        let smooth_area = std::f64::consts::PI * hertz.contact_radius().powi(2);
        let rough = Paraboloid::sphere(radius).plus(Waviness::new(
            0.8 * hertz.approach(),
            1.0 * hertz.contact_radius(),
            1.0 * hertz.contact_radius(),
        ));
        let gap = rough.sample(&grid);

        let config = Config {
            tolerance: 1.0e-9,
            max_iterations: 20_000,
        };
        let bccg = solve_gap(&rough, material, load, grid.clone(), config);
        let problem = Problem::new(grid.clone(), gap, Control::Load(load));
        let dense = DenseReference::new(grid, material.e_star()).solve(&problem, config);

        assert!(bccg.diagnostics().converged, "BCCG did not converge");
        assert!(
            dense.diagnostics().converged,
            "dense reference did not converge",
        );

        // The roughness genuinely fragments the contact: the real area drops
        // well below the smooth Hertz disc and the asperities concentrate the
        // pressure far above the smooth peak. Otherwise this would just be the
        // circular-Hertz test in disguise.
        assert!(
            bccg.contact_area() < 0.6 * smooth_area,
            "roughness should fragment the contact (area {:e} vs smooth {:e})",
            bccg.contact_area(),
            smooth_area,
        );
        assert!(
            bccg.max_pressure() > 1.8 * hertz.max_pressure(),
            "asperities should raise the peak pressure above the smooth Hertz peak",
        );

        // Two unrelated solvers on the same kernel and grid converge to the same
        // discrete solution, so they agree far more tightly than the few-percent
        // grid-discretisation error of either against a continuum reference.
        assert_relative(dense.total_load(), load, 1.0e-3, "dense load");
        assert_relative(
            bccg.contact_area(),
            dense.contact_area(),
            0.02,
            "contact area",
        );
        assert_relative(
            bccg.max_pressure(),
            dense.max_pressure(),
            1.0e-3,
            "peak pressure",
        );
        assert_relative(bccg.approach(), dense.approach(), 1.0e-4, "approach");
    }
}
