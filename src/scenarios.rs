//! High-level scenario constructors (the "analytic shortcut" API layer).
//!
//! These mirror the intended Python entry points (design §8.5): build the gap,
//! the free-space DC-FFT operator, and the problem for a named geometry, then
//! solve in one call.

use ndarray::Array2;

use crate::geometry::{Cone, Gap, GothicArchGroove, Paraboloid, Torus};
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

/// Solves a sphere pressed into a Gothic-arch (ogival) groove.
///
/// The concave counterpart of [`sphere_on_torus`]: the ball sits *inside* a
/// conformal groove whose cross-section is two arcs (two tori overlaid), so the
/// gap is the double-welled [`GothicArchProfile`](crate::geometry::GothicArchProfile).
/// With a large centre offset the ball rides on two well-separated flanks and the
/// contact splits into a pair of elliptic patches at `y = ±y0`, each carrying half
/// the load; with a zero offset it reduces to a single conformal elliptic contact.
/// Between the two, a tightened offset brings the flank contact ellipses to a
/// partial *overlap* — a single connected patch whose two peaks reinforce through
/// the elastic field (no closed form; cross-validated against the dense reference).
#[must_use]
pub fn sphere_in_gothic_arch(
    sphere_radius: f64,
    groove: GothicArchGroove,
    load: f64,
    material: Material,
    grid: Grid,
    config: Config,
) -> Solution {
    solve_gap(
        &groove.against_sphere(sphere_radius),
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
        cone_on_flat, solve_gap, solve_sampled_gap, sphere_in_gothic_arch, sphere_on_flat,
        sphere_on_sphere, sphere_on_torus,
    };
    use crate::geometry::{Gap, GothicArchGroove, Paraboloid, Torus, Waviness};
    use crate::grid::Grid;
    use crate::material::Material;
    use crate::problem::{Control, Problem};
    use crate::reduced::GothicArchLaw;
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

    // A grid for a Gothic contact: two elliptic patches centred at y = ±y0, each
    // sized like `reference` (one flank at half the load). Isotropic spacing
    // resolves the minor (x) semi-axis; the domain spans both flanks plus a clean
    // free-space margin, tall along the split (y) axis and narrow across it.
    fn gothic_grid(reference: &HertzElliptic, offset: f64) -> Grid {
        let spacing = reference.semi_axis_x() / 12.0;
        let margin = 2.0 * reference.semi_axis_x();
        let half_x = reference.semi_axis_x() + margin;
        let half_y = offset + reference.semi_axis_y() + margin;
        let nx = even_ceil(2.0 * half_x / spacing);
        let ny = even_ceil(2.0 * half_y / spacing);
        Grid::new(nx, ny, spacing, spacing)
    }

    // Peak pressure within the y-half on one side of the groove centre (column
    // `j < mid` or `j >= mid`), with its (i, j) location. Isolates a single flank
    // of a split Gothic contact so each patch can be checked on its own.
    fn flank_peak(solution: &crate::solution::Solution, upper: bool) -> (f64, usize, usize) {
        let pressure = solution.pressure();
        let mid = pressure.ncols() / 2;
        let mut best = (0.0_f64, 0, 0);
        for ((i, j), &p) in pressure.indexed_iter() {
            let in_half = if upper { j >= mid } else { j < mid };
            if in_half && p > best.0 {
                best = (p, i, j);
            }
        }
        best
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
    fn sphere_in_gothic_arch_without_offset_matches_elliptic_hertz() {
        // With no centre shim the Gothic groove is a single conformal arc, so the
        // ball-in-groove contact is one elliptic Hertz patch — the concave
        // counterpart of the torus benchmark, validating the groove reduction
        // (concave meridional radius, convex circumferential radius).
        let ball = 4.0e-3;
        let tube = 1.04 * ball; // r/Rs = 1.04: a textbook bearing conformity
        let groove = GothicArchGroove::new(tube, 15.0e-3, 0.0);
        let load = 60.0;
        let material = Material::from_e_star(100.0e9);

        let profile = groove.against_sphere(ball);
        let reference = HertzElliptic::new(
            profile.radius_x(),
            profile.radius_y(),
            load,
            material.e_star(),
        );
        // High conformity makes the patch strongly elliptic (long across groove).
        assert!(
            reference.ellipticity() > 5.0,
            "a conformal groove contact should be strongly elliptic (got {:.2})",
            reference.ellipticity(),
        );

        let grid = gothic_grid(&reference, 0.0);
        let config = Config {
            tolerance: 1.0e-8,
            max_iterations: 10_000,
        };
        let solution = sphere_in_gothic_arch(ball, groove, load, material, grid, config);

        assert!(solution.diagnostics().converged, "solver did not converge");
        assert_relative(solution.total_load(), load, 1.0e-6, "total load");

        // The conformal contact runs long across the groove (meridional y).
        let (a_x, a_y) = solution.contact_half_widths();
        assert!(a_y > a_x, "contact must run long across the groove (y)");
        assert_relative(a_x, reference.semi_axis_x(), 0.03, "semi-axis x");
        assert_relative(a_y, reference.semi_axis_y(), 0.03, "semi-axis y");
        assert_relative(
            solution.max_pressure(),
            reference.max_pressure(),
            0.03,
            "peak pressure",
        );
    }

    #[test]
    fn sphere_in_gothic_arch_splits_into_two_elliptic_contacts() {
        // The defining behaviour of a Gothic arch: a shimmed groove makes the ball
        // ride on two flanks, so the single conformal patch splits into a pair of
        // elliptic contacts at y = ±y0, each carrying half the load. Each flank
        // must therefore match elliptic Hertz at P/2, and the split must lower the
        // peak pressure below the single full-load contact (the design payoff).
        let ball = 4.0e-3;
        let tube = 1.04 * ball;
        let load = 60.0;
        let material = Material::from_e_star(100.0e9);
        let centre_radius = 15.0e-3;

        // Each flank is an elliptic Hertz contact at half the total load.
        let radii = GothicArchGroove::new(tube, centre_radius, 0.0).against_sphere(ball);
        let flank = HertzElliptic::new(
            radii.radius_x(),
            radii.radius_y(),
            load / 2.0,
            material.e_star(),
        );

        // Shim the centres so the flanks sit 1.6 major semi-axes off the centre:
        // comfortably separated, leaving a contact-free Gothic ridge between them.
        let y0 = 1.6 * flank.semi_axis_y();
        let centre_offset = y0 * (tube - ball) / ball;
        let groove = GothicArchGroove::new(tube, centre_radius, centre_offset);
        assert_relative(
            groove.against_sphere(ball).offset(),
            y0,
            1.0e-12,
            "flank offset",
        );

        let grid = gothic_grid(&flank, y0);
        let config = Config {
            tolerance: 1.0e-8,
            max_iterations: 20_000,
        };
        let solution = sphere_in_gothic_arch(ball, groove, load, material, grid.clone(), config);

        assert!(solution.diagnostics().converged, "solver did not converge");
        assert_relative(solution.total_load(), load, 1.0e-6, "total load");

        // Two flanks, each a P/2 elliptic Hertz contact peaking near y = ±y0.
        let (upper_peak, _, j_upper) = flank_peak(&solution, true);
        let (lower_peak, _, j_lower) = flank_peak(&solution, false);
        assert_relative(upper_peak, flank.max_pressure(), 0.09, "upper-flank peak");
        assert_relative(lower_peak, flank.max_pressure(), 0.09, "lower-flank peak");
        assert_relative(upper_peak, lower_peak, 0.03, "flank symmetry");
        assert_relative(grid.y(j_upper), y0, 0.10, "upper-flank location");
        assert_relative(grid.y(j_lower), -y0, 0.10, "lower-flank location");

        // The Gothic ridge carries no load: the centre band is contact-free.
        let peak = upper_peak.max(lower_peak);
        let ridge_pressure = solution
            .pressure()
            .indexed_iter()
            .filter(|&((_, j), _)| grid.y(j).abs() < 0.3 * y0)
            .fold(0.0_f64, |m, (_, &p)| m.max(p));
        assert!(
            ridge_pressure < 0.05 * peak,
            "the Gothic ridge must stay contact-free (ridge {ridge_pressure:e} vs peak {peak:e})",
        );

        // Splitting the load across two patches lowers the peak pressure relative
        // to the single full-load contact, by ~(1/2)^(1/3) for well-separated
        // elliptic patches.
        let single =
            HertzElliptic::new(radii.radius_x(), radii.radius_y(), load, material.e_star());
        assert!(
            peak < 0.9 * single.max_pressure(),
            "the split must lower the peak below the single contact ({:e} vs {:e})",
            peak,
            single.max_pressure(),
        );
        assert!(
            peak > 0.7 * single.max_pressure(),
            "but only by the load-split factor, not collapse",
        );
    }

    #[test]
    fn sphere_in_gothic_arch_half_overlapping_flanks_cross_validate() {
        // A second Gothic-arch pattern. The separated arch above keeps the two
        // flanks far enough apart to leave a contact-free Gothic point between
        // them; here the arc-centre shim is tightened so the two flank contact
        // ellipses *overlap by half* instead. The design target is the meridional
        // flank offset y0 = b/2, where b is the meridional semi-axis of one
        // isolated half-load elliptic flank: two ellipses of semi-axis b whose
        // centres sit b apart share exactly half their meridional extent (each
        // overlaps the other by half).
        //
        // The overlapping regime has no closed form — the two contacts interact
        // through the elastic field, so the load no longer splits cleanly into a
        // pair of P/2 Hertz patches — so it is cross-validated the P4 way: against
        // the independent dense projected-Gauss–Seidel reference on the same grid,
        // built on identical influence coefficients but solved by an unrelated
        // iteration. The physical signatures of the overlap are pinned alongside.
        let ball = 4.0e-3;
        let tube = 1.04 * ball;
        let load = 60.0;
        let material = Material::from_e_star(100.0e9);
        let centre_radius = 15.0e-3;

        // The isolated half-load flank sets the overlap scale b = its meridional
        // semi-axis; the full-load single arc bounds the peak from above.
        let radii = GothicArchGroove::new(tube, centre_radius, 0.0).against_sphere(ball);
        let flank = HertzElliptic::new(
            radii.radius_x(),
            radii.radius_y(),
            load / 2.0,
            material.e_star(),
        );
        let single =
            HertzElliptic::new(radii.radius_x(), radii.radius_y(), load, material.e_star());

        // Half overlap: flank centres one meridional semi-axis apart (y0 = b/2).
        let y0 = 0.5 * flank.semi_axis_y();
        let centre_offset = y0 * (tube - ball) / ball;
        let groove = GothicArchGroove::new(tube, centre_radius, centre_offset);
        assert_relative(
            groove.against_sphere(ball).offset(),
            y0,
            1.0e-12,
            "flank offset",
        );

        // A small *anisotropic* grid: fine across the narrow (x) semi-axis, coarse
        // along the long (y) one the flanks spread over, kept tiny so the O(N^2)
        // dense reference stays cheap while still resolving the saddle.
        let dx = flank.semi_axis_x() / 5.0;
        let dy = flank.semi_axis_y() / 10.0;
        let half_x = 2.0 * flank.semi_axis_x();
        let half_y = y0 + 1.45 * flank.semi_axis_y();
        let nx = even_ceil(2.0 * half_x / dx);
        let ny = even_ceil(2.0 * half_y / dy);
        let grid = Grid::new(nx, ny, dx, dy);

        let config = Config {
            tolerance: 1.0e-9,
            max_iterations: 20_000,
        };
        let gap = groove.against_sphere(ball).sample(&grid);
        let bccg = sphere_in_gothic_arch(ball, groove, load, material, grid.clone(), config);
        let problem = Problem::new(grid.clone(), gap, Control::Load(load));
        let dense = DenseReference::new(grid.clone(), material.e_star()).solve(&problem, config);

        assert!(bccg.diagnostics().converged, "BCCG did not converge");
        assert!(
            dense.diagnostics().converged,
            "dense reference did not converge",
        );
        assert_relative(bccg.total_load(), load, 1.0e-6, "total load");

        // Two symmetric flank peaks, sitting near ±y0 (the elastic interaction
        // nudges them slightly outboard of the geometric offset).
        let (upper_peak, _, j_upper) = flank_peak(&bccg, true);
        let (lower_peak, _, j_lower) = flank_peak(&bccg, false);
        let peak = upper_peak.max(lower_peak);
        assert_relative(upper_peak, lower_peak, 0.02, "flank symmetry");
        assert!(
            grid.y(j_upper) > 0.5 * y0 && grid.y(j_upper) < 1.6 * y0,
            "upper flank must peak near +y0 (got {:e}, y0={y0:e})",
            grid.y(j_upper),
        );
        assert!(
            grid.y(j_lower) < -0.5 * y0 && grid.y(j_lower) > -1.6 * y0,
            "lower flank must peak near -y0 (got {:e})",
            grid.y(j_lower),
        );

        // The defining contrast with the separated arch: the former Gothic point
        // now carries load (the contact is *connected*), yet stays below the
        // flanks, so the two ellipses still read as a saddle-joined pair rather
        // than one merged patch.
        let centre = bccg
            .pressure()
            .indexed_iter()
            .filter(|&((_, j), _)| grid.y(j).abs() < 0.2 * y0)
            .fold(0.0_f64, |m, (_, &p)| m.max(p));
        assert!(
            centre > 0.4 * peak,
            "the overlapped Gothic point must carry load (centre {centre:e} vs peak {peak:e})",
        );
        assert!(
            centre < 0.9 * peak,
            "but stay below the flanks — a saddle, not a single merged peak",
        );

        // The overlap raises the peak above the well-separated (1/2)^(1/3) value
        // (= the isolated half-load flank) but keeps it below the merged
        // single-arc contact: the split still helps, just less than full
        // separation would.
        assert!(
            peak > flank.max_pressure(),
            "overlap must raise the peak above the separated flank value ({:e} vs {:e})",
            peak,
            flank.max_pressure(),
        );
        assert!(
            peak < single.max_pressure(),
            "but the split keeps it below the single-arc peak ({:e} vs {:e})",
            peak,
            single.max_pressure(),
        );

        // Cross-validation: two unrelated solvers on the same kernel and grid
        // land on the same discrete overlapping-contact solution, far tighter
        // than either's grid-discretisation error against a continuum reference.
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
        assert_relative(bccg.approach(), dense.approach(), 1.0e-3, "approach");
    }

    #[test]
    fn gothic_arch_reduced_law_tracks_the_field_solver() {
        // The reduced two-flank force law (crate::reduced) stands in for the field
        // solver in a multibody inner loop, so it must reproduce what the solver
        // computes. Its per-flank stiffness K is calibrated from the elliptic-Hertz
        // flank; here we confirm against the FFT + BCCG solver that (1) the scalar
        // Hertz flank load reproduces the single-arc load–deflection, and (2) two
        // separated flanks superpose — the load doubles at the same approach, each
        // flank carrying half. Together these pin the law the regression rests on:
        // F(δ) = Σ K⌊s_i⌋₊^{3/2} n̂_i with K from one validated elliptic flank.
        let ball = 4.0e-3;
        let tube = 1.04 * ball;
        let centre_radius = 15.0e-3;
        let material = Material::from_e_star(100.0e9);
        let load = 60.0;
        let config = Config {
            tolerance: 1.0e-8,
            max_iterations: 20_000,
        };

        // Calibrate the law from the flank's relative radii (the contact angle is
        // geometric and does not enter the scalar flank stiffness).
        let radii = GothicArchGroove::new(tube, centre_radius, 0.0).against_sphere(ball);
        let law = GothicArchLaw::from_elliptic_flank(
            radii.radius_x(),
            radii.radius_y(),
            material.e_star(),
            0.4,
        );

        // (1) Single arc — one elliptic flank. The law's scalar Hertz load must
        // reproduce the solver's load at the solver's own approach.
        let single_reference =
            HertzElliptic::new(radii.radius_x(), radii.radius_y(), load, material.e_star());
        let single = sphere_in_gothic_arch(
            ball,
            GothicArchGroove::new(tube, centre_radius, 0.0),
            load,
            material,
            gothic_grid(&single_reference, 0.0),
            config,
        );
        assert!(
            single.diagnostics().converged,
            "single arc did not converge"
        );
        assert_relative(
            law.flank_load(single.approach()),
            single.total_load(),
            0.08,
            "single-flank law vs solver",
        );

        // (2) Two separated flanks at the same total load: each carries half, so
        // the law predicts twice the single-flank load at the (smaller) two-flank
        // approach.
        let flank = HertzElliptic::new(
            radii.radius_x(),
            radii.radius_y(),
            load / 2.0,
            material.e_star(),
        );
        let y0 = 1.6 * flank.semi_axis_y();
        let centre_offset = y0 * (tube - ball) / ball;
        let split = sphere_in_gothic_arch(
            ball,
            GothicArchGroove::new(tube, centre_radius, centre_offset),
            load,
            material,
            gothic_grid(&flank, y0),
            config,
        );
        assert!(
            split.diagnostics().converged,
            "split contact did not converge"
        );
        assert_relative(
            2.0 * law.flank_load(split.approach()),
            split.total_load(),
            0.10,
            "two-flank superposition law vs solver",
        );
    }

    #[test]
    fn gothic_flank_pressure_caps_the_field_solver() {
        // The per-flank pressure footprint is the Coulomb-friction cap the reduced
        // law hands a tangential-contact model (|τ| ≤ μ p). In the separated regime
        // each flank is a half-load elliptic-Hertz patch, so the footprint built from
        // the per-flank load must reproduce what the field solver finds — its peak
        // pressure and circumferential half-width — and it integrates to that load
        // exactly. (Where the patches overlap they merge into one connected contact
        // whose seam is not the sum of the two half-ellipsoids; the deferred stage.)
        let ball = 4.0e-3;
        let tube = 1.04 * ball;
        let centre_radius = 15.0e-3;
        let material = Material::from_e_star(100.0e9);
        let load = 60.0;
        let config = Config {
            tolerance: 1.0e-9,
            max_iterations: 40_000,
        };

        let radii = GothicArchGroove::new(tube, centre_radius, 0.0).against_sphere(ball);
        let law = GothicArchLaw::from_elliptic_flank(
            radii.radius_x(),
            radii.radius_y(),
            material.e_star(),
            0.4,
        );

        // Two well-separated flanks, each carrying half the load.
        let flank = HertzElliptic::new(
            radii.radius_x(),
            radii.radius_y(),
            load / 2.0,
            material.e_star(),
        );
        let y0 = 2.0 * flank.semi_axis_y(); // well separated: two distinct patches
        let centre_offset = y0 * (tube - ball) / ball;
        let split = sphere_in_gothic_arch(
            ball,
            GothicArchGroove::new(tube, centre_radius, centre_offset),
            load,
            material,
            gothic_grid(&flank, y0),
            config,
        );
        assert!(
            split.diagnostics().converged,
            "split contact did not converge"
        );

        // The footprint from the per-flank load reproduces the solver's peak pressure
        // (the cap the contact rides under) and its circumferential half-width...
        let footprint = law
            .flank_pressure(split.total_load() / 2.0)
            .expect("the calibrated law has a footprint");
        assert_relative(
            footprint.peak_pressure(),
            split.max_pressure(),
            0.03,
            "flank cap peak vs solver",
        );
        let (a_x, _) = footprint.semi_axes();
        assert_relative(
            a_x,
            split.contact_half_widths().0,
            0.05,
            "flank cap circumferential half-width vs solver",
        );
        // ...and it carries exactly the load it was built from, so ∫ μ p dA = μ Q.
        assert_relative(
            footprint.load(),
            split.total_load() / 2.0,
            1.0e-9,
            "footprint integrates to the flank load",
        );
    }

    #[test]
    fn gothic_groove_pressure_envelope_caps_the_overlap() {
        // The whole-groove Coulomb cap is the *envelope* (pointwise max) of the two
        // per-flank footprints — the dual of the gap's pointwise-min construction.
        // Where the separated test above resolves the two patches, here the arc-centre
        // shim is tightened to a half overlap (y0 = b/2), the regime the naive *sum*
        // of the two half-ellipsoids gets wrong: it double-counts the crossing
        // footprints into an unphysical seam spike. The envelope drops that
        // double-count — its peak tracks the field solver to a few percent (the sum
        // overshoots by ~70%), and it recovers the connected saddle the solver finds.
        let ball = 4.0e-3;
        let tube = 1.04 * ball;
        let centre_radius = 15.0e-3;
        let material = Material::from_e_star(100.0e9);
        let load = 120.0;
        let config = Config {
            tolerance: 1.0e-9,
            max_iterations: 40_000,
        };

        let radii = GothicArchGroove::new(tube, centre_radius, 0.0).against_sphere(ball);
        let flank = HertzElliptic::new(
            radii.radius_x(),
            radii.radius_y(),
            load / 2.0,
            material.e_star(),
        );
        let b = flank.semi_axis_y();
        let ax = flank.semi_axis_x();

        // Half overlap: flank centres one meridional semi-axis apart (y0 = b/2).
        let y0 = 0.5 * b;
        let centre_offset = y0 * (tube - ball) / ball;
        let groove = GothicArchGroove::new(tube, centre_radius, centre_offset);

        let dx = ax / 8.0;
        let dy = b / 12.0;
        let nx = even_ceil(2.0 * 2.5 * ax / dx);
        let ny = even_ceil(2.0 * (y0 + 2.5 * b) / dy);
        let grid = Grid::new(nx, ny, dx, dy);
        let sol = sphere_in_gothic_arch(ball, groove, load, material, grid, config);
        assert!(sol.diagnostics().converged, "solver did not converge");

        // Build the envelope cap from the coupled flank loads at the solver's approach.
        let law = GothicArchLaw::from_elliptic_flank(
            radii.radius_x(),
            radii.radius_y(),
            material.e_star(),
            0.4,
        )
        .with_flank_coupling(material.e_star(), y0);
        let (q_plus, q_minus) = law.coupled_loads(sol.approach(), sol.approach());
        let groove_cap = law
            .groove_pressure(q_plus, q_minus, y0)
            .expect("the calibrated law has a footprint");
        assert!(
            !groove_cap.separated(),
            "the footprints must overlap at y0 = b/2",
        );

        // The envelope peak tracks the solver to a few percent...
        let solver_peak = sol.max_pressure();
        assert_relative(
            groove_cap.peak_pressure(),
            solver_peak,
            0.07,
            "envelope peak vs solver",
        );

        // ...whereas the naive sum double-counts the seam into a spike well above it.
        let (cap_plus, cap_minus) = groove_cap.flanks();
        let sum_seam = cap_plus.pressure_at(0.0, -y0) + cap_minus.pressure_at(0.0, y0);
        assert!(
            sum_seam > 1.3 * solver_peak,
            "the naive sum must over-count the seam (sum {sum_seam:e} vs solver {solver_peak:e})",
        );

        // The envelope is the connected saddle the solver finds: the seam carries
        // load (a connected patch, not the separated arch's contact-free ridge) but
        // stays below the flank crest.
        let seam = groove_cap.pressure_at(0.0, 0.0);
        let crest = groove_cap.peak_pressure();
        assert!(
            seam > 0.4 * crest && seam < crest,
            "connected saddle: seam {seam:e} vs crest {crest:e}",
        );
    }

    #[test]
    fn gothic_overlap_shifts_the_load_centroid_outboard() {
        // The second-order directional signature: the flank *normal* rotates as the
        // contacts overlap. Each flank lifts the inboard side of its neighbour more
        // than the outboard side (the lift `Q/(π E* d)` is steeper the closer the
        // patch), so the load centroid slides *outboard* of the geometric offset
        // y0 — its load sits at a larger effective offset, i.e. a steeper effective
        // contact angle α_eff = arcsin(y_centroid / R_s) > the geometric α. This
        // pins the field-solver evidence for promoting α to an effective α(y0/b):
        // the centroid is outboard in the half overlap and returns to y0 (the
        // geometric angle) once the flanks separate.
        let ball = 4.0e-3;
        let tube = 1.04 * ball;
        let centre_radius = 15.0e-3;
        let material = Material::from_e_star(100.0e9);
        let load = 120.0;
        let config = Config {
            tolerance: 1.0e-9,
            max_iterations: 40_000,
        };
        let radii = GothicArchGroove::new(tube, centre_radius, 0.0).against_sphere(ball);
        let flank = HertzElliptic::new(
            radii.radius_x(),
            radii.radius_y(),
            load / 2.0,
            material.e_star(),
        );
        let b = flank.semi_axis_y();
        let ax = flank.semi_axis_x();

        let centroid_over_y0 = |ratio: f64| {
            let y0 = ratio * b;
            let centre_offset = y0 * (tube - ball) / ball;
            let groove = GothicArchGroove::new(tube, centre_radius, centre_offset);
            let dx = ax / 8.0;
            let dy = b / 16.0;
            let nx = even_ceil(2.0 * 2.5 * ax / dx);
            let ny = even_ceil(2.0 * (y0 + 2.5 * b) / dy);
            let grid = Grid::new(nx, ny, dx, dy);
            let sol = sphere_in_gothic_arch(ball, groove, load, material, grid.clone(), config);
            assert!(sol.diagnostics().converged, "solver did not converge");
            // Load-weighted centroid of the upper (y > 0) flank.
            let (moment, weight) =
                sol.pressure()
                    .indexed_iter()
                    .fold((0.0_f64, 0.0_f64), |(m, w), ((_, j), &p)| {
                        let y = grid.y(j);
                        if y > 0.0 {
                            (m + p * y, w + p)
                        } else {
                            (m, w)
                        }
                    });
            moment / (weight * y0)
        };

        let half = centroid_over_y0(0.5); // half overlap
        let onset = centroid_over_y0(1.0); // patches just meet
        let separated = centroid_over_y0(2.0); // well separated

        // At half overlap the centroid is well outboard — the geometric α clearly
        // understates the effective flank angle ...
        assert!(
            half > 1.2,
            "half-overlap centroid must sit outboard of y0: {half}",
        );
        // ... the shift collapses monotonically as the flanks pull apart ...
        assert!(
            half > onset && onset > separated,
            "the outboard shift must fade with separation: {half} > {onset} > {separated}",
        );
        // ... it grows steeply into the overlap (the half-overlap shift dwarfs the
        // onset one) ...
        assert!(
            half - 1.0 > 5.0 * (onset - 1.0),
            "the shift must grow steeply into overlap: {} vs {}",
            half - 1.0,
            onset - 1.0,
        );
        // ... and the separated limit recovers the geometric offset (α_eff → α).
        assert!(
            separated < 1.01,
            "separated flanks sit at the geometric offset: {separated}",
        );
    }

    #[test]
    fn gothic_coupling_tracks_the_effective_flank_count() {
        // The neighbour-lift coupling (crate::reduced) earns its keep here. As the
        // groove shim is tightened from well-separated flanks down to a half overlap
        // (y0 = b/2), the field solver's effective flank count η = P/(K δ^{3/2})
        // falls from near 2 toward 1: each flank lifts the half-space under the
        // other, so the pair carries less than two independent P/2 patches at the
        // same approach. The uncoupled superposition is frozen at η = 2; the
        // first-order lift `u = Q/(π E* · 2 y0)` closes almost all of that gap — to
        // a few percent through the half-overlap regime and to well under 1% once
        // the flanks separate (where the compact-source approximation is exact).
        let ball = 4.0e-3;
        let tube = 1.04 * ball;
        let centre_radius = 15.0e-3;
        let material = Material::from_e_star(100.0e9);
        let load = 120.0;
        let config = Config {
            tolerance: 1.0e-8,
            max_iterations: 40_000,
        };

        let radii = GothicArchGroove::new(tube, centre_radius, 0.0).against_sphere(ball);
        let flank = HertzElliptic::new(
            radii.radius_x(),
            radii.radius_y(),
            load / 2.0,
            material.e_star(),
        );
        let b = flank.semi_axis_y();
        let ax = flank.semi_axis_x();
        let stiffness = GothicArchLaw::from_elliptic_flank(
            radii.radius_x(),
            radii.radius_y(),
            material.e_star(),
            0.4,
        )
        .stiffness();

        // (y0/b, tolerance): tight once the flanks separate, a few percent at the
        // half overlap where the point-load-at-d=2y0 model is weakest.
        let cases = [
            (0.5_f64, 0.09),
            (0.75, 0.06),
            (1.0, 0.035),
            (1.5, 0.02),
            (2.0, 0.02),
        ];
        let mut last_eta = 0.0_f64;
        for &(ratio, tol) in &cases {
            let y0 = ratio * b;
            let centre_offset = y0 * (tube - ball) / ball;
            let groove = GothicArchGroove::new(tube, centre_radius, centre_offset);
            let dx = ax / 8.0;
            let dy = b / 8.0;
            let nx = even_ceil(2.0 * 2.5 * ax / dx);
            let ny = even_ceil(2.0 * (y0 + 2.5 * b) / dy);
            let grid = Grid::new(nx, ny, dx, dy);
            let sol = sphere_in_gothic_arch(ball, groove, load, material, grid, config);
            assert!(
                sol.diagnostics().converged,
                "solver did not converge at y0/b={ratio}",
            );

            let delta = sol.approach();
            let eta_solver = sol.total_load() / (stiffness * delta.powf(1.5));

            let law = GothicArchLaw::from_elliptic_flank(
                radii.radius_x(),
                radii.radius_y(),
                material.e_star(),
                0.4,
            )
            .with_flank_coupling(material.e_star(), y0);
            let (q_plus, q_minus) = law.coupled_loads(delta, delta);
            let eta_law = (q_plus + q_minus) / (stiffness * delta.powf(1.5));

            // η lies strictly between the merged single arc (1) and two separated
            // flanks (2), and rises monotonically as the flanks pull apart.
            assert!(
                eta_solver > 1.0 && eta_solver < 2.0,
                "η out of (1, 2) at y0/b={ratio}: {eta_solver}",
            );
            assert!(
                eta_solver > last_eta,
                "η must rise as the flanks separate (y0/b={ratio})",
            );
            last_eta = eta_solver;

            // The coupled law tracks the solver to the per-point tolerance ...
            assert_relative(eta_law, eta_solver, tol, "coupled η vs solver");
            // ... and closes most of the gap the uncoupled η = 2 would leave.
            let gap_uncoupled = (2.0 - eta_solver).abs();
            let gap_coupled = (eta_law - eta_solver).abs();
            assert!(
                gap_coupled < 0.4 * gap_uncoupled,
                "coupling must close most of the η gap at y0/b={ratio}: \
                 coupled {gap_coupled:e} vs uncoupled {gap_uncoupled:e}",
            );
        }
    }

    #[test]
    fn gothic_coupling_captures_the_load_split() {
        // The directional check: under an asymmetric drive the load divides
        // unevenly between the flanks, and that split P_+:P_- is the force
        // direction. A lateral drive lowers the + well floor and raises the − one
        // (the half-space stand-in for nudging the ball toward one flank), so the
        // bare flank approaches become s_± = δ ± drive. The coupled law must
        // reproduce the solver's split — including the way the lift *sharpens* it
        // (the heavier flank presses its lighter neighbour down harder) — where the
        // first-order term holds: at the onset of overlap (y0 = b), the uncoupled
        // (s_+/s_-)^{3/2} is already ~20% low, and the coupling all but closes it.
        let ball = 4.0e-3;
        let tube = 1.04 * ball;
        let centre_radius = 15.0e-3;
        let material = Material::from_e_star(100.0e9);
        let load = 120.0;
        let config = Config {
            tolerance: 1.0e-9,
            max_iterations: 40_000,
        };

        let radii = GothicArchGroove::new(tube, centre_radius, 0.0).against_sphere(ball);
        let (radius_x, radius_y) = (radii.radius_x(), radii.radius_y());
        let flank = HertzElliptic::new(radius_x, radius_y, load / 2.0, material.e_star());
        let b = flank.semi_axis_y();
        let ax = flank.semi_axis_x();

        let y0 = b; // onset of overlap: the flank patches just meet at the centre
        let dx = ax / 6.0;
        let dy = b / 10.0;
        let nx = even_ceil(2.0 * 2.5 * ax / dx);
        let ny = even_ceil(2.0 * (y0 + 2.5 * b) / dy);
        let grid = Grid::new(nx, ny, dx, dy);
        let mid = ny / 2;
        let cell = grid.cell_area();
        let law = GothicArchLaw::from_elliptic_flank(radius_x, radius_y, material.e_star(), 0.4)
            .with_flank_coupling(material.e_star(), y0);

        // Solve one asymmetric drive and read the per-half loads the solver splits
        // the contact into (each half-plane of the groove centre is one flank).
        let split_at = |drive: f64| {
            let gap = grid.sample(|x, y| {
                let well_plus = (y - y0).powi(2) / (2.0 * radius_y) - drive;
                let well_minus = (y + y0).powi(2) / (2.0 * radius_y) + drive;
                x * x / (2.0 * radius_x) + well_plus.min(well_minus)
            });
            let sol = solve_sampled_gap(gap, material, load, grid.clone(), config);
            assert!(sol.diagnostics().converged, "split solve did not converge");
            let pressure = sol.pressure();
            let upper: f64 = pressure
                .indexed_iter()
                .filter(|&((_, j), _)| j >= mid)
                .map(|(_, &p)| p * cell)
                .sum();
            let lower: f64 = pressure
                .indexed_iter()
                .filter(|&((_, j), _)| j < mid)
                .map(|(_, &p)| p * cell)
                .sum();
            (upper / lower, sol.approach())
        };

        // A straight push splits the load evenly — the symmetric anchor.
        let (split_sym, _) = split_at(0.0);
        assert_relative(split_sym, 1.0, 0.02, "symmetric split is 1:1");

        // An asymmetric drive: the coupled law lands within a few percent of the
        // solver, while the uncoupled bare ratio is far short.
        let drive = 0.25 * 2.0e-6;
        let (split_solver, delta) = split_at(drive);
        let (q_plus, q_minus) = law.coupled_loads(delta + drive, delta - drive);
        let split_coupled = q_plus / q_minus;
        let split_uncoupled = ((delta + drive) / (delta - drive)).powf(1.5);

        assert!(
            split_solver > 2.0,
            "the drive must produce a clearly asymmetric split (got {split_solver})",
        );
        assert_relative(split_coupled, split_solver, 0.06, "coupled split vs solver");
        assert!(
            (split_uncoupled - split_solver).abs() > 0.12 * split_solver,
            "the uncoupled split must be visibly off ({split_uncoupled} vs {split_solver})",
        );
        assert!(
            (split_coupled - split_solver).abs() < (split_uncoupled - split_solver).abs(),
            "coupling must move the split toward the solver, not away",
        );
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
