//! A reduced, closed-form contact-pressure distribution for one Gothic-arch flank.
//!
//! The companion to [`GothicArchLaw`](crate::reduced::GothicArchLaw). That force
//! law collapses each flank to its *resultant* `Q` — the zeroth moment of the
//! pressure — which is all a frictionless normal contact needs. **Coulomb
//! friction needs the distribution itself**: the tangential traction is bounded
//! pointwise by `‖q‖ ≤ μ p(x, y)`, so the friction force and especially the
//! spin (drilling) moment are integrals of `μ p` over the contact patch, not
//! functions of the net load alone. A point-contact `F(δ)` cannot supply them.
//!
//! This module distils the validated elliptic-Hertz flank into a lightweight
//! `p(x, y)` the multibody inner loop can evaluate in a few `powf`s, alongside
//! the force law, with no FFT solve.
//!
//! # The model
//!
//! A single Hertzian flank carrying load `Q` is the semi-ellipsoidal pressure
//!
//! ```text
//! p(x, y) = p₀ √(1 − (x/a_x)² − (y/a_y)²),   p₀ = 3Q / (2π a_x a_y),
//! ```
//!
//! over the contact ellipse of semi-axes `a_x, a_y` (`x` circumferential, `y`
//! meridional; the patch is centred on the flank). Two facts make this a *closed
//! form in the load*, so it costs only the per-flank `Q` the force law already
//! returns:
//!
//! - **The shape is load-independent.** The contact eccentricity `e` is fixed by
//!   the curvature ratio `R_x / R_y` alone (the same transcendental relation the
//!   elliptic-Hertz reference solves), so only the *size* changes with load.
//! - **The size scales as the Hertzian cube root.** `a_x, a_y ∝ Q^{1/3}` and
//!   hence `p₀ ∝ Q^{1/3}`. Calibrating the semi-axes once at unit load (from the
//!   flank's relative radii and modulus, [`FlankPressure::from_elliptic_flank`])
//!   gives `a(Q) = a₁ Q^{1/3}` for any load.
//!
//! # The friction payoff: the spin moment
//!
//! The friction quantity the *distribution* unlocks — and the force law cannot —
//! is the **spin (drilling) moment**: the limiting friction torque about the
//! contact normal when the patch pivots, `M = μ ∫ p ρ dA` with `ρ` the distance
//! from the patch centroid. For the semi-ellipsoidal pressure this integral has a
//! closed form,
//!
//! ```text
//! M = (3/8) μ Q a E(e),
//! ```
//!
//! with `a` the **major** semi-axis, `e` the contact eccentricity, and `E` the
//! complete elliptic integral of the second kind (the `4 a E(e)` ellipse
//! perimeter falls straight out of the angular part of `∫ p ρ dA`). It reduces to
//! the textbook circular result `M = (3π/16) μ Q a` as `e → 0` (`E(0) = π/2`).
//! The effective friction lever arm is `M / (μ Q) = (3/8) a E(e)`, the
//! [`FlankPressure::spin_radius`]. A conformal bearing flank is strongly elliptic
//! (`a/b > 5`), where the circular-radius stand-in understates this moment by tens
//! of percent — the distribution genuinely matters.
//!
//! # Two flanks, and the C¹ handover
//!
//! Pair this per-flank distribution with the force law's per-flank loads: a
//! separated Gothic contact is two of these semi-ellipsoids, centred at `y = ±y0`
//! and scaled to the loads `Q_±` from
//! [`GothicArchLaw::flank_loads`](crate::reduced::GothicArchLaw::flank_loads). As a
//! flank unloads its patch shrinks (`a ∝ Q^{1/3} → 0`), its peak falls, and its
//! spin moment vanishes (`M ∝ Q^{4/3} → 0`), so the pressure picture follows the
//! force law's `C¹` two-to-one handover without a seam. The closed form is exact
//! per flank where the two patches are separate; in the half-overlap regime the
//! flanks interact through the elastic field, the same caveat the force law
//! carries, and the patch built from the (coupled) `Q` is the first-order stand-in.

use core::f64::consts::PI;

use crate::validation::{complete_elliptic_integrals, HertzElliptic};

/// A reduced, closed-form elliptic-Hertz pressure distribution for one flank.
///
/// A lightweight `p(x, y)` (see the [module docs](self)) built from one flank's
/// contact ellipse, the pressure companion to
/// [`GothicArchLaw`](crate::reduced::GothicArchLaw). Evaluating the field is a
/// couple of `powf`s, and it carries the Coulomb-friction quantities the
/// distribution unlocks — the local traction bound `μ p` and the closed-form spin
/// (drilling) moment — for a multibody inner loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlankPressure {
    /// Circumferential (`x`) contact semi-axis at unit load; `a_x(Q) = a_x₁ Q^{1/3}`.
    semi_axis_x_unit: f64,
    /// Meridional (`y`) contact semi-axis at unit load; `a_y(Q) = a_y₁ Q^{1/3}`.
    semi_axis_y_unit: f64,
    /// Peak pressure at unit load; `p₀(Q) = p₀₁ Q^{1/3}`.
    peak_pressure_unit: f64,
    /// Contact-ellipse eccentricity `e = √(1 − (b/a)²)` (load-independent).
    eccentricity: f64,
    /// `E(e)`, the complete elliptic integral of the second kind, precomputed for
    /// the spin moment `M = (3/8) μ Q a E(e)`.
    spin_integral: f64,
}

impl FlankPressure {
    /// Builds the distribution from a known contact ellipse at a reference load.
    ///
    /// `semi_axis_x`, `semi_axis_y` are the measured (or otherwise known) contact
    /// semi-axes at `reference_load`; since Hertzian semi-axes scale as `Q^{1/3}`,
    /// the unit-load shape is read off by dividing out `reference_load^{1/3}`.
    /// Prefer [`FlankPressure::from_elliptic_flank`], which derives the ellipse
    /// from the flank geometry and material.
    ///
    /// # Panics
    /// Panics if any argument is not strictly positive and finite.
    #[must_use]
    pub fn new(semi_axis_x: f64, semi_axis_y: f64, reference_load: f64) -> Self {
        assert!(
            semi_axis_x > 0.0 && semi_axis_x.is_finite(),
            "contact semi-axis a_x must be positive and finite",
        );
        assert!(
            semi_axis_y > 0.0 && semi_axis_y.is_finite(),
            "contact semi-axis a_y must be positive and finite",
        );
        assert!(
            reference_load > 0.0 && reference_load.is_finite(),
            "reference load must be positive and finite",
        );
        // Hertzian size scaling: a ∝ Q^{1/3}, so the unit-load shape is the patch
        // divided by the cube root of its reference load.
        let scale = reference_load.cbrt();
        let semi_axis_x_unit = semi_axis_x / scale;
        let semi_axis_y_unit = semi_axis_y / scale;
        let major = semi_axis_x_unit.max(semi_axis_y_unit);
        let minor = semi_axis_x_unit.min(semi_axis_y_unit);
        let axis_ratio = minor / major;
        let eccentricity = (1.0 - axis_ratio * axis_ratio).max(0.0).sqrt();
        let (_, spin_integral) = complete_elliptic_integrals(eccentricity);
        // Peak pressure at unit load: p₀ = 3Q / (2π a_x a_y) with Q = 1.
        let peak_pressure_unit = 3.0 / (2.0 * PI * semi_axis_x_unit * semi_axis_y_unit);
        Self {
            semi_axis_x_unit,
            semi_axis_y_unit,
            peak_pressure_unit,
            eccentricity,
            spin_integral,
        }
    }

    /// Calibrates the distribution from one flank's elliptic-Hertz contact.
    ///
    /// The contact ellipse is read off the analytic elliptic-Hertz solution for
    /// the flank's relative radii `radius_x`, `radius_y` and modulus `e_star`.
    /// Since the semi-axes scale as `Q^{1/3}` the shape is evaluated once at unit
    /// load, exactly as [`GothicArchLaw::from_elliptic_flank`] calibrates the
    /// force constant `K` — so the two laws describe the *same* validated flank.
    ///
    /// [`GothicArchLaw::from_elliptic_flank`]: crate::reduced::GothicArchLaw::from_elliptic_flank
    ///
    /// # Panics
    /// Panics if any radius or `e_star` is not strictly positive and finite.
    #[must_use]
    pub fn from_elliptic_flank(radius_x: f64, radius_y: f64, e_star: f64) -> Self {
        let reference = HertzElliptic::new(radius_x, radius_y, 1.0, e_star);
        Self::new(reference.semi_axis_x(), reference.semi_axis_y(), 1.0)
    }

    /// The contact semi-axes `(a_x, a_y)` at load `Q`, scaling as the cube root `Q^{1/3}`.
    ///
    /// A non-positive load means the flank is separated, so the patch has zero
    /// extent.
    #[must_use]
    pub fn semi_axes(&self, load: f64) -> (f64, f64) {
        let scale = load.max(0.0).cbrt();
        (self.semi_axis_x_unit * scale, self.semi_axis_y_unit * scale)
    }

    /// The peak (central) pressure `p₀ = 3Q / (2π a_x a_y)` at load `Q`.
    ///
    /// Scales as the cube root `Q^{1/3}`; zero for a separated flank (non-positive load).
    #[must_use]
    pub fn peak_pressure(&self, load: f64) -> f64 {
        self.peak_pressure_unit * load.max(0.0).cbrt()
    }

    /// The contact-ellipse eccentricity `e = √(1 − (b/a)²)` (load-independent).
    #[must_use]
    pub const fn eccentricity(&self) -> f64 {
        self.eccentricity
    }

    /// The local contact pressure `p(x, y)` at load `Q`, centred on the patch.
    ///
    /// The semi-ellipsoidal Hertz field
    /// `p₀ √(1 − (x/a_x)² − (y/a_y)²)` inside the contact ellipse and `0` outside
    /// it (no adhesion). `x` is circumferential, `y` meridional, both measured
    /// from the flank's own centre. The limiting Coulomb traction at this point is
    /// `friction · pressure_at(..)`.
    #[must_use]
    pub fn pressure_at(&self, load: f64, x: f64, y: f64) -> f64 {
        if load <= 0.0 {
            return 0.0;
        }
        let (semi_axis_x, semi_axis_y) = self.semi_axes(load);
        let rx = x / semi_axis_x;
        let ry = y / semi_axis_y;
        let radial = rx * rx + ry * ry;
        if radial >= 1.0 {
            0.0
        } else {
            self.peak_pressure(load) * (1.0 - radial).sqrt()
        }
    }

    /// The effective spin (drilling) friction radius `(3/8) a E(e)` at load `Q`.
    ///
    /// The lever arm of the spin moment: `M = μ Q · spin_radius`. It is the
    /// pressure-weighted mean pivot distance over the patch, `a` the major
    /// semi-axis, and like the patch it grows as `Q^{1/3}`. Reduces to the
    /// circular `3π/16 · a` as the contact rounds out (`e → 0`).
    #[must_use]
    pub fn spin_radius(&self, load: f64) -> f64 {
        let (semi_axis_x, semi_axis_y) = self.semi_axes(load);
        0.375 * semi_axis_x.max(semi_axis_y) * self.spin_integral
    }

    /// The Coulomb spin (drilling) moment `M = μ ∫ p ρ dA = (3/8) μ Q a E(e)`.
    ///
    /// The limiting friction torque about the contact normal when the patch
    /// pivots (full sliding), with `friction` the coefficient `μ`, `a` the major
    /// semi-axis and `E(e)` the complete elliptic integral of the second kind
    /// (see the [module docs](self#the-friction-payoff-the-spin-moment)). This is
    /// the friction quantity the net force `F(δ)` cannot provide — it is the first
    /// moment of the pressure, not its resultant. Zero for a separated flank.
    ///
    /// # Panics
    /// Panics if `friction` is negative or not finite.
    #[must_use]
    pub fn spin_moment(&self, load: f64, friction: f64) -> f64 {
        assert!(
            friction >= 0.0 && friction.is_finite(),
            "friction coefficient must be non-negative and finite",
        );
        friction * load.max(0.0) * self.spin_radius(load)
    }
}

#[cfg(test)]
mod tests {
    use super::FlankPressure;
    use crate::validation::HertzElliptic;
    use core::f64::consts::PI;

    fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
        let scale = expected.abs().max(1.0e-300);
        let rel_err = (actual - expected).abs() / scale;
        assert!(
            rel_err <= tolerance,
            "{what}: actual={actual:e} expected={expected:e} rel_err={rel_err:e} (> {tolerance:e})",
        );
    }

    // The gallery's conformal Gothic flank: a strongly elliptic patch.
    const RADIUS_X: f64 = 1.6e-3;
    const RADIUS_Y: f64 = 26.0e-3;
    const E_STAR: f64 = 100.0e9;

    fn sample_flank() -> FlankPressure {
        FlankPressure::from_elliptic_flank(RADIUS_X, RADIUS_Y, E_STAR)
    }

    #[test]
    fn pressure_field_matches_elliptic_hertz_at_any_load() {
        // The reduced field is, by construction, the elliptic-Hertz semi-ellipsoid
        // at the flank's own load — so it must agree pointwise with the analytic
        // reference solved directly at that load, at two unrelated loads.
        let flank = sample_flank();
        for &load in &[8.0_f64, 230.0] {
            let reference = HertzElliptic::new(RADIUS_X, RADIUS_Y, load, E_STAR);
            let (a_x, a_y) = flank.semi_axes(load);
            assert_close(a_x, reference.semi_axis_x(), 1.0e-12, "semi-axis x");
            assert_close(a_y, reference.semi_axis_y(), 1.0e-12, "semi-axis y");
            assert_close(
                flank.peak_pressure(load),
                reference.max_pressure(),
                1.0e-12,
                "peak pressure",
            );
            // Sample the field on a grid spanning the patch (and outside it).
            for i in -3..=3 {
                for j in -3..=3 {
                    let x = f64::from(i) * 0.4 * a_x;
                    let y = f64::from(j) * 0.4 * a_y;
                    assert_close(
                        flank.pressure_at(load, x, y),
                        reference.pressure_at(x, y),
                        1.0e-12,
                        "pressure field",
                    );
                }
            }
        }
    }

    #[test]
    fn pressure_integrates_to_the_flank_load() {
        // The zeroth moment is the load itself: the semi-ellipsoid integrates to
        // (2/3) π a_x a_y p₀ = Q. This is the consistency tie to the force law's Q.
        let flank = sample_flank();
        for &load in &[3.0_f64, 47.0, 500.0] {
            let (a_x, a_y) = flank.semi_axes(load);
            let integrated = 2.0 / 3.0 * PI * a_x * a_y * flank.peak_pressure(load);
            assert_close(integrated, load, 1.0e-12, "integrated load");
        }
    }

    #[test]
    fn size_and_peak_scale_as_the_hertzian_cube_root() {
        // a, p₀ ∝ Q^{1/3}: an eightfold load doubles every linear measure.
        let flank = sample_flank();
        let (ax1, ay1) = flank.semi_axes(20.0);
        let (ax8, ay8) = flank.semi_axes(160.0);
        assert_close(ax8 / ax1, 2.0, 1.0e-12, "a_x cube-root scaling");
        assert_close(ay8 / ay1, 2.0, 1.0e-12, "a_y cube-root scaling");
        assert_close(
            flank.peak_pressure(160.0) / flank.peak_pressure(20.0),
            2.0,
            1.0e-12,
            "p0 cube-root scaling",
        );
    }

    // Midpoint quadrature of μ ∫ p ρ dA over the contact ellipse, the first moment
    // of pressure about the patch centre. Polar in the unit disc mapped to the
    // ellipse: x = a_x r cosθ, y = a_y r sinθ, dA = a_x a_y r dr dθ, ρ = √(x²+y²).
    #[allow(
        clippy::cast_precision_loss,
        reason = "the node index and count are tiny relative to f64's integer range"
    )]
    fn spin_moment_quadrature(flank: &FlankPressure, load: f64, friction: f64) -> f64 {
        let (a_x, a_y) = flank.semi_axes(load);
        let (n_r, n_theta) = (2000_usize, 2000_usize);
        let mut integral = 0.0;
        for ir in 0..n_r {
            let r = (ir as f64 + 0.5) / n_r as f64;
            for it in 0..n_theta {
                let theta = 2.0 * PI * (it as f64 + 0.5) / n_theta as f64;
                let x = a_x * r * theta.cos();
                let y = a_y * r * theta.sin();
                integral += flank.pressure_at(load, x, y) * x.hypot(y) * a_x * a_y * r;
            }
        }
        let cell = (1.0 / n_r as f64) * (2.0 * PI / n_theta as f64);
        friction * integral * cell
    }

    #[test]
    fn spin_moment_matches_the_pressure_quadrature() {
        // The headline friction check: the closed form (3/8) μ Q a E(e) equals a
        // direct quadrature of μ ∫ p ρ dA over the contact ellipse.
        let flank = sample_flank();
        let (load, friction) = (120.0, 0.1);
        assert_close(
            flank.spin_moment(load, friction),
            spin_moment_quadrature(&flank, load, friction),
            2.0e-3,
            "spin moment vs quadrature",
        );
    }

    #[test]
    fn spin_moment_reduces_to_the_circular_result() {
        // For a circular contact (equal relative radii) the spin moment must be the
        // textbook 3π/16 μ Q a, the e → 0 limit of (3/8) μ Q a E(e) (E(0) = π/2).
        let flank = FlankPressure::from_elliptic_flank(5.0e-3, 5.0e-3, E_STAR);
        assert!(flank.eccentricity().abs() <= 1.0e-9, "contact is circular");
        let load = 75.0;
        let friction = 0.2;
        let (a, _) = flank.semi_axes(load);
        let expected = 3.0 * PI / 16.0 * friction * load * a;
        assert_close(
            flank.spin_moment(load, friction),
            expected,
            1.0e-9,
            "circular spin moment",
        );
    }

    #[test]
    fn spin_moment_scales_as_load_to_the_four_thirds() {
        // M = (3/8) μ Q a E(e) with a ∝ Q^{1/3}, so M ∝ Q^{4/3}: an eightfold load
        // raises the moment sixteenfold, and the lever arm (spin radius) doubles.
        let flank = sample_flank();
        let friction = 0.15;
        assert_close(
            flank.spin_moment(80.0, friction) / flank.spin_moment(10.0, friction),
            8.0_f64.powf(4.0 / 3.0),
            1.0e-12,
            "spin moment Q^{4/3} scaling",
        );
        assert_close(
            flank.spin_radius(80.0) / flank.spin_radius(10.0),
            2.0,
            1.0e-12,
            "spin radius cube-root scaling",
        );
        // The lever arm is the moment per unit (μ Q).
        assert_close(
            flank.spin_moment(80.0, friction),
            friction * 80.0 * flank.spin_radius(80.0),
            1.0e-12,
            "moment is μ Q times the lever arm",
        );
    }

    #[test]
    fn a_strongly_elliptic_flank_beats_the_circular_radius_stand_in() {
        // The conformal flank is strongly elliptic, so its spin radius (3/8) a E(e)
        // departs markedly from the circular 3π/16 · a_eq of an equal-area disc —
        // the distribution's shape, not just its size, drives the drilling moment.
        let flank = sample_flank();
        assert!(
            flank.eccentricity() > 0.97,
            "the conformal flank must be strongly elliptic: e={}",
            flank.eccentricity(),
        );
        let load = 100.0;
        let (a_x, a_y) = flank.semi_axes(load);
        let a_equiv = (a_x * a_y).sqrt();
        let circular = 3.0 * PI / 16.0 * a_equiv;
        let elliptic = flank.spin_radius(load);
        assert!(
            (elliptic - circular).abs() > 0.2 * circular,
            "the elliptic lever arm must differ from the circular stand-in by >20%: \
             elliptic={elliptic:e} circular={circular:e}",
        );
    }

    #[test]
    fn a_separated_flank_carries_no_pressure() {
        // A non-positive load is a lifted-off flank: zero field, zero patch, zero
        // moment — no adhesion, mirroring the force law.
        let flank = sample_flank();
        assert!(flank.pressure_at(0.0, 0.0, 0.0).abs() <= 1.0e-300);
        assert!(flank.pressure_at(-1.0, 0.0, 0.0).abs() <= 1.0e-300);
        assert_eq!(flank.semi_axes(0.0), (0.0, 0.0));
        assert!(flank.peak_pressure(0.0).abs() <= 1.0e-300);
        assert!(flank.spin_moment(0.0, 0.3).abs() <= 1.0e-300);
    }

    #[test]
    fn from_elliptic_flank_matches_a_hand_built_ellipse() {
        // from_elliptic_flank is new(..) fed the unit-load Hertz semi-axes: the two
        // constructors must agree.
        let reference = HertzElliptic::new(RADIUS_X, RADIUS_Y, 1.0, E_STAR);
        let built = FlankPressure::new(reference.semi_axis_x(), reference.semi_axis_y(), 1.0);
        let calibrated = FlankPressure::from_elliptic_flank(RADIUS_X, RADIUS_Y, E_STAR);
        assert_eq!(built, calibrated);
    }

    #[test]
    fn new_is_reference_load_invariant() {
        // The unit-load shape is recovered whatever reference load it is read at,
        // because the semi-axes are divided by the matching Q^{1/3}.
        let flank = sample_flank();
        let (a_x, a_y) = flank.semi_axes(64.0);
        let rebuilt = FlankPressure::new(a_x, a_y, 64.0);
        assert_close(
            rebuilt.peak_pressure(1.0),
            flank.peak_pressure(1.0),
            1.0e-12,
            "unit peak pressure",
        );
        assert_close(
            rebuilt.spin_radius(1.0),
            flank.spin_radius(1.0),
            1.0e-12,
            "unit spin radius",
        );
    }

    #[test]
    #[should_panic(expected = "friction coefficient")]
    fn spin_moment_rejects_a_negative_friction() {
        let _ = sample_flank().spin_moment(50.0, -0.1);
    }

    #[test]
    #[should_panic(expected = "semi-axis a_x")]
    fn new_rejects_a_non_positive_semi_axis() {
        let _ = FlankPressure::new(0.0, 1.0e-3, 1.0);
    }
}
