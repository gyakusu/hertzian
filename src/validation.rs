//! Analytic Hertz solutions used to validate the solver.
//!
//! Closed-form references for the two validation regimes of the design:
//! - [`HertzCircular`] — circular (axisymmetric) contact (§5.1).
//! - [`HertzElliptic`] — elliptic (non-axisymmetric) contact (§5.2), the
//!   torus-on-sphere benchmark.
//!
//! The elliptic case has no elementary closed form: the contact eccentricity is
//! fixed by a transcendental relation between the principal relative curvatures
//! and complete elliptic integrals `K(e)`, `E(e)`, which are evaluated here by
//! the arithmetic–geometric mean (AGM). The remaining size, pressure and
//! approach then follow in closed form. Every formula reduces to [`HertzCircular`]
//! as the eccentricity tends to zero, which the tests check explicitly.

use core::f64::consts::PI;

/// The analytic Hertz solution for a circular contact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HertzCircular {
    contact_radius: f64,
    max_pressure: f64,
    approach: f64,
}

impl HertzCircular {
    /// Computes the solution for effective radius `R`, load `P`, modulus `E*`.
    ///
    /// # Panics
    /// Panics if any argument is not strictly positive.
    #[must_use]
    pub fn new(effective_radius: f64, load: f64, e_star: f64) -> Self {
        assert!(
            effective_radius > 0.0 && load > 0.0 && e_star > 0.0,
            "Hertz inputs must be positive",
        );
        let contact_radius = (3.0 * load * effective_radius / (4.0 * e_star)).cbrt();
        let max_pressure = 3.0 * load / (2.0 * PI * contact_radius * contact_radius);
        let approach = contact_radius * contact_radius / effective_radius;
        Self {
            contact_radius,
            max_pressure,
            approach,
        }
    }

    /// The combined radius `R` from two radii, `1/R = 1/R1 + 1/R2`.
    #[must_use]
    pub fn combined_radius(radius_1: f64, radius_2: f64) -> f64 {
        1.0 / (1.0 / radius_1 + 1.0 / radius_2)
    }

    /// Contact radius `a`.
    #[must_use]
    pub const fn contact_radius(&self) -> f64 {
        self.contact_radius
    }

    /// Peak pressure `p0`.
    #[must_use]
    pub const fn max_pressure(&self) -> f64 {
        self.max_pressure
    }

    /// Rigid-body approach `delta`.
    #[must_use]
    pub const fn approach(&self) -> f64 {
        self.approach
    }

    /// Pressure at radius `r`: `p0 sqrt(1 - (r/a)^2)`, and `0` for `r >= a`.
    #[must_use]
    pub fn pressure_at(&self, r: f64) -> f64 {
        if r >= self.contact_radius {
            0.0
        } else {
            let ratio = r / self.contact_radius;
            self.max_pressure * (1.0 - ratio * ratio).sqrt()
        }
    }
}

/// The analytic Hertz solution for an elliptic contact.
///
/// Built from the two principal *relative* radii of curvature `R_x`, `R_y` of
/// the gap `h = x^2 / (2 R_x) + y^2 / (2 R_y)`. The contact ellipse is elongated
/// along whichever axis has the larger radius (smaller curvature): with the
/// torus-on-sphere benchmark `R_x` (circumferential) `> R_y` (meridional), so the
/// contact is longer along `x`.
///
/// Internally the major axis `a` lies along the larger-radius direction and the
/// minor axis `b` along the smaller; the per-axis accessors map these back onto
/// the grid's `x`/`y` so they can be compared with the solver directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HertzElliptic {
    semi_axis_x: f64,
    semi_axis_y: f64,
    max_pressure: f64,
    approach: f64,
    eccentricity: f64,
}

impl HertzElliptic {
    /// Computes the solution for principal relative radii `R_x`, `R_y`, load
    /// `P`, and modulus `E*`.
    ///
    /// The radii may be given in either order; the contact simply elongates
    /// along the larger-radius axis. Equal radii recover [`HertzCircular`].
    ///
    /// # Panics
    /// Panics if any argument is not strictly positive and finite.
    #[must_use]
    pub fn new(radius_x: f64, radius_y: f64, load: f64, e_star: f64) -> Self {
        assert!(
            radius_x > 0.0
                && radius_y > 0.0
                && load > 0.0
                && e_star > 0.0
                && radius_x.is_finite()
                && radius_y.is_finite()
                && load.is_finite()
                && e_star.is_finite(),
            "elliptic Hertz inputs must be positive and finite",
        );

        let x_is_major = radius_x >= radius_y;
        let radius_major = radius_x.max(radius_y);
        let radius_minor = radius_x.min(radius_y);

        let (semi_major, semi_minor, max_pressure, approach, eccentricity) =
            solve_elliptic(radius_major, radius_minor, load, e_star);

        let (semi_axis_x, semi_axis_y) = if x_is_major {
            (semi_major, semi_minor)
        } else {
            (semi_minor, semi_major)
        };

        Self {
            semi_axis_x,
            semi_axis_y,
            max_pressure,
            approach,
            eccentricity,
        }
    }

    /// Contact semi-axis along `x` (grid axis 0).
    #[must_use]
    pub const fn semi_axis_x(&self) -> f64 {
        self.semi_axis_x
    }

    /// Contact semi-axis along `y` (grid axis 1).
    #[must_use]
    pub const fn semi_axis_y(&self) -> f64 {
        self.semi_axis_y
    }

    /// Major (longer) contact semi-axis `a`.
    #[must_use]
    pub const fn semi_major(&self) -> f64 {
        self.semi_axis_x.max(self.semi_axis_y)
    }

    /// Minor (shorter) contact semi-axis `b`.
    #[must_use]
    pub const fn semi_minor(&self) -> f64 {
        self.semi_axis_x.min(self.semi_axis_y)
    }

    /// Ellipticity `a / b >= 1`.
    #[must_use]
    pub fn ellipticity(&self) -> f64 {
        self.semi_major() / self.semi_minor()
    }

    /// Eccentricity `e = sqrt(1 - (b/a)^2)` of the contact ellipse.
    #[must_use]
    pub const fn eccentricity(&self) -> f64 {
        self.eccentricity
    }

    /// Peak pressure `p0`.
    #[must_use]
    pub const fn max_pressure(&self) -> f64 {
        self.max_pressure
    }

    /// Rigid-body approach `delta`.
    #[must_use]
    pub const fn approach(&self) -> f64 {
        self.approach
    }

    /// Pressure at `(x, y)`: `p0 sqrt(1 - (x/a_x)^2 - (y/a_y)^2)`, and `0`
    /// outside the contact ellipse.
    #[must_use]
    pub fn pressure_at(&self, x: f64, y: f64) -> f64 {
        let rx = x / self.semi_axis_x;
        let ry = y / self.semi_axis_y;
        let radial = rx * rx + ry * ry;
        if radial >= 1.0 {
            0.0
        } else {
            self.max_pressure * (1.0 - radial).sqrt()
        }
    }
}

/// Sneddon's analytic solution for a rigid cone on an elastic half-space.
///
/// The arbitrary-shape (non-Hertzian) validation reference (design roadmap §3).
/// For a conical gap `h(r) = m r` of small surface slope `m`, pressed by load
/// `P` into a half-space of equivalent modulus `E*` (Sneddon, 1965):
///
/// ```text
/// a  = sqrt(2 P / (π E* m)),   δ = (π/2) m a,   p(r) = (E* m / 2) arccosh(a/r),
/// ```
///
/// so `P = (π/2) E* m a²` and the mean pressure is the constant `E* m / 2`.
/// Unlike Hertz the pressure diverges logarithmically at the apex, so the
/// grid solver is validated on the contact radius, approach and load rather than
/// the (mesh-dependent) peak pressure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SneddonCone {
    contact_radius: f64,
    approach: f64,
    mean_pressure: f64,
    e_star: f64,
    slope: f64,
}

impl SneddonCone {
    /// Computes the solution for surface slope `m`, load `P`, modulus `E*`.
    ///
    /// # Panics
    /// Panics if any argument is not strictly positive and finite.
    #[must_use]
    pub fn new(slope: f64, load: f64, e_star: f64) -> Self {
        assert!(
            slope > 0.0
                && load > 0.0
                && e_star > 0.0
                && slope.is_finite()
                && load.is_finite()
                && e_star.is_finite(),
            "Sneddon cone inputs must be positive and finite",
        );
        let contact_radius = (2.0 * load / (PI * e_star * slope)).sqrt();
        let approach = 0.5 * PI * slope * contact_radius;
        let mean_pressure = 0.5 * e_star * slope;
        Self {
            contact_radius,
            approach,
            mean_pressure,
            e_star,
            slope,
        }
    }

    /// Contact radius `a`.
    #[must_use]
    pub const fn contact_radius(&self) -> f64 {
        self.contact_radius
    }

    /// Rigid-body approach `δ = (π/2) m a`.
    #[must_use]
    pub const fn approach(&self) -> f64 {
        self.approach
    }

    /// Mean contact pressure `P / (π a²) = E* m / 2`.
    #[must_use]
    pub const fn mean_pressure(&self) -> f64 {
        self.mean_pressure
    }

    /// Total normal load `P = (π/2) E* m a²`.
    #[must_use]
    pub fn load(&self) -> f64 {
        0.5 * PI * self.e_star * self.slope * self.contact_radius * self.contact_radius
    }

    /// Pressure at radius `r`: `(E* m / 2) arccosh(a/r)`, and `0` for `r >= a`.
    ///
    /// Diverges as `r -> 0` (the apex singularity), so it is finite only for
    /// `0 < r < a`.
    #[must_use]
    pub fn pressure_at(&self, r: f64) -> f64 {
        if r >= self.contact_radius || r <= 0.0 {
            0.0
        } else {
            0.5 * self.e_star * self.slope * (self.contact_radius / r).acosh()
        }
    }
}

/// Solves the elliptic Hertz problem for ordered radii `R_major >= R_minor`.
///
/// Returns `(a, b, p0, delta, e)` with `a` along the major (larger-radius) axis.
/// The eccentricity satisfies the curvature relation
/// `[E/(1 - e^2) - K] / [K - E] = R_major / R_minor`; the remaining quantities
/// follow from the Hertzian ellipsoidal pressure `p = p0 sqrt(1 - x^2/a^2 -
/// y^2/b^2)` via `a^3 = 3 P R_major (K - E) / (pi e^2 E*)`, `b = a sqrt(1 - e^2)`,
/// `p0 = 3 P / (2 pi a b)` and `delta = a^2 K e^2 / (2 R_major (K - E))`. The
/// near-circular branch is taken analytically to avoid the `0/0` at `e = 0`.
fn solve_elliptic(
    radius_major: f64,
    radius_minor: f64,
    load: f64,
    e_star: f64,
) -> (f64, f64, f64, f64, f64) {
    let curvature_ratio = radius_major / radius_minor;

    // Within this band of the ratio the contact is circular to well under the
    // grid-discretisation error, and the e^2 denominators below are ill-posed.
    if curvature_ratio - 1.0 <= 1.0e-12 {
        let a = (3.0 * load * radius_major / (4.0 * e_star)).cbrt();
        let p0 = 3.0 * load / (2.0 * PI * a * a);
        let delta = a * a / radius_major;
        return (a, a, p0, delta, 0.0);
    }

    let eccentricity = solve_eccentricity(curvature_ratio);
    let (big_k, big_e) = complete_elliptic_integrals(eccentricity);
    let e_sq = eccentricity * eccentricity;
    let axis_ratio = (1.0 - e_sq).sqrt(); // b / a

    let semi_major = (3.0 * load * radius_major * (big_k - big_e) / (PI * e_sq * e_star)).cbrt();
    let semi_minor = semi_major * axis_ratio;
    let max_pressure = 3.0 * load / (2.0 * PI * semi_major * semi_minor);
    let delta = semi_major * semi_major * big_k * e_sq / (2.0 * radius_major * (big_k - big_e));

    (semi_major, semi_minor, max_pressure, delta, eccentricity)
}

/// Solves `[E/(1 - e^2) - K] / [K - E] = ratio` for the eccentricity `e`.
///
/// The left-hand side equals `R_major / R_minor`; it grows monotonically from
/// `1` at `e = 0` to `+infinity` as `e -> 1`, so a bisection on `e in (0, 1)`
/// converges for any `ratio > 1`.
fn solve_eccentricity(ratio: f64) -> f64 {
    let mut low = 0.0_f64;
    let mut high = 1.0 - 1.0e-12;
    // ~60 halvings drive the bracket below f64 precision; 100 is a safe margin.
    for _ in 0..100 {
        let mid = 0.5 * (low + high);
        if elliptic_curvature_ratio(mid) < ratio {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// The curvature ratio `R_major / R_minor` implied by an eccentricity `e > 0`.
fn elliptic_curvature_ratio(eccentricity: f64) -> f64 {
    let (big_k, big_e) = complete_elliptic_integrals(eccentricity);
    let axis_ratio_sq = 1.0 - eccentricity * eccentricity; // (b/a)^2
    (big_e / axis_ratio_sq - big_k) / (big_k - big_e)
}

/// Complete elliptic integrals `K(k)` and `E(k)` of modulus `k` in `[0, 1)`.
///
/// Evaluated by the arithmetic–geometric mean: `K = pi / (2 M(1, k'))` with
/// `k' = sqrt(1 - k^2)`, and `E = K (1 - sum_n 2^(n-1) c_n^2)` accumulated over
/// the same AGM iteration (Abramowitz & Stegun 17.6). The iteration converges
/// quadratically, so a handful of steps reach full `f64` precision.
fn complete_elliptic_integrals(modulus: f64) -> (f64, f64) {
    let mut a = 1.0_f64;
    let mut b = (1.0 - modulus * modulus).max(0.0).sqrt();
    let mut sum = 0.5 * modulus * modulus; // n = 0: 2^(-1) c_0^2, c_0 = k
    let mut two_pow = 1.0_f64; // 2^n for the c_(n+1) term, starting at 2^0

    for _ in 0..60 {
        let a_next = 0.5 * (a + b);
        let b_next = (a * b).sqrt();
        let c_next = 0.5 * (a - b);
        sum += two_pow * c_next * c_next;
        two_pow *= 2.0;
        a = a_next;
        b = b_next;
        if c_next.abs() <= f64::EPSILON * a_next {
            break;
        }
    }

    let big_k = PI / (2.0 * a);
    let big_e = big_k * (1.0 - sum);
    (big_k, big_e)
}

#[cfg(test)]
mod tests {
    use super::{
        complete_elliptic_integrals, elliptic_curvature_ratio, HertzCircular, HertzElliptic,
        SneddonCone,
    };
    use core::f64::consts::PI;

    fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
        let rel_err = (actual - expected).abs() / expected.abs();
        assert!(
            rel_err <= tolerance,
            "{what}: actual={actual:e} expected={expected:e} rel_err={rel_err:e} (> {tolerance:e})",
        );
    }

    #[test]
    fn combined_radius_handles_a_flat() {
        let radius = 12.0e-3;
        let combined = HertzCircular::combined_radius(radius, f64::INFINITY);
        assert!((combined - radius).abs() <= 1e-12 * radius);
    }

    #[test]
    fn hertz_relations_are_self_consistent() {
        // p0 = 2 E* a / (pi R) follows from the three closed forms.
        let radius = 10.0e-3;
        let e_star = 70.0e9;
        let hertz = HertzCircular::new(radius, 50.0, e_star);
        let a = hertz.contact_radius();
        let expected_p0 = 2.0 * e_star * a / (PI * radius);
        assert!((hertz.max_pressure() - expected_p0).abs() <= 1e-6 * expected_p0);
        assert!((hertz.approach() - a * a / radius).abs() <= 1e-12 * hertz.approach());
    }

    #[test]
    fn elliptic_integrals_match_known_values() {
        // K(0) = E(0) = pi/2.
        let (k0, e0) = complete_elliptic_integrals(0.0);
        assert_close(k0, PI / 2.0, 1e-15, "K(0)");
        assert_close(e0, PI / 2.0, 1e-15, "E(0)");

        // Modulus k = 1/sqrt(2): K = Gamma(1/4)^2 / (4 sqrt(pi)) and the
        // companion E are standard reference values.
        let (k, e) = complete_elliptic_integrals(0.5_f64.sqrt());
        assert_close(k, 1.854_074_677_301_372, 1e-13, "K(1/sqrt2)");
        assert_close(e, 1.350_643_881_047_676, 1e-13, "E(1/sqrt2)");
    }

    #[test]
    fn elliptic_reduces_to_circular_for_equal_radii() {
        let radius = 9.0e-3;
        let load = 40.0;
        let e_star = 80.0e9;
        let circular = HertzCircular::new(radius, load, e_star);
        let elliptic = HertzElliptic::new(radius, radius, load, e_star);

        assert_close(
            elliptic.semi_axis_x(),
            circular.contact_radius(),
            1e-12,
            "semi-axis x",
        );
        assert_close(
            elliptic.semi_axis_y(),
            circular.contact_radius(),
            1e-12,
            "semi-axis y",
        );
        assert_close(
            elliptic.max_pressure(),
            circular.max_pressure(),
            1e-12,
            "peak pressure",
        );
        assert_close(elliptic.approach(), circular.approach(), 1e-12, "approach");
        assert!((elliptic.ellipticity() - 1.0).abs() <= 1e-12);
        assert!(elliptic.eccentricity().abs() <= 1e-12);
    }

    #[test]
    fn elliptic_recovers_the_curvature_ratio() {
        // The solved eccentricity must reproduce the input radius ratio through
        // the curvature relation, confirming the transcendental solve.
        let radius_x = 30.0e-3;
        let radius_y = 6.0e-3;
        let elliptic = HertzElliptic::new(radius_x, radius_y, 60.0, 100.0e9);

        let recovered = elliptic_curvature_ratio(elliptic.eccentricity());
        assert_close(recovered, radius_x / radius_y, 1e-9, "curvature ratio");

        // Larger radius => longer axis; here x is the major axis.
        assert!(elliptic.semi_axis_x() > elliptic.semi_axis_y());
        assert_close(
            elliptic.semi_axis_x() / elliptic.semi_axis_y(),
            elliptic.ellipticity(),
            1e-12,
            "x is major",
        );
    }

    #[test]
    fn elliptic_orientation_follows_the_larger_radius() {
        // Swapping the radii swaps the major axis but preserves the shape.
        let load = 25.0;
        let e_star = 90.0e9;
        let wide = HertzElliptic::new(20.0e-3, 5.0e-3, load, e_star);
        let tall = HertzElliptic::new(5.0e-3, 20.0e-3, load, e_star);

        assert!(wide.semi_axis_x() > wide.semi_axis_y(), "wide: x major");
        assert!(tall.semi_axis_y() > tall.semi_axis_x(), "tall: y major");
        assert_close(
            wide.ellipticity(),
            tall.ellipticity(),
            1e-12,
            "ellipticity is orientation-independent",
        );
        assert_close(wide.semi_axis_x(), tall.semi_axis_y(), 1e-12, "major axes");
        assert_close(wide.semi_axis_y(), tall.semi_axis_x(), 1e-12, "minor axes");
    }

    #[test]
    fn elliptic_load_integrates_to_the_applied_load() {
        // The semi-ellipsoidal pressure integrates to (2/3) pi a b p0 = P.
        let load = 75.0;
        let elliptic = HertzElliptic::new(25.0e-3, 8.0e-3, load, 110.0e9);
        let integrated = 2.0 / 3.0
            * PI
            * elliptic.semi_axis_x()
            * elliptic.semi_axis_y()
            * elliptic.max_pressure();
        assert_close(integrated, load, 1e-12, "integrated load");
    }

    #[test]
    fn elliptic_integral_closed_forms_match_numerical_quadrature() {
        // Independent cross-check (design risk 11.2): the closed forms used by
        // the reference, I0 = 2K, Ia = (2/e^2)(K - E), Ib = (2/e^2)(E/m - K),
        // are confirmed against direct quadrature of the underlying Hertz
        // integrals in the regularised variable phi, where
        //   I0 = 2 integral_0^{pi/2} D^{-1/2} dphi,
        //   Ia = 2 integral_0^{pi/2} cos^2 phi D^{-1/2} dphi,
        //   Ib = 2 integral_0^{pi/2} cos^2 phi D^{-3/2} dphi,  D = m cos^2 + sin^2.
        for &ecc in &[0.3_f64, 0.6, 0.85, 0.95] {
            let (big_k, big_e) = complete_elliptic_integrals(ecc);
            let m = 1.0 - ecc * ecc;

            let i0_closed = 2.0 * big_k;
            let ia_closed = 2.0 / (ecc * ecc) * (big_k - big_e);
            let ib_closed = 2.0 / (ecc * ecc) * (big_e / m - big_k);

            let i0_num = quad(|phi| 2.0 / d(phi, m).sqrt());
            let ia_num = quad(|phi| 2.0 * phi.cos().powi(2) / d(phi, m).sqrt());
            let ib_num = quad(|phi| 2.0 * phi.cos().powi(2) / d(phi, m).powf(1.5));

            assert_close(i0_closed, i0_num, 1e-9, "I0 closed vs numeric");
            assert_close(ia_closed, ia_num, 1e-9, "Ia closed vs numeric");
            assert_close(ib_closed, ib_num, 1e-9, "Ib closed vs numeric");
        }
    }

    #[test]
    fn cone_relations_are_self_consistent() {
        // The closed forms must agree with each other: the stored load equals
        // (π/2) E* m a², the mean pressure equals P/(π a²), and the approach
        // follows δ = (π/2) m a.
        let slope = 0.03;
        let load = 45.0;
        let e_star = 90.0e9;
        let cone = SneddonCone::new(slope, load, e_star);
        let a = cone.contact_radius();

        assert_close(cone.load(), load, 1e-12, "stored load");
        assert_close(
            cone.mean_pressure(),
            load / (PI * a * a),
            1e-12,
            "mean pressure",
        );
        assert_close(cone.approach(), 0.5 * PI * slope * a, 1e-12, "approach");
    }

    #[test]
    fn cone_pressure_integrates_to_the_applied_load() {
        // Independent cross-check of the closed forms: quadrature of the
        // arccosh pressure over the contact disc recovers the applied load,
        // P = ∫₀^a p(r) 2π r dr, confirming the a–P–p relation used to validate
        // the grid solver. The integrand vanishes at both ends (apex and edge),
        // so plain composite Simpson is accurate despite the apex singularity.
        let slope = 0.02;
        let load = 60.0;
        let e_star = 100.0e9;
        let cone = SneddonCone::new(slope, load, e_star);
        let a = cone.contact_radius();

        let integrated = simpson(0.0, a, 20_000, |r| 2.0 * PI * r * cone.pressure_at(r));
        assert_close(integrated, load, 1e-4, "integrated cone load");
    }

    // D(phi) = m cos^2 phi + sin^2 phi, the regularised radial factor.
    fn d(phi: f64, m: f64) -> f64 {
        m * phi.cos().powi(2) + phi.sin().powi(2)
    }

    // Composite Simpson quadrature of `f` over [lo, hi] with `n` (even) panels.
    #[allow(
        clippy::cast_precision_loss,
        reason = "the node index and count are tiny relative to f64's integer range"
    )]
    fn simpson<F: Fn(f64) -> f64>(lo: f64, hi: f64, n: usize, f: F) -> f64 {
        let h = (hi - lo) / n as f64;
        let mut acc = f(lo) + f(hi);
        for i in 1..n {
            let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
            acc += weight * f(lo + i as f64 * h);
        }
        acc * h / 3.0
    }

    // Composite Simpson quadrature of `f` over [0, pi/2] with a fine, even mesh.
    #[allow(
        clippy::cast_precision_loss,
        reason = "the node index and count are tiny relative to f64's integer range"
    )]
    fn quad<F: Fn(f64) -> f64>(f: F) -> f64 {
        let n = 4000_usize;
        let h = (PI / 2.0) / n as f64;
        let mut acc = f(0.0) + f(PI / 2.0);
        for i in 1..n {
            let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
            acc += weight * f(i as f64 * h);
        }
        acc * h / 3.0
    }
}
