//! A reduced, closed-form contact law for a ball in a Gothic-arch groove.
//!
//! The FFT + BCCG core solves the *field* problem — a full pressure distribution
//! per configuration — which is the right tool for validation but far too heavy
//! for the inner loop of a multibody simulation, where the same contact is
//! evaluated thousands of times per second. This module distils the validated
//! two-flank Gothic contact into a **lightweight algebraic force law** `F(δ)`
//! that a rigid-body integrator can call directly.
//!
//! # The model
//!
//! A single smooth Hertz contact is the two-input/two-output map `F = k δ^{3/2}`
//! with the force `e ∥ (x − o)` along the line of centres. A Gothic-arch groove
//! breaks that symmetry: the ball rides **two flanks**, so a single algebraic law
//! no longer fits. The reduction keeps the spirit of Hertz but superposes the two
//! flanks. In the groove's meridional plane, a ball-centre displacement
//! `δ = (δ_t, δ_n)` (transverse `t̂`, normal `n̂`) compresses two flanks whose
//! contact normals are tilted by the contact half-angle `±α`,
//! `n̂_± = (±sin α, cos α)`. Each flank sees the projected approach
//!
//! ```text
//! s_± = δ · n̂_± = δ_n cos α ± δ_t sin α,
//! ```
//!
//! carries a Hertzian load along its own normal (no adhesion, so only the
//! positive part `⌊·⌋₊` engages),
//!
//! ```text
//! Q_± = K ⌊s_±⌋₊^{3/2},
//! ```
//!
//! and the net contact force is the vector sum
//!
//! ```text
//! F(δ) = Q_+ n̂_+ + Q_- n̂_-,   F_t = (Q_+ − Q_-) sin α,  F_n = (Q_+ + Q_-) cos α.
//! ```
//!
//! `K` is one flank's elliptic-Hertz load–deflection constant (calibrated from
//! the field solver, [`GothicArchLaw::from_elliptic_flank`]); `α` is the geometric
//! contact angle. The map is two-in/two-out, just like the single Hertz contact it
//! generalises.
//!
//! # The boundary condition this is built to satisfy
//!
//! As the load tilts, the inner flank unloads and at `δ_t = δ_n cot α` it lifts
//! off: the contact passes from **two flanks to one** and the law collapses to the
//! single Hertz contact `F = K s_+^{3/2} n̂_+`. Crucially this transition is
//! **`C¹`** — both the force *and* its Jacobian are continuous across it — because
//! the Hertzian exponent `3/2 > 1` makes a flank engage with zero load *and* zero
//! stiffness: `Q_- ∝ s_-^{3/2} → 0` and `dQ_-/ds_- ∝ s_-^{1/2} → 0` as
//! `s_- → 0⁺`. The `3/2` power is exactly what guarantees the smooth two-to-one
//! handover. It is `C¹` but not `C²`: the tangent stiffness has a `√` cusp
//! (`d²Q_-/ds_-² ∝ s_-^{-1/2} → ∞`), so the contact stiffens with an infinite
//! initial rate, the familiar Hertzian signature. The surviving single-flank
//! branch is precisely the `F = k δ^{3/2}`, `e ∥ (x − o)` law of a single groove.
//!
//! [`GothicArchLaw::jacobian`] returns the analytic tangent stiffness `dF/dδ` for
//! implicit integrators, and is what the `C¹` continuity tests check across the
//! lift-off seam.

use crate::validation::HertzElliptic;

/// The Hertzian load exponent `3/2`. The single value the whole law — and its
/// `C¹` two-to-one transition — turns on, named once so it reads as physics.
const HERTZ_EXPONENT: f64 = 1.5;

/// A reduced two-flank force law for a ball in a Gothic-arch groove.
///
/// A closed-form `F(δ_t, δ_n) → (F_t, F_n)` map (see the [module
/// docs](self)) built from one flank's Hertz stiffness `K` and the contact
/// half-angle `α`. Evaluating it is a couple of `powf`s — cheap enough for a
/// multibody inner loop — and it reproduces, by construction, both the single
/// Hertz contact in the one-flank limit and a `C¹` two-to-one transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GothicArchLaw {
    stiffness: f64,
    contact_angle: f64,
}

impl GothicArchLaw {
    /// Builds the law from a per-flank stiffness `K` and contact half-angle `α`.
    ///
    /// `stiffness` is the Hertzian load–deflection constant of *one* flank
    /// (`Q = K s^{3/2}`, units N·m^−3/2); `contact_angle` is the half-angle `α`
    /// (radians) the flank normals make with the groove's normal axis. Prefer
    /// [`GothicArchLaw::from_elliptic_flank`], which calibrates `K` from the flank
    /// geometry and material.
    ///
    /// # Panics
    /// Panics if `stiffness` is not strictly positive and finite, or if
    /// `contact_angle` is not in the open interval `(0, π/2)` (a degenerate flank
    /// angle gives no transverse force or no normal force).
    #[must_use]
    pub fn new(stiffness: f64, contact_angle: f64) -> Self {
        assert!(
            stiffness > 0.0 && stiffness.is_finite(),
            "flank stiffness must be positive and finite",
        );
        assert!(
            contact_angle > 0.0 && contact_angle < core::f64::consts::FRAC_PI_2,
            "contact half-angle must lie in (0, pi/2)",
        );
        Self {
            stiffness,
            contact_angle,
        }
    }

    /// Calibrates the law from one flank's elliptic-Hertz contact.
    ///
    /// The per-flank stiffness is read off the analytic elliptic-Hertz solution
    /// for the flank's relative radii `radius_x`, `radius_y` and modulus `e_star`:
    /// since the Hertzian approach scales as `δ ∝ P^{2/3}`, the ratio
    /// `K = P / δ^{3/2}` is load-independent, so it is evaluated once at unit
    /// load. `contact_angle` is the geometric flank angle `α` (see
    /// [`contact_half_angle`]). The resulting `K` matches the field solver's
    /// single-arc load–deflection curve (cross-checked in the scenario tests).
    ///
    /// # Panics
    /// Panics if any radius or `e_star` is not strictly positive and finite, or if
    /// `contact_angle` is not in `(0, π/2)`.
    #[must_use]
    pub fn from_elliptic_flank(
        radius_x: f64,
        radius_y: f64,
        e_star: f64,
        contact_angle: f64,
    ) -> Self {
        let reference = HertzElliptic::new(radius_x, radius_y, 1.0, e_star);
        let stiffness = reference.approach().powf(-HERTZ_EXPONENT);
        Self::new(stiffness, contact_angle)
    }

    /// The per-flank Hertz stiffness `K` (N·m^−3/2).
    #[must_use]
    pub const fn stiffness(&self) -> f64 {
        self.stiffness
    }

    /// The contact half-angle `α` (radians).
    #[must_use]
    pub const fn contact_angle(&self) -> f64 {
        self.contact_angle
    }

    /// One flank's Hertz load `Q = K ⌊s⌋₊^{3/2}` for an approach `s`.
    ///
    /// A negative approach means the flank has separated, so the load is zero (no
    /// adhesion). This is the scalar `F = k δ^{3/2}` law of a single contact.
    #[must_use]
    pub fn flank_load(&self, approach: f64) -> f64 {
        self.stiffness * approach.max(0.0).powf(HERTZ_EXPONENT)
    }

    /// The two flank approaches `(s_+, s_-)` for a ball-centre displacement.
    ///
    /// `s_± = δ_n cos α ± δ_t sin α` are the projections of `δ` onto the two flank
    /// normals; a negative value means that flank is separated.
    #[must_use]
    pub fn flank_approaches(&self, delta_t: f64, delta_n: f64) -> (f64, f64) {
        let (sin, cos) = self.contact_angle.sin_cos();
        let along = delta_n * cos;
        let across = delta_t * sin;
        (along + across, along - across)
    }

    /// The two flank loads `(Q_+, Q_-)` for a ball-centre displacement.
    ///
    /// Each is the Hertz load on that flank; in a separated two-flank contact they
    /// are the loads the field solver measures over each half of the groove. When
    /// one is zero the contact has dropped to a single flank.
    #[must_use]
    pub fn flank_loads(&self, delta_t: f64, delta_n: f64) -> (f64, f64) {
        let (s_plus, s_minus) = self.flank_approaches(delta_t, delta_n);
        (self.flank_load(s_plus), self.flank_load(s_minus))
    }

    /// The net contact force `(F_t, F_n)` for a ball-centre displacement `δ`.
    ///
    /// The two-input/two-output law `F(δ_t, δ_n)` of the [module docs](self):
    /// `F_t = (Q_+ − Q_-) sin α`, `F_n = (Q_+ + Q_-) cos α`. Reduces to the single
    /// Hertz contact when one flank is separated, and varies `C¹` through that
    /// two-to-one transition.
    #[must_use]
    pub fn force(&self, delta_t: f64, delta_n: f64) -> (f64, f64) {
        let (q_plus, q_minus) = self.flank_loads(delta_t, delta_n);
        let (sin, cos) = self.contact_angle.sin_cos();
        ((q_plus - q_minus) * sin, (q_plus + q_minus) * cos)
    }

    /// The analytic tangent stiffness `dF/dδ`, as `[[∂F_t/∂δ_t, ∂F_t/∂δ_n], …]`.
    ///
    /// The Jacobian of [`GothicArchLaw::force`], for implicit multibody
    /// integrators and for pinning the law's continuity. With the per-flank
    /// tangent `g_i = dQ_i/ds_i = (3/2) K ⌊s_i⌋₊^{1/2}` it is the symmetric matrix
    ///
    /// ```text
    /// [ (g_+ + g_-) sin²α     (g_+ − g_-) sin α cos α ]
    /// [ (g_+ − g_-) sin α cos α   (g_+ + g_-) cos²α   ].
    /// ```
    ///
    /// Each `g_i → 0` as its flank unloads (`s_i → 0⁺`), so the matrix is
    /// continuous across the two-to-one lift-off — the `C¹` property — while its
    /// own derivative diverges there (`g_i ∝ √s_i`), so the law is not `C²`.
    #[must_use]
    pub fn jacobian(&self, delta_t: f64, delta_n: f64) -> [[f64; 2]; 2] {
        let (s_plus, s_minus) = self.flank_approaches(delta_t, delta_n);
        let tangent = |s: f64| HERTZ_EXPONENT * self.stiffness * s.max(0.0).sqrt();
        let g_plus = tangent(s_plus);
        let g_minus = tangent(s_minus);
        let (sin, cos) = self.contact_angle.sin_cos();
        let sum = g_plus + g_minus;
        let diff = g_plus - g_minus;
        let cross = diff * sin * cos;
        [[sum * sin * sin, cross], [cross, sum * cos * cos]]
    }

    /// The transverse displacement at which the inner flank lifts off.
    ///
    /// For a normal displacement `δ_n > 0`, the contact is two-flanked while
    /// `|δ_t| < δ_n cot α` and single-flanked beyond; this returns that threshold
    /// `δ_t* = δ_n cot α`, the location of the `C¹` two-to-one transition.
    #[must_use]
    pub fn lift_off_transverse(&self, delta_n: f64) -> f64 {
        delta_n / self.contact_angle.tan()
    }
}

/// The geometric contact half-angle of a flank offset `y0` on a ball of radius
/// `R_s`, `α = arcsin(y0 / R_s)`.
///
/// A Gothic flank contact centred a meridional distance `y0` from the groove axis
/// touches the ball where its surface normal is tilted by `α` from that axis. This
/// is the small-shim estimate used to orient the flank normals; the force
/// magnitudes (via the stiffness `K`) come from the field solver, so `α` only
/// sets the *direction* split between transverse and normal force.
///
/// # Panics
/// Panics if `ball_radius` is not strictly positive and finite, or if `offset` is
/// negative or not strictly below `ball_radius` (the flank must sit on the ball).
#[must_use]
pub fn contact_half_angle(offset: f64, ball_radius: f64) -> f64 {
    assert!(
        ball_radius > 0.0 && ball_radius.is_finite(),
        "ball radius must be positive and finite",
    );
    assert!(
        offset >= 0.0 && offset < ball_radius,
        "flank offset must satisfy 0 <= offset < ball_radius",
    );
    (offset / ball_radius).asin()
}

#[cfg(test)]
mod tests {
    use super::{contact_half_angle, GothicArchLaw, HERTZ_EXPONENT};
    use crate::validation::HertzElliptic;
    use core::f64::consts::FRAC_PI_2;

    fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
        let scale = expected.abs().max(1.0e-300);
        let rel_err = (actual - expected).abs() / scale;
        assert!(
            rel_err <= tolerance,
            "{what}: actual={actual:e} expected={expected:e} rel_err={rel_err:e} (> {tolerance:e})",
        );
    }

    // A representative groove flank: the gallery's conformal Gothic arch.
    fn sample_law() -> GothicArchLaw {
        // Relative radii of one flank (circumferential convex, meridional
        // conformal) and a 23-degree contact angle, like the README example.
        GothicArchLaw::from_elliptic_flank(1.6e-3, 26.0e-3, 100.0e9, 0.40)
    }

    #[test]
    fn stiffness_reproduces_the_elliptic_hertz_load() {
        // K is calibrated so that Q = K δ^{3/2} reproduces the elliptic-Hertz
        // load at the contact's own approach — at any load, since K is
        // load-independent. Check it at two unrelated loads.
        let (radius_x, radius_y, e_star) = (1.6e-3, 26.0e-3, 100.0e9);
        let law = GothicArchLaw::from_elliptic_flank(radius_x, radius_y, e_star, 0.40);

        for &load in &[7.0_f64, 250.0] {
            let hertz = HertzElliptic::new(radius_x, radius_y, load, e_star);
            assert_close(
                law.flank_load(hertz.approach()),
                load,
                1.0e-9,
                "flank load reproduces elliptic Hertz",
            );
        }
    }

    #[test]
    fn single_flank_limit_is_a_hertz_contact() {
        // Past lift-off (δ_t beyond δ_n cot α) only the upper flank carries load,
        // and the force collapses to one Hertz contact directed along n̂_+ =
        // (sin α, cos α): the magnitude is the scalar K s_+^{3/2} law and the
        // lower flank contributes nothing.
        let law = sample_law();
        let delta_n = 5.0e-6;
        let delta_t = 2.0 * law.lift_off_transverse(delta_n); // well past lift-off

        let (s_plus, s_minus) = law.flank_approaches(delta_t, delta_n);
        assert!(s_minus < 0.0, "lower flank must be separated past lift-off");

        let (f_t, f_n) = law.force(delta_t, delta_n);
        let magnitude = law.flank_load(s_plus);
        let (sin, cos) = law.contact_angle().sin_cos();
        assert_close(f_t, magnitude * sin, 1.0e-12, "single-flank F_t");
        assert_close(f_n, magnitude * cos, 1.0e-12, "single-flank F_n");
        assert_close(f_t.hypot(f_n), magnitude, 1.0e-12, "single-flank |F|");
    }

    #[test]
    fn symmetric_push_is_purely_normal() {
        // A straight push into the groove (δ_t = 0) loads both flanks equally, so
        // the transverse force cancels and the normal force is
        // F_n = 2 K (δ_n cos α)^{3/2} cos α.
        let law = sample_law();
        let delta_n = 8.0e-6;
        let (f_t, f_n) = law.force(0.0, delta_n);

        let cos = law.contact_angle().cos();
        let expected = 2.0 * law.stiffness() * (delta_n * cos).powf(HERTZ_EXPONENT) * cos;
        assert!(
            f_t.abs() <= 1.0e-18,
            "symmetric push has no transverse force"
        );
        assert_close(f_n, expected, 1.0e-12, "symmetric normal force");
    }

    #[test]
    fn separated_flanks_have_no_adhesion() {
        // Pulling the ball out of the groove (δ_n < 0) separates both flanks: the
        // law returns zero force, never a tensile (negative) one.
        let law = sample_law();
        let (f_t, f_n) = law.force(1.0e-6, -3.0e-6);
        assert!(f_t.abs() <= 1.0e-18 && f_n.abs() <= 1.0e-18, "no adhesion");

        let (q_plus, q_minus) = law.flank_loads(0.0, -1.0e-6);
        assert!(q_plus == 0.0 && q_minus == 0.0, "both flanks carry no load");
    }

    #[test]
    fn jacobian_matches_finite_differences_in_both_regimes() {
        // The analytic tangent stiffness must match a central finite-difference of
        // the force, both in the two-flank region and in the one-flank region past
        // lift-off (where the seam is not crossed by the small step).
        let law = sample_law();
        let delta_n = 6.0e-6;
        let step = 1.0e-11;

        for &delta_t in &[
            0.3 * law.lift_off_transverse(delta_n), // two flanks
            1.8 * law.lift_off_transverse(delta_n), // one flank
        ] {
            let analytic = law.jacobian(delta_t, delta_n);

            let (ft_tp, fn_tp) = law.force(delta_t + step, delta_n);
            let (ft_tm, fn_tm) = law.force(delta_t - step, delta_n);
            let (ft_np, fn_np) = law.force(delta_t, delta_n + step);
            let (ft_nm, fn_nm) = law.force(delta_t, delta_n - step);

            let numeric = [
                [
                    (ft_tp - ft_tm) / (2.0 * step),
                    (ft_np - ft_nm) / (2.0 * step),
                ],
                [
                    (fn_tp - fn_tm) / (2.0 * step),
                    (fn_np - fn_nm) / (2.0 * step),
                ],
            ];
            for row in 0..2 {
                for col in 0..2 {
                    assert_close(
                        numeric[row][col],
                        analytic[row][col],
                        1.0e-4,
                        "jacobian entry vs finite difference",
                    );
                }
            }
        }
    }

    #[test]
    fn jacobian_is_symmetric() {
        // A conservative (potential) contact has a symmetric tangent stiffness.
        let law = sample_law();
        let jac = law.jacobian(2.0e-6, 7.0e-6);
        assert_close(jac[0][1], jac[1][0], 1.0e-12, "Jacobian symmetry");
    }

    #[test]
    fn two_to_one_transition_is_c1_continuous() {
        // The crux: the force and its Jacobian are continuous across the lift-off
        // seam δ_t* = δ_n cot α (the two-to-one transition), because the 3/2
        // exponent makes the unloading flank vanish in load *and* stiffness.
        let law = sample_law();
        let delta_n = 5.0e-6;
        let seam = law.lift_off_transverse(delta_n);
        let step = 1.0e-10;

        // The lower flank is just engaged below the seam and just separated above.
        let (_, s_below) = law.flank_approaches(seam - step, delta_n);
        let (_, s_above) = law.flank_approaches(seam + step, delta_n);
        assert!(
            s_below > 0.0 && s_above < 0.0,
            "step must straddle lift-off"
        );

        let (ft_b, fn_b) = law.force(seam - step, delta_n);
        let (ft_a, fn_a) = law.force(seam + step, delta_n);
        // Force is C0: the two one-sided values agree to the step size.
        assert!(
            (ft_b - ft_a).abs() <= 1.0e-3 * ft_b.abs().max(fn_b.abs()),
            "force is continuous across lift-off",
        );
        assert!((fn_b - fn_a).abs() <= 1.0e-3 * fn_b.abs());

        // Jacobian is C0 too (so the force is C1): the one-sided tangent
        // stiffnesses match across the seam.
        let jac_b = law.jacobian(seam - step, delta_n);
        let jac_a = law.jacobian(seam + step, delta_n);
        let scale = jac_b[1][1].abs();
        for row in 0..2 {
            for col in 0..2 {
                assert!(
                    (jac_b[row][col] - jac_a[row][col]).abs() <= 1.0e-2 * scale,
                    "tangent stiffness is continuous across lift-off (C1)",
                );
            }
        }
    }

    #[test]
    fn the_law_is_c1_but_not_c2_at_the_seam() {
        // The tell-tale of a Hertzian engagement: at lift-off the tangent
        // stiffness has an infinite *slope* (d²Q/ds² ∝ s^{-1/2}), so the law is C¹
        // but not C². The inboard Jacobian jump over a step `h` into the two-flank
        // side is therefore dominated by a √h term, not the O(h) of a smooth
        // function. Halving `h` then scales the jump by √(1/2), not 1/2 — the
        // discriminator between a √-cusp (ratio → √2 ≈ 1.41) and a C² join
        // (ratio → 2). The inboard jump over a step `h` (one-sided, into the
        // two-flank region) is J11(seam) − J11(seam − h).
        let law = sample_law();
        let delta_n = 5.0e-6;
        let seam = law.lift_off_transverse(delta_n);
        let at_seam = law.jacobian(seam, delta_n)[1][1];

        let jump = |h: f64| (at_seam - law.jacobian(seam - h, delta_n)[1][1]).abs();
        let coarse = jump(4.0e-10);
        let fine = jump(2.0e-10);

        // A √-cusp halves to 1/√2 of itself; a C² join would halve to 1/2.
        let ratio = coarse / fine;
        assert!(
            (1.30..1.55).contains(&ratio),
            "inboard Jacobian jump must scale as √h (C¹ but not C²): ratio={ratio:e}",
        );
    }

    #[test]
    fn contact_half_angle_is_the_geometric_arcsine() {
        // α = arcsin(y0 / Rs): a flank a quarter of the ball radius out sits at
        // arcsin(1/4); the offset-free limit is a zero angle.
        assert_close(
            contact_half_angle(1.0e-3, 4.0e-3),
            (0.25_f64).asin(),
            1.0e-12,
            "contact half-angle",
        );
        assert!(
            contact_half_angle(0.0, 4.0e-3).abs() <= 1.0e-18,
            "no offset"
        );
        assert!(contact_half_angle(2.0e-3, 4.0e-3) < FRAC_PI_2, "below pi/2");
    }
}
