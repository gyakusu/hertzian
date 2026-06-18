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
//!
//! # Neighbour coupling: the flanks lift one another
//!
//! Superposing two *independent* Hertz flanks is exact only when they sit far
//! enough apart to ignore one another — the well-separated limit, where each
//! carries half the load and the effective flank count `η = P / (K δ^{3/2})` is
//! `2`. As the groove shim is tightened the two flank contacts close in and their
//! elastic fields overlap: the load `Q` on one flank lifts the half-space under
//! the other, shrinking the neighbour's approach and so its load. To first order
//! each flank sees the Boussinesq far field of the other — a point load `Q` a
//! distance `d = 2 y0` away, since the flank centres sit at `y = ±y0`,
//!
//! ```text
//! u ≈ Q / (π E* d),   d = 2 y0,
//! ```
//!
//! so the two *effective* approaches couple through the loads they themselves set,
//!
//! ```text
//! s_±^eff = s_± − κ Q_∓,   Q_± = K ⌊s_±^eff⌋₊^{3/2},   κ = 1 / (2 π E* y0),
//! ```
//!
//! a small `2×2` self-consistent solve (one [`GothicArchLaw::with_flank_coupling`]
//! call enables it). The lift pulls `η` below `2` in the half-overlap regime — the
//! gap the single-`K` superposition would otherwise fold into its residual — and
//! it sharpens the load split under an asymmetric drive (the heavier flank presses
//! its lighter neighbour down harder). It leaves untouched the two limits the law
//! already nails: as `y0 → ∞`, `κ → 0` and the flanks decouple (`η → 2`); and when
//! a flank lifts off, `Q_∓ → 0` withdraws its lift, so the surviving single Hertz
//! contact — and the `C¹` two-to-one handover — are exactly as before. Coupling is
//! off (`κ = 0`) unless [`GothicArchLaw::with_flank_coupling`] sets it, so the
//! separated two-flank law is the untouched default.
//!
//! This first-order lift is a *magnitude* correction: it acts in the flank-approach
//! frame ([`GothicArchLaw::coupled_loads`]) and so it sets the load split `Q_+ : Q_-`
//! — the direction in which the force divides between the flanks — without touching
//! the flank normals. It tracks the field solver's effective flank count `η(y0/b)`
//! and its load split to a few percent through the half-overlap regime. A *second*,
//! finer directional effect remains: as the contacts overlap, each flank lifts its
//! neighbour's *inboard* side more than its outboard one, so the load centroid
//! slides outboard of the geometric offset (the field solver shows it ~36 % beyond
//! `y0` at half overlap, decaying to `y0` once separated). That is the flank
//! *normal* rotating — a steeper effective contact angle `α_eff(y0/b) =
//! arcsin(y_centroid / R_s)`, with `α_eff → α` separated — which one applies by
//! building the law at that effective angle (via [`contact_half_angle`]). It refines
//! only the `(F_t, F_n)` projection, not the `η`/split this stage validates, and is
//! left, with the full coalescence to `η = 1` (a blend onto the single arch), as the
//! next stage.
//!
//! # Per-flank pressure: the Coulomb-friction cap
//!
//! `F(δ)` is the *resultant*; a Coulomb friction model needs the *distribution* the
//! tangential traction is capped by, `|τ(x, y)| ≤ μ p(x, y)`. Each flank is an
//! elliptic-Hertz contact carrying its (coupled) load `Q`, so its pressure is the
//! half-ellipsoid
//!
//! ```text
//! p(x, y) = p0 √⌊1 − (x/a_x)² − (y/a_y)²⌋₊,   p0 = 3 Q / (2π a_x a_y),
//! ```
//!
//! flank-local (the caller centres it at `±y_c`). The shape is fixed *once*, from the
//! same flank the stiffness `K` is calibrated from: by Hertz's cube-root load scaling
//! the semi-axes are `a = a_unit Q^{1/3}`, so the peak is `p0 = c_p Q^{1/3}` with
//! `c_p = 3 / (2π a_x^unit a_y^unit)`, and [`GothicArchLaw::flank_pressure`] builds the
//! footprint ([`FlankPressure`]) in a couple of `cbrt`s — no eccentricity solve in the
//! inner loop. Because `Q = K s^{3/2}`, the peak is `p0 = c_p K^{1/3} √s`: the cap
//! kisses zero as `√s` at lift-off, the same `3/2`-power signature that makes the
//! force `C¹` there. The half-ellipsoid integrates to `Q` exactly, so the full-sliding
//! friction resultant is `∫ μ p dA = μ Q` per flank.
//!
//! This is exact, per flank, where the two footprints are *resolved* as distinct
//! patches — the separated regime, the common two-flank bearing the section is built
//! around. As the shim closes to half overlap the patches merge into one connected
//! contact whose seam is *not* the sum of the two half-ellipsoids (superposing them
//! double-counts the overlap); that single-patch coalescence is the same next stage
//! as the `η → 1` blend onto the single arch.

use crate::validation::HertzElliptic;

/// The Hertzian load exponent `3/2`. The single value the whole law — and its
/// `C¹` two-to-one transition — turns on, named once so it reads as physics.
const HERTZ_EXPONENT: f64 = 1.5;

/// Newton steps for the coupled two-flank load solve (see [`GothicArchLaw::coupled_loads`]).
///
/// The neighbour-lift fixed point is a well-conditioned `2×2` system in the
/// half-overlap range, so Newton from the uncoupled loads converges in a handful
/// of steps; this cap is a safety net, not the expected count.
const COUPLING_MAX_ITERS: usize = 32;

/// Relative convergence tolerance for the coupled load solve.
const COUPLING_TOL: f64 = 1.0e-13;

/// Floor on the coupled-solve determinant `1 − κ² g_+ g_-`.
///
/// The determinant is positive while the cross-stiffness stays below the
/// self-stiffness — the pull-in bound, which the half-overlap regime sits well
/// inside. Clamping it keeps the Newton step and the analytic Jacobian finite if a
/// caller pushes the coupling past that bound (the deep-merge regime reserved for
/// the next stage), rather than dividing by zero.
const COUPLING_MIN_DET: f64 = 1.0e-6;

/// One flank's contact-ellipse semi-axes at unit load (m·N^−1/3).
///
/// The calibrated shape behind the per-flank pressure model: by Hertz's cube-root
/// load scaling the semi-axes at a flank load `Q` are `a = a_unit · Q^{1/3}`, so
/// these two unit-load semi-axes fix the whole footprint ([`FlankPressure`]). Set by
/// [`GothicArchLaw::from_elliptic_flank`]; absent for the bare [`GothicArchLaw::new`],
/// whose stiffness `K` alone does not pin the contact-ellipse shape.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FlankReference {
    semi_axis_x: f64,
    semi_axis_y: f64,
}

/// A reduced two-flank force law for a ball in a Gothic-arch groove.
///
/// A closed-form `F(δ_t, δ_n) → (F_t, F_n)` map (see the [module
/// docs](self)) built from one flank's Hertz stiffness `K` and the contact
/// half-angle `α`. Evaluating it is a couple of `powf`s — cheap enough for a
/// multibody inner loop — and it reproduces, by construction, both the single
/// Hertz contact in the one-flank limit and a `C¹` two-to-one transition. When
/// calibrated from a flank shape it also yields the per-flank pressure footprint
/// ([`GothicArchLaw::flank_pressure`]), the Coulomb-friction cap `|τ| ≤ μ p`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GothicArchLaw {
    stiffness: f64,
    contact_angle: f64,
    /// Neighbour-lift cross-compliance `κ` (m·N⁻¹): one flank's load `Q` lifts the
    /// half-space under the other by `κ Q`. Zero is the separated default (no
    /// interaction); [`GothicArchLaw::with_flank_coupling`] sets it from the
    /// geometry. See the [module docs](self#neighbour-coupling-the-flanks-lift-one-another).
    coupling: f64,
    /// The flank's unit-load contact-ellipse semi-axes, when calibrated from a flank
    /// shape ([`GothicArchLaw::from_elliptic_flank`]); `None` for the bare
    /// [`GothicArchLaw::new`]. Backs [`GothicArchLaw::flank_pressure`].
    reference: Option<FlankReference>,
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
            coupling: 0.0,
            reference: None,
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
        let hertz = HertzElliptic::new(radius_x, radius_y, 1.0, e_star);
        let stiffness = hertz.approach().powf(-HERTZ_EXPONENT);
        // The unit-load contact semi-axes are the calibrated shape the per-flank
        // pressure model scales by `Q^{1/3}` (see [`GothicArchLaw::flank_pressure`]).
        Self {
            reference: Some(FlankReference {
                semi_axis_x: hertz.semi_axis_x(),
                semi_axis_y: hertz.semi_axis_y(),
            }),
            ..Self::new(stiffness, contact_angle)
        }
    }

    /// Enables the neighbour-lift coupling from the modulus `E*` and flank offset `y0`.
    ///
    /// Sets the cross-compliance `κ = 1 / (2 π E* y0)` — the Boussinesq far-field
    /// lift `u ≈ Q / (π E* d)` of a flank load `Q` at the neighbour's centre, a
    /// distance `d = 2 y0` away (see the [module
    /// docs](self#neighbour-coupling-the-flanks-lift-one-another)). Builder-style so
    /// it composes with the calibrating constructors:
    /// `GothicArchLaw::from_elliptic_flank(..).with_flank_coupling(e_star, y0)`.
    /// Without it the law keeps `κ = 0` — two independent flanks, the exact
    /// well-separated limit.
    ///
    /// # Panics
    /// Panics if `e_star` or `offset` is not strictly positive and finite.
    #[must_use]
    pub fn with_flank_coupling(self, e_star: f64, offset: f64) -> Self {
        assert!(
            e_star > 0.0 && e_star.is_finite(),
            "modulus E* must be positive and finite",
        );
        assert!(
            offset > 0.0 && offset.is_finite(),
            "flank offset y0 must be positive and finite",
        );
        Self {
            coupling: 1.0 / (2.0 * core::f64::consts::PI * e_star * offset),
            ..self
        }
    }

    /// The per-flank Hertz stiffness `K` (N·m^−3/2).
    #[must_use]
    pub const fn stiffness(&self) -> f64 {
        self.stiffness
    }

    /// The neighbour-lift cross-compliance `κ` (m·N⁻¹); `0` when uncoupled.
    ///
    /// One flank's load `Q` lifts the half-space under the other by `κ Q`. Set by
    /// [`GothicArchLaw::with_flank_coupling`]; `0` is the separated default.
    #[must_use]
    pub const fn coupling(&self) -> f64 {
        self.coupling
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

    /// One flank's tangent stiffness `dQ/ds = (3/2) K ⌊s⌋₊^{1/2}` at approach `s`.
    ///
    /// The slope of [`GothicArchLaw::flank_load`]; it vanishes as the flank unloads
    /// (`s → 0⁺`), the `√`-soft engagement that makes the two-to-one handover `C¹`.
    fn flank_tangent(&self, approach: f64) -> f64 {
        HERTZ_EXPONENT * self.stiffness * approach.max(0.0).sqrt()
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

    /// The two flank loads `(Q_+, Q_-)` for prescribed flank approaches `(s_+, s_-)`.
    ///
    /// The self-consistent solution of the coupled pair
    /// `Q_± = K ⌊s_± − κ Q_∓⌋₊^{3/2}` (see the [module
    /// docs](self#neighbour-coupling-the-flanks-lift-one-another)): each flank's
    /// load is reduced by the lift `κ Q_∓` its neighbour raises under it. With
    /// coupling off (`κ = 0`) this is exactly two independent Hertz loads
    /// `(K⌊s_+⌋₊^{3/2}, K⌊s_-⌋₊^{3/2})`, the well-separated superposition. A few
    /// Newton steps from that uncoupled seed converge the `2×2` fixed point; the
    /// system is well-conditioned throughout the half-overlap range.
    #[must_use]
    pub fn coupled_loads(&self, s_plus: f64, s_minus: f64) -> (f64, f64) {
        let mut q_plus = self.flank_load(s_plus);
        let mut q_minus = self.flank_load(s_minus);
        if self.coupling == 0.0 {
            return (q_plus, q_minus);
        }
        let kappa = self.coupling;
        for _ in 0..COUPLING_MAX_ITERS {
            // Residual of Q_± = K⌊s_± − κ Q_∓⌋₊^{3/2} at the current loads.
            let e_plus = s_plus - kappa * q_minus;
            let e_minus = s_minus - kappa * q_plus;
            let r_plus = q_plus - self.flank_load(e_plus);
            let r_minus = q_minus - self.flank_load(e_minus);
            // Newton step: the system matrix is [[1, κ g_+], [κ g_-, 1]], with
            // g_i the per-flank tangent at the effective approach.
            let g_plus = self.flank_tangent(e_plus);
            let g_minus = self.flank_tangent(e_minus);
            let det = (1.0 - kappa * kappa * g_plus * g_minus).max(COUPLING_MIN_DET);
            let dq_plus = (r_plus - kappa * g_plus * r_minus) / det;
            let dq_minus = (r_minus - kappa * g_minus * r_plus) / det;
            q_plus = (q_plus - dq_plus).max(0.0);
            q_minus = (q_minus - dq_minus).max(0.0);
            if dq_plus.abs() + dq_minus.abs()
                <= COUPLING_TOL * (q_plus + q_minus).max(f64::MIN_POSITIVE)
            {
                break;
            }
        }
        (q_plus, q_minus)
    }

    /// The two flank loads `(Q_+, Q_-)` for a ball-centre displacement.
    ///
    /// Projects the displacement onto the flank approaches and applies
    /// [`GothicArchLaw::coupled_loads`]; in a separated two-flank contact these are
    /// the loads the field solver measures over each half of the groove. When one
    /// is zero the contact has dropped to a single flank.
    #[must_use]
    pub fn flank_loads(&self, delta_t: f64, delta_n: f64) -> (f64, f64) {
        let (s_plus, s_minus) = self.flank_approaches(delta_t, delta_n);
        self.coupled_loads(s_plus, s_minus)
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
    /// integrators and for pinning the law's continuity. With the per-flank tangent
    /// `g_i = (3/2) K ⌊s_i^eff⌋₊^{1/2}` at the *effective* approach and the coupled
    /// determinant `D = 1 − κ² g_+ g_-`, differentiating the coupled force through
    /// the implicit pair `Q_± = K⌊s_± − κ Q_∓⌋₊^{3/2}` gives the symmetric matrix
    ///
    /// ```text
    /// 1/D · [ (g_+ + g_- + 2κ g_+ g_-) sin²α        (g_+ − g_-) sin α cos α      ]
    ///       [ (g_+ − g_-) sin α cos α          (g_+ + g_- − 2κ g_+ g_-) cos²α ].
    /// ```
    ///
    /// With coupling off (`κ = 0`) it is `D = 1` and the bare two-flank Jacobian
    /// `[[(g_+ + g_-) sin²α, …], …]`. Each `g_i → 0` as its flank unloads
    /// (`s_i^eff → 0⁺`), so the matrix is continuous across the two-to-one lift-off
    /// — the `C¹` property, coupled or not — while its own derivative diverges there
    /// (`g_i ∝ √s_i^eff`), so the law is not `C²`.
    #[must_use]
    pub fn jacobian(&self, delta_t: f64, delta_n: f64) -> [[f64; 2]; 2] {
        let (s_plus, s_minus) = self.flank_approaches(delta_t, delta_n);
        let (q_plus, q_minus) = self.coupled_loads(s_plus, s_minus);
        let kappa = self.coupling;
        // Tangents at the effective (post-lift) approaches: with κ = 0 these are the
        // bare s_±, so the whole expression collapses to the uncoupled Jacobian.
        let g_plus = self.flank_tangent(s_plus - kappa * q_minus);
        let g_minus = self.flank_tangent(s_minus - kappa * q_plus);
        let (sin, cos) = self.contact_angle.sin_cos();
        let det = (1.0 - kappa * kappa * g_plus * g_minus).max(COUPLING_MIN_DET);
        let sum = g_plus + g_minus;
        let diff = g_plus - g_minus;
        let coupled = 2.0 * kappa * g_plus * g_minus;
        let cross = diff * sin * cos / det;
        [
            [(sum + coupled) * sin * sin / det, cross],
            [cross, (sum - coupled) * cos * cos / det],
        ]
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

    /// One flank's pressure footprint at a flank load `Q` — the Coulomb-friction cap.
    ///
    /// Turns a (coupled) flank load from [`GothicArchLaw::coupled_loads`] /
    /// [`GothicArchLaw::flank_loads`] into the elliptic-Hertz half-ellipsoid
    /// [`FlankPressure`] a Coulomb model rides under, by scaling the calibrated
    /// unit-load semi-axes by `Q^{1/3}` (Hertz) and reading `p0 = 3Q/(2π a_x a_y)`
    /// — a couple of `cbrt`s, no eccentricity solve (see the [module
    /// docs](self#per-flank-pressure-the-coulomb-friction-cap)).
    ///
    /// Returns `None` if the law was built with the bare [`GothicArchLaw::new`]
    /// (stiffness + angle only), which does not fix the contact-ellipse shape;
    /// calibrate with [`GothicArchLaw::from_elliptic_flank`]. A non-positive load
    /// (a lifted-off flank) gives a zero footprint that caps the traction at zero.
    #[must_use]
    pub fn flank_pressure(&self, load: f64) -> Option<FlankPressure> {
        let reference = self.reference?;
        let scale = load.max(0.0).cbrt();
        let semi_axis_x = reference.semi_axis_x * scale;
        let semi_axis_y = reference.semi_axis_y * scale;
        let peak_pressure = if load > 0.0 {
            3.0 * load / (2.0 * core::f64::consts::PI * semi_axis_x * semi_axis_y)
        } else {
            0.0
        };
        Some(FlankPressure {
            peak_pressure,
            semi_axis_x,
            semi_axis_y,
        })
    }
}

/// One flank's elliptic-Hertz pressure footprint — the Coulomb-friction cap.
///
/// The reduced law hands a multibody integrator the *resultant* `F(δ)`; a Coulomb
/// friction model needs the *distribution* the tangential traction is capped by,
/// `|τ| ≤ μ p`. This is that distribution for one flank carrying a (coupled) load
/// `Q`: the flank-local elliptic-Hertz half-ellipsoid (see the [module
/// docs](self#per-flank-pressure-the-coulomb-friction-cap))
///
/// ```text
/// p(x, y) = p0 √⌊1 − (x/a_x)² − (y/a_y)²⌋₊,   p0 = 3 Q / (2π a_x a_y).
/// ```
///
/// Built by [`GothicArchLaw::flank_pressure`] in a couple of `cbrt`s. It is centred
/// on the flank's own contact centre; the caller places it at `±y_c`. It integrates
/// to `Q` exactly, so the full-sliding friction resultant is `∫ μ p dA = μ Q`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlankPressure {
    peak_pressure: f64,
    semi_axis_x: f64,
    semi_axis_y: f64,
}

impl FlankPressure {
    /// Peak (central) contact pressure `p0` (pascals); `0` for a lifted-off flank.
    #[must_use]
    pub const fn peak_pressure(&self) -> f64 {
        self.peak_pressure
    }

    /// Mean contact pressure `(2/3) p0` (pascals) — the half-ellipsoid's load average.
    #[must_use]
    pub fn mean_pressure(&self) -> f64 {
        2.0 / 3.0 * self.peak_pressure
    }

    /// Contact semi-axes `(a_x, a_y)` along the flank's circumferential/meridional
    /// axes (metres); `(0, 0)` for a lifted-off flank.
    #[must_use]
    pub const fn semi_axes(&self) -> (f64, f64) {
        (self.semi_axis_x, self.semi_axis_y)
    }

    /// The flank load `Q = (2/3) π a_x a_y p0` recovered by integrating the
    /// half-ellipsoid over its footprint (newtons).
    ///
    /// Exact by construction — the footprint carries the load it was built from — so
    /// the full-sliding Coulomb resultant integrates to `∫ μ p dA = μ Q`.
    #[must_use]
    pub fn load(&self) -> f64 {
        2.0 / 3.0 * core::f64::consts::PI * self.semi_axis_x * self.semi_axis_y * self.peak_pressure
    }

    /// Pressure at flank-local `(x, y)`: `p0 √⌊1 − (x/a_x)² − (y/a_y)²⌋₊`, and `0`
    /// outside the contact ellipse (or anywhere, for a lifted-off flank).
    #[must_use]
    pub fn pressure_at(&self, x: f64, y: f64) -> f64 {
        if self.peak_pressure <= 0.0 {
            return 0.0;
        }
        let rx = x / self.semi_axis_x;
        let ry = y / self.semi_axis_y;
        let radial = rx * rx + ry * ry;
        if radial >= 1.0 {
            0.0
        } else {
            self.peak_pressure * (1.0 - radial).sqrt()
        }
    }

    /// The Coulomb traction bound `μ p(x, y)` at flank-local `(x, y)` (pascals).
    ///
    /// The cap a tangential-contact model rides under: the local friction stress
    /// cannot exceed it, and integrating it over the footprint gives `μ Q`.
    ///
    /// # Panics
    /// Panics if `mu` is negative or not finite.
    #[must_use]
    pub fn traction_bound(&self, mu: f64, x: f64, y: f64) -> f64 {
        assert!(
            mu >= 0.0 && mu.is_finite(),
            "friction coefficient must be non-negative and finite",
        );
        mu * self.pressure_at(x, y)
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
/// This is the *geometric* (well-separated) angle. When the flanks overlap their
/// load centroid slides outboard of `y0` (see the [module
/// docs](self#neighbour-coupling-the-flanks-lift-one-another)); passing that
/// shifted centroid here yields the steeper *effective* angle `α_eff(y0/b)` for the
/// force projection, which relaxes back to this geometric `α` as the flanks part.
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

    // --- neighbour-lift coupling ------------------------------------------- #

    // The modulus and flank offset used to switch on coupling in these tests; the
    // offset is small enough (relative to the contact) to give a clearly sub-unity
    // β, i.e. the half-overlap regime where the 2×2 solve is well-conditioned.
    const COUPLED_E_STAR: f64 = 100.0e9;
    const COUPLED_OFFSET: f64 = 5.0e-4;

    fn coupled_law() -> GothicArchLaw {
        sample_law().with_flank_coupling(COUPLED_E_STAR, COUPLED_OFFSET)
    }

    #[test]
    fn with_flank_coupling_sets_the_boussinesq_cross_compliance() {
        // κ = 1 / (2 π E* y0): the Boussinesq far-field lift u = Q/(π E* d) of a
        // flank load at its neighbour's centre, a distance d = 2 y0 away.
        let law = coupled_law();
        let expected = 1.0 / (2.0 * core::f64::consts::PI * COUPLED_E_STAR * COUPLED_OFFSET);
        assert_close(law.coupling(), expected, 1.0e-12, "cross-compliance κ");
        // The uncoupled law has κ = 0 (the separated default).
        assert!(sample_law().coupling() == 0.0, "default is uncoupled");
    }

    #[test]
    fn coupling_lowers_the_effective_flank_count_below_two() {
        // A symmetric push loads both flanks equally; each lifts the other, so the
        // pair carries less than the 2 K δ^{3/2} of two independent flanks — the
        // effective flank count η = (Q_+ + Q_-)/(K δ^{3/2}) drops below 2. It must
        // stay above 1 (the flanks still both carry load) and climb back toward 2
        // as the offset grows and the coupling fades.
        let delta = 6.0e-6;
        let eta = |law: &GothicArchLaw| {
            let (q_plus, q_minus) = law.coupled_loads(delta, delta);
            (q_plus + q_minus) / (law.stiffness() * delta.powf(HERTZ_EXPONENT))
        };

        let near = coupled_law();
        let far = sample_law().with_flank_coupling(COUPLED_E_STAR, 50.0e-3);
        let separated = eta(&sample_law());

        assert_close(separated, 2.0, 1.0e-12, "uncoupled η is exactly 2");
        assert!(
            eta(&near) > 1.0 && eta(&near) < 2.0,
            "coupling pulls η into (1, 2): {}",
            eta(&near),
        );
        assert!(
            eta(&far) > eta(&near) && eta(&far) > 1.95,
            "a far offset all but restores the separated η = 2",
        );
    }

    #[test]
    fn coupling_leaves_the_single_flank_limit_untouched() {
        // Past lift-off the lower flank carries no load, so it lifts nothing: the
        // coupled force must be bit-for-bit the uncoupled single Hertz contact.
        let law = sample_law();
        let coupled = coupled_law();
        let delta_n = 5.0e-6;
        let delta_t = 2.0 * law.lift_off_transverse(delta_n); // well past lift-off

        let (_, s_minus) = coupled.flank_approaches(delta_t, delta_n);
        assert!(s_minus < 0.0, "lower flank must be separated past lift-off");

        let (ft_u, fn_u) = law.force(delta_t, delta_n);
        let (ft_c, fn_c) = coupled.force(delta_t, delta_n);
        assert_close(ft_c, ft_u, 1.0e-12, "single-flank F_t is coupling-free");
        assert_close(fn_c, fn_u, 1.0e-12, "single-flank F_n is coupling-free");
    }

    #[test]
    fn coupling_sharpens_the_load_split() {
        // Under an asymmetric drive the heavier flank presses its lighter neighbour
        // down harder than the reverse, so coupling *sharpens* the split Q_+/Q_-
        // beyond the uncoupled (s_+/s_-)^{3/2}; and the lift lowers *both* loads.
        let (s_plus, s_minus) = (9.0e-6, 4.0e-6);
        let (qp_u, qm_u) = sample_law().coupled_loads(s_plus, s_minus); // κ = 0
        let (qp_c, qm_c) = coupled_law().coupled_loads(s_plus, s_minus);

        assert!(
            qp_c / qm_c > qp_u / qm_u,
            "coupling sharpens the split: {} vs {}",
            qp_c / qm_c,
            qp_u / qm_u,
        );
        assert!(
            qp_c < qp_u && qm_c < qm_u,
            "the lift lowers both flank loads"
        );
        // The uncoupled split is exactly the bare Hertz ratio.
        assert_close(
            qp_u / qm_u,
            (s_plus / s_minus).powf(HERTZ_EXPONENT),
            1.0e-12,
            "uncoupled split is (s_+/s_-)^{3/2}",
        );
    }

    #[test]
    fn coupled_jacobian_matches_finite_differences_in_both_regimes() {
        // The analytic coupled tangent stiffness (implicit-function differentiation
        // through the 2×2 solve) must match a central difference of the coupled
        // force, both where two flanks couple and past lift-off where one is off.
        let law = coupled_law();
        let delta_n = 6.0e-6;
        let step = 1.0e-11;

        for &delta_t in &[
            0.3 * law.lift_off_transverse(delta_n), // two coupled flanks
            1.8 * law.lift_off_transverse(delta_n), // one flank (coupling inert)
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
                        "coupled jacobian entry vs finite difference",
                    );
                }
            }
        }
    }

    #[test]
    fn coupled_jacobian_is_symmetric() {
        // The coupled contact is still conservative: the cross terms agree exactly
        // (the κ g_+ g_- pieces cancel out of the off-diagonal).
        let jac = coupled_law().jacobian(2.0e-6, 7.0e-6);
        assert_close(jac[0][1], jac[1][0], 1.0e-12, "coupled Jacobian symmetry");
    }

    #[test]
    fn coupled_two_to_one_transition_stays_c1() {
        // Coupling must not spoil the headline property: the force and its Jacobian
        // are still continuous across the lift-off seam, because the unloading flank
        // vanishes in load *and* stiffness *and* lift as s_-^eff → 0⁺.
        let law = coupled_law();
        let delta_n = 5.0e-6;
        let seam = law.lift_off_transverse(delta_n);
        let step = 1.0e-10;

        let (_, s_below) = law.flank_approaches(seam - step, delta_n);
        let (_, s_above) = law.flank_approaches(seam + step, delta_n);
        assert!(
            s_below > 0.0 && s_above < 0.0,
            "step must straddle lift-off"
        );

        let (ft_b, fn_b) = law.force(seam - step, delta_n);
        let (ft_a, fn_a) = law.force(seam + step, delta_n);
        let scale = ft_b.abs().max(fn_b.abs());
        assert!(
            (ft_b - ft_a).abs() <= 1.0e-3 * scale && (fn_b - fn_a).abs() <= 1.0e-3 * scale,
            "coupled force is continuous across lift-off",
        );

        let jac_b = law.jacobian(seam - step, delta_n);
        let jac_a = law.jacobian(seam + step, delta_n);
        let jac_scale = jac_b[1][1].abs();
        for row in 0..2 {
            for col in 0..2 {
                assert!(
                    (jac_b[row][col] - jac_a[row][col]).abs() <= 1.0e-2 * jac_scale,
                    "coupled tangent stiffness is continuous across lift-off (C¹)",
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "modulus E*")]
    fn with_flank_coupling_rejects_a_non_positive_modulus() {
        let _ = sample_law().with_flank_coupling(0.0, COUPLED_OFFSET);
    }

    #[test]
    #[should_panic(expected = "flank offset")]
    fn with_flank_coupling_rejects_a_non_positive_offset() {
        let _ = sample_law().with_flank_coupling(COUPLED_E_STAR, 0.0);
    }

    // --- per-flank pressure: the Coulomb-friction cap ---------------------- #

    #[test]
    fn flank_pressure_reproduces_the_elliptic_hertz_footprint() {
        // The lightweight cube-root scaling must reproduce, to machine precision, a
        // fresh elliptic-Hertz solve at the flank load — the same closed form the
        // gallery validates the field solver against. Check peak, semi-axes and the
        // pressure field at an interior point, at two unrelated loads.
        let (radius_x, radius_y, e_star) = (1.6e-3, 26.0e-3, 100.0e9);
        let law = GothicArchLaw::from_elliptic_flank(radius_x, radius_y, e_star, 0.40);

        for &load in &[12.0_f64, 540.0] {
            let footprint = law
                .flank_pressure(load)
                .expect("calibrated law has a footprint");
            let hertz = HertzElliptic::new(radius_x, radius_y, load, e_star);
            let (a_x, a_y) = footprint.semi_axes();
            assert_close(a_x, hertz.semi_axis_x(), 1.0e-12, "footprint semi-axis x");
            assert_close(a_y, hertz.semi_axis_y(), 1.0e-12, "footprint semi-axis y");
            assert_close(
                footprint.peak_pressure(),
                hertz.max_pressure(),
                1.0e-12,
                "footprint peak pressure",
            );
            assert_close(
                footprint.pressure_at(0.3 * a_x, 0.2 * a_y),
                hertz.pressure_at(0.3 * a_x, 0.2 * a_y),
                1.0e-12,
                "footprint pressure field",
            );
        }
    }

    #[test]
    fn flank_pressure_integrates_to_the_load() {
        // The half-ellipsoid carries exactly the load it was built from, so the
        // full-sliding Coulomb resultant is μ Q: both the closed form (2/3)π a_x a_y p0
        // and a direct quadrature of the footprint recover Q.
        let law = sample_law();
        let load = 130.0;
        let footprint = law
            .flank_pressure(load)
            .expect("calibrated law has a footprint");
        assert_close(footprint.load(), load, 1.0e-12, "closed-form integral");

        let (a_x, a_y) = footprint.semi_axes();
        let n: u32 = 400; // f64::from(u32) is lossless, so the quadrature is clippy-clean
        let (dx, dy) = (2.0 * a_x / f64::from(n), 2.0 * a_y / f64::from(n));
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                let x = (f64::from(i) + 0.5) * dx - a_x;
                let y = (f64::from(j) + 0.5) * dy - a_y;
                sum += footprint.pressure_at(x, y);
            }
        }
        assert_close(sum * dx * dy, load, 1.0e-3, "quadrature integral");
    }

    #[test]
    fn flank_pressure_is_a_half_ellipsoid() {
        // The footprint peaks at the centre, vanishes on the contact ellipse, is zero
        // outside it, and is symmetric in both axes — the Hertzian half-ellipsoid.
        let footprint = sample_law()
            .flank_pressure(200.0)
            .expect("calibrated law has a footprint");
        let (a_x, a_y) = footprint.semi_axes();
        let p0 = footprint.peak_pressure();

        assert_close(footprint.pressure_at(0.0, 0.0), p0, 1.0e-12, "centre is p0");
        assert!(
            footprint.pressure_at(a_x, 0.0).abs() <= 1.0e-9 * p0,
            "zero on the x rim"
        );
        assert!(
            footprint.pressure_at(0.0, a_y).abs() <= 1.0e-9 * p0,
            "zero on the y rim"
        );
        assert!(footprint.pressure_at(1.1 * a_x, 0.0) == 0.0, "zero outside");
        assert_close(
            footprint.pressure_at(0.4 * a_x, -0.3 * a_y),
            footprint.pressure_at(-0.4 * a_x, 0.3 * a_y),
            1.0e-12,
            "symmetric half-ellipsoid",
        );
        assert_close(
            footprint.mean_pressure(),
            2.0 / 3.0 * p0,
            1.0e-12,
            "mean pressure is 2/3 p0",
        );
    }

    #[test]
    fn flank_pressure_peak_scales_as_the_square_root_of_approach() {
        // Q = K s^{3/2} and p0 = c_p Q^{1/3} give p0 = c_p K^{1/3} √s: the cap kisses
        // zero as √s at lift-off, the same 3/2-power signature behind the C¹ force.
        let law = sample_law();
        let (s_lo, s_hi) = (2.0e-6, 8.0e-6);
        let p_lo = law
            .flank_pressure(law.flank_load(s_lo))
            .expect("footprint")
            .peak_pressure();
        let p_hi = law
            .flank_pressure(law.flank_load(s_hi))
            .expect("footprint")
            .peak_pressure();
        assert_close(p_hi / p_lo, (s_hi / s_lo).sqrt(), 1.0e-12, "p0 ∝ √s");
    }

    #[test]
    fn traction_bound_is_mu_times_the_pressure() {
        // The Coulomb cap is just μ p, so it integrates to μ Q over the footprint.
        let footprint = sample_law()
            .flank_pressure(75.0)
            .expect("calibrated law has a footprint");
        let (a_x, a_y) = footprint.semi_axes();
        let mu = 0.12;
        assert_close(
            footprint.traction_bound(mu, 0.25 * a_x, 0.15 * a_y),
            mu * footprint.pressure_at(0.25 * a_x, 0.15 * a_y),
            1.0e-12,
            "traction bound is μ p",
        );
        // A zero friction coefficient caps the traction at zero everywhere.
        assert!(
            footprint.traction_bound(0.0, 0.0, 0.0) == 0.0,
            "μ = 0 caps at zero"
        );
    }

    #[test]
    fn a_lifted_off_flank_has_a_zero_pressure_cap() {
        // A non-positive load is a separated flank: the footprint is degenerate and
        // caps the traction at zero everywhere, so it composes with the C¹ lift-off.
        let footprint = sample_law()
            .flank_pressure(0.0)
            .expect("a zero load still yields a (zero) footprint");
        assert!(footprint.peak_pressure() == 0.0, "zero peak");
        assert!(footprint.semi_axes() == (0.0, 0.0), "zero footprint");
        assert!(footprint.pressure_at(0.0, 0.0) == 0.0, "zero pressure");
        assert!(footprint.load() == 0.0, "carries no load");
    }

    #[test]
    fn a_bare_law_has_no_pressure_footprint() {
        // The stiffness alone does not fix the contact-ellipse shape, so the bare
        // constructor cannot produce a pressure footprint — only the calibrating one.
        let bare = GothicArchLaw::new(1.0e9, 0.40);
        assert!(
            bare.flank_pressure(100.0).is_none(),
            "bare law has no footprint"
        );
        assert!(
            sample_law().flank_pressure(100.0).is_some(),
            "the calibrated law has one",
        );
    }

    #[test]
    #[should_panic(expected = "friction coefficient")]
    fn traction_bound_rejects_a_negative_friction_coefficient() {
        let footprint = sample_law().flank_pressure(50.0).expect("footprint");
        let _ = footprint.traction_bound(-0.1, 0.0, 0.0);
    }
}
