//! Undeformed gap (surface separation) between the two contacting bodies.

use ndarray::{Array2, ArrayView2};

use crate::grid::Grid;

/// The undeformed gap `h(x, y)`: the separation between the two surfaces before
/// any elastic deformation.
///
/// For a smooth convex contact `h` is non-negative with `h = 0` at the point of
/// first contact, but the solver only ever uses *differences* of the gap (the
/// rigid approach absorbs any constant offset), so a gap built by superposing a
/// roughness onto a base shape need not be re-normalised to a zero minimum.
///
/// A gap is the only thing the solver needs to know about geometry, so new
/// shapes (analytic profiles, measured height fields, roughness) plug in by
/// implementing this trait; [`Gap::plus`] layers them together.
pub trait Gap {
    /// Samples the gap onto the grid, returning an `nx * ny` array.
    #[must_use]
    fn sample(&self, grid: &Grid) -> Array2<f64>;

    /// Superposes `other` on top of this gap (pointwise height-field addition).
    ///
    /// Returns a [`Sum`] sampling as the sum of the two gaps on a shared grid.
    /// This is how roughness is layered onto a smooth base shape, e.g.
    /// `Paraboloid::sphere(r).plus(Waviness::new(..))`: separations of the same
    /// interface add, and the solver is invariant to a constant offset in the
    /// gap, so the operands need not be individually normalised.
    #[must_use]
    fn plus<G: Gap>(self, other: G) -> Sum<Self, G>
    where
        Self: Sized,
    {
        Sum {
            base: self,
            added: other,
        }
    }
}

/// The superposition of two gaps, sampled as their pointwise sum.
///
/// Built by [`Gap::plus`]; the canonical use is `smooth_shape.plus(roughness)`.
/// Sampling evaluates both operands on the same grid and adds them, so the two
/// may be any mix of analytic profiles and sampled height fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sum<A, B> {
    base: A,
    added: B,
}

impl<A: Gap, B: Gap> Gap for Sum<A, B> {
    fn sample(&self, grid: &Grid) -> Array2<f64> {
        let mut heights = self.base.sample(grid);
        heights += &self.added.sample(grid);
        heights
    }
}

/// An arbitrary undeformed gap supplied as a sampled height field.
///
/// The general "any shape" gap: it wraps a precomputed `nx * ny` array of
/// surface separations. Measured profilometry, a meshed surface projected onto
/// the interface, or a generated roughness realisation all enter the solver
/// this way. It composes with the analytic gaps through [`Gap::plus`], so a
/// measured or synthetic roughness can be layered onto a smooth base shape.
#[derive(Debug, Clone, PartialEq)]
pub struct HeightField {
    heights: Array2<f64>,
}

impl HeightField {
    /// Wraps a sampled height field.
    ///
    /// # Panics
    /// Panics if `heights` is empty or contains a non-finite value.
    #[must_use]
    pub fn new(heights: Array2<f64>) -> Self {
        assert!(!heights.is_empty(), "height field must be non-empty");
        assert!(
            heights.iter().all(|h| h.is_finite()),
            "height field must be finite",
        );
        Self { heights }
    }

    /// A view of the wrapped height field.
    #[must_use]
    pub fn heights(&self) -> ArrayView2<'_, f64> {
        self.heights.view()
    }
}

impl Gap for HeightField {
    /// # Panics
    /// Panics if the field's shape does not match `grid`.
    fn sample(&self, grid: &Grid) -> Array2<f64> {
        assert_eq!(
            self.heights.dim(),
            grid.dims(),
            "height-field shape must match the grid",
        );
        self.heights.clone()
    }
}

/// A conical gap `h(r) = m r` of surface slope `m = dh/dr` (Sneddon's cone).
///
/// The axisymmetric *non*-Hertzian benchmark: unlike the paraboloid the
/// curvature is singular at the apex, so the contact pressure carries an
/// (integrable) logarithmic peak at the centre. It validates the arbitrary-gap
/// path against Sneddon's closed-form cone solution
/// ([`SneddonCone`](crate::validation::SneddonCone)). The slope must be small
/// for the half-space approximation to hold (`m = cot` of the semi-apex angle).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cone {
    slope: f64,
}

impl Cone {
    /// Builds a cone from its surface slope `m = dh/dr`.
    ///
    /// # Panics
    /// Panics if `slope` is not strictly positive and finite.
    #[must_use]
    pub fn new(slope: f64) -> Self {
        assert!(
            slope > 0.0 && slope.is_finite(),
            "cone slope must be positive and finite",
        );
        Self { slope }
    }

    /// The surface slope `m = dh/dr`.
    #[must_use]
    pub const fn slope(&self) -> f64 {
        self.slope
    }
}

impl Gap for Cone {
    fn sample(&self, grid: &Grid) -> Array2<f64> {
        grid.sample(|x, y| self.slope * x.hypot(y))
    }
}

/// A doubly-periodic cosine roughness `h = A cos(2π x/λx) cos(2π y/λy)`.
///
/// A deterministic, reproducible roughness contribution for layering onto a
/// smooth gap via [`Gap::plus`]. It is zero-mean and bounded by the amplitude
/// `A`; pressed against a smooth dome it breaks the single Hertzian patch into a
/// regular array of asperity contacts, exercising the multi-contact machinery.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Waviness {
    amplitude: f64,
    wavelength_x: f64,
    wavelength_y: f64,
}

impl Waviness {
    /// Builds a roughness of amplitude `A` and wavelengths `λx`, `λy`.
    ///
    /// # Panics
    /// Panics if any argument is not strictly positive and finite.
    #[must_use]
    pub fn new(amplitude: f64, wavelength_x: f64, wavelength_y: f64) -> Self {
        assert!(
            amplitude > 0.0
                && wavelength_x > 0.0
                && wavelength_y > 0.0
                && amplitude.is_finite()
                && wavelength_x.is_finite()
                && wavelength_y.is_finite(),
            "waviness amplitude and wavelengths must be positive and finite",
        );
        Self {
            amplitude,
            wavelength_x,
            wavelength_y,
        }
    }

    /// The roughness amplitude `A`.
    #[must_use]
    pub const fn amplitude(&self) -> f64 {
        self.amplitude
    }
}

impl Gap for Waviness {
    fn sample(&self, grid: &Grid) -> Array2<f64> {
        let kx = 2.0 * core::f64::consts::PI / self.wavelength_x;
        let ky = 2.0 * core::f64::consts::PI / self.wavelength_y;
        grid.sample(|x, y| self.amplitude * (kx * x).cos() * (ky * y).cos())
    }
}

/// A paraboloidal gap `h(x, y) = x^2 / (2 Rx) + y^2 / (2 Ry)`.
///
/// The standard small-curvature approximation of a smooth convex contact. Equal
/// radii give the axisymmetric (sphere) gap used for circular Hertz contact;
/// distinct radii give the elliptic-contact gap used in the next milestone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Paraboloid {
    radius_x: f64,
    radius_y: f64,
}

impl Paraboloid {
    /// Builds a paraboloid from the two effective radii of curvature.
    ///
    /// # Panics
    /// Panics if either radius is not strictly positive and finite.
    #[must_use]
    pub fn new(radius_x: f64, radius_y: f64) -> Self {
        assert!(
            radius_x > 0.0 && radius_y > 0.0 && radius_x.is_finite() && radius_y.is_finite(),
            "curvature radii must be positive and finite",
        );
        Self { radius_x, radius_y }
    }

    /// Builds an axisymmetric paraboloid (a sphere) of a single effective radius.
    #[must_use]
    pub fn sphere(radius: f64) -> Self {
        Self::new(radius, radius)
    }

    /// The effective radius of curvature along `x`.
    #[must_use]
    pub const fn radius_x(&self) -> f64 {
        self.radius_x
    }

    /// The effective radius of curvature along `y`.
    #[must_use]
    pub const fn radius_y(&self) -> f64 {
        self.radius_y
    }
}

impl Gap for Paraboloid {
    fn sample(&self, grid: &Grid) -> Array2<f64> {
        let half_curvature_x = 0.5 / self.radius_x;
        let half_curvature_y = 0.5 / self.radius_y;
        grid.sample(|x, y| half_curvature_x * x * x + half_curvature_y * y * y)
    }
}

/// A torus, described by its tube radius `r` and centre-circle radius `R0`.
///
/// Used for the elliptic-contact benchmark (design §5.2): a sphere pressed onto
/// the torus's *outer equator*, where both principal directions are convex. The
/// two principal radii of that surface are the tube radius `r` (the tight
/// meridional section) and `R0 + r` (the gentler circumferential hoop). Pairing
/// them with a sphere yields a paraboloidal gap with distinct effective radii,
/// hence an elliptic contact — longer along the circumference.
///
/// The torus is a geometry descriptor, not a [`Gap`] on its own: a gap is the
/// separation between *two* surfaces, so the contacting sphere is supplied
/// separately by [`Torus::against_sphere`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Torus {
    tube_radius: f64,
    centre_radius: f64,
}

impl Torus {
    /// Builds a torus from its tube radius `r` and centre-circle radius `R0`.
    ///
    /// # Panics
    /// Panics if either radius is not strictly positive and finite.
    #[must_use]
    pub fn new(tube_radius: f64, centre_radius: f64) -> Self {
        assert!(
            tube_radius > 0.0
                && centre_radius > 0.0
                && tube_radius.is_finite()
                && centre_radius.is_finite(),
            "torus radii must be positive and finite",
        );
        Self {
            tube_radius,
            centre_radius,
        }
    }

    /// The tube radius `r`.
    #[must_use]
    pub const fn tube_radius(&self) -> f64 {
        self.tube_radius
    }

    /// The centre-circle radius `R0`.
    #[must_use]
    pub const fn centre_radius(&self) -> f64 {
        self.centre_radius
    }

    /// Principal radii `(circumferential, meridional)` of the outer equator.
    ///
    /// The circumferential (hoop) radius is `R0 + r` and the meridional (tube
    /// section) radius is `r`; both are convex, and the circumferential one is
    /// the larger, so a contact there is elongated circumferentially.
    #[must_use]
    pub const fn outer_equator_radii(&self) -> (f64, f64) {
        (self.centre_radius + self.tube_radius, self.tube_radius)
    }

    /// The undeformed gap of a sphere of radius `R_s` on the outer equator.
    ///
    /// Combines each torus principal radius with the sphere via
    /// `1/R = 1/R_s + 1/R_torus` to give the effective relative radii, returning
    /// the corresponding paraboloid. The circumferential radius maps to `x` and
    /// the meridional radius to `y`.
    ///
    /// # Panics
    /// Panics if `sphere_radius` is not strictly positive and finite.
    #[must_use]
    pub fn against_sphere(&self, sphere_radius: f64) -> Paraboloid {
        assert!(
            sphere_radius > 0.0 && sphere_radius.is_finite(),
            "sphere radius must be positive and finite",
        );
        let (circumferential, meridional) = self.outer_equator_radii();
        Paraboloid::new(
            combined_radius(sphere_radius, circumferential),
            combined_radius(sphere_radius, meridional),
        )
    }
}

/// Combined radius of two contacting curvatures, `1/R = 1/R1 + 1/R2`.
fn combined_radius(radius_1: f64, radius_2: f64) -> f64 {
    1.0 / (1.0 / radius_1 + 1.0 / radius_2)
}

/// Conformal (concave) relative radius of a ball in a groove, `1/R = 1/Rs - 1/r`.
///
/// A ball of radius `Rs` nestled in a concave tube of radius `r > Rs` has the
/// large effective radius `Rs r / (r - Rs)`: as the groove osculates the ball
/// (`r -> Rs`) the relative radius diverges and the contact becomes increasingly
/// conformal. Requires `r > Rs`, the precondition for a finite conformal contact.
fn conformal_radius(ball_radius: f64, tube_radius: f64) -> f64 {
    1.0 / (1.0 / ball_radius - 1.0 / tube_radius)
}

/// A Gothic-arch (ogival) groove: two equal tori whose centre circles are
/// displaced symmetrically off a shared reference centre circle.
///
/// A ball-bearing race is often ground not as a single arc but as two arcs of
/// equal radius whose centres are shifted apart by a small "shim" on either side
/// of the groove centre-line, giving the pointed, ogival "Gothic arch" profile. A
/// ball pressed into it rides on the two flanks instead of the bottom, so the
/// single conformal contact splits into two — which is the whole point of the
/// design (it fixes the contact angle and resists axial play).
///
/// Both tori share the tube radius `r` (the groove radius) and a reference
/// centre-circle radius `R0`; the ball centre sits on that reference circle. The
/// two real tori centre circles are displaced by `±centre_offset` from it, so
/// `centre_offset` is exactly the offset of each torus centre curve from the
/// reference torus. `centre_offset = 0` recovers an ordinary single-arc
/// (circular-arch) groove.
///
/// Like [`Torus`], this is a geometry descriptor, not a [`Gap`] on its own: the
/// contacting sphere is supplied by [`GothicArchGroove::against_sphere`], which
/// reduces the ball-in-groove pair to the [`GothicArchProfile`] gap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GothicArchGroove {
    tube_radius: f64,
    centre_radius: f64,
    centre_offset: f64,
}

impl GothicArchGroove {
    /// Builds a Gothic-arch groove from its tube radius `r`, reference
    /// centre-circle radius `R0`, and per-torus centre offset.
    ///
    /// # Panics
    /// Panics if `tube_radius` or `centre_radius` is not strictly positive and
    /// finite, or if `centre_offset` is negative or non-finite (`0` is allowed
    /// and yields a single-arc groove).
    #[must_use]
    pub fn new(tube_radius: f64, centre_radius: f64, centre_offset: f64) -> Self {
        assert!(
            tube_radius > 0.0
                && centre_radius > 0.0
                && tube_radius.is_finite()
                && centre_radius.is_finite(),
            "groove tube and centre radii must be positive and finite",
        );
        assert!(
            centre_offset >= 0.0 && centre_offset.is_finite(),
            "groove centre offset must be non-negative and finite",
        );
        Self {
            tube_radius,
            centre_radius,
            centre_offset,
        }
    }

    /// The tube (groove) radius `r`.
    #[must_use]
    pub const fn tube_radius(&self) -> f64 {
        self.tube_radius
    }

    /// The reference centre-circle radius `R0`.
    #[must_use]
    pub const fn centre_radius(&self) -> f64 {
        self.centre_radius
    }

    /// The offset of each torus centre curve from the reference centre circle.
    #[must_use]
    pub const fn centre_offset(&self) -> f64 {
        self.centre_offset
    }

    /// The undeformed gap of a sphere of radius `R_s` pressed into the groove.
    ///
    /// Reduces the ball-in-groove contact to its principal relative radii: the
    /// conformal meridional radius `R_y = 1/(1/R_s - 1/r)` (concave groove) and
    /// the circumferential radius `R_x = 1/(1/R_s + 1/R0)` (convex race). The
    /// meridional centre offset of each flank contact is the geometric offset
    /// amplified by the conformity, `y0 = centre_offset · R_s / (r - R_s)`: a tiny
    /// shim produces a large flank separation, which is why Gothic grooves are so
    /// sensitive to it. The resulting [`GothicArchProfile`] is the double-welled
    /// gap the solver sees.
    ///
    /// # Panics
    /// Panics if `sphere_radius` is not strictly positive and finite, or if it is
    /// not strictly smaller than the tube radius (a conformal groove contact
    /// exists only for a ball narrower than the groove).
    #[must_use]
    pub fn against_sphere(&self, sphere_radius: f64) -> GothicArchProfile {
        assert!(
            sphere_radius > 0.0 && sphere_radius.is_finite(),
            "sphere radius must be positive and finite",
        );
        assert!(
            sphere_radius < self.tube_radius,
            "ball must be narrower than the groove (sphere_radius < tube_radius) for a conformal contact",
        );
        let radius_y = conformal_radius(sphere_radius, self.tube_radius);
        let radius_x = combined_radius(sphere_radius, self.centre_radius);
        let offset = self.centre_offset * sphere_radius / (self.tube_radius - sphere_radius);
        GothicArchProfile::new(radius_x, radius_y, offset)
    }
}

/// The undeformed gap of a sphere pressed into a [`GothicArchGroove`].
///
/// A double-welled paraboloid
/// `h(x, y) = x^2 / (2 R_x) + (|y| - y0)^2 / (2 R_y)`: a single parabola across
/// the circumferential `x` (effective radius `R_x`), and *two* meridional wells
/// at `y = ±y0` joined by a ridge at the groove centre — the Gothic point, where
/// the gap is `y0^2 / (2 R_y)` and the ball never touches. Each well is locally
/// an elliptic-contact paraboloid of relative radii `(R_x, R_y)`, so a separated
/// Gothic contact is a pair of elliptic Hertz contacts sharing the load. Shrinking
/// `y0` lowers the central ridge until the two flank contacts meet and overlap into
/// a single connected patch; with `y0 = 0` the two wells merge entirely and it is
/// an ordinary [`Paraboloid`].
///
/// This is the geometric superposition the name promises: it equals the pointwise
/// minimum of two [`Paraboloid`]s offset to `y = ±y0` (the surface closest to the
/// ball wins), i.e. two tori overlaid into one concave groove.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GothicArchProfile {
    radius_x: f64,
    radius_y: f64,
    offset: f64,
}

impl GothicArchProfile {
    /// Builds the profile from the circumferential and meridional relative radii
    /// and the meridional flank offset `y0`.
    ///
    /// # Panics
    /// Panics if either radius is not strictly positive and finite, or if
    /// `offset` is negative or non-finite.
    #[must_use]
    pub fn new(radius_x: f64, radius_y: f64, offset: f64) -> Self {
        assert!(
            radius_x > 0.0 && radius_y > 0.0 && radius_x.is_finite() && radius_y.is_finite(),
            "profile relative radii must be positive and finite",
        );
        assert!(
            offset >= 0.0 && offset.is_finite(),
            "profile flank offset must be non-negative and finite",
        );
        Self {
            radius_x,
            radius_y,
            offset,
        }
    }

    /// The circumferential relative radius `R_x` (the `x` well).
    #[must_use]
    pub const fn radius_x(&self) -> f64 {
        self.radius_x
    }

    /// The meridional relative radius `R_y` (each flank well).
    #[must_use]
    pub const fn radius_y(&self) -> f64 {
        self.radius_y
    }

    /// The meridional offset `y0` of each flank contact from the groove centre.
    #[must_use]
    pub const fn offset(&self) -> f64 {
        self.offset
    }
}

impl Gap for GothicArchProfile {
    fn sample(&self, grid: &Grid) -> Array2<f64> {
        let half_curvature_x = 0.5 / self.radius_x;
        let half_curvature_y = 0.5 / self.radius_y;
        let offset = self.offset;
        grid.sample(|x, y| {
            // The meridional well follows the nearer flank: |y| folds the two
            // offset parabolas into a single ridge-at-centre profile.
            let dy = y.abs() - offset;
            half_curvature_x * x * x + half_curvature_y * dy * dy
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{
        Cone, Gap, GothicArchGroove, GothicArchProfile, HeightField, Paraboloid, Torus, Waviness,
    };
    use crate::grid::Grid;

    #[test]
    fn sphere_gap_is_zero_at_apex_and_grows_radially() {
        let grid = Grid::square(9, 1.0e-4);
        let gap = Paraboloid::sphere(5.0e-3).sample(&grid);
        let centre = gap[[4, 4]];
        assert!(centre.abs() < 1e-15, "gap must vanish at the apex");
        assert!(gap[[0, 0]] > centre, "gap must grow away from the apex");
    }

    #[test]
    fn torus_outer_equator_radii_are_convex_and_ordered() {
        let torus = Torus::new(2.0e-3, 10.0e-3);
        let (circumferential, meridional) = torus.outer_equator_radii();
        assert!((meridional - 2.0e-3).abs() <= 1e-18);
        assert!((circumferential - 12.0e-3).abs() <= 1e-18);
        // The circumferential hoop is the gentler (larger) of the two.
        assert!(circumferential > meridional);
    }

    #[test]
    fn sphere_on_torus_gap_is_elongated_circumferentially() {
        // The effective circumferential radius exceeds the meridional one, so
        // the paraboloid is flatter along x and the contact will run long there.
        let torus = Torus::new(3.0e-3, 15.0e-3);
        let gap = torus.against_sphere(8.0e-3);
        assert!(
            gap.radius_x() > gap.radius_y(),
            "circumferential (x) radius must exceed meridional (y)",
        );

        // The sampled gap rises faster along the tighter (y) axis.
        let grid = Grid::square(21, 5.0e-5);
        let sampled = gap.sample(&grid);
        let centre = 10; // middle index of a 21-point axis
        let step = 4;
        let along_x = sampled[[centre + step, centre]];
        let along_y = sampled[[centre, centre + step]];
        assert!(
            along_y > along_x,
            "gap grows faster along the meridional (y) axis",
        );
    }

    #[test]
    fn height_field_round_trips_through_sampling() {
        // A `HeightField` samples back to exactly the array it wraps.
        let grid = Grid::new(4, 6, 1.0e-3, 2.0e-3);
        let original = grid.sample(|x, y| x * x - y);
        let sampled = HeightField::new(original.clone()).sample(&grid);
        assert_eq!(sampled, original);
    }

    #[test]
    #[should_panic(expected = "height-field shape must match the grid")]
    fn height_field_rejects_a_mismatched_grid() {
        let field = HeightField::new(Array2::zeros((4, 4)));
        let _ = field.sample(&Grid::square(8, 1.0e-3));
    }

    #[test]
    fn superposition_adds_the_two_gaps_pointwise() {
        // `smooth.plus(roughness)` must equal the sum of the sampled fields.
        let grid = Grid::square(16, 5.0e-5);
        let sphere = Paraboloid::sphere(5.0e-3);
        let roughness = Waviness::new(1.0e-6, 2.0e-4, 2.0e-4);

        let combined = sphere.plus(roughness).sample(&grid);
        let expected = &sphere.sample(&grid) + &roughness.sample(&grid);

        let max_diff = (&combined - &expected)
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(max_diff <= 1e-18, "superposition must add pointwise");
    }

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "grid point counts are tiny relative to f64's integer range"
    )]
    fn waviness_is_zero_mean_and_bounded_by_its_amplitude() {
        let amplitude = 3.0e-7;
        let grid = Grid::square(64, 1.0e-5);
        let rough = Waviness::new(amplitude, 1.2e-4, 8.0e-5).sample(&grid);

        let peak = rough.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(peak <= amplitude + 1e-18, "amplitude bounds the roughness");
        // The cosine grid sum is close to zero mean (exactly so up to the
        // finite, origin-centred sampling).
        let mean = rough.sum() / rough.len() as f64;
        assert!(
            mean.abs() <= 1e-2 * amplitude,
            "roughness is near zero-mean"
        );
    }

    #[test]
    fn gothic_groove_conformity_amplifies_a_tiny_centre_offset() {
        // r/Rs = 1.04 is a textbook bearing conformity (groove radius 52% of the
        // ball diameter). The meridional relative radius is hugely conformal and
        // a small centre shim is amplified into a large flank offset.
        let ball = 3.5e-3;
        let tube = 1.04 * ball;
        let groove = GothicArchGroove::new(tube, 12.0e-3, 40.0e-6);
        let profile = groove.against_sphere(ball);

        // Meridional conformal radius 1/(1/Rs - 1/r) = Rs r / (r - Rs) = 26 Rs.
        assert!((profile.radius_y() - ball * 1.04 / 0.04).abs() <= 1e-9 * profile.radius_y());
        // Circumferential radius is the convex-convex combination, below Rs.
        assert!(profile.radius_x() < ball);
        // The flank offset is the shim amplified by Rs / (r - Rs) = 25x here.
        let amplification = ball / (tube - ball);
        assert!((profile.offset() - 40.0e-6 * amplification).abs() <= 1e-12);
        assert!(
            profile.offset() > 20.0 * 40.0e-6,
            "offset must be amplified"
        );
    }

    #[test]
    fn gothic_groove_gap_has_a_central_ridge_between_two_wells() {
        // The Gothic point: the gap is a local *maximum* at the groove centre and
        // dips to zero at the two flanks y = ±y0, so a pressed ball bridges the
        // ridge and touches the flanks rather than the bottom.
        let profile = GothicArchProfile::new(2.5e-3, 90.0e-3, 0.30e-3);
        let grid = Grid::square(401, 5.0e-6); // ±1 mm, resolves y0 = 0.30 mm
        let gap = profile.sample(&grid);
        let centre = 200; // middle index of a 401-point axis

        let ridge = gap[[centre, centre]]; // (x, y) = (0, 0): the Gothic point
        let flank = gap[[centre, centre + 60]]; // y = +0.30 mm = y0: a well floor
        assert!(
            (ridge - 0.30e-3 * 0.30e-3 / (2.0 * 90.0e-3)).abs() <= 1e-12,
            "ridge height must be y0^2 / (2 Ry)",
        );
        assert!(flank.abs() <= 1e-12, "the flank well floor must vanish");
        assert!(ridge > flank, "the centre must ride above the flank wells");
        // Symmetric wells: y = -y0 matches y = +y0.
        assert!((gap[[centre, centre - 60]] - flank).abs() <= 1e-15);
    }

    #[test]
    fn gothic_groove_reduces_to_a_paraboloid_without_an_offset() {
        // With no centre shim the two wells merge: the gap is exactly the
        // single-arc elliptic-contact paraboloid of the same relative radii.
        let groove = GothicArchGroove::new(1.04 * 4.0e-3, 15.0e-3, 0.0);
        let profile = groove.against_sphere(4.0e-3);
        assert!(profile.offset().abs() <= 1e-18, "no offset => no split");

        let grid = Grid::square(64, 1.0e-5);
        let gothic = profile.sample(&grid);
        let paraboloid = Paraboloid::new(profile.radius_x(), profile.radius_y()).sample(&grid);
        let max_diff = (&gothic - &paraboloid)
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(max_diff <= 1e-18, "offset-free Gothic gap is a paraboloid");
    }

    #[test]
    fn gothic_groove_gap_is_the_minimum_of_two_offset_paraboloids() {
        // The "two tori overlaid" promise: the groove gap equals the pointwise
        // minimum of two paraboloids shifted to y = ±y0 (closest surface wins).
        let profile = GothicArchProfile::new(3.0e-3, 80.0e-3, 0.25e-3);
        let grid = Grid::square(48, 2.0e-5);
        let gothic = profile.sample(&grid);

        let half_x = 0.5 / profile.radius_x();
        let half_y = 0.5 / profile.radius_y();
        let y0 = profile.offset();
        let expected = grid.sample(|x, y| {
            let well_plus = half_x * x * x + half_y * (y - y0) * (y - y0);
            let well_minus = half_x * x * x + half_y * (y + y0) * (y + y0);
            well_plus.min(well_minus)
        });
        let max_diff = (&gothic - &expected)
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(max_diff <= 1e-18, "Gothic gap must be the min of two wells");
    }

    #[test]
    #[should_panic(expected = "ball must be narrower than the groove")]
    fn gothic_groove_rejects_a_ball_wider_than_the_groove() {
        // A ball at least as wide as the groove has no conformal contact.
        let groove = GothicArchGroove::new(4.0e-3, 15.0e-3, 0.1e-3);
        let _ = groove.against_sphere(4.0e-3);
    }

    #[test]
    fn cone_gap_grows_linearly_with_radius() {
        // A cone rises linearly in r, twice as fast at twice the radius.
        let slope = 0.02;
        let grid = Grid::square(41, 1.0e-4);
        let gap = Cone::new(slope).sample(&grid);
        let centre = 20;
        assert!(gap[[centre, centre]].abs() < 1e-15, "apex gap vanishes");

        let near = gap[[centre + 5, centre]];
        let far = gap[[centre + 10, centre]];
        assert!(
            (far - 2.0 * near).abs() <= 1e-12 * far,
            "cone gap is linear in radius",
        );
        // Slope matches: h = m r, so h / r = m along an axis.
        let r = 5.0 * grid.dx();
        assert!((near / r - slope).abs() <= 1e-12 * slope);
    }
}
