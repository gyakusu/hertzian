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

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{Cone, Gap, HeightField, Paraboloid, Torus, Waviness};
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
