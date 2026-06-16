//! Undeformed gap (surface separation) between the two contacting bodies.

use ndarray::Array2;

use crate::grid::Grid;

/// The undeformed gap `h(x, y) >= 0`: the separation between the two surfaces
/// before any elastic deformation, with `h = 0` at the point of first contact.
///
/// A gap is the only thing the solver needs to know about geometry, so new
/// shapes (height fields, roughness, meshes) plug in by implementing this trait.
pub trait Gap {
    /// Samples the gap onto the grid, returning an `nx * ny` array.
    #[must_use]
    fn sample(&self, grid: &Grid) -> Array2<f64>;
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
    use super::{Gap, Paraboloid, Torus};
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
}
