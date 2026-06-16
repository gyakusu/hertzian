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

#[cfg(test)]
mod tests {
    use super::{Gap, Paraboloid};
    use crate::grid::Grid;

    #[test]
    fn sphere_gap_is_zero_at_apex_and_grows_radially() {
        let grid = Grid::square(9, 1.0e-4);
        let gap = Paraboloid::sphere(5.0e-3).sample(&grid);
        let centre = gap[[4, 4]];
        assert!(centre.abs() < 1e-15, "gap must vanish at the apex");
        assert!(gap[[0, 0]] > centre, "gap must grow away from the apex");
    }
}
