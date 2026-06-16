//! Uniform rectangular sampling grid for the contact interface.
//!
//! A single uniform 2-D grid discretises the (projected) contact plane. Uniform
//! spacing is mandatory: the influence kernel is translation invariant only on a
//! uniform grid, which is precisely what turns the pressure -> displacement
//! relation into a convolution and lets the FFT accelerate it.

use ndarray::Array2;

/// A uniform, origin-centred 2-D grid.
///
/// Array axis 0 indexes the `x` direction (length `nx`) and axis 1 indexes `y`
/// (length `ny`). The grid is centred so that the midpoint of the index range
/// maps to the physical origin `(0, 0)`; this places a centred contact in the
/// middle of the domain.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
}

impl Grid {
    /// Creates a grid of `nx * ny` points with spacings `dx`, `dy`.
    ///
    /// # Panics
    /// Panics if any dimension is zero, or any spacing is not strictly positive
    /// and finite.
    #[must_use]
    pub const fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        assert!(nx > 0 && ny > 0, "grid dimensions must be non-zero");
        assert!(
            dx > 0.0 && dy > 0.0 && dx.is_finite() && dy.is_finite(),
            "grid spacings must be positive and finite"
        );
        Self { nx, ny, dx, dy }
    }

    /// Creates a square grid of `n * n` points with isotropic spacing.
    #[must_use]
    pub const fn square(n: usize, spacing: f64) -> Self {
        Self::new(n, n, spacing, spacing)
    }

    /// Number of grid points along `x`.
    #[must_use]
    pub const fn nx(&self) -> usize {
        self.nx
    }

    /// Number of grid points along `y`.
    #[must_use]
    pub const fn ny(&self) -> usize {
        self.ny
    }

    /// Grid spacing along `x`.
    #[must_use]
    pub const fn dx(&self) -> f64 {
        self.dx
    }

    /// Grid spacing along `y`.
    #[must_use]
    pub const fn dy(&self) -> f64 {
        self.dy
    }

    /// Grid shape as `(nx, ny)`.
    #[must_use]
    pub const fn dims(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// Total number of grid points, `nx * ny`.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nx * self.ny
    }

    /// Always `false`: a constructed [`Grid`] is never empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Area of a single grid cell, `dx * dy`.
    #[must_use]
    pub const fn cell_area(&self) -> f64 {
        self.dx * self.dy
    }

    /// Physical `x` coordinate of index `i`, centred on the origin.
    #[must_use]
    pub const fn x(&self, i: usize) -> f64 {
        Self::coord(i, self.nx, self.dx)
    }

    /// Physical `y` coordinate of index `j`, centred on the origin.
    #[must_use]
    pub const fn y(&self, j: usize) -> f64 {
        Self::coord(j, self.ny, self.dy)
    }

    /// Samples a function of physical coordinates onto the grid.
    ///
    /// `f(x, y)` is evaluated at the centred coordinate of every grid point,
    /// returning an `nx * ny` array. This is the functional entry point used by
    /// gap functions.
    #[must_use]
    pub fn sample<F>(&self, f: F) -> Array2<f64>
    where
        F: Fn(f64, f64) -> f64,
    {
        Array2::from_shape_fn((self.nx, self.ny), |(i, j)| f(self.x(i), self.y(j)))
    }

    // Centred coordinate of index `i` on a length-`n` axis with spacing `d`.
    #[allow(
        clippy::cast_precision_loss,
        reason = "grid indices are tiny relative to f64's 53-bit integer range"
    )]
    const fn coord(i: usize, n: usize, d: f64) -> f64 {
        (i as f64 - (n as f64 - 1.0) * 0.5) * d
    }
}

#[cfg(test)]
mod tests {
    use super::Grid;

    #[test]
    fn centred_coordinates_are_symmetric() {
        let grid = Grid::new(5, 5, 1.0, 1.0);
        assert!((grid.x(0) + grid.x(4)).abs() < 1e-15);
        assert!(grid.x(2).abs() < 1e-15);
        assert!((grid.cell_area() - 1.0).abs() < 1e-15);
        assert_eq!(grid.len(), 25);
    }
}
