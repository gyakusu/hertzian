//! Equivalent elastic modulus for two contacting bodies.

/// The equivalent (contact) modulus `E*`, defined by
/// `1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2`.
///
/// For frictionless normal contact the two bodies reduce to a single half-space
/// problem characterised entirely by `E*`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    e_star: f64,
}

impl Material {
    /// Wraps a precomputed equivalent modulus `E*` (pascals).
    ///
    /// # Panics
    /// Panics if `e_star` is not strictly positive and finite.
    #[must_use]
    pub fn from_e_star(e_star: f64) -> Self {
        assert!(
            e_star > 0.0 && e_star.is_finite(),
            "E* must be positive and finite"
        );
        Self { e_star }
    }

    /// Combines two isotropic materials `(E, nu)` into the equivalent modulus.
    ///
    /// # Panics
    /// Panics if the resulting modulus is not positive and finite.
    #[must_use]
    pub fn from_pair(youngs_1: f64, poisson_1: f64, youngs_2: f64, poisson_2: f64) -> Self {
        let compliance =
            (1.0 - poisson_1 * poisson_1) / youngs_1 + (1.0 - poisson_2 * poisson_2) / youngs_2;
        Self::from_e_star(1.0 / compliance)
    }

    /// The equivalent modulus `E*`.
    #[must_use]
    pub const fn e_star(&self) -> f64 {
        self.e_star
    }
}

#[cfg(test)]
mod tests {
    use super::Material;

    #[test]
    fn identical_bodies_halve_the_modulus() {
        // 1/E* = 2 (1 - nu^2)/E  =>  E* = E / (2 (1 - nu^2)).
        let youngs = 200.0e9;
        let poisson = 0.3;
        let material = Material::from_pair(youngs, poisson, youngs, poisson);
        let expected = youngs / (2.0 * (1.0 - poisson * poisson));
        assert!((material.e_star() - expected).abs() <= 1e-6 * expected);
    }
}
