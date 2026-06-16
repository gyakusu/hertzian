//! Analytic Hertz solution for circular (axisymmetric) contact.
//!
//! Closed-form reference used to validate the solver (design §5.1). For two
//! spheres of radii `R1`, `R2` (a flat is `R2 -> infinity`) under load `P`:
//! `a = (3 P R / (4 E*))^(1/3)`, `p0 = 3 P / (2 pi a^2)`, `delta = a^2 / R`,
//! with `1/R = 1/R1 + 1/R2`, and `p(r) = p0 sqrt(1 - (r/a)^2)`.

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

#[cfg(test)]
mod tests {
    use super::HertzCircular;

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
        let expected_p0 = 2.0 * e_star * a / (core::f64::consts::PI * radius);
        assert!((hertz.max_pressure() - expected_p0).abs() <= 1e-6 * expected_p0);
        assert!((hertz.approach() - a * a / radius).abs() <= 1e-12 * hertz.approach());
    }
}
