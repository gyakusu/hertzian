//! Zero-padded discrete-convolution FFT (DC-FFT) for the free-space kernel.
//!
//! Evaluates `u = K * p` as a *linear* (free-space, non-periodic) convolution. A
//! plain FFT computes a *cyclic* convolution, which corresponds to a periodic
//! tiling of the contact — wrong for an isolated Hertz contact. The fix (Liu,
//! Wang & Liu, 2000) is to zero-pad both operands to twice the grid size and lay
//! the kernel out in wrap-around order, so the cyclic result coincides with the
//! linear convolution on the original region.
//!
//! Real input/output is exploited with a real-to-complex transform along the
//! contiguous axis (about half the work and memory of a full complex FFT).
//!
//! Following the crate's functional-core / imperative-shell convention,
//! [`DcFft::apply`] is referentially transparent (a fresh array out, no
//! observable mutation); the local scratch buffers and in-place transforms are
//! an implementation detail.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    reason = "padded grid sizes are tiny relative to the f64/isize ranges"
)]

use std::sync::Arc;

use ndarray::{s, Array2, ArrayView1, ArrayView2};
use num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};

use crate::grid::Grid;
use crate::kernel::influence_coefficient;

/// Plans and precomputed kernel spectrum for zero-padded DC-FFT convolution.
pub struct DcFft {
    nx: usize,
    ny: usize,
    mx: usize,
    my: usize,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
    fft_col: Arc<dyn Fft<f64>>,
    ifft_col: Arc<dyn Fft<f64>>,
    kernel_spectrum: Array2<Complex<f64>>,
}

impl DcFft {
    /// Builds the convolver for `grid` and equivalent modulus `e_star`.
    ///
    /// The influence kernel is sampled in wrap-around order on the padded grid
    /// and transformed once; [`DcFft::apply`] then reuses this spectrum.
    #[must_use]
    pub fn new(grid: &Grid, e_star: f64) -> Self {
        let (nx, ny) = grid.dims();
        let (mx, my) = (2 * nx, 2 * ny);

        let mut real_planner = RealFftPlanner::<f64>::new();
        let r2c = real_planner.plan_fft_forward(my);
        let c2r = real_planner.plan_fft_inverse(my);

        let mut planner = FftPlanner::<f64>::new();
        let fft_col = planner.plan_fft_forward(mx);
        let ifft_col = planner.plan_fft_inverse(mx);

        let kernel = wrap_around_kernel(grid, e_star, mx, my);
        let kernel_spectrum = forward_2d(r2c.as_ref(), fft_col.as_ref(), kernel);

        Self {
            nx,
            ny,
            mx,
            my,
            r2c,
            c2r,
            fft_col,
            ifft_col,
            kernel_spectrum,
        }
    }

    /// Applies the operator, returning `u = K * p` on the original grid.
    ///
    /// # Panics
    /// Panics if `pressure` does not match the convolver's grid dimensions.
    #[must_use]
    pub fn apply(&self, pressure: ArrayView2<'_, f64>) -> Array2<f64> {
        assert_eq!(
            pressure.dim(),
            (self.nx, self.ny),
            "pressure shape must match the convolver grid",
        );

        let mut padded = Array2::<f64>::zeros((self.mx, self.my));
        padded
            .slice_mut(s![0..self.nx, 0..self.ny])
            .assign(&pressure);

        let mut spectrum = forward_2d(self.r2c.as_ref(), self.fft_col.as_ref(), padded);
        spectrum.zip_mut_with(&self.kernel_spectrum, |value, kernel| *value *= *kernel);

        self.inverse_2d(spectrum)
    }

    // Inverse transform (columns then rows), normalise, crop to the valid region.
    fn inverse_2d(&self, mut spectrum: Array2<Complex<f64>>) -> Array2<f64> {
        for mut col in spectrum.columns_mut() {
            let mut buf = col.to_vec();
            self.ifft_col.process(&mut buf);
            col.assign(&ArrayView1::from(buf.as_slice()));
        }

        let half = self.my / 2 + 1;
        let mut full = Array2::<f64>::zeros((self.mx, self.my));
        let mut input = self.c2r.make_input_vec();
        let mut output = self.c2r.make_output_vec();
        for (mut dst, src) in full.rows_mut().into_iter().zip(spectrum.rows()) {
            input.copy_from_slice(src.as_slice().expect("spectrum row is contiguous"));
            // The half-spectrum of a real signal has real DC and Nyquist terms;
            // clear any rounding residue so the C2R transform stays exact.
            input[0].im = 0.0;
            input[half - 1].im = 0.0;
            self.c2r
                .process(&mut input, &mut output)
                .expect("c2r length invariant holds by construction");
            dst.assign(&ArrayView1::from(output.as_slice()));
        }

        let scale = 1.0 / (self.mx * self.my) as f64;
        full.slice(s![0..self.nx, 0..self.ny])
            .mapv(|value| value * scale)
    }
}

// Forward 2-D real FFT: R2C along the contiguous axis, then C2C along axis 0.
fn forward_2d(
    r2c: &dyn RealToComplex<f64>,
    fft_col: &dyn Fft<f64>,
    mut real: Array2<f64>,
) -> Array2<Complex<f64>> {
    let (mx, my) = real.dim();
    let half = my / 2 + 1;
    let mut spectrum = Array2::<Complex<f64>>::zeros((mx, half));

    let mut output = r2c.make_output_vec();
    for (mut dst, mut src) in spectrum.rows_mut().into_iter().zip(real.rows_mut()) {
        r2c.process(
            src.as_slice_mut().expect("real row is contiguous"),
            &mut output,
        )
        .expect("r2c length invariant holds by construction");
        dst.assign(&ArrayView1::from(output.as_slice()));
    }
    for mut col in spectrum.columns_mut() {
        let mut buf = col.to_vec();
        fft_col.process(&mut buf);
        col.assign(&ArrayView1::from(buf.as_slice()));
    }
    spectrum
}

// The influence kernel sampled on the padded grid in wrap-around order.
fn wrap_around_kernel(grid: &Grid, e_star: f64, mx: usize, my: usize) -> Array2<f64> {
    let (nx, ny) = grid.dims();
    Array2::from_shape_fn((mx, my), |(i, j)| {
        match (wrap_offset(i, nx, mx), wrap_offset(j, ny, my)) {
            (Some(di), Some(dj)) => influence_coefficient(grid, di, dj, e_star),
            _ => 0.0,
        }
    })
}

// Maps a padded index to its signed offset; the single wrap point maps to None.
const fn wrap_offset(i: usize, n: usize, m: usize) -> Option<isize> {
    if i < n {
        Some(i as isize)
    } else if i == n {
        None
    } else {
        Some(i as isize - m as isize)
    }
}
