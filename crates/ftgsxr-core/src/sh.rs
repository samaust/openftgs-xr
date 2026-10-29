// SPDX-License-Identifier: Apache-2.0

//! The real spherical-harmonics basis and E4's color: the basis in gsplat's band order and
//! signs (the Condon–Shortley phase kept), the sum over the loaded degree L′, the +0.5
//! offset and the clamp at 0 per channel, with no clamp above 1. Evaluated in float64 and
//! rounded to float32 once (Q-T6.1).

use glam::Vec3;

/// C₀ = 1/(2√π), the l = 0 real SH constant (E4).
#[expect(
    clippy::excessive_precision,
    reason = "§2.1's literal, E4's C₀ to float64 precision"
)]
pub const SH_C0: f32 = 0.282_094_791_773_878_14;

/// The highest degree E4 evaluates (§1.2: L ∈ {0, 1, 2, 3}).
const MAX_DEGREE: u8 = 3;

// E4's constants, named C<l>_<|m|>, in float64.
const C0: f64 = 0.282_094_791_773_878_14; // 1/(2√π)
const C1: f64 = 0.488_602_511_902_919_9; // √(3/(4π))
const C2_0: f64 = 0.315_391_565_252_520_05; // ¼√(5/π)
const C2_1: f64 = 1.092_548_430_592_079_2; // ½√(15/π)
const C2_2: f64 = 0.546_274_215_296_039_6; // ¼√(15/π)
const C3_0: f64 = 0.373_176_332_590_115_4; // ¼√(7/π)
const C3_1: f64 = 0.457_045_799_464_465_8; // ¼√(21/(2π))
const C3_2: f64 = 1.445_305_721_320_277; // ¼√(105/π)
const C3_3: f64 = 0.590_043_589_926_643_5; // ¼√(35/(2π))

/// E4's offset added to the SH sum before the clamp at 0.
const COLOR_OFFSET: f64 = 0.5;

/// Y_lm(dir) for l ≤ degree in E4's order and signs; `dir` is normalized here; entries past (degree+1)² are 0.
///
/// A zero `dir` (the eye at the primitive's center) is normalized to the zero vector, so
/// its basis is E4's table at (0, 0, 0) (Q-T6.1).
///
/// # Panics
///
/// When `degree` > 3 (Q-T6.1).
pub fn sh_basis(degree: u8, dir: Vec3) -> [f32; 16] {
    basis_f64(degree, dir).map(|y| y as f32)
}

/// E4's color for one primitive: `coeffs` holds (degree+1)²·3 values, coefficient-major.
///
/// The color is max(0, Σ c_lm Y_lm(d) + 0.5) per channel, with d = `dir` normalized as
/// [`sh_basis`] does; a channel may exceed 1.
///
/// # Panics
///
/// When `degree` > 3, or `coeffs.len()` is not (degree + 1)²·3 (Q-T6.1).
pub fn sh_color(degree: u8, coeffs: &[f32], dir: Vec3) -> [f32; 3] {
    let k = (usize::from(degree) + 1).pow(2);
    // Q-T6.1: a coefficient count other than the degree's is a caller's error.
    assert_eq!(
        coeffs.len(),
        3 * k,
        "sh_color: degree {degree} takes {} coefficients, got {}",
        3 * k,
        coeffs.len()
    );
    let basis = basis_f64(degree, dir);
    let mut sum = [0.0f64; 3];
    let (rows, _) = coeffs.as_chunks::<3>();
    for (y, c) in basis[..k].iter().zip(rows) {
        for (s, &c) in sum.iter_mut().zip(c) {
            *s += y * f64::from(c);
        }
    }
    sum.map(|s| (s + COLOR_OFFSET).max(0.0) as f32)
}

/// E4's basis in float64 for `dir` normalized in float64.
fn basis_f64(degree: u8, dir: Vec3) -> [f64; 16] {
    // Q-T6.1: degrees above E4's table are a caller's error.
    assert!(
        degree <= MAX_DEGREE,
        "sh_basis: degree {degree} > {MAX_DEGREE}"
    );
    let d = dir.as_dvec3();
    let len = d.length();
    // Q-T6.1: a zero direction stays zero rather than becoming NaN.
    let d = if len > 0.0 { d / len } else { d };
    let (x, y, z) = (d.x, d.y, d.z);

    let mut out = [0.0f64; 16];
    out[0] = C0;
    if degree >= 1 {
        out[1] = -C1 * y;
        out[2] = C1 * z;
        out[3] = -C1 * x;
    }
    if degree >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        out[4] = C2_1 * x * y;
        out[5] = -C2_1 * y * z;
        out[6] = C2_0 * (3.0 * zz - 1.0);
        out[7] = -C2_1 * x * z;
        out[8] = C2_2 * (xx - yy);
        if degree >= 3 {
            out[9] = -C3_3 * y * (3.0 * xx - yy);
            out[10] = 2.0 * C3_2 * x * y * z;
            out[11] = -C3_1 * y * (5.0 * zz - 1.0);
            out[12] = C3_0 * z * (5.0 * zz - 3.0);
            out[13] = -C3_1 * x * (5.0 * zz - 1.0);
            out[14] = C3_2 * z * (xx - yy);
            out[15] = -C3_3 * x * (xx - 3.0 * yy);
        }
    }
    out
}
