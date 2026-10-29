// SPDX-License-Identifier: Apache-2.0

//! U-9's independent real spherical harmonics (E4, Q-1), evaluated in float64 from the
//! general definition rather than from E4's Cartesian table, so it shares no code and no
//! transcription with `ftgsxr_core::sh`.

use ftgsxr_core::Vec3;

/// The highest degree E4 evaluates.
const MAX_DEGREE: usize = 3;

/// The real spherical harmonics Y_lm(d), d = `dir` normalized in float64, in E4's order
/// (l = 0 … `degree`, m = −l … l) and with its signs (the Condon–Shortley phase kept);
/// entries past (`degree` + 1)² are 0. Written from the general definition: the
/// normalization √((2l + 1)/(4π) · (l − |m|)!/(l + |m|)!), the associated Legendre
/// functions P_l^|m|(z) by the standard recurrence, and √2 cos(mφ) for m > 0,
/// √2 sin(|m|φ) for m < 0, with φ = atan2(y, x).
///
/// # Panics
///
/// When `degree` > 3, or `dir` is zero or not finite.
//
// Adapted from openftgs (https://github.com/samaust/openftgs), commit 9362202,
// tests/real_sh.py (`_legendre_cs`, `real_sh`), Copyright 2026 Samuel Austin,
// Apache-2.0 (source log 29, NOTICE).
#[expect(
    clippy::needless_range_loop,
    reason = "the recurrences index by l and m"
)]
pub fn real_sh(degree: u8, dir: Vec3) -> [f64; 16] {
    let degree = usize::from(degree);
    assert!(
        degree <= MAX_DEGREE,
        "real_sh: degree {degree} > {MAX_DEGREE}"
    );
    let d = dir.as_dvec3();
    let norm = d.length();
    assert!(
        norm.is_finite() && norm > 0.0,
        "real_sh: the direction {dir} has no normalization"
    );
    let d = d / norm;
    let phi = d.y.atan2(d.x);
    let p = legendre_cs(d.z);

    let mut out = [0.0f64; 16];
    let mut k = 0;
    for l in 0..=degree {
        for m in -(l as i64)..=(l as i64) {
            let a = m.unsigned_abs() as usize;
            let norm = ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI) * factorial(l - a)
                / factorial(l + a))
            .sqrt();
            let angular = match m {
                0 => 1.0,
                m if m > 0 => std::f64::consts::SQRT_2 * (a as f64 * phi).cos(),
                _ => std::f64::consts::SQRT_2 * (a as f64 * phi).sin(),
            };
            out[k] = norm * angular * p[l][a];
            k += 1;
        }
    }
    out
}

/// P_l^m(x) for 0 ≤ m ≤ l ≤ 3 with the Condon–Shortley phase (−1)^m, indexed `[l][m]`:
/// P_m^m = −(2m − 1)√(1 − x²) P_{m−1}^{m−1}, P_{m+1}^m = (2m + 1) x P_m^m, and
/// (l − m) P_l^m = (2l − 1) x P_{l−1}^m − (l + m − 1) P_{l−2}^m.
#[expect(
    clippy::needless_range_loop,
    reason = "the recurrences index by l and m"
)]
fn legendre_cs(x: f64) -> [[f64; MAX_DEGREE + 1]; MAX_DEGREE + 1] {
    let mut p = [[0.0f64; MAX_DEGREE + 1]; MAX_DEGREE + 1];
    let s = (1.0 - x * x).max(0.0).sqrt();
    p[0][0] = 1.0;
    for m in 1..=MAX_DEGREE {
        p[m][m] = -((2 * m - 1) as f64) * s * p[m - 1][m - 1];
    }
    for m in 0..MAX_DEGREE {
        p[m + 1][m] = (2 * m + 1) as f64 * x * p[m][m];
    }
    for m in 0..=MAX_DEGREE {
        for l in m + 2..=MAX_DEGREE {
            p[l][m] = ((2 * l - 1) as f64 * x * p[l - 1][m] - (l + m - 1) as f64 * p[l - 2][m])
                / (l - m) as f64;
        }
    }
    p
}

/// n! in float64 (exact for the n ≤ 6 used here).
fn factorial(n: usize) -> f64 {
    (1..=n).map(|i| i as f64).product()
}
