// SPDX-License-Identifier: Apache-2.0

//! U-9 (`cpu` lane): `sh::sh_basis` and `sh::sh_color` (E4) against hand-worked literals,
//! the properties the row states, and `ftgsxr_testkit::real_sh`, the independent real-SH
//! evaluation; with `Model::cap_sh` (§1.2 load option 1) for the cap clause.

use approx::assert_abs_diff_eq;
use ftgsxr_core::Vec3;
use ftgsxr_core::sh::{SH_C0, sh_basis, sh_color};
use ftgsxr_testkit::{random_model, real_sh};
use proptest::prelude::*;

/// U-9's tolerance for every clause but the cap's.
const TOL: f32 = 1e-6;

/// U-9: a degree-0 color is C₀·c₀₀ + 0.5 per channel, whatever the direction.
#[test]
fn u09_degree_0_is_c0_times_dc_plus_half() {
    let dc = [1.0f32, -0.5, 0.25];
    for dir in [
        Vec3::X,
        Vec3::new(-3.0, 2.0, 0.5),
        Vec3::new(0.0, 0.0, -1.0e-3),
    ] {
        let c = sh_color(0, &dc, dir);
        // 1/(2√π) = 0.28209479177387814: 0.28209479 + 0.5, −0.14104740 + 0.5, 0.07052370 + 0.5.
        assert_abs_diff_eq!(c[0], 0.782_094_8, epsilon = TOL);
        assert_abs_diff_eq!(c[1], 0.358_952_6, epsilon = TOL);
        assert_abs_diff_eq!(c[2], 0.570_523_7, epsilon = TOL);
    }
    assert_abs_diff_eq!(SH_C0, 0.282_094_8, epsilon = TOL);
}

/// A random direction of random length: a nonzero vector in [−1, 1]³ scaled by 10^k,
/// k ∈ [−3, 3], so the normalization `sh_basis` does is exercised too.
fn direction() -> impl Strategy<Value = Vec3> {
    (-1.0f32..1.0, -1.0f32..1.0, -1.0f32..1.0, -3.0f32..3.0)
        .prop_filter("a direction of length ≥ 10⁻²", |(x, y, z, _)| {
            Vec3::new(*x, *y, *z).length() >= 1e-2
        })
        .prop_map(|(x, y, z, k)| Vec3::new(x, y, z) * 10f32.powf(k))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// U-9: for degrees 1–3, `sh_basis` equals the independent real-SH evaluation within
    /// 10⁻⁶ at 1,000 random directions, and its entries past (degree + 1)² are 0.
    #[test]
    fn u09_basis_matches_real_sh(dir in direction()) {
        for degree in 1u8..=3 {
            let got = sh_basis(degree, dir);
            let want = real_sh(degree, dir);
            let k = (usize::from(degree) + 1).pow(2);
            for j in 0..16 {
                if j < k {
                    prop_assert!(
                        (f64::from(got[j]) - want[j]).abs() <= f64::from(TOL),
                        "degree {degree}, entry {j}: {} against {}", got[j], want[j]
                    );
                } else {
                    prop_assert_eq!(got[j], 0.0);
                }
            }
        }
    }

    /// U-9: for degrees 1–3, `sh_color` equals max(0, Σ c_lm Y_lm(d) + 0.5) per channel,
    /// with Y_lm the independent real-SH evaluation, within 10⁻⁶ at 1,000 random directions.
    #[test]
    fn u09_color_matches_real_sh_sum(
        dir in direction(),
        coeffs in prop::collection::vec(-1.0f32..1.0, 48),
    ) {
        for degree in 1u8..=3 {
            let k = (usize::from(degree) + 1).pow(2);
            let row = &coeffs[..3 * k];
            let y = real_sh(degree, dir);
            let got = sh_color(degree, row, dir);
            for ch in 0..3 {
                let sum: f64 = (0..k).map(|j| f64::from(row[3 * j + ch]) * y[j]).sum();
                let want = (sum + 0.5).max(0.0);
                prop_assert!(
                    (f64::from(got[ch]) - want).abs() <= f64::from(TOL),
                    "degree {degree}, channel {ch}: {} against {want}", got[ch]
                );
            }
        }
    }
}

/// U-9: the l = 1 response changes sign when the eye passes the primitive. The eye moves
/// along a line through the primitive's center μ; on either side d = (μ − o)/‖μ − o‖ is
/// opposite, so the l = 1 basis entries and a band-1-only color's deviation from 0.5 flip.
#[test]
fn u09_l1_response_changes_sign_when_the_eye_passes() {
    let mu = Vec3::new(0.2, -0.4, 1.0);
    let line = Vec3::new(0.3, 0.5, -0.8);
    // Band 1 only, small enough that no channel reaches the clamp (|Σ| ≤ C₁·0.4 < 0.5).
    #[rustfmt::skip]
    let coeffs = [
        0.0, 0.0, 0.0,
        0.1, -0.2, 0.15,
        0.2, 0.1, -0.1,
        -0.1, 0.05, 0.2,
    ];
    for t in [2.0f32, 0.5, 0.01] {
        let before = mu - (mu + t * line);
        let after = mu - (mu - t * line);
        for degree in 1u8..=3 {
            let (a, b) = (sh_basis(degree, before), sh_basis(degree, after));
            for j in 1..4 {
                assert!(
                    a[j] != 0.0 && a[j].signum() == -b[j].signum(),
                    "degree {degree}, entry {j}"
                );
                assert_abs_diff_eq!(a[j], -b[j], epsilon = TOL);
            }
        }
        let (a, b) = (sh_color(1, &coeffs, before), sh_color(1, &coeffs, after));
        for ch in 0..3 {
            let (da, db) = (a[ch] - 0.5, b[ch] - 0.5);
            assert!(
                da != 0.0 && da.signum() == -db.signum(),
                "t {t}, channel {ch}: {da}, {db}"
            );
            assert_abs_diff_eq!(da, -db, epsilon = TOL);
        }
    }
}

/// U-9: a negative sum clamps to 0 in its channel, the other channels unaffected; there is
/// no clamp above 1 (E4).
#[test]
fn u09_negative_sums_clamp_to_zero() {
    // Degree 0: C₀·(−3, 1, 4) + 0.5 = (−0.346, 0.782, 1.628).
    let c = sh_color(0, &[-3.0, 1.0, 4.0], Vec3::Y);
    assert_eq!(c[0], 0.0);
    assert_abs_diff_eq!(c[1], 0.782_094_8, epsilon = TOL);
    assert_abs_diff_eq!(c[2], 1.628_379_2, epsilon = TOL);

    // Degree 2 at d = +z: Y = (C₀, 0, C₁, 0, 0, 0, 2C₂,₀, 0, 0). Red: C₀·(−2) + C₁·(−0.5)
    // + 2C₂,₀·(−0.5) + 0.5 = −0.623 < 0; green: C₁·1 + 0.5 = 0.989; blue: 2C₂,₀·2 + 0.5 = 1.762.
    let mut coeffs = [0.0f32; 27];
    coeffs[0] = -2.0;
    coeffs[2 * 3] = -0.5;
    coeffs[2 * 3 + 1] = 1.0;
    coeffs[6 * 3] = -0.5;
    coeffs[6 * 3 + 2] = 2.0;
    let c = sh_color(2, &coeffs, Vec3::new(0.0, 0.0, 3.0));
    assert_eq!(c[0], 0.0);
    assert_abs_diff_eq!(c[1], 0.988_602_5, epsilon = TOL);
    assert_abs_diff_eq!(c[2], 1.761_566_3, epsilon = TOL);
}

/// U-9 with §1.2 load option 1: a degree-3 model capped at degree 2 equals the degree-2
/// model holding the same first nine coefficients of every row, and its E4 colors equal,
/// exactly, those of the degree-3 rows with their l = 3 band dropped (set to 0).
#[test]
fn u09_degree_2_cap_equals_degree_2_model() {
    let full = random_model(9, 64, 3);
    let mut capped = full.clone();
    capped.cap_sh(2);

    let mut degree_2 = full.clone();
    degree_2.sh_degree = 2;
    degree_2.sh = (0..64)
        .flat_map(|i| full.sh[i * 48..i * 48 + 27].iter().copied())
        .collect();
    assert_eq!(capped, degree_2);

    let dirs = [
        Vec3::new(0.3, -0.7, 0.2),
        Vec3::new(-1.0, 0.1, 0.9),
        Vec3::new(0.05, 2.0, -0.4),
        Vec3::NEG_Z,
    ];
    for i in 0..64 {
        let mut dropped = full.sh[i * 48..(i + 1) * 48].to_vec();
        dropped[27..].fill(0.0);
        for dir in dirs {
            assert_eq!(
                sh_color(2, &capped.sh[i * 27..(i + 1) * 27], dir),
                sh_color(3, &dropped, dir),
                "row {i}, direction {dir}"
            );
        }
    }
}

/// Q-T6.1: a zero direction (the eye at the primitive's center) normalizes to the zero
/// vector, so the basis is E4's table at (0, 0, 0): C₀, the l = 2, m = 0 entry −C₂,₀, the
/// rest 0; the color is finite.
#[test]
fn sh_zero_direction_gives_the_table_at_the_origin() {
    let y = sh_basis(3, Vec3::ZERO);
    for (j, v) in y.iter().enumerate() {
        match j {
            0 => assert_abs_diff_eq!(*v, 0.282_094_8, epsilon = TOL),
            6 => assert_abs_diff_eq!(*v, -0.315_391_57, epsilon = TOL),
            _ => assert_eq!(*v, 0.0, "entry {j}"),
        }
    }
    let c = sh_color(1, &[0.5; 12], Vec3::ZERO);
    assert!(c.iter().all(|v| v.is_finite()));
}

/// Q-T6.1: a degree above 3 panics.
#[test]
#[should_panic(expected = "degree 4")]
fn sh_degree_above_3_panics() {
    let _ = sh_basis(4, Vec3::X);
}

/// Q-T6.1: a coefficient count other than (degree + 1)²·3 panics.
#[test]
#[should_panic(expected = "takes 27 coefficients")]
fn sh_wrong_coefficient_count_panics() {
    let _ = sh_color(2, &[0.0; 48], Vec3::X);
}
