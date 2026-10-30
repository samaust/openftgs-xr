// SPDX-License-Identifier: Apache-2.0

//! U-19 (E13's weight, its CPU mirror `covariance::occlusion_weight` of Q-63, `cpu` lane):
//! hard: 1 up to z_e + δ and 0 beyond; soft: 1, 0.5 and 0 at z_e − δ, z_e and z_e + δ; an
//! invalid sample gives 1. 10⁻⁶. At δ ≤ 0 both modes are the hard step at z_e (Q-63).
//!
//! The depths are exact binary fractions (z_e = 1 m, δ = 0.25 m), so z_e ± δ is exact in f32
//! and the hard edge is tested at the edge itself and one ulp beyond it.

// Q-63
use approx::assert_abs_diff_eq;
use ftgsxr_core::occlusion_weight;

/// The row's tolerance.
const TOL: f32 = 1e-6;
/// The environment depth z_e (m).
const Z_E: f32 = 1.0;
/// The margin δ (m).
const DELTA: f32 = 0.25;

#[test]
fn u19_hard_weight_is_one_up_to_z_e_plus_delta_and_zero_beyond() {
    for z_f in [0.1, 0.75, 1.0, 1.25] {
        assert_eq!(
            occlusion_weight(z_f, Some(Z_E), DELTA, false),
            1.0,
            "z_f = {z_f}"
        );
    }
    let beyond = f32::from_bits(1.25_f32.to_bits() + 1);
    for z_f in [beyond, 1.5, 3.0] {
        assert_eq!(
            occlusion_weight(z_f, Some(Z_E), DELTA, false),
            0.0,
            "z_f = {z_f}"
        );
    }
}

#[test]
fn u19_soft_weight_is_one_half_and_zero_at_z_e_minus_delta_z_e_and_z_e_plus_delta() {
    for (z_f, w) in [(0.75, 1.0), (1.0, 0.5), (1.25, 0.0)] {
        assert_abs_diff_eq!(
            occlusion_weight(z_f, Some(Z_E), DELTA, true),
            w,
            epsilon = TOL
        );
    }
}

#[test]
fn u19_soft_weight_is_one_minus_smoothstep_inside_and_clamped_outside() {
    // t = (0.875 − 0.75)/0.5 = 0.25: smoothstep = t²(3 − 2t) = 0.15625, so w = 0.84375;
    // t = 0.75 mirrors it: w = 0.15625.
    assert_abs_diff_eq!(
        occlusion_weight(0.875, Some(Z_E), DELTA, true),
        0.84375,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        occlusion_weight(1.125, Some(Z_E), DELTA, true),
        0.15625,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        occlusion_weight(0.1, Some(Z_E), DELTA, true),
        1.0,
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        occlusion_weight(3.0, Some(Z_E), DELTA, true),
        0.0,
        epsilon = TOL
    );
}

#[test]
fn u19_soft_weight_at_zero_delta_is_the_hard_step() {
    // Q-63 (owner answer): δ = 0 gives 1 for z_f ≤ z_e and 0 beyond, never NaN.
    let beyond = f32::from_bits(Z_E.to_bits() + 1);
    for (z_f, w) in [(0.5, 1.0), (Z_E, 1.0), (beyond, 0.0), (2.0, 0.0)] {
        assert_eq!(
            occlusion_weight(z_f, Some(Z_E), 0.0, true),
            w,
            "z_f = {z_f}"
        );
    }
}

#[test]
fn u19_a_negative_delta_is_the_hard_step_at_z_e_in_both_modes() {
    // Q-63: at δ ≤ 0 the weight is the hard step at z_e itself, 1 for z_f ≤ z_e and 0
    // beyond, in hard and soft mode alike; a negative δ does not move the edge to z_e + δ.
    let beyond = f32::from_bits(Z_E.to_bits() + 1);
    for soft in [false, true] {
        for (z_f, w) in [
            (0.5, 1.0),
            (Z_E - DELTA, 1.0),
            (0.9, 1.0),
            (Z_E, 1.0),
            (beyond, 0.0),
            (2.0, 0.0),
        ] {
            assert_eq!(
                occlusion_weight(z_f, Some(Z_E), -DELTA, soft),
                w,
                "z_f = {z_f}, soft = {soft}"
            );
        }
    }
}

#[test]
fn u19_an_invalid_or_absent_sample_gives_one() {
    for soft in [false, true] {
        for z_f in [0.1, 1.0, 3.0] {
            assert_eq!(
                occlusion_weight(z_f, None, DELTA, soft),
                1.0,
                "z_f = {z_f}, soft = {soft}"
            );
        }
    }
}
