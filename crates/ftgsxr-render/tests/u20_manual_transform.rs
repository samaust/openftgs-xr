// SPDX-License-Identifier: Apache-2.0

//! U-20, the manual case (§3.1, E14, `cpu` lane; Q-250): `ColorTransform::from_manual`
//! gives g = 2^EV with no clamp and w = (1 + 0.4 t, 1 − 0.4 u, 1 − 0.4 t) scaled to mean 1,
//! the gain g w; the identity at zero. Every value is worked by hand, within 10⁻⁶.

use ftgsxr_core::Vec3;
use ftgsxr_io::ManualLighting;
use ftgsxr_render::ColorTransform;

/// U-20's tolerance.
const TOL: f32 = 1e-6;

fn gain(exposure_ev: f32, temperature: f32, tint: f32) -> Vec3 {
    ColorTransform::from_manual(&ManualLighting {
        exposure_ev,
        temperature,
        tint,
    })
    .gain
}

fn assert_near(got: Vec3, want: Vec3, case: &str) {
    assert!(
        (got - want).abs().max_element() <= TOL,
        "{case}: got {got}, by hand {want}"
    );
}

// Q-250
#[test]
fn u20_manual_mode_gives_e14s_manual_form() {
    // Zero → the identity.
    assert_eq!(
        ColorTransform::from_manual(&ManualLighting::default()),
        ColorTransform::IDENTITY
    );
    // EV 1 → g = 2 (U-20's manual case).
    assert_near(gain(1.0, 0.0, 0.0), Vec3::splat(2.0), "EV 1");
    // EV ±4 → g = 16 and 1/16: no clamp to [0.25, 4] in manual mode (Q-250).
    assert_near(gain(4.0, 0.0, 0.0), Vec3::splat(16.0), "EV 4");
    assert_near(gain(-4.0, 0.0, 0.0), Vec3::splat(0.0625), "EV -4");
    // t = 1 → w = (1.4, 1, 0.6), already of mean 1: warmer.
    assert_near(gain(0.0, 1.0, 0.0), Vec3::new(1.4, 1.0, 0.6), "t = 1");
    // u = 1 → w = (1, 0.6, 1) of mean 2.6 / 3, scaled by 3 / 2.6 = 1.153846…: less green.
    assert_near(
        gain(0.0, 0.0, 1.0),
        Vec3::new(1.153_846_2, 0.692_307_7, 1.153_846_2),
        "u = 1",
    );
}
