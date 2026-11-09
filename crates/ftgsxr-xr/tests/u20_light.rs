// SPDX-License-Identifier: Apache-2.0

//! U-20 (§3.1, E14, D-38) through `ftgsxr_xr`'s `light` items, and the smoother's time
//! constant (PRD #664 candidate 1, Q-382), on the host (`cpu`). U-20's clause "manual EV 1
//! with auto g = √2 → g = 2√2" (E14, revision 52) is checked in `light::control`'s
//! `step_manual_offset_composes_with_the_auto_gains`: the product is formed only inside
//! the light control, and no public item composes the two gains (#696).

use ftgsxr_app::{CameraThumb, FeatureState};
use ftgsxr_io::{EnvironmentSettings, ManualLighting};
use ftgsxr_xr::{
    LightEstimate, LightSmoother, LightingMode, band_mask, grey_world_gains, light_gains,
    manual_gains,
};

const TOL: f32 = 1e-6;

fn close(a: f32, b: f32) {
    assert!((a - b).abs() <= TOL, "{a} vs {b}");
}

fn est(luma: f32, rgb: [f32; 3]) -> LightEstimate {
    LightEstimate { luma, rgb }
}

fn thumb(pixels: &[[f32; 3]], width: u32) -> CameraThumb {
    CameraThumb {
        width,
        height: pixels.len() as u32 / width,
        rgb: pixels.iter().flatten().copied().collect(),
        at_s: 0.0,
        camera: None,
    }
}

#[test]
fn u20_reference_equal_to_estimate_gives_unit_gains() {
    let e = est(0.4, [0.5, 0.4, 0.3]);
    let t = light_gains(&e, &e, 0.5);
    for c in t.gain.to_array() {
        close(c, 1.0);
    }
}

#[test]
fn u20_luma_doubled_at_half_strength_gives_sqrt2() {
    let r = est(0.2, [0.2, 0.2, 0.2]);
    let e = est(0.4, [0.4, 0.4, 0.4]);
    let t = light_gains(&e, &r, 0.5);
    for c in t.gain.to_array() {
        close(c, 2f32.sqrt());
    }
}

#[test]
fn u20_chroma_shift_gives_w_of_mean_one() {
    let r = est(0.4, [0.4, 0.4, 0.4]);
    let e = est(0.4, [0.6, 0.4, 0.2]);
    let t = light_gains(&e, &r, 0.5);
    let g = t.gain.to_array();
    close((g[0] + g[1] + g[2]) / 3.0, 1.0);
    // w_c ∝ c̄/Ȳ over the reference's: (1.5, 1, 0.5), already of mean 1.
    close(g[0], 1.5);
    close(g[1], 1.0);
    close(g[2], 0.5);
}

#[test]
fn u20_gains_clamp_at_quarter_and_four() {
    let r = est(0.01, [0.01, 0.01, 0.01]);
    let e = est(1.0, [1.0, 1.0, 1.0]);
    for c in light_gains(&e, &r, 1.0).gain.to_array() {
        close(c, 4.0);
    }
    for c in light_gains(&r, &e, 1.0).gain.to_array() {
        close(c, 0.25);
    }
    // A blue starved of light: its w falls below 0.25 before the clamp.
    let grey = est(0.5, [0.5, 0.5, 0.5]);
    let dim_blue = est(0.5, [0.5, 0.5, 0.001]);
    let g = light_gains(&dim_blue, &grey, 0.5).gain.to_array();
    close(g[2], 0.25);
}

#[test]
fn u20_manual_ev_one_gives_gain_two() {
    let m = ManualLighting {
        exposure_ev: 1.0,
        temperature: 0.0,
        tint: 0.0,
    };
    for c in manual_gains(&m).gain.to_array() {
        close(c, 2.0);
    }
}

#[test]
fn u20_grey_world_on_reddish_thumbnail_lowers_red() {
    let th = thumb(&[[0.6, 0.3, 0.3]; 4], 2);
    let e = LightEstimate::from_thumb(&th, None).unwrap();
    let g = grey_world_gains(&e).gain.to_array();
    assert!(g[0] < 1.0 && g[0] < g[1] && g[0] < g[2], "{g:?}");
    close((g[0] + g[1] + g[2]) / 3.0, 1.0);
}

#[test]
fn u20_thumbnail_statistics_global_and_band_mask() {
    let near = [0.8, 0.6, 0.4];
    let far = [0.2, 0.1, 0.0];
    let th = thumb(&[near, far, near, far, near, far, near, far], 4);
    let all = LightEstimate::from_thumb(&th, None).unwrap();
    let mean = [0.5, 0.35, 0.2];
    for (a, m) in all.rgb.into_iter().zip(mean) {
        close(a, m);
    }
    // E14 luma by hand: 0.2126·0.5 + 0.7152·0.35 + 0.0722·0.2 = 0.10630 + 0.25032 + 0.01444.
    close(all.luma, 0.37106);

    let depths = [1.0, 5.0, 1.2, f32::NAN, 0.8, 5.0, 1.0, 9.0];
    let mask = band_mask(&depths, 1.0, 0.5).unwrap();
    assert_eq!(mask, [true, false, true, false, true, false, true, false]);
    let local = LightEstimate::from_thumb(&th, Some(&mask)).unwrap();
    for (a, m) in local.rgb.into_iter().zip(near) {
        close(a, m);
    }
    // By hand: 0.2126·0.8 + 0.7152·0.6 + 0.0722·0.4 = 0.17008 + 0.42912 + 0.02888.
    close(local.luma, 0.62808);
}

#[test]
fn u20_statistics_guards() {
    assert!(LightEstimate::from_thumb(&thumb(&[], 1), None).is_none());
    let th = thumb(&[[0.5; 3]; 4], 2);
    assert!(LightEstimate::from_thumb(&th, Some(&[true; 3])).is_none());
    // Fewer than 1 % of 200 pixels in the band: the caller takes the frame.
    let mut depths = vec![9.0; 200];
    depths[0] = 1.0;
    assert!(band_mask(&depths, 1.0, 0.5).is_none());
    depths[1] = 1.0;
    assert!(band_mask(&depths, 1.0, 0.5).is_some());
}

#[test]
fn smoother_step_held_for_tau_moves_by_one_minus_inverse_e() {
    let mut s = LightSmoother::new(0.5);
    assert!(s.current().is_none());
    let a = est(0.2, [0.2, 0.3, 0.4]);
    let b = est(0.6, [1.0, 0.7, 0.4]);
    assert_eq!(s.push(a, 10.0), a);
    let out = s.push(b, 10.5);
    let k = 1.0 - (-1.0f32).exp();
    close(out.luma, 0.2 + k * 0.4);
    close(out.rgb[0], 0.2 + k * 0.8);
    close(out.rgb[1], 0.3 + k * 0.4);
    close(out.rgb[2], 0.4);
    assert_eq!(s.current(), Some(out));
    // Δt ≤ 0 leaves the estimate unchanged.
    assert_eq!(s.push(a, 10.5), out);
    assert_eq!(s.push(a, 10.0), out);
    s.reset();
    assert!(s.current().is_none());
    assert_eq!(s.push(b, 0.0), b);
}

#[test]
fn lighting_mode_from_settings_and_the_camera_gate() {
    let mut env = EnvironmentSettings::default();
    let manual = ManualLighting {
        exposure_ev: 1.0,
        temperature: 0.5,
        tint: -0.5,
    };
    env.lighting_manual = manual.clone();
    let m = |env: &EnvironmentSettings, c| LightingMode::from_settings(env, c);
    assert_eq!(m(&env, FeatureState::Available), LightingMode::Off);
    env.lighting_mode = "manual".into();
    assert_eq!(
        m(&env, FeatureState::Absent),
        LightingMode::Manual(manual.clone())
    );
    env.lighting_mode = "auto-global".into();
    assert_eq!(m(&env, FeatureState::Available), LightingMode::AutoGlobal);
    assert_eq!(
        m(&env, FeatureState::Denied),
        LightingMode::Manual(manual.clone())
    );
    env.lighting_mode = "auto-local".into();
    assert_eq!(m(&env, FeatureState::Available), LightingMode::AutoLocal);
    assert_eq!(m(&env, FeatureState::Absent), LightingMode::Manual(manual));
}
