// SPDX-License-Identifier: Apache-2.0

//! U-20 (§3.1, E14, D-38) through `ftgsxr_xr`'s `light` items, and the smoother's time
//! constant (PRD #664 candidate 1, Q-382), on the host (`cpu`). U-20's clause "manual EV 1
//! with auto g = √2 → g = 2√2" (E14, revision 52) is checked in `light::control`'s
//! `step_manual_offset_composes_with_the_auto_gains`: the product is formed only inside
//! the light control, and no public item composes the two gains (#696).

use bevy::math::{URect, UVec2};
use ftgsxr_app::{CameraThumb, FeatureState};
use ftgsxr_io::{EnvironmentSettings, ManualLighting};
use ftgsxr_render::LayerSums;
use ftgsxr_xr::{
    ExposureMatch, LightEstimate, LightSmoother, LightingMode, band_mask, grey_world_gains,
    light_gains, linear_luma, manual_gains, matched_exposure_ev,
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
    // E14 luma by hand: 0.299·0.5 + 0.587·0.35 + 0.114·0.2 = 0.1495 + 0.20545 + 0.0228.
    close(all.luma, 0.37775);

    let depths = [1.0, 5.0, 1.2, f32::NAN, 0.8, 5.0, 1.0, 9.0];
    let mask = band_mask(&depths, 1.0, 0.5).unwrap();
    assert_eq!(mask, [true, false, true, false, true, false, true, false]);
    let local = LightEstimate::from_thumb(&th, Some(&mask)).unwrap();
    for (a, m) in local.rgb.into_iter().zip(near) {
        close(a, m);
    }
    // By hand: 0.299·0.8 + 0.587·0.6 + 0.114·0.4 = 0.2392 + 0.3522 + 0.0456.
    close(local.luma, 0.6370);
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

/// The 64 × 48 grid's sums over a layer of `layer` pixels, each cell's from `cell(i, j)`.
fn sums(layer: UVec2, cell: impl Fn(u32, u32) -> [f32; 4]) -> LayerSums {
    let cells = (0..48)
        .flat_map(|j| (0..64).map(move |i| (i, j)))
        .map(|(i, j)| cell(i, j))
        .collect();
    LayerSums {
        cells,
        width: 64,
        height: 48,
        layer,
    }
}

/// A cell of `px` pixels of linear grey `c` at coverage α: (Σ α c, Σ α c, Σ α c, Σ α).
fn grey(c: f32, alpha: f32, px: f32) -> [f32; 4] {
    let a = alpha * px;
    [c * a, c * a, c * a, a]
}

fn set(m: ExposureMatch) -> (f32, bool) {
    match m {
        ExposureMatch::Set { ev, clamped } => (ev, clamped),
        ExposureMatch::NotInView => panic!("NotInView"),
    }
}

#[test]
fn u20_exposure_match_half_the_room_moves_ev_by_one() {
    // One pixel per cell, linear grey 0.2 at α = 1: L_model = 0.2; L_room = 0.4.
    let model = sums(UVec2::new(64, 48), |_, _| grey(0.2, 1.0, 1.0));
    for (old, new) in [(0.0, 1.0), (1.5, 2.5)] {
        let (ev, clamped) = set(matched_exposure_ev(old, 0.4, &model, None));
        close(ev, new);
        assert!(!clamped);
    }
}

#[test]
fn u20_exposure_match_clamps_to_four() {
    let model = sums(UVec2::new(64, 48), |_, _| grey(0.4, 1.0, 1.0));
    // log2(64) = 6 and log2(1/64) = −6, beyond the range.
    assert_eq!(
        set(matched_exposure_ev(0.0, 0.4 * 64.0, &model, None)),
        (4.0, true)
    );
    assert_eq!(
        set(matched_exposure_ev(0.0, 0.4 / 64.0, &model, None)),
        (-4.0, true)
    );
    // 3 + log2(4) = 5.
    assert_eq!(
        set(matched_exposure_ev(3.0, 1.6, &model, None)),
        (4.0, true)
    );
}

#[test]
fn u20_exposure_match_under_one_percent_alpha_is_not_in_view() {
    // 640 × 480: 100 pixels per cell. α = 0.009 everywhere is 0.9 % of the pixels.
    let layer = UVec2::new(640, 480);
    let faint = sums(layer, |_, _| grey(0.2, 0.009, 100.0));
    assert_eq!(
        matched_exposure_ev(0.0, 0.4, &faint, None),
        ExposureMatch::NotInView
    );
    let enough = sums(layer, |_, _| grey(0.2, 0.011, 100.0));
    close(set(matched_exposure_ev(0.0, 0.4, &enough, None)).0, 1.0);
    // Zero or non-finite lumas, and no cell, change nothing.
    let model = sums(UVec2::new(64, 48), |_, _| grey(0.2, 1.0, 1.0));
    for room in [0.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            matched_exposure_ev(0.0, room, &model, None),
            ExposureMatch::NotInView,
            "{room}"
        );
    }
    let black = sums(UVec2::new(64, 48), |_, _| grey(0.0, 1.0, 1.0));
    assert_eq!(
        matched_exposure_ev(0.0, 0.4, &black, None),
        ExposureMatch::NotInView
    );
    let none = URect::new(5, 5, 5, 9);
    assert_eq!(
        matched_exposure_ev(0.0, 0.4, &model, Some(none)),
        ExposureMatch::NotInView
    );
}

#[test]
fn u20_exposure_match_counts_only_the_cells_in_the_rectangle() {
    let inside = URect::new(10, 20, 30, 40);
    // Grey 0.2 inside the rectangle, 0.8 outside: only 0.2 counts.
    let model = sums(UVec2::new(64, 48), |i, j| {
        let c = if (10..30).contains(&i) && (20..40).contains(&j) {
            0.2
        } else {
            0.8
        };
        grey(c, 1.0, 1.0)
    });
    close(
        set(matched_exposure_ev(0.0, 0.4, &model, Some(inside))).0,
        1.0,
    );
    // Over all cells: (400·0.2 + 2672·0.8) / 3072, the rectangle 20 × 20 cells.
    let all = (400.0 * 0.2 + 2672.0 * 0.8) / 3072.0_f32;
    close(
        set(matched_exposure_ev(0.0, 0.4, &model, None)).0,
        (0.4 / all).log2(),
    );
}

#[test]
fn u20_linear_luma_of_a_flat_half_code_thumbnail() {
    // The sRGB decode of 0.5: ((0.5 + 0.055) / 1.055)^2.4.
    let th = thumb(&[[0.5; 3]; 8], 4);
    close(linear_luma(&th, None).unwrap(), 0.214_041_14);
    let mask = [true, false, true, false, true, false, true, false];
    close(linear_luma(&th, Some(&mask)).unwrap(), 0.214_041_14);
    // The linear segment: 0.02 / 12.92, and pixels the mask leaves out don't count.
    let th = thumb(&[[0.02; 3], [1.0; 3]], 2);
    close(
        linear_luma(&th, Some(&[true, false])).unwrap(),
        0.02 / 12.92,
    );
    assert!(linear_luma(&thumb(&[], 1), None).is_none());
    assert!(linear_luma(&th, Some(&[true])).is_none());
    assert!(linear_luma(&th, Some(&[false, false])).is_none());
}
