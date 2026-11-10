// SPDX-License-Identifier: Apache-2.0

//! R-21, the layer probe (§3.2, E14's exposure match, §2.1 `HeadlessRenderer::probe`,
//! `LayerSums`): a flat grey primitive of known α in the left half of a 200 × 150 layer
//! (not a multiple of the 64 × 48 grid, so E14's partition ⌊64x/W⌋, ⌊48y/H⌋ is uneven).
//! The cells' Σα and Σα·c_lin equal the values computed by hand, independently of the
//! draw: α at each pixel from E6 and E7 in f64 (the primitive's known σ, μ′ and Σ′, worked
//! below), c_lin the grey decoded, within 10⁻³ relative (D-35; Σα within one half-float
//! ulp more, Q-T707.1); the cells of the right half, which the primitive does not reach,
//! are zero.

use crate::display_support;

use display_support::{Row, model};
use ftgsxr_core::{Camera, DMat3, DMat4, DVec3, Pose};
use ftgsxr_render::{HeadlessRenderer, QualityTier};
use ftgsxr_testkit::layer_messages;

const W: u32 = 200;
const H: u32 = 150;

/// probe_camera's axes (x_c = x, y_c = −z, z_c = y, the eye 4 m away), at 200 × 150 with
/// f = 100 px and the principal point at the center.
fn camera() -> Camera {
    let r = DMat3::from_cols(
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(0.0, -1.0, 0.0),
    );
    let mut w2c = DMat4::from_mat3(r);
    w2c.w_axis = DVec3::new(0.0, 0.0, 4.0).extend(1.0);
    Camera {
        width: W,
        height: H,
        k: DMat3::from_cols(
            DVec3::new(100.0, 0.0, 0.0),
            DVec3::new(0.0, 100.0, 0.0),
            DVec3::new(100.0, 75.0, 1.0),
        ),
        world_to_camera: w2c,
    }
}

#[test]
#[ignore = "gpu"]
fn gpu_r21_layer_probe_sums_the_decoded_colors_by_alpha_per_cell() {
    let _layer = layer_messages(); // R-16, Q-105
    let grey = 0.04;
    let opacity = 0.8;
    // x = −2 projects to u = 100 − 100·2/4 = 50, a quarter across; σ = 100·0.4/4 = 10 px,
    // so α < 1/255 (E7's skip) beyond about 3.2σ ≈ 32 px: the footprint ends before
    // u = 100, the layer's middle.
    let row = Row {
        position: [-2.0, 0.0, 0.0],
        velocity: [0.0; 3],
        time_center: 0.95,
        duration: 100.0,
        scale: 0.4,
        opacity,
        color: [grey; 3],
    };
    let m = model(&[row]);
    let mut renderer = HeadlessRenderer::new(QualityTier::conformance()).expect("an adapter");
    let id = renderer.load(&m, None).expect("loads");
    let instances = [(id, Pose::IDENTITY)];
    let cam = camera();
    let sums = renderer.probe(&instances, &cam, 0.95).expect("probes");
    assert_eq!((sums.width, sums.height), (64, 48));
    assert_eq!((sums.layer.x, sums.layer.y), (W, H));
    assert_eq!(sums.cells.len(), 64 * 48);

    // By hand, from E6 and E7 in f64. The eye is 4 m from the primitive (x_c = −2,
    // y_c = 0, z_c = 4): μ′ = (100 − 100·2/4, 75) = (50, 75). Σ = 0.4² I and the EWA
    // Jacobian J = [[f/z, 0, −f x_c/z²], [0, f/z, −f y_c/z²]] = [[25, 0, 12.5], [0, 25, 0]]
    // give Σ′ = 0.16 · diag(625 + 156.25, 625) + 0.3 I = diag(125.3, 100.3); σ_eff = σ = 0.8
    // at the time center (E2). α(p) = 0.8 · exp(−½ (dx²/125.3 + dy²/100.3)) at the pixel
    // centers (u + 0.5, v + 0.5), skipped below 1/255 (E7). The opacity-aware box,
    // e = √(2 ln(0.8 · 255)) ≈ 3.26 standard deviations, is the skip's own ellipse, so it
    // clips nothing more. The grey 0.04 is in the decode's linear segment, so
    // α·c_lin = α·0.04/12.92 = C/12.92: the layer's half-float rounding of C and A, 2⁻¹¹
    // relative each (D-35), stays within the row's 10⁻³ in Σα·c_lin.
    //
    // Q-T707.1: Σα gets one half-float ulp of A, 2⁻¹⁰ relative, plus 2·10⁻⁴ for E6 and
    // E7 evaluated in f32 against this f64: the layer's A comes out up to one ulp below α,
    // not rounded to nearest (0.79785156 where α = 0.798207 rounds to 0.79833984), so a
    // cell of a few pixels in the low part of their binade misses the row's 10⁻³.
    const TOL_ALPHA: f64 = 1.0 / 1024.0 + 2e-4;
    let lin = f64::from(grey) / 12.92;
    let mut hand = vec![[0.0f64; 4]; 64 * 48];
    for y in 0..H as usize {
        for x in 0..W as usize {
            let (dx, dy) = (x as f64 + 0.5 - 50.0, y as f64 + 0.5 - 75.0);
            let a = f64::from(opacity) * (-0.5 * (dx * dx / 125.3 + dy * dy / 100.3)).exp();
            if a >= 1.0 / 255.0 {
                let cell = (48 * y / H as usize) * 64 + 64 * x / W as usize;
                for sum in hand[cell].iter_mut().take(3) {
                    *sum += a * lin;
                }
                hand[cell][3] += a;
            }
        }
    }
    let mut covered = 0;
    for (i, (got, want)) in sums.cells.iter().zip(&hand).enumerate() {
        for k in 0..4 {
            let (g, w) = (f64::from(got[k]), want[k]);
            let tol = if k == 3 { TOL_ALPHA } else { 1e-3 };
            assert!(
                (g - w).abs() <= tol * w.abs(),
                "cell ({}, {}) channel {k}: probed {g}, by hand {w}",
                i % 64,
                i / 64
            );
        }
        if i % 64 >= 32 {
            assert_eq!(
                *got,
                [0.0; 4],
                "cell ({}, {}) is outside the footprint",
                i % 64,
                i / 64
            );
        }
        if want[3] > 0.0 {
            covered += 1;
        }
    }
    assert!(covered >= 16, "{covered} cells covered");

    // The decode's linear segment cannot tell lin(C/α)·α from lin(C): the same primitive
    // in grey 0.4, whose decode ((0.4 + 0.055)/1.055)^2.4 = 0.132868 is curved there, has
    // the same Σα, and each well-covered cell's Σα·c_lin / Σα is 0.132868 (lin(C) would
    // give lin(0.4α)/α, 0.11 or less). Q-T707.1: within the layer's half-float rounding of
    // C and A, 2⁻¹⁰ relative in C/A, times the decode's slope there, d ln lin / d ln c =
    // 2.4 · 0.4/0.455 ≈ 2.1: 2.5·10⁻³.
    let id = renderer
        .load(
            &model(&[Row {
                color: [0.4; 3],
                ..row
            }]),
            None,
        )
        .expect("loads");
    let sums = renderer
        .probe(&[(id, Pose::IDENTITY)], &cam, 0.95)
        .expect("probes");
    let lin = 0.132_868;
    let mut well = 0;
    for (got, want) in sums.cells.iter().zip(&hand) {
        let (g, w) = (f64::from(got[3]), want[3]);
        assert!((g - w).abs() <= TOL_ALPHA * w, "Σα {g}, by hand {w}");
        if w > 1.0 {
            well += 1;
            for c in &got[..3] {
                let mean = f64::from(*c) / g;
                assert!(
                    (mean - lin).abs() <= 2.5e-3 * lin,
                    "mean {mean}, by hand {lin}"
                );
            }
        }
    }
    assert!(well >= 16, "{well} cells well covered");
}
