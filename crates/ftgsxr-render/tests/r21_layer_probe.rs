// SPDX-License-Identifier: Apache-2.0

//! R-21, the layer probe (§3.2, E14's exposure match, §2.1 `HeadlessRenderer::probe`,
//! `LayerSums`): a flat grey primitive of known α in the left half of a 200 × 150 layer
//! (not a multiple of the 64 × 48 grid, so E14's partition ⌊64x/W⌋, ⌊48y/H⌋ is uneven).
//! The cells' Σα and Σα·c_lin equal the values computed by hand from the same render's
//! layer (`render` over a black background gives the premultiplied C and A: un-premultiply,
//! decode, weight by α), within 10⁻³ relative (D-35); the cells of the right half, which
//! the primitive does not reach, are zero; and every covered cell's mean color is the grey
//! decoded.

use crate::display_support;

use display_support::{Row, model, srgb_to_linear};
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
    let grey = 0.4;
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

    // By hand, from the same render's layer.
    let image = renderer
        .render(&instances, &cam, 0.95, [0.0; 3])
        .expect("renders");
    let mut hand = vec![[0.0f64; 4]; 64 * 48];
    let mut peak = 0.0f32;
    for y in 0..H as usize {
        for x in 0..W as usize {
            let p = y * W as usize + x;
            let a = image.alpha[p];
            peak = peak.max(a);
            if a > 0.0 {
                let cell = (48 * y / H as usize) * 64 + 64 * x / W as usize;
                for (k, sum) in hand[cell].iter_mut().take(3).enumerate() {
                    let c = image.rgb[3 * p + k] / a;
                    *sum += f64::from(a) * f64::from(srgb_to_linear(c.max(0.0)));
                }
                hand[cell][3] += f64::from(a);
            }
        }
    }
    assert!(
        (peak - opacity).abs() < 0.01,
        "the known α: peak {peak}, opacity {opacity}"
    );
    for (i, (got, want)) in sums.cells.iter().zip(&hand).enumerate() {
        for k in 0..4 {
            let (g, w) = (f64::from(got[k]), want[k]);
            assert!(
                (g - w).abs() <= 1e-3 * w.abs() + 1e-7,
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
    }
    // The left half holds the footprint; each well-covered cell's mean is the grey decoded,
    // within the layer's half-float rounding of C (2⁻¹¹ relative, D-35) times the decode's
    // slope (2.4): 2.5·10⁻³. The row's 10⁻³ applies to the sums against the hand values.
    let lin = f64::from(srgb_to_linear(grey));
    let mut covered = 0;
    for got in &sums.cells {
        if got[3] > 1.0 {
            covered += 1;
            for k in 0..3 {
                let mean = f64::from(got[k]) / f64::from(got[3]);
                assert!(
                    (mean - lin).abs() <= 2.5e-3 * lin,
                    "mean {mean}, by hand {lin}"
                );
            }
        }
    }
    assert!(covered >= 16, "{covered} cells covered");
}
