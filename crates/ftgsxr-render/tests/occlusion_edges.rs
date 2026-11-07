// SPDX-License-Identifier: Apache-2.0

//! E13's occlusion edge across a step in the depth image (D-16's smooth edges, A-5): the
//! weight is filtered over the 2 × 2 nearest texels (Q-340, #601), so at a silhouette it
//! ramps from 0 to 1 over about one depth texel instead of stepping along the texel grid.
//! Drawn through `SplatPasses::encode_draw` with an `OcclusionInput`, each occluded frame
//! compared with the same frame drawn without one, as R-12 does.

use ftgsxr_core::{Camera, DMat3, DMat4, DVec3, UVec2};
use ftgsxr_render::{ColorTransform, RenderMode};
use ftgsxr_testkit::{gpu_context, layer_messages};

use crate::display_support::{
    Occlusion, Row, Run, draw, model, render_occluded, stored_depth, upload,
};

const W: u32 = 64;
const H: u32 = 48;

/// A pinhole camera of `w` × `h` at the origin with the world's axes and focal `f`, the
/// principal point at the center (E6).
fn camera(w: u32, h: u32, f: f64) -> Camera {
    Camera {
        width: w,
        height: h,
        k: DMat3::from_cols(
            DVec3::new(f, 0.0, 0.0),
            DVec3::new(0.0, f, 0.0),
            DVec3::new(f64::from(w) / 2.0, f64::from(h) / 2.0, 1.0),
        ),
        world_to_camera: DMat4::IDENTITY,
    }
}

/// The weight at a vertical step in the depth image. The view is 64 × 48 at focal 60;
/// the depth camera shares its pose with a 16 × 12 image at focal 15, so one depth texel
/// spans 4 × 4 view pixels and the view's sample point u + ½ projects to q = (u + ½)/4.
/// Texel columns 0–7 hold a near occluder at 1 m, columns 8–15 a far surface at 3 m
/// (stored with nearZ 0.25 and farZ 20), and a large primitive at 2 m lies between them.
/// Hard (δ = 0) and soft (δ = 5 cm, far from both surfaces) alike, the weight (the ratio of
/// the occluded and unoccluded alphas) is 0 up to the centre of texel 7 (q = 7.5,
/// u + ½ = 30) and 1 from the centre of texel 8 (q = 8.5, u + ½ = 34), and between them
/// q − 7.5 by the bilinear blend of the 2 × 2 nearest texels: 0.125, 0.375, 0.625 and
/// 0.875 at u = 30 to 33, within 10⁻³; it rises monotonically along each row, with values
/// strictly between 0 and 1, where the nearest texel gives a step from 0 to 1.
#[test]
#[ignore = "gpu"]
fn gpu_occlusion_weight_ramps_over_one_depth_texel_at_a_step() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let (near, far) = (0.25, 20.0);
    let (dw, dh) = (16u32, 12u32);
    let (occluder, surface) = (stored_depth(1.0, near, far), stored_depth(3.0, near, far));
    let depth: Vec<f32> = (0..dw * dh)
        .map(|t| if t % dw < 8 { occluder } else { surface })
        .collect();
    let rows = [Row {
        position: [0.0, 0.0, 2.0],
        velocity: [0.0; 3],
        time_center: 0.95,
        duration: 100.0,
        scale: 0.5,
        opacity: 0.999,
        color: [0.8, 0.6, 0.4],
    }];
    let m = upload(&dev, &model(&rows));
    let draws = [draw(&m, 0.95, RenderMode::Splats, ColorTransform::IDENTITY)];
    let run = Run {
        resolve: None,
        depth: false,
    };
    let view = camera(W, H, 60.0);
    let alpha = |o: Option<&Occlusion>| -> Vec<f32> {
        render_occluded(&dev, &draws, &view, &run, o)
            .layer
            .iter()
            .map(|p| p[3])
            .collect()
    };
    let reference = alpha(None);
    for (soft, delta_m) in [(false, 0.0), (true, 0.05)] {
        let occlusion = Occlusion {
            depth: depth.clone(),
            size: UVec2::new(dw, dh),
            format: wgpu::TextureFormat::R32Float,
            camera: camera(dw, dh, 15.0),
            near: near as f32,
            far: far as f32,
            soft,
            delta_m,
        };
        let occluded = alpha(Some(&occlusion));
        let mut rows_checked = 0;
        for v in 0..H {
            let at = |u: u32| (v * W + u) as usize;
            // The row's pixels around the step, all covered by the primitive.
            if (24..40).any(|u| reference[at(u)] <= 0.5) {
                continue;
            }
            rows_checked += 1;
            let weights: Vec<f32> = (24..40)
                .map(|u| occluded[at(u)] / reference[at(u)])
                .collect();
            for (k, u) in (24..40).enumerate() {
                let q = (f64::from(u) + 0.5) / 4.0;
                let expected = (q - 7.5).clamp(0.0, 1.0) as f32;
                assert!(
                    (weights[k] - expected).abs() <= 1e-3,
                    "soft {soft}, row {v}, pixel {u}: weight {}, expected {expected}",
                    weights[k]
                );
            }
            assert!(
                weights.windows(2).all(|p| p[0] <= p[1]),
                "soft {soft}, row {v}: the weight rises monotonically: {weights:?}"
            );
            assert!(
                weights.iter().any(|&w| w > 0.05 && w < 0.95),
                "soft {soft}, row {v}: intermediate weights across the step: {weights:?}"
            );
        }
        assert!(
            rows_checked > 0,
            "soft {soft}: the primitive covers the step"
        );
    }
}
