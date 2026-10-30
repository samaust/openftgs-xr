// SPDX-License-Identifier: Apache-2.0

//! R-14, the light transform (§3.2, E14; Q-80): gains (2, 1, 1) on a flat grey primitive,
//! drawn and resolved onto an sRGB view target: the displayed value equals the file's grey
//! decoded, multiplied by 2 in red, clamped and encoded, by hand, within 1/255.

mod display_support;

use display_support::{
    Run, center_block, draw, flat_probe, linear_to_srgb, probe_camera, render, srgb_to_linear,
    upload,
};
use ftgsxr_core::Vec3;
use ftgsxr_render::{ColorTransform, RenderMode};
use ftgsxr_testkit::gpu_context;

#[test]
#[ignore = "gpu"]
fn gpu_r14_gains_apply_in_linear_light() {
    let dev = gpu_context();
    let grey = 0.4;
    let model = flat_probe([grey; 3]);
    let gpu_model = upload(&dev, &model);
    let transform = ColorTransform {
        gain: Vec3::new(2.0, 1.0, 1.0),
    };
    let draws = [draw(&gpu_model, 0.95, RenderMode::Splats, transform)];
    let frame = render(
        &dev,
        &draws,
        &probe_camera(),
        &Run {
            resolve: Some(Some([0.0; 3])),
            depth: false,
        },
    );
    let red = linear_to_srgb((2.0 * srgb_to_linear(grey)).clamp(0.0, 1.0));
    let expected = [red, grey, grey];
    assert!(
        red - grey > 0.1,
        "the gain is visible ({red} against {grey})"
    );
    let target = frame.target.expect("resolved");
    for i in center_block() {
        for (k, want) in expected.iter().enumerate() {
            let shown = f32::from(target[i][k]) / 255.0;
            assert!(
                (shown - want).abs() <= 1.0 / 255.0,
                "pixel {i} channel {k}: shown {shown}, by hand {want}"
            );
        }
    }
}
