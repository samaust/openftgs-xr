// SPDX-License-Identifier: Apache-2.0

//! R-11, the color space (§3.2, D-6; §1.4 stage 7, §1.14 Color): a flat-color probe drawn
//! into the file-space layer and resolved onto an sRGB view target shows the file's color
//! within 1/255; a linear-light misreading (the file's values taken as linear) would differ
//! by more. The background (VR mode) and passthrough forms of the resolve are both checked
//! against E7's C + (1 − A)·c_bg, formed from the layer (Q-T116.4).

mod display_support;

use display_support::{
    Run, center_block, draw, flat_probe, linear_to_srgb, probe_camera, render, upload,
};
use ftgsxr_render::{ColorTransform, RenderMode};
use ftgsxr_testkit::gpu_context;

const TOLERANCE: f32 = 1.0 / 255.0;

fn byte(v: u8) -> f32 {
    f32::from(v) / 255.0
}

#[test]
#[ignore = "gpu"]
fn gpu_r11_resolve_shows_the_files_color_on_an_srgb_target() {
    let dev = gpu_context();
    let color = [0.5, 0.25, 0.8];
    // The misreading would store encode(c): the tolerance tells the two apart.
    let misread = color.map(linear_to_srgb);
    assert!(
        color
            .iter()
            .zip(misread)
            .any(|(c, m)| (m - c).abs() > TOLERANCE),
        "the linear-light misreading differs from the file's color by more than 1/255"
    );
    let model = flat_probe(color);
    let gpu_model = upload(&dev, &model);
    let draws = [draw(
        &gpu_model,
        0.95,
        RenderMode::Splats,
        ColorTransform::IDENTITY,
    )];
    let cam = probe_camera();

    // VR mode over a grey background, and passthrough (no background).
    for background in [Some([0.1, 0.1, 0.1]), None] {
        let frame = render(
            &dev,
            &draws,
            &cam,
            &Run {
                resolve: Some(background),
                depth: false,
            },
        );
        let target = frame.target.as_ref().expect("resolved");
        for i in center_block() {
            let [r, g, b, a] = target[i];
            for (shown, file) in [byte(r), byte(g), byte(b)].into_iter().zip(color) {
                assert!(
                    (shown - file).abs() <= TOLERANCE,
                    "{background:?} pixel {i}: shown {shown}, the file's {file}"
                );
            }
            assert!((byte(a) - 1.0).abs() <= TOLERANCE, "alpha {a}");
        }
        // Every pixel: the shown color is E7's composite of the layer (Q-T116.4); without
        // a background the target keeps the layer's alpha and shows C/A where covered.
        for (i, (&[c0, c1, c2, a], &shown)) in frame.layer.iter().zip(target).enumerate() {
            let c = [c0, c1, c2];
            match background {
                Some(bg) => {
                    for k in 0..3 {
                        let expected = c[k] + (1.0 - a) * bg[k];
                        assert!(
                            (byte(shown[k]) - expected).abs() <= TOLERANCE,
                            "pixel {i} channel {k}: {} against {expected}",
                            byte(shown[k])
                        );
                    }
                    assert_eq!(shown[3], 255);
                }
                None => {
                    assert!((byte(shown[3]) - a).abs() <= TOLERANCE, "pixel {i} alpha");
                    if a > 0.5 {
                        for k in 0..3 {
                            // The target holds encode(lin(C/A)·A): decoded, ÷ A, encoded.
                            let lin = display_support::srgb_to_linear(byte(shown[k])) / a;
                            let expected = c[k] / a;
                            assert!(
                                (linear_to_srgb(lin) - expected).abs() <= 2.0 * TOLERANCE,
                                "pixel {i} channel {k}"
                            );
                        }
                    }
                }
            }
        }
    }
}
