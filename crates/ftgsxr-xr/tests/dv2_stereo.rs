// SPDX-License-Identifier: Apache-2.0

//! Dv-2 (§3.4, `dv` lane; §1.4 Stereo, §1.13 GPU validation, D-4, Q-399, Q-123, spike S5):
//! two eyes draw a probe for 10 s at the default 72 Hz, with no Vulkan validation message
//! in logcat in the debug build. The driver pushes P1 as `dv/probe.safetensors`, enables
//! the bundled Khronos layer through Android's GPU debug-layer settings for the app (reset
//! afterwards, also when the test fails), launches `dv2` and reads its verdict; it fails on
//! any line under the tag `VALIDATION`, checks that Android loaded the layer for the app
//! (its `GraphicsEnvironment` line naming the layer, the line the first device run showed),
//! and fails on any adapter warning about a splat camera (#224's "not a perspective
//! projection", Q-91's target format). It also checks the pixels: a screenshot while both
//! eyes draw must show the probe in each eye (Q-141). Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::path::PathBuf;
use std::time::Duration;

use dv_support::{DebugLayers, Validation, apk, device, install, launch, log, push, stop, verdict};

#[test]
#[ignore = "dv"]
fn dv_dv2_two_eyes_draw_a_probe_for_10_s_without_validation_messages() {
    let apk = apk();
    device();
    install(&apk);
    let probe = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/probes/P1.safetensors");
    push(&probe, "probe.safetensors");
    let layers = DebugLayers::enable();
    let start = launch("dv2");
    let shot = screenshot_while_drawing(start);
    let result = verdict("dv2", start, Duration::from_secs(60));
    stop();
    let validation = Validation::read();
    let warnings: Vec<String> = log()
        .into_iter()
        // The adapter's warnings (`W` lines), not dv2's own info lines.
        .filter(|l| l.contains(" W ") && l.contains("splat camera"))
        .collect();
    drop(layers);

    let (pass, detail, _) = result.expect("a dv2 verdict line in logcat");
    validation.assert_clean();
    assert!(warnings.is_empty(), "no splat camera warning: {warnings:?}");
    assert!(pass, "dv2 FAIL {detail}");
    let shot = shot.expect("a screenshot while both eyes draw");
    for (eye, lit) in probe_pixels(&shot).into_iter().enumerate() {
        assert!(
            lit >= MIN_PROBE_PIXELS,
            "eye {eye} shows the probe: {lit} pixels apart from the background"
        );
    }
}

/// The pixels each eye's probe must cover in the screenshot. P1's splat (σ 0.08 m,
/// 1 m ahead) spans some 5° of the eye's field, well above this.
const MIN_PROBE_PIXELS: usize = 200;
/// How far (8-bit units, any channel) a pixel must lie from the background to count.
const PROBE_CONTRAST: i32 = 40;

/// A screenshot of the compositor's view (both eyes side by side) 3 s after `dv2` reports
/// that both eyes draw, as decoded RGBA. Q-141
fn screenshot_while_drawing(start: std::time::Instant) -> Option<image::RgbaImage> {
    while start.elapsed() < Duration::from_secs(40) {
        if log()
            .iter()
            .any(|l| l.contains("both eyes are splat cameras"))
        {
            std::thread::sleep(Duration::from_secs(3));
            let out = std::process::Command::new(dv_support::adb_path())
                .args(["exec-out", "screencap", "-p"])
                .output()
                .expect("adb screencap runs");
            return image::load_from_memory(&out.stdout)
                .ok()
                .map(|i| i.to_rgba8());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    None
}

/// For each eye (the left and right halves), the pixels of the central fifth of the half
/// (where the head-locked probe stands) that differ from the background, the median of
/// that region, by more than `PROBE_CONTRAST`.
fn probe_pixels(shot: &image::RgbaImage) -> [usize; 2] {
    let (w, h) = shot.dimensions();
    let half = w / 2;
    let mut out = [0; 2];
    for (eye, n) in out.iter_mut().enumerate() {
        let x0 = eye as u32 * half + half * 2 / 5;
        let (x1, y0, y1) = (x0 + half / 5, h * 2 / 5, h * 3 / 5);
        let mut lum: Vec<[u8; 3]> = Vec::new();
        for y in y0..y1 {
            for x in x0..x1 {
                let p = shot.get_pixel(x, y).0;
                lum.push([p[0], p[1], p[2]]);
            }
        }
        let mut sorted: Vec<u32> = lum
            .iter()
            .map(|p| p.iter().map(|&c| u32::from(c)).sum())
            .collect();
        sorted.sort_unstable();
        let median_sum = sorted[sorted.len() / 2];
        let bg = lum
            .iter()
            .find(|p| p.iter().map(|&c| u32::from(c)).sum::<u32>() == median_sum)
            .copied()
            .unwrap_or([0; 3]);
        *n = lum
            .iter()
            .filter(|p| {
                p.iter()
                    .zip(bg)
                    .any(|(&c, b)| (i32::from(c) - i32::from(b)).abs() > PROBE_CONTRAST)
            })
            .count();
    }
    out
}
