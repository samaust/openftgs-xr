// SPDX-License-Identifier: Apache-2.0

//! Dv-2 (§3.4, `dv` lane; §1.4 Stereo, §1.13 GPU validation, D-4 per view, Q-121, Q-123):
//! two eyes draw a probe for 10 s at the default 72 Hz, with no Vulkan validation message
//! in logcat in the debug build. The driver pushes P1 as `dv/probe.safetensors`, enables
//! the bundled Khronos layer through Android's GPU debug-layer settings for the app (reset
//! afterwards, also when the test fails), launches `dv2` and reads its verdict; it fails on
//! any line under the tag `VALIDATION`, checks that Android loaded the layer for the app
//! (its `GraphicsEnvironment` line naming the layer, the line the first device run showed),
//! and fails on any adapter warning about a splat camera (#224's "not a perspective
//! projection", Q-91's target format). Prerequisites: `dv_support`'s.

mod dv_support;

use std::path::PathBuf;
use std::time::Duration;

use dv_support::{PACKAGE, adb, apk, device, install, launch, log, push, stop, verdict};

/// The layer the debug APK bundles (Q-123).
const LAYER: &str = "VK_LAYER_KHRONOS_validation";

/// Android's GPU debug-layer settings for the app, reset when dropped.
struct DebugLayers;

impl DebugLayers {
    fn enable() -> DebugLayers {
        adb(&[
            "shell",
            "settings",
            "put",
            "global",
            "enable_gpu_debug_layers",
            "1",
        ]);
        adb(&[
            "shell",
            "settings",
            "put",
            "global",
            "gpu_debug_app",
            PACKAGE,
        ]);
        adb(&[
            "shell",
            "settings",
            "put",
            "global",
            "gpu_debug_layers",
            LAYER,
        ]);
        DebugLayers
    }
}

impl Drop for DebugLayers {
    fn drop(&mut self) {
        for key in [
            "enable_gpu_debug_layers",
            "gpu_debug_app",
            "gpu_debug_layers",
        ] {
            adb(&["shell", "settings", "delete", "global", key]);
        }
    }
}

/// The lines of `tag` in logcat so far.
fn tagged(tag: &str) -> Vec<String> {
    adb(&["logcat", "-d", "-s", &format!("{tag}:*")])
        .lines()
        .filter(|l| !l.starts_with("---------"))
        .map(str::to_owned)
        .collect()
}

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
    let result = verdict("dv2", start, Duration::from_secs(60));
    stop();
    let validation = tagged("VALIDATION");
    let announced = tagged("GraphicsEnvironment")
        .iter()
        .any(|l| l.contains("Vulkan debug layer list") && l.contains(LAYER));
    let warnings: Vec<String> = log()
        .into_iter()
        .filter(|l| l.contains("splat camera"))
        .collect();
    drop(layers);

    let (pass, detail, _) = result.expect("a dv2 verdict line in logcat");
    assert!(announced, "Android loaded {LAYER} for the app");
    assert!(
        validation.is_empty(),
        "no validation message ({} lines), first: {:?}",
        validation.len(),
        validation.iter().take(6).collect::<Vec<_>>()
    );
    assert!(warnings.is_empty(), "no splat camera warning: {warnings:?}");
    assert!(pass, "dv2 FAIL {detail}");
}
