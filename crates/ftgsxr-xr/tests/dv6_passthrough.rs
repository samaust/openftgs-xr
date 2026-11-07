// SPDX-License-Identifier: Apache-2.0

//! Dv-6 (§3.4, `dv` lane; §1.9 Environment modes, D-14, Q-346, Q-122): switching to
//! passthrough submits the passthrough layer, and back; the blend mode stays OPAQUE. The
//! device flips `[environment] mode` to passthrough and back twice and prints PASS when,
//! in passthrough mode, the render world's layer list holds `passthrough`'s provider
//! first and the layer is resumed, in VR mode the projection layer alone and the layer
//! paused, the blend mode OPAQUE in every frame, and every pause, resume and style call
//! accepted (the style reported, Q-337). The driver installs, launches `dv6`, reads the
//! verdict and, for PRD #565's spike check 3, scans the error-level log of the run for
//! the crate's end-frame failure and passthrough errors. Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::time::Duration;

use dv_support::{adb, apk, device, install, launch, log, stop, verdict};

/// The run's error-level lines from every tag that name the frame's end or passthrough:
/// a compositor error over the switches (spike check 3).
fn compositor_errors() -> Vec<String> {
    adb(&["logcat", "-d", "*:E"])
        .lines()
        .filter(|l| {
            let l = l.to_ascii_lowercase();
            l.contains("passthrough") || l.contains("endframe") || l.contains("end frame")
        })
        .map(str::to_owned)
        .collect()
}

#[test]
#[ignore = "dv"]
fn dv_dv6_passthrough_layer_submitted_and_removed_blend_mode_opaque() {
    let apk = apk();
    device();
    install(&apk);
    let start = launch("dv6");
    let result = verdict("dv6", start, Duration::from_secs(120));
    let lines = log();
    let errors = compositor_errors();
    stop();
    let (pass, detail, _) = result.unwrap_or_else(|| panic!("a dv6 verdict: {lines:?}"));
    // The spike check's result is printed whatever the verdict.
    println!("dv6 {} {detail}", if pass { "PASS" } else { "FAIL" });
    println!("dv6 spike check 3: {} error lines {errors:?}", errors.len());
    assert!(pass, "dv6 FAIL {detail}");
    assert!(detail.contains("cycle 1 vr"), "both cycles ran: {detail}");
    assert!(
        detail.contains("style applied"),
        "the style was sent: {detail}"
    );
    assert!(
        errors.is_empty(),
        "no compositor error over the switches: {errors:?}"
    );
}
