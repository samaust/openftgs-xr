// SPDX-License-Identifier: Apache-2.0

//! Dv-6 (§3.4, `dv` lane; §1.9 Environment modes, D-14, Q-346, Q-122): switching to
//! passthrough submits the passthrough layer, and back; the blend mode stays OPAQUE. The
//! device flips `[environment] mode` to passthrough and back twice and prints PASS when,
//! in passthrough mode, the render world's layer list holds `passthrough`'s provider
//! first, the layer resumed and the passthrough feature started, in VR mode the
//! projection layer alone and the layer and the feature paused (owner, #568), the blend mode OPAQUE in every frame, and every pause, resume and style call
//! accepted (the style reported, Q-337). The driver installs, launches `dv6`, reads the
//! verdict and, for PRD #565's spike check 3, checks the app's log of the run has no
//! error line (an end-frame failure, a failed passthrough call) and prints the system's
//! passthrough error lines for the record. Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::time::Duration;

use dv_support::{adb, apk, device, install, launch, log, stop, verdict};

/// The run's error-level lines from every tag that name the frame's end or passthrough,
/// printed for PRD #565's spike check 3. The system's own lines (the camera pipeline's,
/// MRSS's watchdog timings while passthrough starts) appear here without any fault of
/// the app, so they are a record, not a criterion.
fn system_errors() -> Vec<String> {
    adb(&["logcat", "-d", "*:E"])
        .lines()
        .filter(|l| {
            let l = l.to_ascii_lowercase();
            l.contains("passthrough") || l.contains("endframe") || l.contains("end frame")
        })
        .map(str::to_owned)
        .collect()
}

/// The app's error-level lines (its tag): a compositor error on a submitted frame
/// reaches the app as xrEndFrame's result, which bevy\_mod\_openxr logs as "Failed to end
/// frame stream", and `passthrough` logs its failed calls (spike check 3's criterion).
fn app_errors(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|l| l.contains(" E ") && !l.contains("ftgsxr-test: dv6"))
        .cloned()
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
    let system = system_errors();
    let errors = app_errors(&lines);
    stop();
    let (pass, detail, _) = result.unwrap_or_else(|| panic!("a dv6 verdict: {lines:?}"));
    // The spike check's result is printed whatever the verdict.
    println!("dv6 {} {detail}", if pass { "PASS" } else { "FAIL" });
    println!(
        "dv6 spike check 3: app errors {errors:?}; system lines ({}) {system:?}",
        system.len()
    );
    assert!(pass, "dv6 FAIL {detail}");
    assert!(detail.contains("cycle 1 vr"), "both cycles ran: {detail}");
    assert!(
        detail.contains("layer and passthrough paused"),
        "the feature paused in VR mode: {detail}"
    );
    assert!(
        detail.contains("style applied"),
        "the style was sent: {detail}"
    );
    assert!(
        errors.is_empty(),
        "no error in the app's log over the switches: {errors:?}"
    );
}
