// SPDX-License-Identifier: Apache-2.0

//! Dv-8 (§3.4, `dv` lane; §1.8 Hands and Controllers, D-25, D-44, Appendix D.3, Q-272,
//! Q-122): joints and the aim state arrive, an index pinch above 0.7 produces `Select`,
//! a grab start's 20 ms pulse reaches the controller's haptic output, as Q-272 reads the
//! row without a person: both trackers locate for 5 s with every locate succeeding (the
//! verdict reports whether a hand was seen, the aim state's COMPUTED and VALID flags,
//! PRD #452's spike check 2, the simultaneous hands-and-controllers resume, spike check
//! 5, and the wrist rate under each frequency hint, Q-275); a synthetic pinch of 0.8
//! writes `Select`; the pulse call on the active controller succeeds, or "no controller".
//! The driver grants the hand-tracking permission (so no dialog waits for a person),
//! launches `dv8` and reads its verdict. Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::process::Command;
use std::time::Duration;

use dv_support::{PACKAGE, adb_path, apk, device, install, launch, log, stop, verdict};

/// The hand-tracking permission of §2.5's manifest.
const PERMISSION: &str = "com.oculus.permission.HAND_TRACKING";

/// `pm grant`, whose failure (a permission the system grants at install) is only printed.
fn grant() {
    let out = Command::new(adb_path())
        .args(["shell", "pm", "grant", PACKAGE, PERMISSION])
        .output();
    match out {
        Ok(o) if o.status.success() => {}
        Ok(o) => println!(
            "dv8: pm grant {PERMISSION}: {}{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        ),
        Err(e) => println!("dv8: pm grant {PERMISSION}: {e}"),
    }
}

#[test]
#[ignore = "dv"]
fn dv_dv8_joints_and_aim_state_arrive_a_pinch_selects_a_grab_pulses() {
    let apk = apk();
    device();
    install(&apk);
    grant();
    let start = launch("dv8");
    let result = verdict("dv8", start, Duration::from_secs(150));
    let lines = log();
    stop();
    let (pass, detail, _) = result.unwrap_or_else(|| panic!("a dv8 verdict: {lines:?}"));
    // The spike checks' results are in the detail; print it whatever the verdict.
    println!("dv8 {} {detail}", if pass { "PASS" } else { "FAIL" });
    assert!(pass, "dv8 FAIL {detail}");
    assert!(
        detail.contains("joints: "),
        "the joints clause ran: {detail}"
    );
    assert!(
        detail.contains("0 failed"),
        "every locate succeeded: {detail}"
    );
    assert!(
        detail.contains("aim state: "),
        "spike check 2 reported: {detail}"
    );
    assert!(
        detail.contains("simultaneous resume: "),
        "spike check 5 reported: {detail}"
    );
    assert!(
        detail.contains("wrote Select"),
        "the pinch selected: {detail}"
    );
    assert!(
        detail.contains("pulse: no controller") || detail.contains("succeeded"),
        "the pulse clause: {detail}"
    );
}
