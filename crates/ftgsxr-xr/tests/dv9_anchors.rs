// SPDX-License-Identifier: Apache-2.0

//! Dv-9 (§3.4, `dv` lane; §1.9 Floor, shadows and anchors, D-26, Q-346, Q-122): create,
//! persist, restart the app, locate, within 5 cm. The driver grants USE\_ANCHOR\_API (so
//! no dialog waits for a person) and launches `dv9` twice: the first launch creates an
//! anchor 1 m ahead of the content group's origin, persists it and prints
//! `uuid=<uuid> pose=<x>,<y>,<z>,<qx>,<qy>,<qz>,<qw>` in its verdict's detail with the
//! creation's and the save's latencies (PRD #565's spike check 4); the driver passes that
//! part to the second launch in the extra `ftgsxr-anchor`, which queries the UUID (spike
//! check 5), locates it, PASS within 5 cm, and erases it. Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::time::{Duration, Instant};

use dv_support::{ACTIVITY, adb, apk, device, install, log, stop, verdict};

/// The anchors permission of §2.5's manifest.
const PERMISSION: &str = "com.oculus.permission.USE_ANCHOR_API";

/// Launches `dv9`, restarting the app, with `anchor` in the extra `ftgsxr-anchor`.
fn launch(anchor: Option<&str>) -> Instant {
    adb(&["logcat", "-c"]);
    let start = Instant::now();
    // `am start` runs in the device's shell: the detail is quoted for it.
    let quoted = anchor.map(|a| format!("'{a}'"));
    let mut args = vec![
        "shell",
        "am",
        "start",
        "-S",
        "-n",
        ACTIVITY,
        "--es",
        "ftgsxr-test",
        "dv9",
    ];
    if let Some(q) = &quoted {
        args.extend(["--es", "ftgsxr-anchor", q.as_str()]);
    }
    adb(&args);
    start
}

#[test]
#[ignore = "dv"]
fn dv_dv9_an_anchor_persists_across_a_restart_within_5_cm() {
    let apk = apk();
    device();
    install(&apk);
    dv_support::pm("dv9", "grant", &[PERMISSION]);

    let start = launch(None);
    let first = verdict("dv9", start, Duration::from_secs(120));
    let lines = log();
    stop();
    let (pass, detail, _) = first.unwrap_or_else(|| panic!("a first dv9 verdict: {lines:?}"));
    // Spike check 4's latencies are in the detail; print it whatever the verdict.
    println!("dv9 first {} {detail}", if pass { "PASS" } else { "FAIL" });
    assert!(pass, "dv9 first launch FAIL {detail}");
    let anchor = detail
        .find("uuid=")
        .map(|i| detail[i..].to_owned())
        .unwrap_or_else(|| panic!("the anchor in the detail: {detail}"));
    assert!(anchor.contains(" pose="), "the created pose: {detail}");

    let start = launch(Some(&anchor));
    let second = verdict("dv9", start, Duration::from_secs(120));
    let lines = log();
    stop();
    let (pass, detail, _) = second.unwrap_or_else(|| panic!("a second dv9 verdict: {lines:?}"));
    // Spike check 5 (found by its UUID after the restart) and the distance.
    println!("dv9 second {} {detail}", if pass { "PASS" } else { "FAIL" });
    assert!(pass, "dv9 second launch FAIL {detail}");
    assert!(
        detail.contains("found by its UUID"),
        "the query found it: {detail}"
    );
    assert!(
        detail.contains("within 0.05 m: true"),
        "within 5 cm: {detail}"
    );
    assert!(detail.contains("erased"), "the anchor erased: {detail}");
}
