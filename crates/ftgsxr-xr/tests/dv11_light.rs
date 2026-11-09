// SPDX-License-Identifier: Apache-2.0

//! Dv-11 (§3.4, `dv` lane; §1.9 Light estimation, E14, Q-390, Q-122): with the camera
//! permissions granted, thumbnails arrive at 30 Hz or more; covering the cameras lowers
//! the luma estimate by half or more within 1 s. The driver grants both camera
//! permissions with `pm grant` (so no dialog waits for a person), installs, launches
//! `dv11`, prints the cue for the person when the device logs `dv11: cover`, reads the
//! verdict, requires clause 1 and reports clause 2's state: a clause 2 "not run" (no
//! cover within the device's 20 s) leaves Dv-11 unverified until the owner runs it, not
//! failed (Q-390, Q-273). It prints the shim's logged YUV data space and range (Q-386).
//! Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::process::Command;
use std::time::{Duration, Instant};

use dv_support::{PACKAGE, adb_path, apk, device, install, launch, log, stop, verdict};

/// The camera permissions of §2.5's manifest, which light estimation needs (§1.9).
const PERMISSIONS: [&str; 2] = [
    "android.permission.CAMERA",
    "horizonos.permission.HEADSET_CAMERA",
];

/// `pm grant` (or `revoke`) of each camera permission; a failure is only printed.
fn pm(verb: &str) {
    for p in PERMISSIONS {
        let out = Command::new(adb_path())
            .args(["shell", "pm", verb, PACKAGE, p])
            .output();
        match out {
            Ok(o) if o.status.success() => {}
            Ok(o) => println!(
                "dv11: pm {verb} {p}: {}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            ),
            Err(e) => println!("dv11: pm {verb} {p}: {e}"),
        }
    }
}

fn wait_for(line: &str, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if log().iter().any(|l| l.contains(line)) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

#[test]
#[ignore = "dv"]
fn dv_dv11_thumbnails_at_30_hz_and_the_cover_halves_the_luma() {
    let apk = apk();
    device();
    install(&apk);
    pm("grant");
    let start = launch("dv11");
    if wait_for("dv11: cover", Duration::from_secs(90)) {
        println!("dv11: COVER THE HEADSET'S CAMERAS NOW (20 s)");
    }
    let result = verdict("dv11", start, Duration::from_secs(150));
    let lines = log();
    stop();
    pm("revoke");
    for l in lines
        .iter()
        .filter(|l| l.contains("passthrough camera data space"))
    {
        println!("dv11 Q-386: {l}");
    }
    let (pass, detail, _) = result.unwrap_or_else(|| panic!("a dv11 verdict: {lines:?}"));
    println!("dv11 {} {detail}", if pass { "PASS" } else { "FAIL" });
    assert!(detail.contains("clause 1 PASS"), "clause 1: {detail}");
    assert!(pass, "dv11 FAIL {detail}");
    if detail.contains("clause 2 not run") {
        println!("dv11: clause 2 not run; Dv-11 is unverified until the owner covers the cameras");
    }
}
