// SPDX-License-Identifier: Apache-2.0

//! Dv-7 (§3.4, `dv` lane; §1.9 Depth occlusion, E13, Q-346, Q-122): after USE\_SCENE is
//! granted the provider starts, a depth frame with its camera and planes arrives every
//! frame for both views, the provider stops cleanly. The driver grants USE\_SCENE (so no
//! dialog waits for a person), launches `dv7` and reads the verdict; the device sets
//! passthrough mode and occlusion on, counts for 3 s the frames that had two depth frames
//! with finite planes and distinct cameras, turns occlusion off and checks the provider
//! stopped with no failed call. For PRD #565's spike checks 1 and 2 the detail gives the
//! swapchain's size and length and whether the views' GPU textures kept their ids, and the
//! driver scans the run's log for Vulkan validation messages (the layer the debug APK
//! bundles, Q-123: a format, layer-count or image-layout mismatch of the copies, Q-356).
//! Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::process::Command;
use std::time::Duration;

use dv_support::{PACKAGE, adb, adb_path, apk, device, install, launch, log, stop, verdict};

/// The scene permission of §2.5's manifest, which the Depth API needs (Appendix E.1).
const PERMISSION: &str = "com.oculus.permission.USE_SCENE";

/// `pm grant`, whose failure (a permission the system grants at install) is only printed.
fn grant() {
    let out = Command::new(adb_path())
        .args(["shell", "pm", "grant", PACKAGE, PERMISSION])
        .output();
    match out {
        Ok(o) if o.status.success() => {}
        Ok(o) => println!(
            "dv7: pm grant {PERMISSION}: {}{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        ),
        Err(e) => println!("dv7: pm grant {PERMISSION}: {e}"),
    }
}

/// The run's Vulkan validation messages (spike checks 1 and 2, Q-356).
fn validation_messages() -> Vec<String> {
    adb(&["logcat", "-d"])
        .lines()
        .filter(|l| l.contains("VUID") || l.contains("Validation Error"))
        .map(str::to_owned)
        .collect()
}

#[test]
#[ignore = "dv"]
fn dv_dv7_depth_frames_every_frame_and_a_clean_stop() {
    let apk = apk();
    device();
    install(&apk);
    grant();
    let start = launch("dv7");
    let result = verdict("dv7", start, Duration::from_secs(120));
    let lines = log();
    let validation = validation_messages();
    stop();
    let (pass, detail, _) = result.unwrap_or_else(|| panic!("a dv7 verdict: {lines:?}"));
    // The spike checks' observations are printed whatever the verdict.
    println!("dv7 {} {detail}", if pass { "PASS" } else { "FAIL" });
    println!(
        "dv7 spike checks 1 and 2: {} validation lines {validation:?}",
        validation.len()
    );
    assert!(pass, "dv7 FAIL {detail}");
    assert!(detail.contains("provider stopped"), "stopped: {detail}");
    assert!(
        detail.contains("kept across frames: true"),
        "spike check 2: {detail}"
    );
}
