// SPDX-License-Identifier: Apache-2.0

//! Dv-7 (§3.4, `dv` lane; §1.9 Depth occlusion, E13, Q-346, Q-122): after USE\_SCENE is
//! granted the provider starts, a depth frame with its camera and planes arrives every
//! frame for both views, the provider stops cleanly. The driver grants USE\_SCENE (so no
//! dialog waits for a person), launches `dv7` and reads the verdict; the device sets
//! passthrough mode and occlusion on, counts for 3 s the frames that had two depth frames
//! with finite planes and distinct cameras, turns occlusion off and checks the provider
//! stopped with no failed call. For PRD #565's spike checks 1 and 2 the detail gives the
//! swapchain's size and length and whether the views' GPU textures kept their ids, and the
//! driver runs with the validation layer the debug APK bundles (Q-123) and requires it
//! loaded and silent (a format, layer-count, usage or image-layout mismatch of the
//! conversion pass, Q-356, Q-360).
//! Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::time::Duration;

use dv_support::{DebugLayers, Validation, apk, device, install, launch, log, stop, verdict};

/// The scene permission of §2.5's manifest, which the Depth API needs (Appendix E.1).
const PERMISSION: &str = "com.oculus.permission.USE_SCENE";

#[test]
#[ignore = "dv"]
fn dv_dv7_depth_frames_every_frame_and_a_clean_stop() {
    let apk = apk();
    device();
    install(&apk);
    dv_support::pm("dv7", "grant", &[PERMISSION]);
    // Q-123: the bundled validation layer, whose messages are spike checks 1 and 2's.
    let layers = DebugLayers::enable();
    let start = launch("dv7");
    let result = verdict("dv7", start, Duration::from_secs(120));
    let lines = log();
    stop();
    let validation = Validation::read();
    drop(layers);
    let (pass, detail, _) = result.unwrap_or_else(|| panic!("a dv7 verdict: {lines:?}"));
    // The spike checks' observations are printed whatever the verdict.
    println!("dv7 {} {detail}", if pass { "PASS" } else { "FAIL" });
    println!(
        "dv7 spike checks 1 and 2: layer loaded {}, {} validation lines {:?}",
        validation.announced,
        validation.messages.len(),
        validation.messages.iter().take(12).collect::<Vec<_>>()
    );
    assert!(pass, "dv7 FAIL {detail}");
    assert!(detail.contains("provider stopped"), "stopped: {detail}");
    assert!(
        detail.contains("kept across frames: true"),
        "spike check 2: {detail}"
    );
    // Spike check 1 and Q-356: no format, usage or layout message about the conversion pass.
    validation.assert_clean();
}
