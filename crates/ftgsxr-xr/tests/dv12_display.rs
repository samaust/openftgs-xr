// SPDX-License-Identifier: Apache-2.0

//! Dv-12 (§3.4, `dv` lane, #228): 90 Hz selected takes effect; every foveation level is
//! accepted; an `[xr] render_scale` of 0.8 re-creates the swapchain, and 1.0 re-creates it
//! again, both sizes logged (#729, Q-412). The runner's `dv12` drives the settings; this
//! driver reads `display`'s log lines (Q-124, Q-131, Q-132). Prerequisites: `dv_support`'s.

use crate::dv_support;

use std::time::Duration;

use dv_support::{apk, device, install, launch, log, stop, verdict};

/// `(w, h)` from the `WxH` after `prefix` in `line`.
fn size_after(line: &str, prefix: &str) -> Option<(u32, u32)> {
    let rest = line.split_once(prefix)?.1;
    let word = rest.split_whitespace().next()?;
    let (w, h) = word.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

#[test]
#[ignore = "dv"]
fn dv_dv12_90_hz_every_foveation_level_and_render_scale_0_8_then_1_0() {
    let apk = apk();
    device();
    install(&apk);
    let start = launch("dv12");
    let result = verdict("dv12", start, Duration::from_secs(120));
    let lines = log();
    stop();
    let (pass, detail, _) = result.expect("a dv12 verdict line in logcat");
    assert!(pass, "dv12 FAIL {detail}");

    assert!(
        lines
            .iter()
            .any(|l| l.contains("display refresh rate") && l.contains("-> 90 Hz")),
        "the runtime's change event to 90 Hz"
    );
    for level in ["Off", "Low", "Medium", "High", "Dynamic"] {
        assert!(
            lines
                .iter()
                .any(|l| l.contains(&format!("foveation {level} applied to the swapchain"))),
            "foveation {level} accepted"
        );
    }
    // `display`'s line at each session: `swapchain WxH at render scale s`.
    let swapchains: Vec<(u32, u32)> = lines
        .iter()
        .filter(|l| l.contains("at render scale"))
        .filter_map(|l| size_after(l, "swapchain "))
        .collect();
    let requested: Vec<(u32, u32)> = lines
        .iter()
        .filter_map(|l| size_after(l, "re-creating the session with a "))
        .collect();
    assert_eq!(
        requested.len(),
        2,
        "two re-creations, 0.8 then 1.0: {requested:?}"
    );
    assert_eq!(
        swapchains.len(),
        3,
        "the launch's swapchain and two re-created: {swapchains:?}"
    );
    eprintln!("dv12: swapchains {swapchains:?}, requested {requested:?}");
    // Each re-created swapchain has the size requested of it (Q-132's base, Q-412).
    assert_eq!(swapchains[1], requested[0], "the 0.8 swapchain");
    assert_eq!(swapchains[2], requested[1], "the 1.0 swapchain");
    let (first, small, back) = (swapchains[0], swapchains[1], swapchains[2]);
    assert!(
        small.0 < first.0 && small.1 < first.1,
        "0.8's {small:?} under the launch's {first:?}"
    );
    assert!(
        small.0 < back.0 && small.1 < back.1,
        "0.8's {small:?} under 1.0's {back:?}"
    );
    let errors: Vec<&String> = lines.iter().filter(|l| l.contains("E-XR-")).collect();
    assert!(errors.is_empty(), "no XR error: {errors:?}");
}
