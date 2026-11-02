// SPDX-License-Identifier: Apache-2.0

//! Dv-12 (§3.4, `dv` lane, #228): 90 Hz selected takes effect; every foveation level is
//! accepted; a render scale of 0.8 re-creates the swapchain, the new size logged. The
//! runner's `dv12` drives the settings; this driver reads `display`'s log lines (Q-124,
//! Q-131, Q-132). Prerequisites: `dv_support`'s.

mod dv_support;

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
fn dv_dv12_90_hz_every_foveation_level_and_render_scale_0_8() {
    let apk = apk();
    device();
    install(&apk);
    let start = launch("dv12");
    let result = verdict("dv12", start, Duration::from_secs(90));
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
    let swapchains: Vec<(u32, u32)> = lines
        .iter()
        .filter_map(|l| size_after(l, "swapchain "))
        .collect();
    let requested = lines
        .iter()
        .find_map(|l| size_after(l, "re-creating the session with a "))
        .expect("the re-creation line");
    assert!(
        swapchains.len() >= 2,
        "two sessions' swapchains: {swapchains:?}"
    );
    let (first, last) = (swapchains[0], *swapchains.last().unwrap());
    assert_eq!(last, requested, "the new swapchain has the requested size");
    assert!(
        last.0 < first.0 && last.1 < first.1,
        "{last:?} under {first:?}"
    );
    let errors: Vec<&String> = lines.iter().filter(|l| l.contains("E-XR-")).collect();
    assert!(errors.is_empty(), "no XR error: {errors:?}");
}
