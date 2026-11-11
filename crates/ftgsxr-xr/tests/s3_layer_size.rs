// SPDX-License-Identifier: Apache-2.0

//! Spike S3 (PRD #718, #729, `dv` lane; §1.12 Headset display at revision 56, Q-164,
//! Q-412): the runner's `s3` runs the app in `quest-low` with `[xr] render_scale` unset;
//! from the log, the swapchain is at the view configuration's recommended size (scale 1)
//! and each eye's splat layer at 0.8 of it, rounded (Q-164's `layer_size`). The driver
//! prints both sizes for the PRD. Prerequisites: `dv_support`'s.

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
fn dv_s3_quest_low_swapchain_at_the_recommended_size_and_the_splat_layer_at_0_8() {
    let apk = apk();
    device();
    install(&apk);
    let start = launch("s3");
    let result = verdict("s3", start, Duration::from_secs(60));
    let lines = log();
    stop();
    let (pass, detail, _) = result.expect("an s3 verdict line in logcat");
    assert!(pass, "s3 FAIL {detail}");

    // `display`'s line: `swapchain WxH at render scale s (§1.12); the view configuration
    // recommends RxR`.
    let session = lines
        .iter()
        .rfind(|l| l.contains("at render scale") && l.contains("recommends"))
        .expect("display's swapchain line");
    let swapchain = size_after(session, "swapchain ").expect("the swapchain's size");
    let recommended = size_after(session, "recommends ").expect("the recommendation");
    // The adapter's line: `splat layer LxL for a VxV view at render scale 0.8`.
    let layers: Vec<((u32, u32), (u32, u32))> = lines
        .iter()
        .filter(|l| l.contains("splat layer") && l.contains("at render scale 0.8"))
        .filter_map(|l| Some((size_after(l, "splat layer ")?, size_after(l, "for a ")?)))
        .collect();
    eprintln!(
        "s3: swapchain {swapchain:?}, recommended {recommended:?}, layers (layer, view) {layers:?}"
    );
    assert_eq!(
        swapchain, recommended,
        "the swapchain at the recommended size"
    );
    assert!(
        session.contains("at render scale 1 "),
        "the swapchain at [xr] render_scale 1.0: {session}"
    );
    assert!(!layers.is_empty(), "the adapter's layer line at 0.8");
    let scaled = |v: u32| ((v as f32 * 0.8).round() as u32).max(1);
    for (layer, view) in layers {
        assert_eq!(view, swapchain, "the eye's view is the swapchain's image");
        assert_eq!(layer, (scaled(view.0), scaled(view.1)), "the layer at 0.8");
    }
}
