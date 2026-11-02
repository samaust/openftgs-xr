// SPDX-License-Identifier: Apache-2.0

//! `display` on Monado (`xr` lane, #228's second criterion as the owner rewrote it on
//! 2026-11-02, Q-124): `DisplaySettings.available_rates` is what the runtime reports;
//! foveation and the recommended layer resolution are unsupported, and a request for
//! either logs E-XR-UNSUPPORTED with the setting kept. Prerequisites: `xr_support`'s.

mod xr_support;

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_openxr::resources::OxrInstance;
use bevy_mod_openxr::session::OxrSession;
use bevy_mod_xr::session::XrState;
use ftgsxr_render::{Foveation, SortMode};
use ftgsxr_xr::{DisplaySettings, XrPlugin};
use tracing::Level;
use xr_support::{Monado, capture_log, lines, xr_app};

fn run_until_running(app: &mut App) {
    let start = Instant::now();
    while app.world().get_resource::<XrState>() != Some(&XrState::Running) {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "the session runs within 20 s"
        );
        app.update();
    }
}

fn unsupported_lines(what: &str) -> usize {
    lines()
        .iter()
        .filter(|l| {
            l.level == Level::WARN
                && l.target == "ftgsxr_xr::display"
                && l.message.contains("E-XR-UNSUPPORTED")
                && l.message.contains(what)
        })
        .count()
}

#[test]
#[ignore = "xr"]
fn xr_display_rates_are_the_runtimes_and_foveation_and_layer_resolution_are_unsupported() {
    capture_log();
    let _monado = Monado::start();
    let mut app = xr_app(
        XrPlugin {
            passthrough: false,
            hands: false,
            sort: SortMode::PerView,
        },
        |_| {},
    );
    run_until_running(&mut app);
    for _ in 0..5 {
        app.update();
    }

    // The rates are what the runtime reports (an empty list without the extension).
    let has_ext = app
        .world()
        .resource::<OxrInstance>()
        .exts()
        .fb_display_refresh_rate
        .is_some();
    let reported = if has_ext {
        app.world()
            .resource::<OxrSession>()
            .enumerate_display_refresh_rates()
            .expect("the runtime's rates")
    } else {
        Vec::new()
    };
    assert_eq!(
        app.world().resource::<DisplaySettings>().available_rates,
        reported
    );

    // Foveation: requested, unsupported, kept.
    let before = unsupported_lines("XR_FB_foveation");
    app.world_mut().resource_mut::<DisplaySettings>().foveation = Foveation::Low;
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(unsupported_lines("XR_FB_foveation"), before + 1);
    assert_eq!(
        app.world().resource::<DisplaySettings>().foveation,
        Foveation::Low
    );

    // The recommended layer resolution: a render-scale request logs it unsupported and
    // the scale is kept (the re-creation uses the view configuration's size, Q-T228.2).
    let before = unsupported_lines("XR_META_recommended_layer_resolution");
    app.world_mut()
        .resource_mut::<DisplaySettings>()
        .render_scale = 0.8;
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(
        unsupported_lines("XR_META_recommended_layer_resolution"),
        before + 1
    );
    assert_eq!(app.world().resource::<DisplaySettings>().render_scale, 0.8);

    // The re-creation completes: a second session runs with the first swapchain's size
    // times 0.8 (no META recommendation on Monado, Q-T228.2).
    let sizes = || -> Vec<(u32, u32)> {
        lines()
            .iter()
            .filter(|l| l.target == "ftgsxr_xr::display")
            .filter_map(|l| {
                let rest = l.message.strip_prefix("swapchain ")?;
                let (w, h) = rest.split_whitespace().next()?.split_once('x')?;
                Some((w.parse().ok()?, h.parse().ok()?))
            })
            .collect()
    };
    let first = sizes()[0];
    let start = Instant::now();
    while sizes().len() < 2 || app.world().get_resource::<XrState>() != Some(&XrState::Running) {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "the session is re-created within 20 s: {:?}",
            sizes()
        );
        app.update();
    }
    let scaled = |v: u32| (v as f32 * 0.8).round() as u32;
    assert_eq!(sizes()[1], (scaled(first.0), scaled(first.1)));
}
