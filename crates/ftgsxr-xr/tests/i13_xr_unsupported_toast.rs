// SPDX-License-Identifier: Apache-2.0

//! I-13 extended (`xr` lane, #289, Q-161): on Monado, a refresh rate and a foveation
//! asked for in `[xr]` trigger one E-XR-UNSUPPORTED `Toast` per absent extension, a
//! second request in the same session triggers no second one (owner, #306), and a launch
//! with the defaults triggers none. Prerequisites: `xr_support`'s.

use crate::xr_support;

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_openxr::resources::OxrInstance;
use bevy_mod_xr::session::XrState;
use ftgsxr_app::{SettingsRes, Toast, ToastLevel};
use ftgsxr_io::Settings;
use ftgsxr_render::SortMode;
use ftgsxr_xr::{DisplaySettings, Foveation, XrPlugin};
use xr_support::{Monado, capture_log, xr_app};

/// The toasts triggered so far.
#[derive(Resource, Default)]
struct Triggered(Vec<Toast>);

fn record(toast: On<Toast>, mut log: ResMut<Triggered>) {
    log.0.push(toast.event().clone());
}

/// The app on Monado with `settings`, run until its session runs and a few frames more;
/// its triggered toasts, and whether the runtime has XR\_FB\_display\_refresh\_rate.
fn toasts_with(settings: Settings) -> (Vec<Toast>, bool) {
    toasts_after(settings, |_| {})
}

/// As `toasts_with`, with `change` made to the display settings once the session runs,
/// and a few frames more.
fn toasts_after(settings: Settings, change: impl Fn(&mut DisplaySettings)) -> (Vec<Toast>, bool) {
    capture_log();
    let _monado = Monado::start();
    let mut app = xr_app(
        XrPlugin {
            passthrough: false,
            hands: false,
            sort: SortMode::PerView,
        },
        |app| {
            app.insert_resource(SettingsRes(settings))
                .init_resource::<Triggered>()
                .add_observer(record);
        },
    );
    let start = Instant::now();
    while app.world().get_resource::<XrState>() != Some(&XrState::Running) {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "the session runs within 20 s"
        );
        app.update();
    }
    for _ in 0..10 {
        app.update();
    }
    change(&mut app.world_mut().resource_mut::<DisplaySettings>());
    for _ in 0..10 {
        app.update();
    }
    let rates = app
        .world()
        .resource::<OxrInstance>()
        .exts()
        .fb_display_refresh_rate
        .is_some();
    (app.world().resource::<Triggered>().0.clone(), rates)
}

#[test]
#[ignore = "xr"]
fn xr_i13_an_explicit_request_on_monado_is_one_unsupported_toast_per_extension() {
    let mut settings = Settings::default();
    settings.xr.refresh_rate = Some(90.0);
    settings.xr.foveation = Some("low".to_owned());
    let (toasts, rates) = toasts_with(settings);
    assert_one_per_extension(toasts, rates);
}

/// A change to both while the session runs is a second request for each absent extension
/// in the same session: its warning is logged, but no second toast (Q-161, owner #306).
#[test]
#[ignore = "xr"]
fn xr_i13_a_second_request_in_the_session_is_no_second_toast() {
    let mut settings = Settings::default();
    settings.xr.refresh_rate = Some(90.0);
    settings.xr.foveation = Some("low".to_owned());
    let (toasts, rates) = toasts_after(settings, |d| {
        d.refresh_rate = Some(72.0);
        d.foveation = Foveation::High;
    });
    assert_one_per_extension(toasts, rates);
}

/// The toasts are exactly one E-XR-UNSUPPORTED per absent extension: XR\_FB\_foveation,
/// and XR\_FB\_display\_refresh\_rate unless the runtime has it (`rates`).
fn assert_one_per_extension(toasts: Vec<Toast>, rates: bool) {
    let mut expected = vec![Toast {
        text: "E-XR-UNSUPPORTED: Not available on this runtime (XR_FB_foveation)".to_owned(),
        level: ToastLevel::Warning,
    }];
    if !rates {
        expected.insert(
            0,
            Toast {
                text: "E-XR-UNSUPPORTED: Not available on this runtime \
                       (XR_FB_display_refresh_rate)"
                    .to_owned(),
                level: ToastLevel::Warning,
            },
        );
    }
    let mut got = toasts.clone();
    got.sort_by(|a, b| a.text.cmp(&b.text));
    expected.sort_by(|a, b| a.text.cmp(&b.text));
    assert_eq!(
        got, expected,
        "Monado has the refresh-rate extension: {rates}"
    );
}

#[test]
#[ignore = "xr"]
fn xr_i13_a_default_launch_on_monado_is_no_toast() {
    let (toasts, _) = toasts_with(Settings::default());
    assert_eq!(toasts, Vec::new());
}
