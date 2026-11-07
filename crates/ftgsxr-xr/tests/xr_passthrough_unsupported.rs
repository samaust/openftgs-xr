// SPDX-License-Identifier: Apache-2.0

//! Q-161's clause for passthrough (`xr` lane, #568, Q-335): on Monado, which offers no
//! XR\_FB\_passthrough, a settings file asking for passthrough on the headset gives one
//! E-XR-UNSUPPORTED `Toast` in the session, however many frames it runs, `[environment]
//! mode` returns to `vr` and the mode stays VR. Prerequisites: `xr_support`'s.

use crate::xr_support;

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::session::XrState;
use ftgsxr_app::{Feature, Headset, Intent, SettingsRes, Toast, ToastLevel};
use ftgsxr_io::Settings;
use ftgsxr_render::SortMode;
use ftgsxr_xr::{Environment, EnvironmentMode, XrPlugin};
use xr_support::{Monado, capture_log, xr_app};

/// The toasts triggered so far.
#[derive(Resource, Default)]
struct Triggered(Vec<Toast>);

#[test]
#[ignore = "xr"]
fn xr_passthrough_asked_on_monado_is_one_unsupported_toast_and_vr() {
    capture_log();
    let _monado = Monado::start();
    let mut settings = Settings::default();
    settings.environment.mode = "passthrough".to_owned();
    let mut app = xr_app(
        XrPlugin {
            passthrough: true,
            hands: false,
            sort: SortMode::PerView,
        },
        |app| {
            app.insert_resource(SettingsRes(settings))
                .insert_resource(Headset)
                .init_resource::<Triggered>()
                .add_observer(|t: On<Toast>, mut log: ResMut<Triggered>| {
                    log.0.push(t.event().clone());
                });
        },
    );
    assert!(
        !app.world().resource::<Environment>().passthrough_available,
        "Monado offers no XR_FB_passthrough"
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
        assert_eq!(
            app.world().resource::<Environment>().mode,
            EnvironmentMode::Vr
        );
    }
    // A second request in the session (the toggle): written back, no second toast.
    app.world_mut()
        .write_message(Intent::Toggle(Feature::Passthrough));
    for _ in 0..10 {
        app.update();
    }
    assert_eq!(
        app.world().resource::<SettingsRes>().0.environment.mode,
        "vr",
        "the setting returns to vr"
    );
    assert_eq!(
        app.world().resource::<Environment>().mode,
        EnvironmentMode::Vr
    );
    assert_eq!(
        app.world().resource::<Triggered>().0,
        vec![Toast {
            text: "E-XR-UNSUPPORTED: Not available on this runtime (XR_FB_passthrough)".to_owned(),
            level: ToastLevel::Warning,
        }]
    );
}
