// SPDX-License-Identifier: Apache-2.0

//! I-17 (§3.3, `xr` lane): with `[xr] reference_space` excluding STAGE in the test's
//! settings, the fallback chain picks LOCAL\_FLOOR, then LOCAL, with the warning (D-24).
//! Monado 21.0.0 offers no XR\_EXT\_local\_floor, so on it the chain ends at LOCAL: the
//! warning is logged and the tracking root stands 1.5 m up (PRD #221's Notes); the
//! LOCAL\_FLOOR step is the device's. Prerequisites: `xr_support`'s.

mod xr_support;

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::session::{XrState, XrTrackingRoot};
use bevy_mod_xr::spaces::XrPrimaryReferenceSpace;
use ftgsxr_app::SettingsRes;
use ftgsxr_io::Settings;
use ftgsxr_render::SortMode;
use ftgsxr_xr::XrPlugin;
use tracing::Level;
use xr_support::{Monado, capture_log, lines, xr_app};

#[test]
#[ignore = "xr"]
fn xr_i17_without_stage_the_chain_falls_back_with_the_warning() {
    capture_log();
    let _monado = Monado::start();
    let mut settings = Settings::default();
    settings.xr.reference_space = vec!["local-floor".into(), "local".into()];
    let mut app = xr_app(
        XrPlugin {
            passthrough: false,
            hands: false,
            sort: SortMode::PerView,
        },
        |app| {
            app.insert_resource(SettingsRes(settings));
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

    assert!(
        app.world()
            .get_resource::<XrPrimaryReferenceSpace>()
            .is_some()
    );
    let log = lines();
    let local_floor = !log
        .iter()
        .any(|l| l.message.contains("XR_EXT_local_floor") && l.message.contains("not available"));
    let warned = log.iter().any(|l| {
        l.level == Level::WARN && l.target == "ftgsxr_xr::session" && l.message.contains("LOCAL")
    });
    let root_y = app
        .world_mut()
        .query_filtered::<&Transform, With<XrTrackingRoot>>()
        .single(app.world())
        .expect("one tracking root")
        .translation
        .y;
    if local_floor {
        // A runtime with the extension: LOCAL_FLOOR, no warning, the root at the floor.
        assert!(!warned, "LOCAL_FLOOR is not the fallback");
        assert_eq!(root_y, 0.0);
    } else {
        assert!(warned, "the LOCAL fallback warns");
        assert_eq!(root_y, 1.5, "the floor 1.5 m below the session's origin");
    }
}
