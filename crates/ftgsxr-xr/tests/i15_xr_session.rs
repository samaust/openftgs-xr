// SPDX-License-Identifier: Apache-2.0

//! I-15 (§3.3, `xr` lane): on Monado, the session reaches running, two `SplatCamera` eyes
//! exist, 2 s of frames render without errors, passthrough is reported unavailable and
//! the mode stays VR. Through `add_xr` (Q-120), `XrPlugin`, `SplatPlugin`, `SplatCamera`
//! and `Environment` (PRD #221). Prerequisites: `xr_support`'s.

use crate::xr_support;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::session::XrState;
use ftgsxr_io::ModelBytes;
use ftgsxr_render::{SortMode, SplatCamera, SplatEvent, SplatLoad, SplatStats, SplatTime};
use ftgsxr_testkit::fixtures_dir;
use ftgsxr_xr::{Environment, EnvironmentMode, XrPlugin};
use tracing::Level;
use xr_support::{Monado, RenderErrored, capture_log, lines, xr_app};

#[derive(Resource, Clone, Default)]
struct Failed(Arc<Mutex<Vec<String>>>);

#[test]
#[ignore = "xr"]
fn xr_i15_session_runs_two_splat_eyes_in_vr_without_passthrough() {
    capture_log();
    let _monado = Monado::start();
    let failed = Failed::default();
    let mut app = xr_app(
        XrPlugin {
            passthrough: false,
            hands: false,
            sort: SortMode::PerView,
        },
        |_| {},
    );
    app.insert_resource(failed.clone());
    app.add_observer(|event: On<SplatEvent>, failed: Res<Failed>| {
        if let SplatEvent::Failed { error, .. } = event.event() {
            failed.0.lock().unwrap().push(error.clone());
        }
    });
    let bytes = std::fs::read(fixtures_dir().join("probes/P1.safetensors")).expect("P1's file");
    app.world_mut().spawn((
        SplatLoad {
            bytes: ModelBytes::Owned(bytes),
            kind_hint: None,
            reduce_to: None,
        },
        SplatTime {
            tau: 0.0,
            extrapolating: false,
        },
        Transform::from_xyz(0.0, 1.5, -2.0),
    ));

    // Until the session runs (Monado needs a few seconds), then 2 s of frames.
    let start = Instant::now();
    while app.world().get_resource::<XrState>() != Some(&XrState::Running) {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "the session runs within 20 s"
        );
        app.update();
    }
    let running = Instant::now();
    let mut frames = 0;
    while running.elapsed() < Duration::from_secs(2) {
        app.update();
        frames += 1;
        assert_eq!(
            app.world().get_resource::<XrState>(),
            Some(&XrState::Running),
            "the session stays running"
        );
    }
    assert!(frames > 10, "frames ran: {frames}");

    let eyes = app
        .world_mut()
        .query::<&SplatCamera>()
        .iter(app.world())
        .filter(|c| c.sort == SortMode::PerView)
        .count();
    assert_eq!(eyes, 2, "two SplatCamera eyes");
    let env = app.world().resource::<Environment>();
    assert!(
        !env.passthrough_available,
        "passthrough unavailable on Monado"
    );
    assert_eq!(env.mode, EnvironmentMode::Vr);
    // The frames drew: the eyes' statistics count P1's splat (the model stands in front of
    // the simulated HMD).
    let stats = app.world().resource::<SplatStats>().0;
    assert!(
        stats.is_some_and(|s| s.visible_splats > 0 && s.non_finite == 0),
        "the eyes drew P1: {stats:?}"
    );
    assert!(
        app.world().get_resource::<RenderErrored>().is_none(),
        "no render error"
    );
    assert!(failed.0.lock().unwrap().is_empty(), "no SplatEvent::Failed");
    let errors: Vec<_> = lines()
        .into_iter()
        .filter(|l| l.level == Level::ERROR)
        .collect();
    assert!(errors.is_empty(), "no error-level log: {errors:?}");
}
