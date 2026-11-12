// SPDX-License-Identifier: Apache-2.0

//! The eyes in shared-center mode on Monado (`xr` lane, #727; D-4, §1.4 Stereo, §2.4
//! Frame loop hooks, Q-399): with `XrPlugin.sort` at `SharedCenter`, and a tier that
//! sorts shared-center so the eyes keep it (`session` follows `SplatTierState.sort_mode`),
//! the session renders 2 s without errors, both eyes are `SharedCenter` splat cameras, and
//! `SplatStats` reports visible splats from one statistics block a frame. Two instances
//! of P1 side by side under a capacity of 1 overflow every block (D-34), so the
//! `SplatEvent::Overflow` count bounds the blocks copied: at most one a frame for the
//! eyes' one group, where two per-view eyes copy two (§1.4 stage 8, Q-403). Prerequisites:
//! `xr_support`'s.

use crate::xr_support;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::session::XrState;
use ftgsxr_io::ModelBytes;
use ftgsxr_render::{
    Capacity, QualityTier, SortMode, SplatCamera, SplatEvent, SplatLoad, SplatStats, SplatTime,
};
use ftgsxr_testkit::fixtures_dir;
use ftgsxr_xr::XrPlugin;
use tracing::Level;
use xr_support::{Monado, RenderErrored, capture_log, lines, xr_app_in};

#[derive(Resource, Clone, Default)]
struct Seen {
    failed: Arc<Mutex<Vec<String>>>,
    overflows: Arc<Mutex<usize>>,
}

#[test]
#[ignore = "xr"]
fn xr_shared_center_eyes_render_from_one_statistics_block_a_frame() {
    capture_log();
    let _monado = Monado::start();
    let seen = Seen::default();
    let tier = QualityTier {
        sort_mode: SortMode::SharedCenter,
        capacity: Capacity::Fixed(1),
        ..QualityTier::conformance()
    };
    let mut app = xr_app_in(
        tier,
        XrPlugin {
            passthrough: false,
            hands: false,
            sort: SortMode::SharedCenter,
        },
        |_| {},
    );
    app.insert_resource(seen.clone());
    app.add_observer(
        |event: On<SplatEvent>, seen: Res<Seen>| match event.event() {
            SplatEvent::Failed { error, .. } => seen.failed.lock().unwrap().push(error.clone()),
            SplatEvent::Overflow { .. } => *seen.overflows.lock().unwrap() += 1,
            SplatEvent::Uploaded { .. } => {}
        },
    );
    let bytes = std::fs::read(fixtures_dir().join("probes/P1.safetensors")).expect("P1's file");
    for x in [-0.1, 0.1] {
        app.world_mut().spawn((
            SplatLoad {
                bytes: ModelBytes::Owned(bytes.clone()),
                kind_hint: None,
                reduce_to: None,
            },
            SplatTime {
                tau: 0.0,
                extrapolating: false,
            },
            Transform::from_xyz(x, 1.5, -2.0),
        ));
    }

    let start = Instant::now();
    while app.world().get_resource::<XrState>() != Some(&XrState::Running) {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "the session runs within 20 s"
        );
        app.update();
    }
    // Until the eyes draw both instances (the uploads and the first blocks), then 2 s.
    let settle = Instant::now();
    while *seen.overflows.lock().unwrap() == 0 {
        assert!(
            settle.elapsed() < Duration::from_secs(10),
            "the two instances overflow the capacity of 1 within 10 s: {:?}",
            app.world().resource::<SplatStats>().0
        );
        app.update();
    }
    *seen.overflows.lock().unwrap() = 0;
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
    let overflows = *seen.overflows.lock().unwrap();
    eprintln!("{overflows} overflow events in {frames} frames");
    // Pipelined rendering can land a block's event one update late at either edge of
    // the window; two per-view eyes would still raise about two a frame.
    assert!(
        (1..=frames + 2).contains(&overflows),
        "one statistics block a frame: {overflows} overflow events in {frames} frames"
    );

    let eyes: Vec<SortMode> = app
        .world_mut()
        .query_filtered::<&SplatCamera, With<bevy_mod_xr::camera::XrCamera>>()
        .iter(app.world())
        .map(|c| c.sort)
        .collect();
    assert_eq!(eyes, [SortMode::SharedCenter; 2], "two shared-center eyes");
    let stats = app.world().resource::<SplatStats>().0;
    assert!(
        stats.is_some_and(|s| s.visible_splats > 0 && s.overflow && s.non_finite == 0),
        "the eyes drew P1: {stats:?}"
    );
    assert!(
        app.world().get_resource::<RenderErrored>().is_none(),
        "no render error"
    );
    assert!(
        seen.failed.lock().unwrap().is_empty(),
        "no SplatEvent::Failed"
    );
    let errors: Vec<_> = lines()
        .into_iter()
        .filter(|l| l.level == Level::ERROR)
        .collect();
    assert!(errors.is_empty(), "no error-level log: {errors:?}");
}
