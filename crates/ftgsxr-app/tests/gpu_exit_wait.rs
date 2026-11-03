// SPDX-License-Identifier: Apache-2.0

//! #376 (§1.7 Command line, §2.4 Threads, §1.11, Q-236, `gpu` lane): `AppPlugin` holds an
//! `AppExit` while render pipelines are queued or compiling, until a render frame after
//! it reports none, at most Q-236's 2 s; then lets it through with the code it had, and
//! the settings are saved at that exit, not before. An exit the window's close action
//! writes (bevy's `exit_on_all_closed`, in `Last`'s `ExitSystems`) is held the same way. An exit with nothing
//! compiling passes within the frame after it.

mod support;

use std::time::{Duration, Instant};

use bevy::app::{AppExit, Last};
use bevy::ecs::prelude::{IntoScheduleConfigs, Res, Resource};
use bevy::window::{ExitSystems, exit_on_all_closed};
use support::TestApp;

/// Q-236's cap, and the slack a loaded machine may add to it.
// Q-236
const CAP: Duration = Duration::from_secs(2);
const SLACK: Duration = Duration::from_secs(1);

/// Runs frames until a pipeline compiles, at most 10; whether one does.
fn until_compiling(t: &mut TestApp) -> bool {
    for _ in 0..10 {
        t.app.update();
        if t.pipelines_compiling() > 0 {
            return true;
        }
    }
    false
}

/// Runs frames until the exit passes, at most the cap and its slack: the exit, and
/// whether a pipeline was compiling after the last frame that held it (false when the
/// cap released it).
fn until_released(t: &mut TestApp, start: Instant) -> (Option<AppExit>, bool) {
    let mut compiling_when_held = t.pipelines_compiling() > 0;
    while start.elapsed() <= CAP + SLACK {
        t.app.update();
        if let Some(exit) = t.app.should_exit() {
            let capped = start.elapsed() >= CAP;
            return (Some(exit), compiling_when_held && !capped);
        }
        compiling_when_held = t.pipelines_compiling() > 0;
        std::thread::sleep(Duration::from_millis(1));
    }
    (None, false)
}

#[test]
#[ignore = "gpu"]
fn gpu_an_exit_while_pipelines_compile_waits_for_them() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu("gpu-exit-wait-held");
    let settings = t.dir.join("config/settings.toml");
    assert!(
        until_compiling(&mut t),
        "a pipeline compiles in the first frames (bevy's multi_threaded)"
    );
    assert!(!settings.exists(), "a first run has no settings file yet");
    // An error over a success, as `App::should_exit` chooses.
    t.app.world_mut().write_message(AppExit::Success);
    t.app.world_mut().write_message(AppExit::from_code(3));
    let start = Instant::now();
    t.app.update();
    assert_eq!(t.app.should_exit(), None, "held on the frame it arrives");
    assert!(!settings.exists(), "not saved while held");
    let (exit, early) = until_released(&mut t, start);
    assert_eq!(exit, Some(AppExit::from_code(3)), "released within the cap");
    assert!(!early, "not released while a pipeline compiled");
    assert!(settings.exists(), "saved at the released exit");
}

/// Whether bevy's `exit_on_all_closed` runs this frame (the gpu app has no window, so
/// it would exit from the first frame).
#[derive(Resource)]
struct CloseNow(bool);

fn close_now(now: Res<CloseNow>) -> bool {
    now.0
}

#[test]
#[ignore = "gpu"]
fn gpu_an_exit_from_the_window_systems_waits_too() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu("gpu-exit-wait-window");
    t.app.insert_resource(CloseNow(false)).add_systems(
        Last,
        exit_on_all_closed.run_if(close_now).in_set(ExitSystems),
    );
    assert!(
        until_compiling(&mut t),
        "a pipeline compiles in the first frames (bevy's multi_threaded)"
    );
    t.app.insert_resource(CloseNow(true));
    let start = Instant::now();
    t.app.update();
    assert_eq!(t.app.should_exit(), None, "held on the frame it arrives");
    let (exit, early) = until_released(&mut t, start);
    assert_eq!(exit, Some(AppExit::Success), "released within the cap");
    assert!(!early, "not released while a pipeline compiled");
}

#[test]
#[ignore = "gpu"]
fn gpu_an_exit_with_nothing_compiling_passes_at_once() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu("gpu-exit-wait-quiet");
    assert!(
        t.settle_pipelines(),
        "the first pipelines compile within 10 s"
    );
    // `settle_pipelines` waits for those compiling; a few more frames let those still
    // queued (the depth downsampling's, say) compile too, so nothing is pending.
    t.frames(30);
    t.app.world_mut().write_message(AppExit::Success);
    t.app.update();
    if t.app.should_exit().is_none() {
        t.app.update();
    }
    assert_eq!(
        t.app.should_exit(),
        Some(AppExit::Success),
        "through within the frame after the exit"
    );
}
