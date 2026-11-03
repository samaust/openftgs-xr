// SPDX-License-Identifier: Apache-2.0

//! #376 (§1.7 Command line, §2.4 Threads, §1.11, Q-229, `gpu` lane): `AppPlugin` holds an
//! `AppExit` while render pipelines are queued or compiling, until a render frame after
//! it reports none, at most Q-229's 2 s; then lets it through with the code it had, and
//! the settings are saved at that exit, not before. An exit with nothing compiling passes
//! within the frame after it.

mod support;

use std::time::{Duration, Instant};

use bevy::app::AppExit;
use support::TestApp;

/// Q-229's cap, and the slack a loaded machine may add to it.
const CAP: Duration = Duration::from_secs(2);
const SLACK: Duration = Duration::from_secs(1);

#[test]
#[ignore = "gpu"]
fn gpu_an_exit_while_pipelines_compile_waits_for_them() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu("gpu-exit-wait-held");
    let settings = t.dir.join("config/settings.toml");
    let mut compiling = 0;
    for _ in 0..10 {
        t.app.update();
        compiling = t.pipelines_compiling();
        if compiling > 0 {
            break;
        }
    }
    assert!(
        compiling > 0,
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
    let released = t.until_for(CAP + SLACK, |app| app.should_exit().is_some());
    assert!(released, "released within the cap");
    assert!(start.elapsed() <= CAP + SLACK);
    assert_eq!(t.app.should_exit(), Some(AppExit::from_code(3)));
    assert!(settings.exists(), "saved at the released exit");
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
