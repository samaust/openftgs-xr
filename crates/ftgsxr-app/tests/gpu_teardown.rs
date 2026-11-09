// SPDX-License-Identifier: Apache-2.0

//! #291 (§2.4 Threads, `gpu` lane): the teardown crash's cause. On bevy's `multi_threaded`
//! task pools a `gpu_` app still has render pipelines compiling on the async compute pool a
//! frame after it starts; a test returning then let the process exit with a worker inside
//! the Vulkan driver (SIGSEGV after the tests passed). `TestApp`'s drop lets them finish.
//! The adapter's warm-up threads (Q-325) compile beside them, outside `PipelineCache`: one
//! still running after the app was dropped held the last reference to the device and
//! destroyed the Vulkan instance during the process's exit (#650); the adapter joins them
//! when the app drops (Q-378).
//! A process's exit right after such a frame is the crash's check: the crate's tests share
//! one test binary (#413), so the test also runs again in a child process of it, alone, which
//! exits once the test returns.

use crate::support;

use std::process::Command;

use support::TestApp;

/// Set in the child's environment: run the test alone, without a child of its own.
const CHILD: &str = "FTGSXR_GPU_TEARDOWN_CHILD";

/// Runs the test `name` of this file again, alone, in a child process of this test binary
/// with `CHILD` set, and fails unless the child ran it, passed and exited cleanly.
fn run_in_child(name: &str) {
    let path = format!("{}::{name}", crate::test_file_stem());
    let out = Command::new(std::env::current_exe().expect("the test binary"))
        .args([path.as_str(), "--exact", "--ignored", "--test-threads", "1"])
        .env(CHILD, "1")
        .output()
        .expect("the child run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("test result: ok. 1 passed"),
        "the child's exit ({:?}):\n{stdout}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
#[ignore = "gpu"]
fn gpu_an_app_dropped_while_a_pipeline_compiles_exits_cleanly() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
    if std::env::var_os(CHILD).is_none() {
        run_in_child("gpu_an_app_dropped_while_a_pipeline_compiles_exits_cleanly");
    }
    let mut t = TestApp::gpu("gpu-teardown");
    let mut seen = 0;
    for _ in 0..10 {
        t.app.update();
        seen = seen.max(t.pipelines_compiling());
        if seen > 0 {
            break;
        }
    }
    assert!(
        seen > 0,
        "a pipeline compiles beside the frames (bevy's multi_threaded)"
    );
    // Returns with the compile in flight: `TestApp`'s drop must let it finish, or the
    // child crashes at its exit.
}
