// SPDX-License-Identifier: Apache-2.0

//! #291 (§2.4 Threads, `gpu` lane): the teardown crash's cause. On bevy's `multi_threaded`
//! task pools a `gpu_` app still has render pipelines compiling on the async compute pool a
//! frame after it starts; a test returning then let the process exit with a worker inside
//! the Vulkan driver (SIGSEGV after the tests passed). `TestApp`'s drop lets them finish.
//! This binary's own exit, right after such a frame, is the crash's check.

use crate::support;

use support::TestApp;

#[test]
#[ignore = "gpu"]
fn gpu_an_app_dropped_while_a_pipeline_compiles_exits_cleanly() {
    let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
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
    // Returns with the compile in flight: `TestApp`'s drop must let it finish, or this
    // binary crashes at exit.
}
