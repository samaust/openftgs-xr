// SPDX-License-Identifier: Apache-2.0

//! Q-378 (#650, §2.4 Threads, Q-325, `gpu`): the adapter's warm-up threads end with the
//! app. An app dropped while its warm-up compiles leaves no warm-up thread running, so
//! none can hold the device after the drop and release it while the process exits
//! (a SIGSEGV in the Vulkan driver's exit-time teardown).

use std::time::{Duration, Instant};

use bevy::MinimalPlugins;
use bevy::app::{App, PluginsState};
use bevy::asset::AssetPlugin;
use bevy::camera::CameraPlugin;
use bevy::core_pipeline::CorePipelinePlugin;
use bevy::image::ImagePlugin;
use bevy::mesh::MeshPlugin;
use bevy::render::RenderPlugin;
use bevy::transform::TransformPlugin;
use bevy::window::{ExitCondition, WindowPlugin};
use ftgsxr_render::{QualityTier, SplatPlugin};
use ftgsxr_testkit::layer_messages;

/// The warm-up threads' name ("ftgsxr splat passes") as Linux keeps it: its first 15
/// bytes.
const WARM_UP_COMM: &str = "ftgsxr splat pa";

/// The threads of this process named as the warm-up's, from `/proc/self/task`.
fn warm_up_threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("/proc/self/task")
        .filter_map(|task| std::fs::read_to_string(task.ok()?.path().join("comm")).ok())
        .filter(|comm| comm.trim_end() == WARM_UP_COMM)
        .count()
}

/// A headless app with the adapter, the conformance tier, no model and no camera: its
/// device's warm-up starts in its first render frames (Q-325).
fn app() -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        TransformPlugin,
        AssetPlugin::default(),
        WindowPlugin {
            primary_window: None,
            exit_condition: ExitCondition::DontExit,
            close_when_requested: false,
            ..WindowPlugin::default()
        },
        RenderPlugin::default(),
        ImagePlugin::default(),
        MeshPlugin,
        CameraPlugin,
        CorePipelinePlugin,
        SplatPlugin {
            tier: QualityTier::conformance(),
        },
    ));
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
    }
    app.finish();
    app.cleanup();
    app
}

#[test]
#[ignore = "gpu"]
fn gpu_an_app_dropped_while_its_warm_up_compiles_leaves_no_warm_up_thread() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut app = app();
    let start = Instant::now();
    let mut running = 0;
    while running == 0 && start.elapsed() < Duration::from_secs(10) {
        app.update();
        running = warm_up_threads();
    }
    assert!(running > 0, "the warm-up started (Q-325)");
    drop(app);
    assert_eq!(
        warm_up_threads(),
        0,
        "a warm-up thread outlived the app (Q-378)"
    );
}
