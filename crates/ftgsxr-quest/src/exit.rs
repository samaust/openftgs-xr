// SPDX-License-Identifier: Apache-2.0

//! E-XR-RUNTIME once the app runs (§1.15, §1.7, #660): a session created without its
//! primary reference space ends the app through an `AppExit` with the Quest's code 2, so
//! Q-236's hold and the settings' save at exit see it and the render world is dropped
//! before the process exits, which joins the adapter's warm-up threads (Q-381, §2.4
//! Threads). Built on the host too, so its tests run there.

use bevy::app::AppExit;
use bevy::ecs::prelude::{MessageReader, MessageWriter, Res};
use bevy_mod_xr::session::XrSessionCreatedMessage;
use bevy_mod_xr::spaces::XrPrimaryReferenceSpace;

/// The exit code of E-XR-RUNTIME on the Quest (§1.15).
pub(crate) const EXIT_ENVIRONMENT: u8 = 2;

/// E-XR-RUNTIME when the session cannot be created (§1.15): the XR crate logs it and
/// inserts the primary reference space only once a session exists, so a session-created
/// message without that space is the failure (or a session left without its reference
/// space, which is E-XR-RUNTIME too). The app ends through the exit, not the process,
/// which `main` ends once `App::run` returns with the app dropped (#660).
pub(crate) fn exit_without_session(
    mut created: MessageReader<XrSessionCreatedMessage>,
    space: Option<Res<XrPrimaryReferenceSpace>>,
    mut exit: MessageWriter<AppExit>,
) {
    if created.read().count() > 0 && space.is_none() {
        exit.write(AppExit::from_code(EXIT_ENVIRONMENT));
    }
}

/// The process's exit code for the app's `AppExit` (§1.7, §1.15): 0 for a success, else
/// the exit's own code.
pub(crate) fn exit_code(exit: &AppExit) -> i32 {
    match exit {
        AppExit::Success => 0,
        AppExit::Error(code) => i32::from(code.get()),
    }
}

#[cfg(test)]
mod tests {
    use bevy::app::{App, Update};

    use super::*;

    fn app() -> App {
        let mut app = App::new();
        app.add_message::<XrSessionCreatedMessage>()
            .add_systems(Update, exit_without_session);
        app
    }

    #[test]
    fn exit_without_session_ends_the_app_with_code_2_without_a_reference_space() {
        let mut app = app();
        app.world_mut().write_message(XrSessionCreatedMessage);
        app.update();
        assert_eq!(app.should_exit(), Some(AppExit::from_code(2)), "§1.15");
    }

    #[test]
    fn exit_code_is_the_app_exit_s_code() {
        assert_eq!(exit_code(&AppExit::Success), 0);
        assert_eq!(exit_code(&AppExit::from_code(EXIT_ENVIRONMENT)), 2, "§1.15");
        assert_eq!(exit_code(&AppExit::error()), 1);
    }

    #[test]
    fn exit_without_session_writes_no_exit_without_a_session_created() {
        let mut app = app();
        app.update();
        assert_eq!(app.should_exit(), None);
    }

    /// The `gpu_` variant (Q-381, Q-236, Q-325, §2.4 Threads): `AppPlugin` on a headless
    /// render app, as the Quest's `main` builds it but without XR, its warm-up started.
    mod gpu {
        use std::sync::Arc;
        use std::task::Poll;
        use std::time::{Duration, Instant};

        use bevy::MinimalPlugins;
        use bevy::app::{App, PluginsState, Update};
        use bevy::asset::AssetPlugin;
        use bevy::camera::CameraPlugin;
        use bevy::core_pipeline::CorePipelinePlugin;
        use bevy::image::ImagePlugin;
        use bevy::input::InputPlugin;
        use bevy::mesh::MeshPlugin;
        use bevy::render::RenderPlugin;
        use bevy::transform::TransformPlugin;
        use bevy::window::{ExitCondition, WindowPlugin};
        use ftgsxr_app::{
            AppPaths, AppPlugin, AudioOutputRes, CameraFrames, DocumentProvider, ModelRef,
            Permission, PermissionHandle, PickHandle, PickKind, Platform, QualityTier,
            StubAudioOutput,
        };
        use ftgsxr_testkit::{layer_messages, scratch_dir};

        use super::*;

        /// The warm-up threads' name ("ftgsxr splat passes") as Linux keeps it: its
        /// first 15 bytes (Q-381's test in `ftgsxr-render`).
        const WARM_UP_COMM: &str = "ftgsxr splat pa";

        /// The threads of this process named as the warm-up's, from `/proc/self/task`.
        fn warm_up_threads() -> usize {
            std::fs::read_dir("/proc/self/task")
                .expect("/proc/self/task")
                .filter_map(|task| std::fs::read_to_string(task.ok()?.path().join("comm")).ok())
                .filter(|comm| comm.trim_end() == WARM_UP_COMM)
                .count()
        }

        /// A platform with its directories under a scratch folder and nothing else: no
        /// pick, permission, document or camera is asked for here (a smaller
        /// `StubPlatform` of `ftgsxr-app`'s test support, which this crate cannot use).
        struct NoPlatform(AppPaths);

        impl Platform for NoPlatform {
            fn paths(&self) -> &AppPaths {
                &self.0
            }
            fn pick(&self, _: PickKind) -> PickHandle {
                PickHandle(0)
            }
            fn poll_pick(&self, _: PickHandle) -> Poll<Result<Vec<ModelRef>, String>> {
                Poll::Ready(Ok(Vec::new()))
            }
            fn request_permission(&self, _: Permission) -> PermissionHandle {
                PermissionHandle(0)
            }
            fn poll_permission(&self, _: PermissionHandle) -> Poll<bool> {
                Poll::Ready(false)
            }
            fn document_provider(&self) -> Option<Arc<dyn DocumentProvider>> {
                None
            }
            fn camera_frames(&self) -> Option<Arc<dyn CameraFrames>> {
                None
            }
        }

        fn app() -> App {
            let dir = scratch_dir(
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/tmp"),
                "ftgsxr_quest_unit",
                "exit_gpu",
            );
            let paths = AppPaths {
                config: dir.join("config"),
                cache: dir.join("cache"),
                models: dir.join("models"),
                scenes: dir.join("config/scenes"),
                screenshots: dir.join("pictures"),
                logs: dir.join("cache/logs"),
            };
            let mut app = App::new();
            app.add_plugins((
                MinimalPlugins,
                AssetPlugin::default(),
                TransformPlugin,
                InputPlugin,
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
            ));
            app.insert_resource(AudioOutputRes(Arc::new(StubAudioOutput::new())));
            app.add_plugins(AppPlugin {
                platform: Arc::new(NoPlatform(paths)),
                tier: QualityTier::by_name("quest-balanced").expect("§1.12's tier"),
                initial: Vec::new(),
                scene: None,
            });
            app.add_message::<XrSessionCreatedMessage>()
                .add_systems(Update, exit_without_session);
            while app.plugins_state() == PluginsState::Adding {
                bevy::tasks::tick_global_task_pools_on_main_thread();
            }
            app.finish();
            app.cleanup();
            app
        }

        #[test]
        #[ignore = "gpu"]
        fn gpu_exit_without_session_during_the_warm_up_leaves_no_warm_up_thread() {
            let _layer = layer_messages(); // R-16, Q-105
            let mut app = app();
            let start = Instant::now();
            let mut running = 0;
            while running == 0 && start.elapsed() < Duration::from_secs(10) {
                app.update();
                running = warm_up_threads();
            }
            assert!(running > 0, "the warm-up started (Q-325)");
            // The session fails while the warm-up compiles: the app ends, not the process.
            // Whether a
            // warm-up thread is still running when Q-236's hold lets the exit through
            // depends on the GPU (on an RTX 4090 the warm-up has ended by then); the join
            // itself is Q-381's own test, in `ftgsxr-render`.
            app.world_mut().write_message(XrSessionCreatedMessage);
            let failed = Instant::now();
            let mut exit = None;
            // Q-236's cap, and a second of slack for a loaded machine.
            while exit.is_none() && failed.elapsed() < Duration::from_secs(6) {
                app.update();
                exit = app.should_exit();
            }
            assert_eq!(
                exit,
                Some(AppExit::from_code(EXIT_ENVIRONMENT)),
                "§1.15, Q-236"
            );
            assert_eq!(exit_code(&exit.unwrap()), 2, "§1.15");
            drop(app);
            // A joined thread can stay listed in `/proc/self/task` for a moment after its
            // join returns: allow it 10 ms, far less than a warm-up compile.
            let joined = Instant::now();
            let mut left = warm_up_threads();
            while left > 0 && joined.elapsed() < Duration::from_millis(10) {
                std::thread::sleep(Duration::from_millis(1));
                left = warm_up_threads();
            }
            assert_eq!(left, 0, "a warm-up thread outlived the app (Q-381)");
        }
    }
}
