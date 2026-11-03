// SPDX-License-Identifier: Apache-2.0

//! The test support of PRD #275 ("Test support"), the base of every CPU I-n of M-4: a
//! headless `App` without `RenderPlugin` (`MinimalPlugins`, the asset, transform and input
//! plugins, `AppPlugin` with a stub `Platform`), the fixture server, frames advanced
//! under a controlled `Time`; its `gpu_` variant adds `RenderPlugin` and the camera and
//! pipeline plugins, as the adapter's own tests do (`gpu_` tests hold Q-105's layer
//! guard). Each test file includes it with `mod support;`. The app runs on
//! `StubAudioOutput` (#281); later tickets add the toasts' drain as their modules arrive.

#![allow(dead_code)]

pub mod desktop;
pub mod ui;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::Duration;

use bevy::MinimalPlugins;
use bevy::app::{App, PluginsState};
use bevy::asset::AssetPlugin;
use bevy::input::InputPlugin;
use bevy::time::TimeUpdateStrategy;
use bevy::transform::TransformPlugin;
use ftgsxr_app::{
    AppPaths, AppPlugin, AudioOutputRes, CameraFrames, DocumentProvider, ModelRef, Permission,
    PermissionHandle, PickHandle, PickKind, Platform, QualityTier, StubAudioOutput,
};
use ftgsxr_testkit::{LocalServer, scratch_dir, serve_fixtures};

/// The test's frame period: 60 Hz.
pub const FRAME: Duration = Duration::from_micros(16_667);

/// A scratch directory of this test binary (Q-62).
pub fn scratch(name: &str) -> PathBuf {
    scratch_dir(env!("CARGO_TARGET_TMPDIR"), env!("CARGO_CRATE_NAME"), name)
}

/// The player's directories under `root` (§1.11's layout), none of them created.
pub fn paths_under(root: &Path) -> AppPaths {
    AppPaths {
        config: root.join("config"),
        cache: root.join("cache"),
        models: root.join("models"),
        scenes: root.join("config/scenes"),
        screenshots: root.join("pictures"),
        logs: root.join("cache/logs"),
    }
}

/// A `Platform` for the tests: picks answer what the test queued (`queue_pick`), else an
/// empty pick; every permission is granted; no documents and no camera.
pub struct StubPlatform {
    paths: AppPaths,
    next: AtomicU32,
    queued: Mutex<Vec<Result<Vec<ModelRef>, String>>>,
    picks: Mutex<HashMap<u32, Result<Vec<ModelRef>, String>>>,
    /// Every pick opened, in order.
    pub opened: Mutex<Vec<PickKind>>,
    /// Every permission requested, in order.
    pub requested: Mutex<Vec<Permission>>,
}

impl StubPlatform {
    /// A stub whose directories are under `root`.
    pub fn new(root: &Path) -> Arc<StubPlatform> {
        Self::with_paths(paths_under(root))
    }

    /// A stub with the given directories (I-19).
    pub fn with_paths(paths: AppPaths) -> Arc<StubPlatform> {
        Arc::new(StubPlatform {
            paths,
            next: AtomicU32::new(0),
            queued: Mutex::new(Vec::new()),
            picks: Mutex::new(HashMap::new()),
            opened: Mutex::new(Vec::new()),
            requested: Mutex::new(Vec::new()),
        })
    }

    /// What the next pick answers.
    pub fn queue_pick(&self, answer: Result<Vec<ModelRef>, String>) {
        self.queued.lock().unwrap().push(answer);
    }
}

impl Platform for StubPlatform {
    fn paths(&self) -> &AppPaths {
        &self.paths
    }

    fn pick(&self, kind: PickKind) -> PickHandle {
        self.opened.lock().unwrap().push(kind);
        let h = self.next.fetch_add(1, Ordering::Relaxed);
        let mut queued = self.queued.lock().unwrap();
        let answer = if queued.is_empty() {
            Ok(Vec::new())
        } else {
            queued.remove(0)
        };
        self.picks.lock().unwrap().insert(h, answer);
        PickHandle(h)
    }

    fn poll_pick(&self, h: PickHandle) -> Poll<Result<Vec<ModelRef>, String>> {
        match self.picks.lock().unwrap().remove(&h.0) {
            Some(answer) => Poll::Ready(answer),
            None => Poll::Ready(Err(format!("unknown pick {}", h.0))),
        }
    }

    fn request_permission(&self, p: Permission) -> PermissionHandle {
        self.requested.lock().unwrap().push(p);
        PermissionHandle(self.next.fetch_add(1, Ordering::Relaxed))
    }

    fn poll_permission(&self, _: PermissionHandle) -> Poll<bool> {
        Poll::Ready(true)
    }

    fn document_provider(&self) -> Option<Arc<dyn DocumentProvider>> {
        None
    }

    fn camera_frames(&self) -> Option<Arc<dyn CameraFrames>> {
        None
    }
}

/// A headless player: the `App`, its stub platform, its scratch directory and the
/// fixture server.
pub struct TestApp {
    /// The app, built and finished.
    pub app: App,
    /// Its platform.
    pub platform: Arc<StubPlatform>,
    /// The scratch directory holding the platform's directories.
    pub dir: PathBuf,
    /// The fixture server (`serve_fixtures`).
    pub server: LocalServer,
    /// The audio output (#281), inserted before `before` runs, which may replace it.
    pub audio: Arc<StubAudioOutput>,
}

/// What a test sets before `AppPlugin` builds.
pub struct Setup {
    /// The tier (`desktop` by default).
    pub tier: QualityTier,
    /// `AppPlugin.initial`.
    pub initial: Vec<ModelRef>,
    /// `AppPlugin.scene`.
    pub scene: Option<PathBuf>,
}

impl Default for Setup {
    fn default() -> Self {
        Setup {
            tier: QualityTier::by_name("desktop").expect("the desktop tier"),
            initial: Vec::new(),
            scene: None,
        }
    }
}

impl TestApp {
    /// The CPU app in the scratch directory `name`, without `RenderPlugin`.
    pub fn new(name: &str) -> TestApp {
        Self::with(name, Setup::default(), |_| {})
    }

    /// The CPU app, with `before` run on the `App` before `AppPlugin` is added (to insert
    /// `SettingsPath` or `SettingsRes`, say).
    pub fn with(name: &str, setup: Setup, before: impl FnOnce(&mut App)) -> TestApp {
        Self::build(name, setup, before, false)
    }

    /// The `gpu_` variant: the same app with `RenderPlugin` and the camera and pipeline
    /// plugins, no window (PRD #275's spike check 1). Its tests hold Q-105's layer guard.
    pub fn gpu(name: &str) -> TestApp {
        Self::build(name, Setup::default(), |_| {}, true)
    }

    /// The `gpu_` variant with a `Setup` (a tier with a small budget, say).
    pub fn gpu_with(name: &str, setup: Setup) -> TestApp {
        Self::build(name, setup, |_| {}, true)
    }

    /// The CPU app on a given platform (I-19).
    pub fn on(name: &str, platform: Arc<StubPlatform>) -> TestApp {
        Self::build_on(name, platform, Setup::default(), |_| {}, false)
    }

    fn build(name: &str, setup: Setup, before: impl FnOnce(&mut App), gpu: bool) -> TestApp {
        let dir = scratch(name);
        let platform = StubPlatform::new(&dir);
        Self::build_on(name, platform, setup, before, gpu)
    }

    fn build_on(
        name: &str,
        platform: Arc<StubPlatform>,
        setup: Setup,
        before: impl FnOnce(&mut App),
        gpu: bool,
    ) -> TestApp {
        let dir = scratch(name);
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            AssetPlugin::default(),
            TransformPlugin,
            InputPlugin,
        ));
        if gpu {
            add_render(&mut app);
        }
        app.insert_resource(TimeUpdateStrategy::ManualDuration(FRAME));
        let audio = Arc::new(StubAudioOutput::new());
        app.insert_resource(AudioOutputRes(audio.clone()));
        before(&mut app);
        app.add_plugins(AppPlugin {
            platform: platform.clone(),
            tier: setup.tier,
            initial: setup.initial,
            scene: setup.scene,
        });
        while app.plugins_state() == PluginsState::Adding {
            bevy::tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();
        TestApp {
            app,
            platform,
            dir,
            server: serve_fixtures(),
            audio,
        }
    }

    /// Runs frames (sleeping a millisecond between them, for the task pools) until
    /// `done` holds, at most `max`; whether it held.
    pub fn until(&mut self, max: usize, mut done: impl FnMut(&mut App) -> bool) -> bool {
        for _ in 0..max {
            self.app.update();
            if done(&mut self.app) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        false
    }

    /// Runs frames (sleeping a millisecond between them) until `done` holds, for at most
    /// `limit` of wall time; whether it held. For work on the task pools whose length is
    /// wall time, not frames (#291, §2.4 Threads): with bevy's `multi_threaded` a load runs
    /// beside the frames instead of inside one.
    pub fn until_for(&mut self, limit: Duration, mut done: impl FnMut(&mut App) -> bool) -> bool {
        let start = std::time::Instant::now();
        while start.elapsed() < limit {
            self.app.update();
            if done(&mut self.app) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        false
    }

    /// The render pipelines still compiling on the async compute pool (0 for a CPU app).
    pub fn pipelines_compiling(&self) -> usize {
        use bevy::render::RenderApp;
        use bevy::render::render_resource::{CachedPipelineState, PipelineCache};
        let Some(render) = self.app.get_sub_app(RenderApp) else {
            return 0;
        };
        render
            .world()
            .resource::<PipelineCache>()
            .pipelines()
            .filter(|p| matches!(p.state, CachedPipelineState::Creating(_)))
            .count()
    }

    /// Runs frames until no render pipeline is compiling, at most 10 s; whether none is.
    pub fn settle_pipelines(&mut self) -> bool {
        let limit = Duration::from_secs(10);
        let start = std::time::Instant::now();
        while self.pipelines_compiling() > 0 {
            if start.elapsed() > limit {
                return false;
            }
            self.app.update();
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    }

    /// Writes a library command.
    pub fn command(&mut self, c: ftgsxr_app::LibraryCommand) {
        self.app.world_mut().write_message(c);
    }

    /// Runs `n` frames of `FRAME` each.
    pub fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.app.update();
        }
    }
}

/// #291 (§2.4 Threads): with bevy's `multi_threaded`, `PipelineCache` compiles pipelines on
/// the async compute pool. A test that returns while one is compiling (bevy's tonemapping
/// pipeline, say) lets the harness exit the process with that worker inside the Vulkan
/// driver, and the driver's exit-time destructors crash (SIGSEGV after the tests passed).
/// Dropping a `gpu_` app first lets its compiles finish. Not while panicking: a failed test
/// must report, not abort on a second panic.
impl Drop for TestApp {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.settle_pipelines();
        }
    }
}

/// The render half the adapter's own tests add (`ftgsxr-render`'s `bevy_adapter`).
fn add_render(app: &mut App) {
    use bevy::camera::CameraPlugin;
    use bevy::core_pipeline::CorePipelinePlugin;
    use bevy::image::ImagePlugin;
    use bevy::mesh::MeshPlugin;
    use bevy::render::RenderPlugin;
    use bevy::window::{ExitCondition, WindowPlugin};
    app.add_plugins((
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
}
