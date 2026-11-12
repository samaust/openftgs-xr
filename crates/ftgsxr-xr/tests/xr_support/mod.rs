// SPDX-License-Identifier: Apache-2.0

//! The `xr` lane's support (§2.7, D-28): Monado started for one test, the log captured,
//! and an `App` built through `add_xr` (Q-120) without winit.
//!
//! Prerequisites: Monado's packages (`monado-service` and the runtime manifest
//! `/usr/share/openxr/1/openxr_monado.json`), a desktop session (`DISPLAY` or
//! `WAYLAND_DISPLAY`: the compositor runs in a window, `XRT_COMPOSITOR_FORCE_XCB`), the
//! systemd unit `monado.socket` inactive, and a Vulkan GPU. The runtime is selected for
//! this process through `XR_RUNTIME_JSON`; the user's active-runtime file is never read or
//! written (PRD #221). Monado's simulated HMD serves when no hardware is found.

#![allow(dead_code)]

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use bevy::app::{App, PluginsState};
use bevy::asset::AssetPlugin;
use bevy::camera::CameraPlugin;
use bevy::core_pipeline::CorePipelinePlugin;
use bevy::image::ImagePlugin;
use bevy::input::InputPlugin;
use bevy::mesh::MeshPlugin;
use bevy::prelude::{MinimalPlugins, PluginGroup, Resource};
use bevy::render::RenderPlugin;
use bevy::render::error_handler::{RenderErrorHandler, RenderErrorPolicy};
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::transform::TransformPlugin;
use bevy::window::{ExitCondition, WindowPlugin};
use ftgsxr_app::{
    AppPaths, AppPlugin, AudioOutputRes, CameraFrames, DocumentProvider, ModelRef, Permission,
    PermissionHandle, PickHandle, PickKind, Platform, StubAudioOutput,
};
use ftgsxr_render::{QualityTier, SplatPlugin};
use ftgsxr_xr::{XrPlugin, add_xr};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;

/// Monado's runtime manifest from Ubuntu's `libopenxr1-monado`.
const MONADO_JSON: &str = "/usr/share/openxr/1/openxr_monado.json";

/// A running `monado-service`, stopped (Enter on its standard input, its clean shutdown)
/// when dropped.
pub struct Monado {
    child: Child,
}

fn socket() -> PathBuf {
    let dir = std::env::var_os("XDG_RUNTIME_DIR").expect("XDG_RUNTIME_DIR for Monado's socket");
    PathBuf::from(dir).join("monado_comp_ipc")
}

impl Monado {
    /// Starts the service and waits for its socket; selects Monado for this process.
    pub fn start() -> Monado {
        // SAFETY: `set_var` races with other threads reading the environment; the lane
        // runs with `--test-threads 1` and this runs before the App, the only reader.
        unsafe { std::env::set_var("XR_RUNTIME_JSON", MONADO_JSON) };
        let child = Command::new("monado-service")
            .env("XRT_COMPOSITOR_FORCE_XCB", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("monado-service runs (the xr lane's prerequisite)");
        let start = Instant::now();
        while !socket().exists() || start.elapsed() < Duration::from_millis(500) {
            assert!(
                start.elapsed() < Duration::from_secs(20),
                "monado-service's socket within 20 s"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
        // The compositor finishes its own start after the socket appears.
        std::thread::sleep(Duration::from_secs(2));
        Monado { child }
    }
}

impl Drop for Monado {
    fn drop(&mut self) {
        if let Some(stdin) = self.child.stdin.as_mut() {
            let _ = stdin.write_all(b"\n");
        }
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(10) {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One captured log line.
#[derive(Clone, Debug)]
pub struct Line {
    pub level: Level,
    pub target: String,
    pub message: String,
}

static LINES: Mutex<Vec<Line>> = Mutex::new(Vec::new());

#[derive(Default)]
struct Record(String);

impl Visit for Record {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.0, "{value:?}");
        } else {
            let _ = write!(self.0, " {}={value:?}", field.name());
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.0.push_str(value);
        } else {
            let _ = write!(self.0, " {}={value}", field.name());
        }
    }
}

struct Collector;

impl<S: Subscriber> Layer<S> for Collector {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut record = Record::default();
        event.record(&mut record);
        let line = Line {
            level: *event.metadata().level(),
            target: event.metadata().target().to_owned(),
            message: record.0,
        };
        eprintln!("{} {}: {}", line.level, line.target, line.message);
        LINES
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(line);
    }
}

/// Installs the capturing subscriber (info and above) once per process, and empties the
/// capture, so that a test reads its own lines only: the crate's tests share one test
/// binary (#413), and the lane runs them one after another in its process.
pub fn capture_log() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        tracing_subscriber::registry()
            .with(Collector.with_filter(LevelFilter::INFO))
            .try_init()
            .expect("the xr test's log capture holds the process's subscriber");
    });
    LINES.lock().unwrap_or_else(PoisonError::into_inner).clear();
}

/// The lines captured so far.
pub fn lines() -> Vec<Line> {
    LINES.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

/// The swapchain sizes `display` logged, one per session, in order (`swapchain WxH …`).
pub fn swapchain_sizes() -> Vec<(u32, u32)> {
    lines()
        .iter()
        .filter(|l| l.target == "ftgsxr_xr::display")
        .filter_map(|l| {
            let rest = l.message.strip_prefix("swapchain ")?;
            let (w, h) = rest.split_whitespace().next()?.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .collect()
}

/// Set when the render world reports an error.
#[derive(Resource)]
pub struct RenderErrored;

/// An `App` with `SplatPlugin { tier: conformance }` and `xr`, built through `add_xr`
/// from a minimal plugin group without winit or Bevy's log plugin (the test's subscriber
/// holds the log), finished and cleaned up.
pub fn xr_app(xr: XrPlugin, setup: impl FnOnce(&mut App)) -> App {
    xr_app_in(QualityTier::conformance(), xr, setup)
}

/// As [`xr_app`], with `SplatPlugin { tier }`.
pub fn xr_app_in(tier: QualityTier, xr: XrPlugin, setup: impl FnOnce(&mut App)) -> App {
    let mut app = App::new();
    setup(&mut app);
    app.add_plugins(add_xr(xr_group(), xr))
        .add_plugins(SplatPlugin { tier });
    finish(app)
}

/// The player on the headset's composition (#461): as [`xr_app`], with `AppPlugin` (which
/// adds `SplatPlugin` itself) over a stub `Platform` whose directories are under `root`,
/// the stub audio output, and Bevy's input, light, PBR and gizmo plugins, so the meshes
/// the headset draws (the comfort quads) exist as in the player.
pub fn xr_player_app(xr: XrPlugin, root: &std::path::Path) -> App {
    let mut app = App::new();
    app.insert_resource(AudioOutputRes(std::sync::Arc::new(StubAudioOutput::new())));
    app.add_plugins(add_xr(
        xr_group()
            .add(InputPlugin)
            .add(bevy::light::LightPlugin)
            .add(bevy::pbr::PbrPlugin::default())
            .add(bevy::gizmos::GizmoPlugin),
        xr,
    ))
    .add_plugins(AppPlugin {
        platform: std::sync::Arc::new(StubPlatform(paths_under(root))),
        tier: QualityTier::conformance(),
        initial: Vec::new(),
        scene: None,
    });
    finish(app)
}

/// The player's directories under `root` (§1.11's layout).
fn paths_under(root: &std::path::Path) -> AppPaths {
    AppPaths {
        config: root.join("config"),
        cache: root.join("cache"),
        models: root.join("models"),
        scenes: root.join("config/scenes"),
        screenshots: root.join("pictures"),
        logs: root.join("cache/logs"),
    }
}

/// A `Platform` whose picks are empty, permissions granted, no documents or camera.
struct StubPlatform(AppPaths);

impl Platform for StubPlatform {
    fn paths(&self) -> &AppPaths {
        &self.0
    }
    fn pick(&self, _: PickKind) -> PickHandle {
        PickHandle(0)
    }
    fn poll_pick(&self, _: PickHandle) -> std::task::Poll<Result<Vec<ModelRef>, String>> {
        std::task::Poll::Ready(Ok(Vec::new()))
    }
    fn request_permission(&self, _: Permission) -> PermissionHandle {
        PermissionHandle(0)
    }
    fn poll_permission(&self, _: PermissionHandle) -> std::task::Poll<bool> {
        std::task::Poll::Ready(true)
    }
    fn document_provider(&self) -> Option<std::sync::Arc<dyn DocumentProvider>> {
        None
    }
    fn camera_frames(&self) -> Option<std::sync::Arc<dyn CameraFrames>> {
        None
    }
}

/// The minimal plugin group without winit or Bevy's log plugin.
fn xr_group() -> bevy::app::PluginGroupBuilder {
    MinimalPlugins
        .build()
        .add(TransformPlugin)
        .add(AssetPlugin::default())
        .add(WindowPlugin {
            primary_window: None,
            exit_condition: ExitCondition::DontExit,
            close_when_requested: false,
            ..WindowPlugin::default()
        })
        .add(RenderPlugin::default())
        .add(ImagePlugin::default())
        .add(PipelinedRenderingPlugin)
        .add(MeshPlugin)
        .add(CameraPlugin)
        .add(CorePipelinePlugin)
}

/// Inserts the render error handler, finishes and cleans the app up.
fn finish(mut app: App) -> App {
    app.insert_resource(RenderErrorHandler(|error, main_world, _| {
        tracing::error!("render error: {:?}: {}", error.ty, error.description);
        main_world.insert_resource(RenderErrored);
        RenderErrorPolicy::StopRendering
    }));
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
    }
    app.finish();
    app.cleanup();
    app
}
