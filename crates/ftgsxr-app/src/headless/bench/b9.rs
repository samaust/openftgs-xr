// SPDX-License-Identifier: Apache-2.0

//! B-9 (§3.5, Q-155): the desktop's startup, in-process. From `App::new()` to the first
//! frame drawn after the first `Uploaded` of a probe loaded through `AppPlugin.initial`, on
//! an offscreen camera, without and with the UI; reported, not budgeted. The Quest's half
//! is M-8's device benchmarks.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::{Duration, Instant};

use bevy::app::{App, PluginsState};
use bevy::asset::{AssetPlugin, Assets, RenderAssetUsages};
use bevy::camera::{
    Camera, Camera3d, CameraPlugin, PerspectiveProjection, Projection, RenderTarget,
};
use bevy::core_pipeline::CorePipelinePlugin;
use bevy::image::{Image, ImagePlugin};
use bevy::input::InputPlugin;
use bevy::math::Vec3;
use bevy::mesh::MeshPlugin;
use bevy::prelude::{MinimalPlugins, On, Transform, TransformPlugin, With};
use bevy::render::RenderPlugin;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::window::{ExitCondition, WindowPlugin};
use bevy_egui::PrimaryEguiContext;
use ftgsxr_core::write_exchange;
use ftgsxr_io::{AppPaths, DocumentProvider, ModelRef};
use ftgsxr_render::{OutputSpace, QualityTier, SortMode, SplatCamera, SplatEvent, SplatStats};

use super::super::synthetic::{Persistence, synthetic_model};
use super::super::{HeadlessError, core_input, input_error};
use super::{Given, Gpu, HEIGHT, Json, Measured, WIDTH};
use crate::platform::{CameraFrames, Permission, PermissionHandle, PickHandle, PickKind, Platform};
use crate::plugin::AppPlugin;

/// The probe's N and degree when no `--model` is given (Q-222).
const B9_N: usize = 100_000;
/// As `B9_N`.
const B9_DEGREE: u8 = 2;
/// The startups timed per variant, the median kept (Q-102).
const B9_REPEATS: usize = 3;
/// A startup that draws nothing within this is an environment failure.
const B9_TIMEOUT: Duration = Duration::from_secs(300);

/// The bench's platform: the player's directories under a scratch directory, a picker
/// that returns nothing.
struct BenchPlatform {
    paths: AppPaths,
}

impl Platform for BenchPlatform {
    fn paths(&self) -> &AppPaths {
        &self.paths
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
        Poll::Ready(true)
    }
    fn document_provider(&self) -> Option<Arc<dyn DocumentProvider>> {
        None
    }
    fn camera_frames(&self) -> Option<Arc<dyn CameraFrames>> {
        None
    }
}

fn platform(dir: &Path) -> BenchPlatform {
    let at = |d: &str| dir.join(d);
    BenchPlatform {
        paths: AppPaths {
            config: at("config"),
            cache: at("cache"),
            models: at("models"),
            scenes: at("scenes"),
            screenshots: at("screenshots"),
            logs: at("logs"),
        },
    }
}

/// One startup's times, in seconds from `App::new()`.
struct Startup {
    uploaded_s: f64,
    first_frame_s: f64,
}

/// One startup: the app the desktop builds without its window (the render plugins of
/// `DefaultPlugins` that `AppPlugin` needs, Q-155), `model` in `initial`, an offscreen
/// 1920 × 1080 splat camera; `ui` keeps egui's context on the camera.
fn startup(model: &Path, dir: &Path, ui: bool) -> Result<Startup, HeadlessError> {
    let start = Instant::now();
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
    let uploaded: Arc<Mutex<Option<f64>>> = Arc::new(Mutex::new(None));
    let failed = Arc::new(AtomicBool::new(false));
    {
        let (up, fail) = (uploaded.clone(), failed.clone());
        app.add_observer(move |e: On<SplatEvent>| match e.event() {
            SplatEvent::Uploaded { .. } => {
                let mut u = up.lock().expect("invariant: the lock is not poisoned");
                u.get_or_insert(start.elapsed().as_secs_f64());
            }
            SplatEvent::Failed { .. } => fail.store(true, Ordering::Relaxed),
            SplatEvent::Overflow { .. } => {}
        });
    }
    app.add_plugins(AppPlugin {
        platform: Arc::new(platform(dir)),
        tier: QualityTier::desktop(),
        initial: vec![ModelRef::File(model.to_path_buf())],
        scene: None,
    });
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
    }
    app.finish();
    app.cleanup();
    if !ui
        && let Some(mut s) = app
            .world_mut()
            .get_resource_mut::<bevy_egui::EguiGlobalSettings>()
    {
        s.auto_create_primary_context = false;
    }
    let mut image = Image::new_target_texture(WIDTH, HEIGHT, TextureFormat::Rgba8UnormSrgb, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    app.world_mut().spawn((
        Camera3d::default(),
        Camera::default(),
        RenderTarget::Image(target.into()),
        Projection::Perspective(PerspectiveProjection {
            near: 0.01,
            ..PerspectiveProjection::default()
        }),
        Transform::from_xyz(0.0, 0.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        SplatCamera {
            sort: SortMode::PerView,
            output: OutputSpace::LinearOver { background: None },
        },
    ));
    // Q-222: the statistics arrive two frames late (§1.4 stage 8), so the frame drawn
    // is the one two updates before the statistics first count a splat.
    let mut ends: Vec<f64> = Vec::new();
    loop {
        app.update();
        ends.push(start.elapsed().as_secs_f64());
        if failed.load(Ordering::Relaxed) {
            return Err(HeadlessError::Input(format!(
                "{}: the probe failed to load",
                model.display()
            )));
        }
        let up = *uploaded
            .lock()
            .expect("invariant: the lock is not poisoned");
        let drawn = app
            .world()
            .get_resource::<SplatStats>()
            .and_then(|s| s.0)
            .is_some_and(|s| s.visible_splats > 0);
        if let Some(uploaded_s) = up
            && drawn
        {
            // With the UI, egui's primary context must sit on the camera, else the
            // figure would be labelled with a UI it did not draw.
            let contexts = app
                .world_mut()
                .query_filtered::<(), With<PrimaryEguiContext>>()
                .iter(app.world())
                .count();
            if ui != (contexts > 0) {
                return Err(HeadlessError::Environment(format!(
                    "B-9: ui {ui} but {contexts} egui primary contexts"
                )));
            }
            let frame = ends.len().saturating_sub(3);
            return Ok(Startup {
                uploaded_s,
                first_frame_s: ends[frame].max(uploaded_s),
            });
        }
        if start.elapsed() > B9_TIMEOUT {
            return Err(HeadlessError::Environment(format!(
                "B-9: no frame drew the probe within {} s",
                B9_TIMEOUT.as_secs()
            )));
        }
    }
}

/// B-9 (§3.5, Q-155): the median of `B9_REPEATS` startups without the UI and with it, and
/// the first startup of the process apart (it pays the one-time costs).
// Q-155, Q-222
pub(super) fn bench_b9(_: &Gpu, given: Option<&Given>) -> Result<Measured, HeadlessError> {
    let dir = std::env::temp_dir().join(format!("ftgsxr-bench-b9-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| input_error(&dir, e))?;
    let result = (|| {
        let (path, label, n): (PathBuf, Json, usize) = match given {
            Some(g) => (g.path.clone(), g.label(), g.model.len()),
            None => {
                let path = dir.join("b9-probe.ftgs-if");
                let model = synthetic_model(1, B9_N, B9_DEGREE, Persistence::All);
                write_exchange(&path, &model).map_err(|e| core_input(&path, &e))?;
                (path, Json::str("synthetic"), B9_N)
            }
        };
        let mut m = Measured::default();
        let mut first = true;
        for ui in [false, true] {
            let mut uploads = Vec::new();
            let mut frames = Vec::new();
            for k in 0..B9_REPEATS {
                let run = dir.join(format!("run-{ui}-{k}"));
                let s = startup(&path, &run, ui)?;
                let _ = std::fs::remove_dir_all(&run);
                if first {
                    let setup = setup(&label, n, ui, true);
                    m.figure(&setup, "uploaded_s", s.uploaded_s, "s");
                    m.figure(&setup, "first_frame_s", s.first_frame_s, "s");
                    first = false;
                }
                uploads.push(s.uploaded_s);
                frames.push(s.first_frame_s);
            }
            let setup = setup(&label, n, ui, false);
            m.figure(&setup, "uploaded_s", super::median(&mut uploads), "s");
            m.figure(&setup, "first_frame_s", super::median(&mut frames), "s");
        }
        Ok(Measured {
            notes: vec![B9_NOTE],
            ..m
        })
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn setup(label: &Json, n: usize, ui: bool, cold: bool) -> Json {
    Json::obj([
        ("model", label.clone()),
        ("n", Json::Num(n as f64)),
        ("tier", Json::str("desktop")),
        ("width", Json::Num(f64::from(WIDTH))),
        ("height", Json::Num(f64::from(HEIGHT))),
        ("ui", Json::Bool(ui)),
        ("run", Json::str(if cold { "first" } else { "median" })),
    ])
}

/// The note on B-9's figures.
const B9_NOTE: &str = "B-9 times the app in-process from App::new() without a window (Q-155): the window's creation and the swapchain are not counted; the first frame drawn is the one two updates before the statistics first count a splat (Q-222); the probe file was just written, so its read may come from the page cache";
