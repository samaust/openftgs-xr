// SPDX-License-Identifier: Apache-2.0

//! A tracked hand drawn in passthrough on the GPU (§1.9 Depth occlusion, §1.4 Depth and
//! occlusion, D-16, Q-364, #608): a headless app with `SplatPlugin`,
//! `ContactShadowPlugin` and Bevy's PBR, an image-target `SplatCamera` of
//! `LinearOver { background: None }` (D-14) looking straight down at a model standing on
//! the floor, and a capsule of the hands' grey material (`tracking::grey`) between them.
//! Where the capsule is, the target is the capsule's one opaque grey, the primitive and
//! the contact shadow behind it hidden; everywhere else the target's color and alpha are
//! the render's without the capsule, bit for bit.
//!
//! Private, since the hands' material is crate-private; under `passthrough`, which may
//! use `tracking` (§2.1's module order) and owns the shadow.

use std::sync::{Arc, Mutex};

use bevy::MinimalPlugins;
use bevy::app::{App, PluginsState};
use bevy::asset::{AssetPlugin, Assets, RenderAssetUsages};
use bevy::camera::visibility::Visibility;
use bevy::camera::{
    Camera, Camera3d, CameraPlugin, PerspectiveProjection, Projection, RenderTarget,
};
use bevy::core_pipeline::CorePipelinePlugin;
use bevy::ecs::prelude::*;
use bevy::image::{Image, ImagePlugin};
use bevy::math::primitives::Capsule3d;
use bevy::math::{Quat, Vec3};
use bevy::mesh::{Mesh, Mesh3d, MeshPlugin};
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::render::error_handler::{RenderErrorHandler, RenderErrorPolicy};
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::{RenderApp, RenderPlugin};
use bevy::transform::TransformPlugin;
use bevy::transform::components::Transform;
use bevy::window::{ExitCondition, WindowPlugin};
use ftgsxr_app::{Headset, SettingsRes};
use ftgsxr_core::{Model, ModelKind, RenderConventions, TimeSpan};
use ftgsxr_io::{ModelBytes, Settings};
use ftgsxr_render::{
    OutputSpace, QualityTier, SortMode, SplatAsset, SplatCamera, SplatEvent, SplatLoad, SplatModel,
    SplatPlugin,
};
use ftgsxr_testkit::{exchange_bytes, layer_messages};

use super::{ContactShadowPlugin, Environment, EnvironmentMode, PassthroughStyle};
use crate::tracking::grey;

const MAX_FRAMES: usize = 600;
const SETTLE_FRAMES: usize = 4;
/// The target's side, odd so that the middle pixel's centre lies on the view's axis.
const SIDE: u32 = 129;
/// The camera's height above the floor, looking straight down, and its field of view.
const HEIGHT: f32 = 2.0;
const FOV: f32 = std::f32::consts::FRAC_PI_3;
/// The capsule: along x at this height over z = 0, of this radius and half length.
const CAPSULE_Y: f32 = 1.0;
const CAPSULE_R: f32 = 0.05;
const CAPSULE_HALF: f32 = 0.6;
/// Pixels whose ray passes within this distance of the capsule's surface are neither
/// inside nor outside it (a pixel spans about 9 mm at the capsule).
const EDGE_M: f32 = 0.015;
/// The spread of Bevy's deband dither on an 8-bit channel (one step each way).
const DITHER: u8 = 1;

#[derive(Resource)]
struct RenderErrored;

#[derive(Resource, Clone, Default)]
struct Seen {
    uploaded: Arc<Mutex<bool>>,
    failed: Arc<Mutex<Option<String>>>,
    readback: Arc<Mutex<Option<Vec<u8>>>>,
}

/// A primitive of scale 5 cm over the centre, and two of 2 cm at opposite corners that
/// widen B_m, so the shadow's 0.3 shows around the centre one; all persistent over the
/// span. The corners lie away from the capsule's band.
fn three_primitives() -> Model {
    Model {
        position: vec![[0.0, 0.5, 0.0], [-0.3, 0.5, -0.3], [0.3, 0.5, 0.3]],
        time_center: vec![0.95; 3],
        duration: vec![10.0; 3],
        velocity: vec![[0.0; 3]; 3],
        scale: vec![[0.05; 3], [0.02; 3], [0.02; 3]],
        rotation: vec![[1.0, 0.0, 0.0, 0.0]; 3],
        opacity: vec![0.9; 3],
        sh: vec![1.0; 9],
        sh_degree: 0,
        time: TimeSpan {
            tau0: 0.0,
            tau1: 1.9,
            time_unit: "s".to_owned(),
        },
        conventions: RenderConventions::default(),
        producer: "ftgsxr-testkit".to_owned(),
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    }
}

fn settle_pipelines(app: &mut App) {
    use bevy::render::render_resource::{CachedPipelineState, PipelineCache};
    let start = std::time::Instant::now();
    while app
        .sub_app(RenderApp)
        .world()
        .resource::<PipelineCache>()
        .pipelines()
        .any(|p| {
            matches!(
                p.state,
                CachedPipelineState::Queued | CachedPipelineState::Creating(_)
            )
        })
        && start.elapsed() < std::time::Duration::from_secs(10)
    {
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

struct Scene {
    app: App,
    seen: Seen,
    capsule: Entity,
}

impl Drop for Scene {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            settle_pipelines(&mut self.app);
        }
    }
}

impl Scene {
    /// The app, the primitive loaded and standing on the floor (B_m's bottom at y = 0),
    /// the capsule hidden; passthrough, the marker and the shadows off.
    fn settled() -> Scene {
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
            // As R-18 (#291): pipelines compiled at once, so the frames that follow a
            // change draw it.
            RenderPlugin {
                synchronous_pipeline_compilation: true,
                ..RenderPlugin::default()
            },
            ImagePlugin::default(),
            MeshPlugin,
            CameraPlugin,
            CorePipelinePlugin,
            bevy::light::LightPlugin,
            bevy::pbr::PbrPlugin::default(),
            bevy::gizmos::GizmoPlugin,
            SplatPlugin {
                tier: QualityTier::conformance(),
            },
            ContactShadowPlugin,
        ))
        .insert_resource(RenderErrorHandler(|error, main_world, _| {
            eprintln!("render error: {:?}: {}", error.ty, error.description);
            main_world.insert_resource(RenderErrored);
            RenderErrorPolicy::StopRendering
        }));
        while app.plugins_state() == PluginsState::Adding {
            bevy::tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();

        let mut settings = Settings::default();
        settings.environment.contact_shadows = false;
        app.insert_resource(SettingsRes(settings))
            .insert_resource(Headset)
            .insert_resource(Environment {
                mode: EnvironmentMode::Passthrough,
                passthrough_available: true,
                style: PassthroughStyle::default(),
            });

        let seen = Seen::default();
        app.insert_resource(seen.clone());
        app.add_observer(
            |event: On<SplatEvent>, seen: Res<Seen>| match event.event() {
                SplatEvent::Uploaded { .. } => *seen.uploaded.lock().unwrap() = true,
                SplatEvent::Failed { error, .. } => {
                    *seen.failed.lock().unwrap() = Some(error.clone());
                }
                SplatEvent::Overflow { .. } => {}
            },
        );

        let mut image = Image::new_target_texture(SIDE, SIDE, TextureFormat::Rgba8UnormSrgb, None);
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        image.asset_usage = RenderAssetUsages::RENDER_WORLD;
        let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
        app.world_mut().spawn((
            Camera3d::default(),
            Camera::default(),
            RenderTarget::Image(target.clone().into()),
            Projection::Perspective(PerspectiveProjection {
                fov: FOV,
                aspect_ratio: 1.0,
                near: 0.01,
                ..PerspectiveProjection::default()
            }),
            // Image right is +x, image down is +z.
            Transform::from_xyz(0.0, HEIGHT, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
            SplatCamera {
                sort: SortMode::PerView,
                output: OutputSpace::LinearOver { background: None },
            },
        ));
        app.world_mut().spawn(Readback::texture(target)).observe(
            |event: On<ReadbackComplete>, seen: Res<Seen>| {
                *seen.readback.lock().unwrap() = Some(event.data.clone());
            },
        );

        let instance = app
            .world_mut()
            .spawn((
                SplatLoad {
                    bytes: ModelBytes::Owned(exchange_bytes(&three_primitives())),
                    kind_hint: None,
                    reduce_to: None,
                },
                Transform::IDENTITY,
                Visibility::Visible,
            ))
            .id();
        // The capsule: its +y along x, hidden until `show_capsule`.
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Capsule3d::new(CAPSULE_R, 2.0 * CAPSULE_HALF));
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(grey());
        let capsule = app
            .world_mut()
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::from_xyz(0.0, CAPSULE_Y, 0.0)
                    .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
                Visibility::Hidden,
            ))
            .id();
        let mut scene = Scene { app, seen, capsule };
        for _ in 0..MAX_FRAMES {
            scene.frames(1);
            if let Some(e) = scene.seen.failed.lock().unwrap().clone() {
                panic!("the load failed: {e}");
            }
            if *scene.seen.uploaded.lock().unwrap() {
                break;
            }
        }
        assert!(
            *scene.seen.uploaded.lock().unwrap(),
            "the instance uploaded within {MAX_FRAMES} frames"
        );
        let handle = scene
            .app
            .world()
            .get::<SplatModel>(instance)
            .expect("the instance's model")
            .0
            .clone();
        let bounds = scene
            .app
            .world()
            .resource::<Assets<SplatAsset>>()
            .get(&handle)
            .expect("the model's asset")
            .bounds;
        *scene
            .app
            .world_mut()
            .get_mut::<Transform>(instance)
            .unwrap() = Transform::from_xyz(0.0, -bounds.min.y, 0.0);
        scene.frames(SETTLE_FRAMES);
        settle_pipelines(&mut scene.app);
        scene
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.app.update();
            assert!(
                !self.app.world().contains_resource::<RenderErrored>(),
                "Bevy reported a render error"
            );
        }
    }

    /// The target's pixels from a frame rendered after this call, row-major.
    fn pixels(&mut self) -> Vec<[u8; 4]> {
        *self.seen.readback.lock().unwrap() = None;
        settle_pipelines(&mut self.app);
        self.frames(2 + SETTLE_FRAMES);
        let bytes = self
            .seen
            .readback
            .lock()
            .unwrap()
            .clone()
            .expect("a read back of the target");
        let side = SIDE as usize;
        let row = bytes.len() / side;
        (0..side)
            .flat_map(|y| (0..side).map(move |x| y * row + 4 * x))
            .map(|at| [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
            .collect()
    }

    fn show_capsule(&mut self, shown: bool) {
        *self
            .app
            .world_mut()
            .get_mut::<Visibility>(self.capsule)
            .unwrap() = if shown {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }

    fn set_shadows(&mut self, on: bool) {
        self.app
            .world_mut()
            .resource_mut::<SettingsRes>()
            .0
            .environment
            .contact_shadows = on;
    }
}

/// Where pixel (u, v)'s ray passes the capsule: `Some(true)` inside it, `Some(false)`
/// outside it, `None` within `EDGE_M` of its surface. The ray leaves the camera through
/// the pixel's centre; the distance to the axis is the least over points 2.5 mm apart.
fn inside(u: usize, v: usize) -> Option<bool> {
    let tan = (FOV / 2.0).tan();
    let ndc = |p: usize| (p as f32 + 0.5) / SIDE as f32 * 2.0 - 1.0;
    let origin = Vec3::new(0.0, HEIGHT, 0.0);
    let dir = Vec3::new(ndc(u) * tan, -1.0, ndc(v) * tan).normalize();
    let steps = (2.0 * CAPSULE_HALF / 0.0025) as usize;
    let d = (0..=steps)
        .map(|k| {
            let x = -CAPSULE_HALF + 2.0 * CAPSULE_HALF * k as f32 / steps as f32;
            (Vec3::new(x, CAPSULE_Y, 0.0) - origin).cross(dir).length()
        })
        .fold(f32::INFINITY, f32::min);
    if d < CAPSULE_R - EDGE_M {
        Some(true)
    } else if d > CAPSULE_R + EDGE_M {
        Some(false)
    } else {
        None
    }
}

/// Checks the capsule's render against the one without it: inside the capsule one opaque
/// grey (alpha 255, each channel within `DITHER` of the first inside pixel's, Bevy's
/// deband dither on the unlit color), which it returns; outside, the same pixels. Also
/// returns how many inside pixels had alpha ≥ `covered` before.
fn compare(without: &[[u8; 4]], with: &[[u8; 4]], covered: f32) -> ([u8; 4], usize) {
    let side = SIDE as usize;
    let mut hidden = 0;
    let (mut ins, mut outs) = (0, 0);
    let mut grey: Option<[u8; 4]> = None;
    for v in 0..side {
        for u in 0..side {
            let (a, b) = (without[v * side + u], with[v * side + u]);
            match inside(u, v) {
                Some(true) => {
                    ins += 1;
                    let g = *grey.get_or_insert(b);
                    assert!(
                        b[3] == 255 && (0..3).all(|c| b[c].abs_diff(g[c]) <= DITHER),
                        "one grey inside the capsule at ({u}, {v}): {b:?}, not {g:?}"
                    );
                    if f32::from(a[3]) / 255.0 >= covered {
                        hidden += 1;
                    }
                }
                Some(false) => {
                    outs += 1;
                    assert_eq!(b, a, "outside the capsule at ({u}, {v})");
                }
                None => {}
            }
        }
    }
    assert!(ins > 100 && outs > 10_000, "{ins} inside, {outs} outside");
    let g = grey.expect("pixels inside");
    assert!(
        g[3] == 255 && g[0].abs_diff(g[1]) <= DITHER && g[1].abs_diff(g[2]) <= DITHER,
        "grey: {g:?}"
    );
    (g, hidden)
}

#[test]
#[ignore = "gpu"]
fn gpu_drawn_hand_capsule_hides_what_is_behind_it_and_nothing_else() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut scene = Scene::settled();

    // A capsule of the hands' material in front of a primitive hides it behind its grey,
    // and leaves the target's color and alpha untouched elsewhere.
    let splat = scene.pixels();
    scene.show_capsule(true);
    let (grey, hidden) = compare(&splat, &scene.pixels(), 0.1);
    assert!(
        hidden > 50,
        "the capsule covers the primitive over pixels: {hidden}"
    );

    // The contact shadow under it is hidden too (Q-364): the same grey there, the
    // shadow's 0.3 elsewhere.
    scene.show_capsule(false);
    scene.set_shadows(true);
    let shadowed = scene.pixels();
    let at_shadow = shadowed
        .iter()
        .filter(|p| (f32::from(p[3]) / 255.0 - 0.3).abs() <= 1.5 / 255.0)
        .count();
    assert!(at_shadow > 50, "the shadow's 0.3 over pixels: {at_shadow}");
    scene.show_capsule(true);
    let (over_shadow, hidden) = compare(&shadowed, &scene.pixels(), 0.29);
    assert!(
        (0..4).all(|c| over_shadow[c].abs_diff(grey[c]) <= DITHER),
        "the capsule's grey whatever is behind it: {over_shadow:?}, {grey:?}"
    );
    assert!(
        hidden > 50,
        "the capsule covers the shadow over pixels: {hidden}"
    );
}
