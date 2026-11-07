// SPDX-License-Identifier: Apache-2.0

//! R-18, the contact shadow (§1.9 Floor, shadows and anchors; §3.2 R-18; Q-342, #569):
//! a headless app with `ContactShadowPlugin`, `Environment` in passthrough mode, the
//! `Headset` marker (Q-266), `SettingsRes` and an image-target `SplatCamera` of
//! `LinearOver { background: None }` (D-14) looking straight down at an instance standing
//! on the floor. The target's alpha along the footprint's x axis is Q-342's ellipse: 0.3
//! at the center within 1/255, falling by smoothstep to 0 over the outer 10 cm of the
//! radius, monotone; two overlapping footprints give the larger alpha where they meet;
//! with `[environment] contact_shadows` off, in VR mode or without the marker, nothing is
//! drawn.
//!
//! The model is two small primitives at opposite top corners of its box, so the splats
//! never cover the row the test reads; a render with the shadows off checks that.

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
use bevy::math::Vec3;
use bevy::mesh::MeshPlugin;
use bevy::render::error_handler::{RenderErrorHandler, RenderErrorPolicy};
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::{RenderApp, RenderPlugin};
use bevy::transform::TransformPlugin;
use bevy::transform::components::Transform;
use bevy::window::{ExitCondition, WindowPlugin};
use ftgsxr_app::{Headset, SettingsRes};
use ftgsxr_core::{Aabb, Model, ModelKind, RenderConventions, TimeSpan};
use ftgsxr_io::{ModelBytes, Settings};
use ftgsxr_render::{
    OutputSpace, QualityTier, SortMode, SplatAsset, SplatCamera, SplatEvent, SplatLoad, SplatModel,
    SplatPlugin,
};
use ftgsxr_testkit::{exchange_bytes, layer_messages};
use ftgsxr_xr::{ContactShadowPlugin, Environment, EnvironmentMode, PassthroughStyle};

const MAX_FRAMES: usize = 600;
const SETTLE_FRAMES: usize = 4;
/// The target's side, odd so that the middle pixel's center lies on the view's axis.
const SIDE: u32 = 257;
/// The camera's height above the floor, looking straight down, and its field of view.
const HEIGHT: f32 = 2.0;
const FOV: f32 = std::f32::consts::FRAC_PI_3;
/// R-18's tolerance at the center, and the per-pixel tolerance of the feather on the 8-bit
/// target (one step of rounding each way).
const TOL: f32 = 1.0 / 255.0;
/// The second instance's offset along x, so that its feather crosses the first's body.
const OFFSET_X: f32 = 0.6;

#[derive(Resource)]
struct RenderErrored;

#[derive(Resource, Clone, Default)]
struct Seen {
    uploaded: Arc<Mutex<Vec<Entity>>>,
    failed: Arc<Mutex<Option<String>>>,
    readback: Arc<Mutex<Option<Vec<u8>>>>,
}

/// Two primitives of scale 1 cm at (−0.4, 0.5, −0.25) and (0.4, 0.5, 0.25), persistent over
/// the span: B_m spans x ∈ [−0.43, 0.43], z ∈ [−0.28, 0.28] (the centers expanded by
/// 3·max scale, §1.3), well away from the line z = 0 the test reads.
fn two_corner_model() -> Model {
    Model {
        position: vec![[-0.4, 0.5, -0.25], [0.4, 0.5, 0.25]],
        time_center: vec![0.95, 0.95],
        duration: vec![10.0, 10.0],
        velocity: vec![[0.0; 3]; 2],
        scale: vec![[0.01; 3]; 2],
        rotation: vec![[1.0, 0.0, 0.0, 0.0]; 2],
        opacity: vec![0.9, 0.9],
        sh: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
    /// The two instances: the first shown, the second hidden until `show_second`.
    instances: [Entity; 2],
    /// B_m of the model, read from its asset once loaded.
    bounds: Aabb,
}

impl Drop for Scene {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            settle_pipelines(&mut self.app);
        }
    }
}

impl Scene {
    /// The app, both instances loaded and placed standing on the floor (B_m's bottom at
    /// y = 0), the second at x = `OFFSET_X` and hidden; the passthrough environment, the
    /// marker and the settings with `contact_shadows` off.
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
            // #291 (§2.4 Threads): with bevy's `multi_threaded`, which the workspace's
            // build unifies in, `PipelineCache` compiles on the async pool, and the
            // shadow's pipeline is queued a frame or more after the switch turns on, after
            // `pixels`' settle has returned; compiled at once, the frames it runs draw it.
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
                SplatEvent::Uploaded { entity, .. } => seen.uploaded.lock().unwrap().push(*entity),
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

        let bytes = exchange_bytes(&two_corner_model());
        let instances = [Visibility::Visible, Visibility::Hidden].map(|visibility| {
            app.world_mut()
                .spawn((
                    SplatLoad {
                        bytes: ModelBytes::Owned(bytes.clone()),
                        kind_hint: None,
                        reduce_to: None,
                    },
                    Transform::IDENTITY,
                    visibility,
                ))
                .id()
        });
        let mut scene = Scene {
            app,
            seen,
            instances,
            bounds: Aabb {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
            },
        };
        for _ in 0..MAX_FRAMES {
            scene.frames(1);
            if let Some(e) = scene.seen.failed.lock().unwrap().clone() {
                panic!("the load failed: {e}");
            }
            if scene.seen.uploaded.lock().unwrap().len() == 2 {
                break;
            }
        }
        assert_eq!(
            scene.seen.uploaded.lock().unwrap().len(),
            2,
            "both instances uploaded within {MAX_FRAMES} frames"
        );
        let handle = scene
            .app
            .world()
            .get::<SplatModel>(instances[0])
            .expect("the first instance's model")
            .0
            .clone();
        scene.bounds = scene
            .app
            .world()
            .resource::<Assets<SplatAsset>>()
            .get(&handle)
            .expect("the model's asset")
            .bounds;
        let lift = -scene.bounds.min.y;
        for (e, x) in instances.into_iter().zip([0.0, OFFSET_X]) {
            *scene.app.world_mut().get_mut::<Transform>(e).unwrap() =
                Transform::from_xyz(x, lift, 0.0);
        }
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

    /// The target's alphas from a frame rendered after this call, row-major.
    fn alphas(&mut self) -> Vec<f32> {
        self.pixels()
            .iter()
            .map(|p| f32::from(p[3]) / 255.0)
            .collect()
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

    fn set_shadows(&mut self, on: bool) {
        self.app
            .world_mut()
            .resource_mut::<SettingsRes>()
            .0
            .environment
            .contact_shadows = on;
    }

    fn set_mode(&mut self, mode: EnvironmentMode) {
        self.app.world_mut().resource_mut::<Environment>().mode = mode;
    }

    fn show_second(&mut self) {
        *self
            .app
            .world_mut()
            .get_mut::<Visibility>(self.instances[1])
            .unwrap() = Visibility::Visible;
    }
}

/// The floor's x under the center of pixel column `u` of the middle row (z = 0).
fn floor_x(u: usize) -> f32 {
    let ndc = (u as f32 + 0.5) / SIDE as f32 * 2.0 - 1.0;
    ndc * (FOV / 2.0).tan() * HEIGHT
}

/// Q-342's alpha on the footprint's x axis, worked from its text: at distance `d` from
/// the center along x the ellipse's radius is the half extent `hx`, the feather's width
/// f = min(0.1 m, hx), and the alpha 0.3·(1 − smoothstep(hx − f, hx, d)), 0 beyond hx.
fn axis_alpha(d: f32, hx: f32) -> f32 {
    let f = hx.min(0.1);
    let t = ((d - (hx - f)) / f).clamp(0.0, 1.0);
    0.3 * (1.0 - t * t * (3.0 - 2.0 * t))
}

/// The middle row's alphas, from the center column rightwards, with their floor x.
fn axis_row(alphas: &[f32]) -> Vec<(f32, f32)> {
    let side = SIDE as usize;
    let mid = side / 2;
    (mid..side)
        .map(|u| (floor_x(u), alphas[mid * side + u]))
        .collect()
}

#[test]
#[ignore = "gpu"]
fn gpu_r18_contact_shadow_is_a_feathered_dark_ellipse_under_the_footprint_in_passthrough() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut scene = Scene::settled();
    let b = scene.bounds;
    let hx = 0.5 * (b.max.x - b.min.x);
    let cx = 0.5 * (b.max.x + b.min.x);
    assert!(
        (hx - 0.43).abs() < 1e-3 && cx.abs() < 1e-3,
        "B_m as built: {b:?}"
    );

    // The switch off: no shadow, and the splats leave the row the test reads uncovered.
    let baseline = scene.alphas();
    assert!(
        baseline.iter().filter(|&&a| a > 0.1).count() > 4,
        "the splats are drawn"
    );
    assert!(
        axis_row(&baseline).iter().all(|&(_, a)| a == 0.0),
        "the splats leave the middle row uncovered"
    );

    // R-18: 0.3 at the center within 1/255, Q-342's feather over the outer 10 cm,
    // monotone, 0 beyond the footprint.
    scene.set_shadows(true);
    let pixels = scene.pixels();
    let alphas: Vec<f32> = pixels.iter().map(|p| f32::from(p[3]) / 255.0).collect();
    let row = axis_row(&alphas);
    let (x0, a0) = row[0];
    assert!(x0.abs() < 1e-6, "the middle column lies on the axis");
    assert!((a0 - 0.3).abs() <= TOL, "0.3 at the center: {a0}");
    for w in row.windows(2) {
        assert!(w[1].1 <= w[0].1, "monotone outwards: {row:?}");
    }
    let mut feathered = 0;
    for &(x, a) in &row {
        let want = axis_alpha(x - cx, hx);
        assert!(
            (a - want).abs() <= TOL + 0.5 / 255.0,
            "at x = {x}: {a}, Q-342's {want}"
        );
        if want > 0.01 && want < 0.29 {
            feathered += 1;
        }
    }
    assert!(feathered >= 5, "the feather spans pixels: {feathered}");
    assert!(
        row.iter().any(|&(x, a)| x > hx + 0.02 && a == 0.0),
        "0 beyond the footprint"
    );
    // A dark disc: premultiplied black, so the compositor darkens the room under it.
    let mid = SIDE as usize / 2;
    assert_eq!(pixels[mid * SIDE as usize + mid][..3], [0, 0, 0], "black");

    // Two overlapping footprints: the larger alpha where they meet (the alpha maximum,
    // Q-342), so the first's body keeps 0.3 under the second's feather.
    scene.show_second();
    let row = axis_row(&scene.alphas());
    let mut met = 0;
    for &(x, a) in &row {
        let (first, second) = (
            axis_alpha(x - cx, hx),
            axis_alpha((OFFSET_X + cx - x).abs(), hx),
        );
        let want = first.max(second);
        assert!(
            (a - want).abs() <= TOL + 0.5 / 255.0,
            "at x = {x}: {a}, the maximum of {first} and {second}"
        );
        if first > 0.29 && second > 0.01 {
            met += 1;
        }
    }
    assert!(met >= 3, "the footprints meet over pixels: {met}");

    // Nothing with the switch off, in VR mode, or without the marker: the target is the
    // splats' alone.
    scene.set_shadows(false);
    let splats = scene.alphas();
    assert!(
        axis_row(&splats).iter().all(|&(_, a)| a == 0.0),
        "contact_shadows off: nothing drawn"
    );
    scene.set_shadows(true);
    scene.set_mode(EnvironmentMode::Vr);
    assert_eq!(scene.alphas(), splats, "VR mode: nothing drawn");
    scene.set_mode(EnvironmentMode::Passthrough);
    scene.app.world_mut().remove_resource::<Headset>();
    assert_eq!(scene.alphas(), splats, "no Headset marker: nothing drawn");
}
