// SPDX-License-Identifier: Apache-2.0

//! #722's adapter tests of §2.4's shared prepare (sources of truth: §1.4 Stereo, D-4,
//! §2.4 Frame loop hooks, §1.4 stage 8; Q-399, Q-403, Q-100's skip, Q-125's order): two
//! `SplatCamera`s whose `sort` is `SharedCenter` form one group, prepared once with both
//! eyes as cull views and drawn by both. The scene is hand-built so the eyes' visible sets
//! differ: 3 rows only the left eye sees, 5 only the right, 7 both see and 4 behind
//! them, so one prepare culling for both eyes counts the 15 of the union, where a prepare
//! per eye would publish the first eye's 10, and a sum 22. Each eye's target matches
//! `HeadlessRenderer::render_stereo`'s shared-center render of the same eyes at E15's
//! tier, and the group copies one statistics block a frame, so a capacity below the
//! visible count raises `SplatEvent::Overflow` once a frame, where two per-view cameras
//! raise it twice. Moving only the second eye runs stages 1–4 again (the skip applies to
//! the group as a whole).

use std::sync::{Arc, Mutex};

use bevy::MinimalPlugins;
use bevy::app::{App, Last, PluginsState};
use bevy::asset::{AssetPlugin, Assets, RenderAssetUsages};
use bevy::camera::{
    Camera, Camera3d, CameraPlugin, PerspectiveProjection, Projection, RenderTarget,
};
use bevy::core_pipeline::CorePipelinePlugin;
use bevy::ecs::prelude::*;
use bevy::image::{Image, ImagePlugin};
use bevy::math::{Mat3, Quat, Vec3};
use bevy::mesh::MeshPlugin;
use bevy::render::error_handler::{RenderErrorHandler, RenderErrorPolicy};
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::renderer::RenderDevice;
use bevy::render::settings::WgpuFeatures;
use bevy::render::{RenderApp, RenderPlugin};
use bevy::transform::TransformPlugin;
use bevy::transform::components::Transform;
use bevy::window::{ExitCondition, WindowPlugin};
use ftgsxr_core::{Camera as FtgsCamera, DMat3, DMat4, DVec3, Model, Pose, RenderImage};
use ftgsxr_io::ModelBytes;
use ftgsxr_render::{
    Capacity, FrameStats, HeadlessRenderer, OutputSpace, QualityTier, SortMode, SplatCamera,
    SplatEvent, SplatLoad, SplatPlugin, SplatStats, SplatTime,
};
use ftgsxr_testkit::{exchange_bytes, layer_messages, metrics};

use crate::display_support::{Row, model};

const MAX_FRAMES: usize = 600;
const SETTLE_FRAMES: usize = 4;
const TAU: f64 = 0.95;
const W: u32 = 64;
const H: u32 = 48;
const F: f64 = 60.0;
/// E15's tier.
const MIN_PSNR: f64 = 45.0;
const MAX_SHARE: f64 = 0.005;
/// The rows each eye sees and their union (the module's docs).
const LEFT: u32 = 10;
const RIGHT: u32 = 12;
const UNION: u32 = 15;

#[derive(Resource)]
struct RenderErrored;

#[derive(Resource, Clone, Default)]
struct Seen {
    uploaded: Arc<Mutex<bool>>,
    failed: Arc<Mutex<Option<String>>>,
    overflows: Arc<Mutex<usize>>,
    stats: Arc<Mutex<Vec<FrameStats>>>,
    /// The last read back of each eye's target: left, right.
    readbacks: [Arc<Mutex<Option<Vec<u8>>>>; 2],
}

/// An eye at (x, 0, 0) looking along +z, y down in the image (openftgs's camera with the
/// identity rotation), 64 × 48 at focal 60 with a centered principal point.
fn eye(x: f64) -> FtgsCamera {
    let mut w2c = DMat4::IDENTITY;
    w2c.w_axis = DVec3::new(-x, 0.0, 0.0).extend(1.0);
    FtgsCamera {
        width: W,
        height: H,
        k: DMat3::from_cols(
            DVec3::new(F, 0.0, 0.0),
            DVec3::new(0.0, F, 0.0),
            DVec3::new(f64::from(W) / 2.0, f64::from(H) / 2.0, 1.0),
        ),
        world_to_camera: w2c,
    }
}

/// The left and right eyes, 1 m apart.
fn eyes() -> [FtgsCamera; 2] {
    [eye(-0.5), eye(0.5)]
}

/// The module's scene at z = 2 m (z = −2 m behind the eyes): at x = −1.3 the left eye's
/// u is 8 and the right's −22, so 3 rows there are the left eye's only; 5 at x = 1.3 the
/// right's only; 7 at x = 0 both eyes'; 4 behind, neither's. Each row is small (3σ ≈ 4.5
/// pixels), so it lies far inside or far outside each image.
fn scene_model() -> Model {
    let row = |x: f32, y: f32, z: f32, color: [f32; 3]| Row {
        position: [x, y, z],
        velocity: [0.0; 3],
        time_center: TAU,
        duration: 100.0,
        scale: 0.05,
        opacity: 0.8,
        color,
    };
    let mut rows = Vec::new();
    for y in [-0.3, 0.0, 0.3] {
        rows.push(row(-1.3, y, 2.0, [0.9, 0.2, 0.2]));
    }
    for y in [-0.3, -0.15, 0.0, 0.15, 0.3] {
        rows.push(row(1.3, y, 2.0, [0.2, 0.2, 0.9]));
    }
    for y in [-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45] {
        rows.push(row(0.0, y, 2.0, [0.2, 0.9, 0.2]));
    }
    for x in [-1.0, -0.3, 0.3, 1.0] {
        rows.push(row(x, 0.0, -2.0, [0.9, 0.9, 0.9]));
    }
    model(&rows)
}

/// openftgs's camera as a Bevy transform and projection (centered principal point).
fn bevy_camera(camera: &FtgsCamera) -> (Transform, Projection) {
    let r = DMat3::from_mat4(camera.world_to_camera);
    let t = camera.world_to_camera.w_axis.truncate();
    let center = -(r.transpose() * t);
    let flip = DMat3::from_diagonal(DVec3::new(1.0, -1.0, -1.0));
    let m = (r.transpose() * flip).as_mat3();
    let transform = Transform {
        translation: center.as_vec3(),
        rotation: Quat::from_mat3(&Mat3::from_cols(m.x_axis, m.y_axis, m.z_axis)),
        scale: Vec3::ONE,
    };
    let (w, h) = (f64::from(camera.width), f64::from(camera.height));
    let projection = Projection::Perspective(PerspectiveProjection {
        fov: (2.0 * (h / 2.0 / camera.k.y_axis.y).atan()) as f32,
        aspect_ratio: (w / h) as f32,
        near: 0.01,
        ..PerspectiveProjection::default()
    });
    (transform, projection)
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
    /// The eyes' camera entities: left, right.
    eyes: [Entity; 2],
}

impl Drop for Scene {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            settle_pipelines(&mut self.app);
        }
    }
}

impl Scene {
    /// The scene seen by the left eye (order 0) and the right eye (order 1), both of sort
    /// `sort`, each writing `FileEncoded` into its own `Rgba16Float` image, under the
    /// conformance tier with `capacity`; the right eye spawned first when `right_first`,
    /// so entity order cannot stand in for camera order.
    fn settled(sort: SortMode, capacity: Capacity, right_first: bool) -> Scene {
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
                tier: QualityTier {
                    capacity,
                    ..QualityTier::conformance()
                },
            },
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
        let seen = Seen::default();
        app.insert_resource(seen.clone());
        app.add_observer(
            |event: On<SplatEvent>, seen: Res<Seen>| match event.event() {
                SplatEvent::Uploaded { .. } => *seen.uploaded.lock().unwrap() = true,
                SplatEvent::Failed { error, .. } => {
                    *seen.failed.lock().unwrap() = Some(error.clone());
                }
                SplatEvent::Overflow { .. } => *seen.overflows.lock().unwrap() += 1,
            },
        );
        app.add_systems(Last, |stats: Res<SplatStats>, seen: Res<Seen>| {
            if stats.is_changed()
                && let Some(s) = stats.0
            {
                seen.stats.lock().unwrap().push(s);
            }
        });
        let mut entities = [Entity::PLACEHOLDER; 2];
        let order: [usize; 2] = if right_first { [1, 0] } else { [0, 1] };
        for slot in order {
            let (transform, projection) = bevy_camera(&eyes()[slot]);
            let mut image = Image::new_target_texture(W, H, TextureFormat::Rgba16Float, None);
            image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
            image.asset_usage = RenderAssetUsages::RENDER_WORLD;
            let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
            entities[slot] = app
                .world_mut()
                .spawn((
                    Camera3d::default(),
                    Camera {
                        order: slot as isize,
                        ..Camera::default()
                    },
                    RenderTarget::Image(target.clone().into()),
                    projection,
                    transform,
                    SplatCamera {
                        sort,
                        output: OutputSpace::FileEncoded,
                    },
                ))
                .id();
            app.world_mut().spawn(Readback::texture(target)).observe(
                move |event: On<ReadbackComplete>, seen: Res<Seen>| {
                    *seen.readbacks[slot].lock().unwrap() = Some(event.data.clone());
                },
            );
        }
        app.world_mut().spawn((
            SplatLoad {
                bytes: ModelBytes::Owned(exchange_bytes(&scene_model())),
                kind_hint: None,
                reduce_to: None,
            },
            SplatTime {
                tau: TAU,
                extrapolating: false,
            },
            Transform::IDENTITY,
        ));
        let mut scene = Scene {
            app,
            seen,
            eyes: entities,
        };
        let mut uploaded_at = None;
        for frame in 0..MAX_FRAMES {
            scene.frames(1);
            if let Some(e) = scene.seen.failed.lock().unwrap().clone() {
                panic!("the load failed: {e}");
            }
            if uploaded_at.is_none() && *scene.seen.uploaded.lock().unwrap() {
                uploaded_at = Some(frame);
            }
            if uploaded_at.is_some_and(|f| frame >= f + SETTLE_FRAMES) {
                break;
            }
        }
        assert!(uploaded_at.is_some(), "uploaded within {MAX_FRAMES} frames");
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

    /// Both eyes' targets from a frame rendered after this call, composited over black as
    /// a headless render is (E7).
    fn images(&mut self) -> [RenderImage; 2] {
        for slot in &self.seen.readbacks {
            *slot.lock().unwrap() = None;
        }
        settle_pipelines(&mut self.app);
        self.frames(2 + SETTLE_FRAMES);
        std::array::from_fn(|i| {
            let bytes = self.seen.readbacks[i]
                .lock()
                .unwrap()
                .clone()
                .expect("a read back of the target");
            layer_image(&bytes)
        })
    }

    /// The statistics published since the last call.
    fn take_stats(&self) -> Vec<FrameStats> {
        std::mem::take(&mut *self.seen.stats.lock().unwrap())
    }

    fn timestamps(&self) -> bool {
        self.app
            .sub_app(RenderApp)
            .world()
            .resource::<RenderDevice>()
            .features()
            .contains(WgpuFeatures::TIMESTAMP_QUERY)
    }
}

/// IEEE binary16 to f32.
fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let frac = f32::from(bits & 0x3ff);
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 if frac == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        e => (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    }
}

/// A read back `Rgba16Float` layer (C, A) as an image over black: rgb = C, alpha = A.
fn layer_image(bytes: &[u8]) -> RenderImage {
    let row = bytes.len() / H as usize;
    let (mut rgb, mut alpha) = (Vec::new(), Vec::new());
    for y in 0..H as usize {
        for x in 0..W as usize {
            let texel = &bytes[y * row + 8 * x..][..8];
            let v = |i: usize| half(u16::from_le_bytes([texel[2 * i], texel[2 * i + 1]]));
            rgb.extend([v(0), v(1), v(2)]);
            alpha.push(v(3));
        }
    }
    RenderImage {
        width: W,
        height: H,
        rgb,
        alpha,
    }
}

/// The eyes' shared-center render by the headless renderer (D-4), over black.
fn reference_pair() -> [RenderImage; 2] {
    let mut renderer = HeadlessRenderer::new(QualityTier::conformance()).expect("an adapter");
    let id = renderer
        .load(&scene_model(), None)
        .expect("the model loads");
    renderer
        .render_stereo(
            &[(id, Pose::IDENTITY)],
            &eyes(),
            TAU,
            [0.0; 3],
            SortMode::SharedCenter,
        )
        .expect("the reference render")
}

#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_shared_center_pair_prepares_once_and_both_eyes_draw() {
    let _layer = layer_messages(); // R-16, Q-105
    let reference = reference_pair();
    for right_first in [false, true] {
        let mut scene = Scene::settled(SortMode::SharedCenter, Capacity::SumOfModels, right_first);
        let images = scene.images();
        for (k, (ours, theirs)) in images.iter().zip(&reference).enumerate() {
            let m = metrics(theirs, ours);
            eprintln!(
                "eye {k} (right spawned first: {right_first}): PSNR {:.2} dB, s {:.4}, s_alpha {:.4}",
                m.psnr, m.share, m.alpha
            );
            assert!(
                m.psnr >= MIN_PSNR && m.share <= MAX_SHARE && m.alpha <= MAX_SHARE,
                "eye {k} against render_stereo's, E15's tier: {m:?}"
            );
            assert!(ours.alpha.iter().any(|&a| a > 0.1), "eye {k} draws");
        }
        assert_ne!(
            images[0].alpha, images[1].alpha,
            "each eye draws its own view"
        );

        // One prepare culling for both eyes: the union, counted once (§2.4, Q-399).
        scene.take_stats();
        scene.frames(SETTLE_FRAMES);
        let stats = scene.take_stats();
        assert!(!stats.is_empty(), "statistics two frames late");
        for s in &stats {
            assert_eq!(
                s.visible_splats, UNION,
                "the union of the eyes' {LEFT} and {RIGHT}, once (right spawned first: {right_first}): {s:?}"
            );
        }

        // Unchanged: stages 1–4 skipped for the group (§1.4, Q-100).
        if !scene.timestamps() {
            eprintln!("no TIMESTAMP_QUERY: the skip is not checked by the GPU times");
            continue;
        }
        scene.frames(8);
        scene.take_stats();
        scene.frames(SETTLE_FRAMES);
        let skipped = scene.take_stats();
        assert!(!skipped.is_empty(), "statistics of the unchanged frames");
        assert!(
            skipped
                .iter()
                .all(|s| (s.gpu_times_ms.cull, s.gpu_times_ms.sort) == (0.0, 0.0)),
            "an unchanged group skips stages 1–4: {skipped:?}"
        );
        // Only the second eye moves: the group prepares again.
        let right = scene.eyes[1];
        scene
            .app
            .world_mut()
            .get_mut::<Transform>(right)
            .expect("the right eye")
            .translation
            .x += 0.01;
        scene.frames(2 + SETTLE_FRAMES);
        let moved = scene.take_stats();
        assert!(
            moved.iter().any(|s| s.gpu_times_ms.cull > 0.0),
            "moving the second eye runs stages 1–4: {moved:?}"
        );
    }
}

/// Frames counted for the overflow events, after the settle.
const COUNTED_FRAMES: usize = 24;

/// Overflow events over `COUNTED_FRAMES` frames, at a capacity of 5 below every view's
/// visible count.
fn overflows_per_frames(sort: SortMode) -> usize {
    let mut scene = Scene::settled(sort, Capacity::Fixed(5), false);
    scene.frames(SETTLE_FRAMES);
    *scene.seen.overflows.lock().unwrap() = 0;
    scene.frames(COUNTED_FRAMES);
    let n = *scene.seen.overflows.lock().unwrap();
    eprintln!("{sort:?}: {n} overflow events in {COUNTED_FRAMES} frames");
    n
}

#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_shared_center_pair_copies_one_statistics_block_a_frame() {
    let _layer = layer_messages(); // R-16, Q-105
    // The group: one block a frame, so at most one event a frame (§1.4 stage 8, D-34).
    let shared = overflows_per_frames(SortMode::SharedCenter);
    assert!(
        (1..=COUNTED_FRAMES).contains(&shared),
        "one block a frame: {shared} in {COUNTED_FRAMES}"
    );
    // The control: two per-view cameras copy a block each.
    let per_view = overflows_per_frames(SortMode::PerView);
    assert!(
        per_view > COUNTED_FRAMES,
        "two blocks a frame per view: {per_view} in {COUNTED_FRAMES}"
    );
}
