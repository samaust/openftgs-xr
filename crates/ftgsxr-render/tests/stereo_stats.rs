// SPDX-License-Identifier: Apache-2.0

//! #220's detecting test of Q-125 (E6, §1.4 Stereo and stage 8, D-4, Q-118): in per-view
//! stereo each eye runs its own statistics block, and `SplatStats` publishes the first
//! eye's (the splat camera of lowest order), so twelve non-finite rows count 12, not 24,
//! nor the other eye's count.
//!
//! The loader refuses non-finite values (§1.2), so the twelve rows are finite in the file
//! and overflow float32 at projection, which Q-118 counts as non-finite (#219): six at
//! depth 1 and six at depth 0.02. Eye 1 (order 1) stands 5 cm ahead of eye 0 (order 0),
//! so the near cull drops the six close rows for it and its block counts 6. The test
//! spawns the eyes in both orders, so whichever block arrives last, only eye 0's may be
//! published.

use bevy::MinimalPlugins;
use bevy::app::{App, Last, PluginsState};
use bevy::asset::{AssetPlugin, Assets, RenderAssetUsages};
use bevy::camera::{
    Camera, Camera3d, CameraPlugin, PerspectiveProjection, Projection, RenderTarget,
};
use bevy::core_pipeline::CorePipelinePlugin;
use bevy::ecs::prelude::*;
use bevy::image::{Image, ImagePlugin};
use bevy::math::Vec3;
use bevy::mesh::MeshPlugin;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::TextureFormat;
use bevy::transform::TransformPlugin;
use bevy::transform::components::Transform;
use bevy::window::{ExitCondition, WindowPlugin};
use ftgsxr_io::ModelBytes;
use ftgsxr_render::{
    FrameStats, OutputSpace, QualityTier, SortMode, SplatCamera, SplatEvent, SplatLoad,
    SplatPlugin, SplatStats, SplatTime,
};
use ftgsxr_testkit::{exchange_bytes, layer_messages, random_model};
use std::sync::{Arc, Mutex};

const W: u32 = 320;
const H: u32 = 240;
const F: f32 = 300.0;
const TAU: f64 = 0.7;
const MAX_FRAMES: usize = 600;

#[derive(Resource, Clone, Default)]
struct Seen {
    stats: Arc<Mutex<Vec<FrameStats>>>,
    events: Arc<Mutex<Vec<SplatEvent>>>,
}

/// An eye at `z` on the world's z axis looking along +z with y down in the image
/// (openftgs's camera, identity rotation): Bevy's view looks along −z with y up, so the
/// view is turned π about x.
fn eye(app: &mut App, z: f32, order: isize) {
    let image = Image::new_target_texture(W, H, TextureFormat::Rgba8UnormSrgb, None);
    let mut image = image;
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    app.world_mut().spawn((
        Camera3d::default(),
        Camera {
            order,
            ..Camera::default()
        },
        RenderTarget::Image(target.into()),
        Projection::Perspective(PerspectiveProjection {
            fov: 2.0 * (H as f32 / 2.0 / F).atan(),
            aspect_ratio: W as f32 / H as f32,
            near: 0.01,
            ..PerspectiveProjection::default()
        }),
        Transform::from_xyz(0.0, 0.0, z).looking_to(Vec3::Z, Vec3::NEG_Y),
        SplatCamera {
            sort: SortMode::PerView,
            output: OutputSpace::LinearOver { background: None },
        },
    ));
}

#[test]
#[ignore = "gpu"]
fn gpu_q125_per_view_stereo_counts_twelve_non_finite_rows_once() {
    let _layer = layer_messages(); // R-16, Q-105
    for eye_1_first in [false, true] {
        stereo_counts_twelve(eye_1_first);
    }
}

fn stereo_counts_twelve(eye_1_first: bool) {
    let mut model = random_model(0x2_20, 2_000, 0);
    let rows: [[f32; 3]; 12] = [
        [1e37, 0.0, 1.0],
        [-1e37, 0.0, 1.0],
        [0.0, 1e37, 1.0],
        [0.0, -1e37, 1.0],
        [1e37, 1e37, 1.0],
        [-1e37, -1e37, 1.0],
        [1e37, 0.0, 0.02],
        [-1e37, 0.0, 0.02],
        [0.0, 1e37, 0.02],
        [0.0, -1e37, 0.02],
        [1e37, 1e37, 0.02],
        [-1e37, -1e37, 0.02],
    ];
    for (i, p) in rows.iter().enumerate() {
        model.time_center[i] = TAU;
        model.duration[i] = 10.0;
        model.velocity[i] = [0.0; 3];
        model.scale[i] = [0.05; 3];
        model.opacity[i] = 0.9;
        model.position[i] = *p;
    }

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
    let seen = Seen::default();
    app.insert_resource(seen.clone());
    app.add_observer(|event: On<SplatEvent>, seen: Res<Seen>| {
        seen.events.lock().unwrap().push(event.event().clone());
    });
    app.add_systems(Last, |stats: Res<SplatStats>, seen: Res<Seen>| {
        if stats.is_changed()
            && let Some(s) = stats.0
        {
            seen.stats.lock().unwrap().push(s);
        }
    });
    // Both spawn orders, so neither entity order can pass for the rule.
    if eye_1_first {
        eye(&mut app, 0.05, 1);
        eye(&mut app, 0.0, 0);
    } else {
        eye(&mut app, 0.0, 0);
        eye(&mut app, 0.05, 1);
    }
    app.world_mut().spawn((
        SplatLoad {
            bytes: ModelBytes::Owned(exchange_bytes(&model)),
            kind_hint: None,
            reduce_to: None,
        },
        SplatTime {
            tau: TAU,
            extrapolating: false,
        },
        Transform::IDENTITY,
    ));

    let mut uploaded = None;
    for frame in 0..MAX_FRAMES {
        app.update();
        let events = seen.events.lock().unwrap().clone();
        if let Some(SplatEvent::Failed { error, .. }) = events
            .iter()
            .find(|e| matches!(e, SplatEvent::Failed { .. }))
        {
            panic!("the load failed: {error}");
        }
        if uploaded.is_none()
            && events
                .iter()
                .any(|e| matches!(e, SplatEvent::Uploaded { .. }))
        {
            uploaded = Some(frame);
            seen.stats.lock().unwrap().clear();
        }
        if uploaded.is_some_and(|f| frame >= f + 12) {
            break;
        }
    }
    assert!(uploaded.is_some(), "the model uploaded");
    let stats = seen.stats.lock().unwrap().clone();
    // The first blocks after the upload may predate it (two frames late).
    let drawn: Vec<_> = stats.iter().filter(|s| s.visible_splats > 0).collect();
    assert!(drawn.len() >= 4, "statistics of drawn frames: {stats:?}");
    for s in drawn {
        assert_eq!(
            s.non_finite, 12,
            "the first eye's count, once per row (eye 1 spawned first: {eye_1_first}): {s:?}"
        );
    }
}
