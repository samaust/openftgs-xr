// SPDX-License-Identifier: Apache-2.0

//! #722's in-crate `gpu` test of the published statistics in per-view stereo (Q-403,
//! Q-125): `SplatStats` shows only what is published, so the test reads each unit's
//! block where the render world polls it. Two per-view cameras 1 m apart see different
//! rows (the left 5, the right 7); every published `FrameStats` holds the first camera's
//! counts and, for each GPU time, the sum of the two cameras' blocks of that poll.

use std::sync::{Arc, Mutex};

use ::bevy::MinimalPlugins;
use ::bevy::app::{App, Last, PluginsState};
use ::bevy::asset::{AssetPlugin, Assets, RenderAssetUsages};
use ::bevy::camera::{
    Camera, Camera3d, CameraPlugin, PerspectiveProjection, Projection, RenderTarget,
};
use ::bevy::core_pipeline::CorePipelinePlugin;
use ::bevy::ecs::prelude::*;
use ::bevy::image::{Image, ImagePlugin};
use ::bevy::math::Vec3;
use ::bevy::mesh::MeshPlugin;
use ::bevy::render::renderer::RenderDevice;
use ::bevy::render::settings::WgpuFeatures;
use ::bevy::render::{Render, RenderApp, RenderPlugin, RenderSystems};
use ::bevy::transform::TransformPlugin;
use ::bevy::transform::components::Transform;
use ::bevy::window::{ExitCondition, WindowPlugin};
use ftgsxr_io::ModelBytes;
use ftgsxr_testkit::{exchange_bytes, layer_messages, random_model};

use super::{Views, poll_stats};
use crate::bevy::{
    OutputSpace, SplatCamera, SplatEvent, SplatLoad, SplatPlugin, SplatStats, SplatTime,
};
use crate::gpu::{FrameStats, QualityTier, SortMode};

const W: u32 = 64;
const H: u32 = 48;
const F: f32 = 60.0;
const TAU: f64 = 0.7;
const MAX_FRAMES: usize = 600;

#[derive(Resource, Clone, Default)]
struct Seen {
    uploaded: Arc<Mutex<bool>>,
    published: Arc<Mutex<Vec<FrameStats>>>,
    /// Each frame's newest blocks of the two units, after the poll.
    polled: Arc<Mutex<Vec<(FrameStats, FrameStats)>>>,
}

/// An eye at (x, 0, 0) looking along +z with y down in the image.
fn eye(app: &mut App, x: f32, order: isize) {
    let mut image = Image::new_target_texture(
        W,
        H,
        ::bevy::render::render_resource::TextureFormat::Rgba16Float,
        None,
    );
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
        Transform::from_xyz(x, 0.0, 0.0).looking_to(Vec3::Z, Vec3::NEG_Y),
        SplatCamera {
            sort: SortMode::PerView,
            output: OutputSpace::FileEncoded,
        },
    ));
}

#[test]
#[ignore = "gpu"]
fn gpu_per_view_stereo_publishes_the_first_counts_and_the_summed_gpu_times() {
    let _layer = layer_messages(); // R-16, Q-105
    // 3 rows only the left eye sees (its u = 8, the right's −22), 5 only the right's, 2
    // both see, the rest behind the eyes.
    let mut model = random_model(0x7_22, 40, 0);
    for i in 0..model.len() {
        let (x, y, z) = match i {
            0..3 => (-1.3, 0.3 * (i as f32 - 1.0), 2.0),
            3..8 => (1.3, 0.15 * (i as f32 - 5.0), 2.0),
            8..10 => (0.0, 0.3 * (i as f32 - 8.5), 2.0),
            _ => (0.0, 0.0, -2.0),
        };
        model.position[i] = [x, y, z];
        model.time_center[i] = TAU;
        model.duration[i] = 10.0;
        model.velocity[i] = [0.0; 3];
        model.scale[i] = [0.05; 3];
        model.opacity[i] = 0.8;
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
        ::bevy::tasks::tick_global_task_pools_on_main_thread();
    }
    app.finish();
    app.cleanup();
    let seen = Seen::default();
    app.insert_resource(seen.clone());
    let polled = Arc::clone(&seen.polled);
    app.sub_app_mut(RenderApp).add_systems(
        Render,
        (move |views: Res<Views>| {
            if let [a, b] = views.units.as_slice()
                && let (Some(a), Some(b)) = (a.last_stats, b.last_stats)
            {
                polled.lock().unwrap().push((a, b));
            }
        })
        .in_set(RenderSystems::Cleanup)
        .after(poll_stats),
    );
    app.add_observer(|event: On<SplatEvent>, seen: Res<Seen>| {
        if let SplatEvent::Uploaded { .. } = event.event() {
            *seen.uploaded.lock().unwrap() = true;
        }
    });
    app.add_systems(Last, |stats: Res<SplatStats>, seen: Res<Seen>| {
        if stats.is_changed()
            && let Some(s) = stats.0
        {
            seen.published.lock().unwrap().push(s);
        }
    });
    // The right eye spawned first, so entity order cannot stand in for camera order.
    eye(&mut app, 0.5, 1);
    eye(&mut app, -0.5, 0);
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
    let mut uploaded_at = None;
    for frame in 0..MAX_FRAMES {
        app.update();
        if uploaded_at.is_none() && *seen.uploaded.lock().unwrap() {
            uploaded_at = Some(frame);
        }
        if uploaded_at.is_some_and(|f| frame >= f + 8) {
            break;
        }
    }
    assert!(uploaded_at.is_some(), "uploaded within {MAX_FRAMES} frames");
    seen.published.lock().unwrap().clear();
    for _ in 0..16 {
        app.update();
    }
    let timestamps = app
        .sub_app(RenderApp)
        .world()
        .resource::<RenderDevice>()
        .features()
        .contains(WgpuFeatures::TIMESTAMP_QUERY);
    let published = seen.published.lock().unwrap().clone();
    let polled = seen.polled.lock().unwrap().clone();
    assert!(
        published.len() >= 4,
        "statistics two frames late: {published:?}"
    );
    for p in &published {
        assert_eq!(
            p.visible_splats, 5,
            "the first (left) camera's count: {p:?}"
        );
        let source = polled.iter().find(|(a, b)| {
            let (ta, tb) = (a.gpu_times_ms, b.gpu_times_ms);
            *a == FrameStats {
                gpu_times_ms: a.gpu_times_ms,
                ..*p
            } && p.gpu_times_ms.cull == ta.cull + tb.cull
                && p.gpu_times_ms.sort == ta.sort + tb.sort
                && p.gpu_times_ms.draw == ta.draw + tb.draw
                && p.gpu_times_ms.total == ta.total + tb.total
        });
        let Some((a, b)) = source else {
            panic!(
                "{p:?} is the first block's counts with the summed times of no poll: {polled:?}"
            );
        };
        assert_eq!(
            b.visible_splats, 7,
            "the second (right) camera's block: {b:?}"
        );
        if timestamps {
            assert!(
                b.gpu_times_ms.total > 0.0 && p.gpu_times_ms.total > a.gpu_times_ms.total,
                "the second camera's time is in the sum: {p:?} {b:?}"
            );
        }
    }
}
