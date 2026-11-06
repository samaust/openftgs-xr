// SPDX-License-Identifier: Apache-2.0

//! PRD #565's adapter test of the occlusion path (#566; sources of truth: §2.1
//! `SplatOcclusion` "one per view, the runtime's left and right", PRD #565's "Which frame
//! an eye draws against", Q-125's camera order, Q-338's render side): the draw node takes
//! `SplatOcclusion.frames[i]` from the render world for the i-th live splat camera in
//! camera order and binds its image for E13, so a frame whose depth lies at its near plane
//! hides the probe; a camera without a frame, an empty `frames`, and a running tier whose
//! `occlusion` flag is false draw it unoccluded, as without the resource.

use std::sync::{Arc, Mutex};

use bevy::MinimalPlugins;
use bevy::app::{App, PluginsState};
use bevy::asset::{AssetPlugin, Assets, Handle, RenderAssetUsages};
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
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::{RenderApp, RenderPlugin};
use bevy::transform::TransformPlugin;
use bevy::transform::components::Transform;
use bevy::window::{ExitCondition, WindowPlugin};
use ftgsxr_core::{Camera as FtgsCamera, DMat3};
use ftgsxr_io::ModelBytes;
use ftgsxr_render::{
    DepthFrame, OutputSpace, QualityTier, SortMode, SplatCamera, SplatEvent, SplatLoad,
    SplatOcclusion, SplatPlugin, SplatTierState, SplatTime,
};
use ftgsxr_testkit::{fixtures_dir, layer_messages, probe};

const MAX_FRAMES: usize = 600;
const SETTLE_FRAMES: usize = 4;

#[derive(Resource)]
struct RenderErrored;

#[derive(Resource, Clone, Default)]
struct Seen {
    uploaded: Arc<Mutex<bool>>,
    failed: Arc<Mutex<Option<String>>>,
    /// The last read back of each camera's target, by camera order.
    readbacks: [Arc<Mutex<Option<Vec<u8>>>>; 2],
}

/// The probes' camera (openftgs's) as a Bevy transform and projection.
fn probe_camera(camera: &FtgsCamera) -> (Transform, Projection) {
    let r = DMat3::from_mat4(camera.world_to_camera);
    let t = camera.world_to_camera.w_axis.truncate();
    let center = -(r.transpose() * t);
    let flip = DMat3::from_diagonal(ftgsxr_core::DVec3::new(1.0, -1.0, -1.0));
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
    /// The camera of the probe's item, the views' camera in openftgs's conventions.
    camera: FtgsCamera,
    /// A 1 × 1 depth image holding 0: every surface at the frame's near plane.
    wall: Handle<Image>,
}

impl Drop for Scene {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            settle_pipelines(&mut self.app);
        }
    }
}

impl Scene {
    /// P1 at its first item's time, seen from that item's camera by two splat cameras of
    /// orders 0 and 1, each writing `FileEncoded` into its own `Rgba16Float` image, under
    /// the conformance tier with `occlusion: true` (Q-347).
    fn settled() -> Scene {
        let (_, request) = probe("P1");
        let item = &request.items[0];
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
                    occlusion: true,
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
                SplatEvent::Overflow { .. } => {}
            },
        );
        let (w, h) = (item.camera.width, item.camera.height);
        let (transform, projection) = probe_camera(&item.camera);
        for order in 0..2 {
            let mut image = Image::new_target_texture(w, h, TextureFormat::Rgba16Float, None);
            image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
            image.asset_usage = RenderAssetUsages::RENDER_WORLD;
            let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
            app.world_mut().spawn((
                Camera3d::default(),
                Camera {
                    order,
                    ..Camera::default()
                },
                RenderTarget::Image(target.clone().into()),
                projection.clone(),
                transform,
                SplatCamera {
                    sort: SortMode::PerView,
                    output: OutputSpace::FileEncoded,
                },
            ));
            let slot = order as usize;
            app.world_mut().spawn(Readback::texture(target)).observe(
                move |event: On<ReadbackComplete>, seen: Res<Seen>| {
                    *seen.readbacks[slot].lock().unwrap() = Some(event.data.clone());
                },
            );
        }
        let bytes = std::fs::read(fixtures_dir().join("probes/P1.safetensors")).expect("P1");
        app.world_mut().spawn((
            SplatLoad {
                bytes: ModelBytes::Owned(bytes),
                kind_hint: None,
                reduce_to: None,
            },
            SplatTime {
                tau: item.timestamp,
                extrapolating: false,
            },
            Transform::IDENTITY,
        ));
        let wall = Image::new(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            0f32.to_le_bytes().to_vec(),
            TextureFormat::R32Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        let wall = app.world_mut().resource_mut::<Assets<Image>>().add(wall);
        let mut scene = Scene {
            app,
            seen,
            camera: item.camera,
            wall,
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

    /// Both cameras' targets' alphas from a frame rendered after this call.
    fn alphas(&mut self) -> [Vec<f32>; 2] {
        for slot in &self.seen.readbacks {
            *slot.lock().unwrap() = None;
        }
        settle_pipelines(&mut self.app);
        self.frames(2 + SETTLE_FRAMES);
        let (w, h) = (self.camera.width as usize, self.camera.height as usize);
        std::array::from_fn(|i| {
            let bytes = self.seen.readbacks[i]
                .lock()
                .unwrap()
                .clone()
                .expect("a read back of the target");
            let row = bytes.len() / h;
            (0..h)
                .flat_map(|y| (0..w).map(move |x| y * row + 8 * x + 6))
                .map(|at| half(u16::from_le_bytes([bytes[at], bytes[at + 1]])))
                .collect()
        })
    }

    /// `SplatOcclusion` in the render world, with `frames` frames of the wall.
    fn occlusion(&mut self, frames: usize) {
        let frame = DepthFrame {
            image: self.wall.clone(),
            camera: self.camera,
            near: 0.01,
            far: 100.0,
            age_frames: 0,
        };
        self.app
            .sub_app_mut(RenderApp)
            .world_mut()
            .insert_resource(SplatOcclusion {
                frames: vec![frame; frames],
                soft: true,
                delta_m: 0.05,
            });
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

#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_occludes_the_ith_camera_with_frames_i_unless_none_or_the_tier_is_off() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut scene = Scene::settled();
    let [base0, base1] = scene.alphas();
    assert_eq!(base0, base1, "both cameras see the same image");
    assert!(
        base0.iter().filter(|&&a| a > 0.1).count() > 10,
        "the probe is drawn"
    );
    let hidden = |a: &[f32]| a.iter().all(|&x| x == 0.0);

    // One frame: the first camera (order 0) draws against it, the second has none.
    scene.occlusion(1);
    let [a0, a1] = scene.alphas();
    assert!(hidden(&a0), "frames[0] hides the probe from camera 0");
    assert_eq!(a1, base1, "camera 1 has no frame: unoccluded");

    // Two frames: each camera its own.
    scene.occlusion(2);
    let [a0, a1] = scene.alphas();
    assert!(
        hidden(&a0) && hidden(&a1),
        "frames[i] hides the probe from camera i"
    );

    // No frame (one that has not arrived yet): unoccluded.
    scene.occlusion(0);
    let [a0, a1] = scene.alphas();
    assert_eq!(
        (a0, a1),
        (base0.clone(), base1.clone()),
        "empty frames: unoccluded"
    );

    // The running tier's flag off (Q-338): unoccluded whatever the frames.
    scene.occlusion(2);
    *scene.app.world_mut().resource_mut::<SplatTierState>() =
        SplatTierState::of(&QualityTier::conformance());
    let [a0, a1] = scene.alphas();
    assert_eq!(
        (a0, a1),
        (base0, base1),
        "the tier's occlusion off: unoccluded"
    );
}
