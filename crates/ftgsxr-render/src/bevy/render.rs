// SPDX-License-Identifier: Apache-2.0

//! The render world's part of the adapter: extraction of the instances and splat
//! cameras, one `FrameBuffers` per view, the two render nodes inside each splat camera's
//! 3D pass (§2.4 Frame loop hooks), the skip of stages 1–4 when nothing changed (§1.4,
//! §2.4 rule 2) and the statistics two frames late (§1.4 stage 8).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use ::bevy::app::SubApp;
use ::bevy::camera::visibility::InheritedVisibility;
use ::bevy::core_pipeline::core_3d::{main_opaque_pass_3d, main_transparent_pass_3d};
use ::bevy::core_pipeline::{Core3d, Core3dSystems};
use ::bevy::ecs::prelude::*;
use ::bevy::log::{error, warn};
use ::bevy::math::UVec2;
use ::bevy::render::camera::ExtractedCamera;
use ::bevy::render::render_resource::{StoreOp, TextureFormat};
use ::bevy::render::renderer::{RenderContext, ViewQuery};
use ::bevy::render::sync_world::RenderEntity;
use ::bevy::render::view::{ExtractedView, Msaa, ViewDepthTexture, ViewTarget};
use ::bevy::render::{Extract, ExtractSchedule, Render, RenderSystems};
use ::bevy::transform::components::GlobalTransform;
use ftgsxr_core::Pose;

use super::camera::view_camera;
use super::load::{GpuModels, SplatGpu};
use super::{
    OutputSpace, SplatCamera, SplatEvent, SplatMode, SplatModel, SplatOverlays, SplatStats,
    SplatTime, SplatTransform,
};
use crate::gpu::device::SplatDevice;
use crate::gpu::frame::TIMESTAMPS;
use crate::gpu::headless::sort_pose;
use crate::gpu::passes::ENGINE_NEAR;
use crate::gpu::{
    Capacity, ColorTransform, FrameBuffers, FrameStats, GpuModel, ModelDraw, Overlays, QualityTier,
    RenderMode, SortMode, SortPose, SplatPasses, ViewParams,
};

/// The layer's format (§1.4 stage 5, D-35): 16-bit float RGBA.
const LAYER_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// The view target the resolve is compiled for (Q-91).
const RESOLVE_TARGET: TextureFormat = TextureFormat::Rgba8UnormSrgb;
/// Bevy's 3D depth format (`bevy_core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT`).
const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

/// Statistics on their way from the render world to `SplatStats`, in arrival order.
#[derive(Resource, Clone, Default)]
pub(crate) struct StatsInbox(Arc<Mutex<Vec<FrameStats>>>);

impl StatsInbox {
    fn push(&self, stats: FrameStats) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(stats);
    }

    fn drain(&self) -> Vec<FrameStats> {
        std::mem::take(&mut *self.0.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

/// Main world: the statistics that arrived into `SplatStats` (the last one wins), and
/// `SplatEvent::Overflow` for each frame that dropped records (D-34).
pub(crate) fn publish_stats(
    inbox: Res<StatsInbox>,
    mut stats: ResMut<SplatStats>,
    mut commands: Commands,
) {
    for s in inbox.drain() {
        stats.0 = Some(s);
        if s.overflow {
            commands.trigger(SplatEvent::Overflow {
                visible: s.visible_splats + s.visible_centers,
                capacity: s.capacity,
            });
        }
    }
}

/// The plugin's tier, in the render world.
#[derive(Resource)]
struct RenderTier(QualityTier);

/// One instance as extracted: its model on the GPU and its frame inputs (§1.4 stage 1).
#[derive(Clone)]
struct Instance {
    model: Arc<GpuModel>,
    pose: Pose,
    tau: f64,
    mode: RenderMode,
    transform: ColorTransform,
    overlays: Overlays,
}

/// What the frame's extraction found: the device, the visible instances whose models are
/// on the GPU (in entity order), and the splat cameras by render entity.
#[derive(Resource, Default)]
struct Extracted {
    dev: Option<Arc<SplatDevice>>,
    instances: Vec<Instance>,
    cameras: HashMap<Entity, SplatCamera>,
}

/// What stages 1–4 depend on (§1.4): when it equals the previous frame's, they are
/// skipped and the previous frame's buffers are drawn again.
#[derive(Clone, PartialEq)]
struct PrepareKey {
    view: ViewParams,
    instances: Vec<(usize, Pose, u64, RenderMode, ColorTransform, u8)>,
}

/// One splat camera's state: its passes and frame buffers, the layer (`LinearOver`),
/// this frame's inputs and whether stages 1–4 run.
struct ViewState {
    camera: SplatCamera,
    passes: Arc<SplatPasses>,
    frame: FrameBuffers,
    layer: Option<(wgpu::TextureView, UVec2)>,
    view: ViewParams,
    sort_pose: SortPose,
    instances: Vec<Instance>,
    key: Option<PrepareKey>,
    prepare: bool,
}

/// The splat cameras' states by render entity, and the passes by (layer format, shared
/// center), made on the device they belong to.
#[derive(Resource, Default)]
struct Views {
    dev: Option<Arc<SplatDevice>>,
    passes: HashMap<(TextureFormat, bool), Arc<SplatPasses>>,
    states: HashMap<Entity, ViewState>,
    /// What was warned about once: (render entity, reason).
    warned: HashSet<(Entity, &'static str)>,
}

impl Views {
    fn warn_once(&mut self, entity: Entity, reason: &'static str) {
        if self.warned.insert((entity, reason)) {
            warn!("splat camera {entity}: {reason}");
        }
    }
}

/// The render app's part of `SplatPlugin::build`.
pub(crate) fn build(render_app: &mut SubApp, tier: QualityTier, inbox: StatsInbox) {
    render_app
        .insert_resource(RenderTier(tier))
        .insert_resource(inbox)
        .init_resource::<Extracted>()
        .init_resource::<Views>()
        .add_systems(ExtractSchedule, extract)
        .add_systems(
            Render,
            (
                prepare_views.in_set(RenderSystems::PrepareResources),
                poll_stats.in_set(RenderSystems::Cleanup),
            ),
        );
}

/// The render app's part of `SplatPlugin::finish`: the two nodes in `Core3d`, once every
/// plugin has built (the core pipeline's schedule exists then).
// Q-100
pub(crate) fn finish(render_app: &mut SubApp) {
    render_app.add_systems(
        Core3d,
        (
            prepare_node
                .in_set(Core3dSystems::MainPass)
                .before(main_opaque_pass_3d),
            draw_node
                .in_set(Core3dSystems::MainPass)
                .after(main_opaque_pass_3d)
                .before(main_transparent_pass_3d),
        ),
    );
}

/// P_m from the instance's global transform (E5): translation, rotation and the x scale
/// as the uniform s_m (Q-99).
fn pose_of(t: &GlobalTransform) -> Pose {
    let (scale, rotation, translation) = t.to_scale_rotation_translation();
    Pose {
        translation,
        rotation,
        scale: scale.x,
    }
}

type InstanceQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static SplatModel,
        &'static GlobalTransform,
        Option<&'static InheritedVisibility>,
        Option<&'static SplatTime>,
        Option<&'static SplatMode>,
        Option<&'static SplatTransform>,
        Option<&'static SplatOverlays>,
    ),
>;

fn extract(
    mut out: ResMut<Extracted>,
    gpu: Extract<Option<Res<SplatGpu>>>,
    models: Extract<Res<GpuModels>>,
    instances: Extract<InstanceQuery>,
    cameras: Extract<Query<(&RenderEntity, &SplatCamera)>>,
) {
    out.dev = gpu.as_ref().map(|g| Arc::clone(&g.0));
    let mut found: Vec<(Entity, Instance)> = instances
        .iter()
        .filter(|(.., vis, _, _, _, _)| vis.is_none_or(|v| v.get()))
        .filter_map(
            |(entity, model, transform, _, time, mode, color, overlays)| {
                let asset = model.0.id();
                let gpu_model = models.ready.get(&asset)?;
                Some((
                    entity,
                    Instance {
                        model: Arc::clone(gpu_model),
                        pose: pose_of(transform),
                        // Q-99: without a time, the model's τ₀.
                        tau: time.map_or(gpu_model.tau0, |t| t.tau),
                        mode: mode.map_or(RenderMode::Splats, |m| m.0),
                        transform: color.map_or(ColorTransform::IDENTITY, |c| c.0),
                        overlays: overlays.map_or(Overlays::default(), |o| o.0),
                    },
                ))
            },
        )
        .collect();
    found.sort_by_key(|(e, _)| *e);
    out.instances = found.into_iter().map(|(_, i)| i).collect();
    out.cameras = cameras.iter().map(|(r, c)| (r.id(), *c)).collect();
}

/// Each splat camera's passes, frame buffers, layer and inputs, and whether stages 1–4
/// run this frame.
fn prepare_views(
    extracted: Res<Extracted>,
    tier: Res<RenderTier>,
    mut views: ResMut<Views>,
    cameras: Query<(&ExtractedView, &ExtractedCamera, Option<&Msaa>)>,
) {
    let Some(dev) = extracted.dev.clone() else {
        return;
    };
    if views.dev.as_ref().is_none_or(|d| !Arc::ptr_eq(d, &dev)) {
        // A new device: everything made on the old one goes.
        *views = Views {
            dev: Some(Arc::clone(&dev)),
            ..Views::default()
        };
    }
    let live: HashSet<Entity> = extracted.cameras.keys().copied().collect();
    views.states.retain(|e, _| live.contains(e));
    for (&entity, &camera) in &extracted.cameras {
        let Ok((view, extracted_camera, msaa)) = cameras.get(entity) else {
            views.states.remove(&entity);
            continue;
        };
        match prepare_view(
            &mut views,
            &dev,
            &tier.0,
            &extracted.instances,
            entity,
            camera,
            view,
            extracted_camera,
            msaa,
        ) {
            Ok(()) => {}
            Err(reason) => {
                views.warn_once(entity, reason);
                views.states.remove(&entity);
            }
        }
    }
}

/// The capacity C of a view's frame buffers (§1.12, D-34): the tier's, or the drawn
/// instances' total N (Q-86).
fn capacity(tier: &QualityTier, instances: &[Instance]) -> u32 {
    match tier.capacity {
        Capacity::Fixed(c) => c,
        Capacity::SumOfModels => instances
            .iter()
            .map(|i| i.model.n())
            .fold(0u32, u32::saturating_add),
    }
}

/// Whether `FileEncoded` draws into a view target of `format`: exactly on 16-bit float,
/// rounded on another format (the warning), or not at all on sRGB, whose encode on store
/// and decode for blending would turn the file's values into linear light (the reason).
// Q-100
fn file_encoded_rule(format: TextureFormat) -> Result<Option<&'static str>, &'static str> {
    if format.is_srgb() {
        Err(
            "FileEncoded needs a target that is not sRGB; an sRGB target would encode the file's values (D-6, Q-100)",
        )
    } else if format != LAYER_FORMAT {
        Ok(Some(
            "FileEncoded on a target that is not 16-bit float rounds the layer",
        ))
    } else {
        Ok(None)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "one view's inputs, gathered by the system above"
)]
fn prepare_view(
    views: &mut Views,
    dev: &Arc<SplatDevice>,
    tier: &QualityTier,
    instances: &[Instance],
    entity: Entity,
    camera: SplatCamera,
    view: &ExtractedView,
    extracted_camera: &ExtractedCamera,
    msaa: Option<&Msaa>,
) -> Result<(), &'static str> {
    if msaa.is_some_and(|m| *m != Msaa::Off) {
        return Err("MSAA is on; splat cameras draw without it (D-6)");
    }
    let size = extracted_camera
        .physical_viewport_size
        .ok_or("no viewport size")?;
    let core_camera = view_camera(view.world_from_view.to_matrix(), view.clip_from_view, size)
        .ok_or("not a perspective projection")?;
    // Q-87, Q-100: the splats test depth at the engine's near plane.
    let near = view.clip_from_view.w_axis.z;
    if (near - ENGINE_NEAR).abs() > ENGINE_NEAR * 1e-3 {
        views.warn_once(
            entity,
            "the projection's near plane is not 0.01 m (§1.7); the splats' depth test assumes it",
        );
    }
    let layer_format = match camera.output {
        OutputSpace::FileEncoded => {
            if let Some(warning) = file_encoded_rule(view.target_format)? {
                views.warn_once(entity, warning);
            }
            view.target_format
        }
        OutputSpace::LinearOver { .. } => {
            if view.target_format != RESOLVE_TARGET {
                return Err("LinearOver needs an Rgba8UnormSrgb view target (Q-91)");
            }
            LAYER_FORMAT
        }
    };
    let mut view_tier = tier.clone();
    view_tier.sort_mode = camera.sort;
    let key = (layer_format, camera.sort == SortMode::SharedCenter);
    let passes = match views.passes.get(&key) {
        Some(p) => Arc::clone(p),
        None => {
            let p = SplatPasses::new(dev, layer_format, Some(DEPTH_FORMAT), &view_tier).map_err(
                |e| {
                    error!("splat passes: {}: {}", e.code(), e.user_text());
                    "the splat passes did not compile"
                },
            )?;
            let p = Arc::new(p);
            views.passes.insert(key, Arc::clone(&p));
            p
        }
    };
    let c = capacity(&view_tier, instances);
    let params = ViewParams {
        camera: core_camera,
        target_size: size,
    };
    let new_key = PrepareKey {
        view: params,
        instances: instances
            .iter()
            .map(|i| {
                (
                    Arc::as_ptr(&i.model) as usize,
                    i.pose,
                    i.tau.to_bits(),
                    i.mode,
                    i.transform,
                    i.model.sh_degree,
                )
            })
            .collect(),
    };
    let state = match views.states.remove(&entity) {
        Some(s)
            if Arc::ptr_eq(&s.passes, &passes) && s.camera == camera && s.frame.capacity() == c =>
        {
            s
        }
        _ => ViewState {
            camera,
            passes: Arc::clone(&passes),
            frame: FrameBuffers::new(dev, &view_tier, c),
            layer: None,
            view: params,
            sort_pose: sort_pose(&core_camera),
            instances: Vec::new(),
            key: None,
            prepare: true,
        },
    };
    let mut state = state;
    // §1.4, §2.4 rule 2, Q-100: stages 1–4 only when an input changed.
    // The key names GPU models by address: safe because `state.instances` still holds
    // the previous frame's `Arc`s here, so no address can have been reused yet.
    state.prepare = state.key.as_ref() != Some(&new_key);
    state.key = Some(new_key);
    state.view = params;
    state.sort_pose = sort_pose(&core_camera);
    state.instances = instances.to_vec();
    if matches!(camera.output, OutputSpace::LinearOver { .. })
        && state.layer.as_ref().is_none_or(|(_, s)| *s != size)
    {
        let texture = dev.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("ftgsxr splat layer"),
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LAYER_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        state.layer = Some((
            texture.create_view(&wgpu::TextureViewDescriptor::default()),
            size,
        ));
    }
    views.states.insert(entity, state);
    Ok(())
}

/// The instances of a view as `ModelDraw`s, at their loaded degrees.
fn draws(instances: &[Instance]) -> Vec<ModelDraw<'_>> {
    instances
        .iter()
        .map(|i| ModelDraw {
            model: &i.model,
            pose: i.pose,
            tau: i.tau,
            mode: i.mode,
            transform: i.transform,
            sh_degree: i.model.sh_degree,
            overlays: i.overlays,
        })
        .collect()
}

/// The prepare node (§2.4 Frame loop hooks): stages 1–4 for the view, before the opaque
/// pass, unless nothing changed since the last frame.
fn prepare_node(view: ViewQuery<&ExtractedView>, views: Res<Views>, mut ctx: RenderContext) {
    let Some(state) = views.states.get(&view.entity()) else {
        return;
    };
    if !state.prepare {
        return;
    }
    let draws = draws(&state.instances);
    state.passes.encode_prepare(
        ctx.command_encoder(),
        &state.frame,
        &draws,
        &state.sort_pose,
        &[state.view],
    );
}

/// The draw node (§2.4 Frame loop hooks): stages 5 and 6 after the opaque pass, tested
/// against its depth, then the resolve (`LinearOver`) or none (`FileEncoded`), then stage 8.
fn draw_node(
    view: ViewQuery<(&ExtractedCamera, &ViewTarget, &ViewDepthTexture)>,
    views: Res<Views>,
    mut ctx: RenderContext,
) {
    let Some(state) = views.states.get(&view.entity()) else {
        return;
    };
    let (camera, target, depth) = view.into_inner();
    let frame = &state.frame;
    // The draw's timestamps, indices 4 and 5 (Q-81, Q-87): this node owns the pass.
    let timestamp_writes =
        frame
            .set
            .queries
            .as_ref()
            .map(|(queries, _)| wgpu::RenderPassTimestampWrites {
                query_set: queries,
                beginning_of_pass_write_index: Some(4),
                end_of_pass_write_index: Some(5),
            });
    if timestamp_writes.is_some() {
        frame.ring.written.store(TIMESTAMPS, Ordering::Release);
    }
    let depth_attachment = Some(depth.get_attachment(StoreOp::Store));
    match (state.camera.output, &state.layer) {
        (OutputSpace::FileEncoded, _) => {
            let color = [Some(target.get_color_attachment())];
            let mut pass = ctx.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
                label: Some("ftgsxr splats"),
                color_attachments: &color,
                depth_stencil_attachment: depth_attachment,
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(viewport) = &camera.viewport {
                pass.set_camera_viewport(viewport);
            }
            state
                .passes
                .encode_draw(pass.wgpu_pass(), frame, &state.view, None);
        }
        (OutputSpace::LinearOver { background }, Some((layer, _))) => {
            {
                let color = [Some(wgpu::RenderPassColorAttachment {
                    view: layer,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })];
                let mut pass = ctx.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
                    label: Some("ftgsxr splats"),
                    color_attachments: &color,
                    depth_stencil_attachment: depth_attachment,
                    timestamp_writes,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                state
                    .passes
                    .encode_draw(pass.wgpu_pass(), frame, &state.view, None);
            }
            let color = [Some(target.get_color_attachment())];
            let mut pass = ctx.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
                label: Some("ftgsxr splat resolve"),
                color_attachments: &color,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(viewport) = &camera.viewport {
                pass.set_camera_viewport(viewport);
            }
            state
                .passes
                .encode_resolve(pass.wgpu_pass(), layer, background);
        }
        (OutputSpace::LinearOver { .. }, None) => return,
    }
    state.passes.encode_stats(ctx.command_encoder(), frame);
}

/// Stage 8's read back, never blocking (Q-81): each view's statistics of two frames ago,
/// once mapped, to the main world.
fn poll_stats(views: Res<Views>, inbox: Res<StatsInbox>) {
    let Some(dev) = &views.dev else {
        return;
    };
    let mut entities: Vec<&Entity> = views.states.keys().collect();
    entities.sort();
    for entity in entities {
        if let Some(stats) = FrameStats::poll(&views.states[entity].frame, dev) {
            inbox.push(stats);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_encoded_draws_unwarned_on_a_16_bit_float_target() {
        assert_eq!(file_encoded_rule(TextureFormat::Rgba16Float), Ok(None));
    }

    #[test]
    fn file_encoded_draws_warned_on_an_8_bit_linear_target() {
        assert!(matches!(
            file_encoded_rule(TextureFormat::Rgba8Unorm),
            Ok(Some(_))
        ));
    }

    #[test]
    fn file_encoded_is_refused_on_srgb_targets() {
        for format in [TextureFormat::Rgba8UnormSrgb, TextureFormat::Bgra8UnormSrgb] {
            assert!(file_encoded_rule(format).is_err(), "{format:?}");
        }
    }
}
