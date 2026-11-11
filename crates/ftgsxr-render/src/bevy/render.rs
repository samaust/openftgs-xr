// SPDX-License-Identifier: Apache-2.0

//! The render world's part of the adapter: extraction of the instances and splat
//! cameras, one `FrameBuffers` per prepare unit (a per-view camera, or the shared-center
//! group, D-4), the two render nodes inside each splat camera's 3D pass (§2.4 Frame loop
//! hooks), the skip of stages 1–4 when nothing changed (§1.4, §2.4 rule 2) and the
//! statistics two frames late (§1.4 stage 8).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};

use ::bevy::app::SubApp;
use ::bevy::camera::visibility::InheritedVisibility;
// The depth the splats test against is Bevy's 3D depth (#204).
use ::bevy::core_pipeline::core_3d::{
    CORE_3D_DEPTH_FORMAT, main_opaque_pass_3d, main_transparent_pass_3d,
};
use ::bevy::core_pipeline::{Core3d, Core3dSystems};
use ::bevy::ecs::prelude::*;
use ::bevy::log::{error, warn};
use ::bevy::math::UVec2;
use ::bevy::render::camera::ExtractedCamera;
use ::bevy::render::render_asset::RenderAssets;
use ::bevy::render::render_resource::{StoreOp, TextureFormat};
use ::bevy::render::renderer::{RenderContext, ViewQuery};
use ::bevy::render::storage::GpuShaderBuffer;
use ::bevy::render::sync_world::RenderEntity;
use ::bevy::render::texture::GpuImage;
use ::bevy::render::view::{ExtractedView, Msaa, ViewDepthTexture, ViewTarget};
use ::bevy::render::{Extract, ExtractSchedule, Render, RenderSystems};
use ::bevy::transform::components::GlobalTransform;
use ftgsxr_core::{Pose, Trim};

use super::camera::view_camera;
use super::depth_copy::DepthCopy;
use super::load::{GpuModels, SplatGpu};
use super::probe::{ProbeTarget, extract_target};
use super::{
    DepthFrame, OutputSpace, SplatCamera, SplatEvent, SplatMode, SplatModel, SplatOcclusion,
    SplatOverlays, SplatShLimit, SplatStats, SplatTierState, SplatTime, SplatTransform, SplatTrim,
};
use crate::gpu::device::SplatDevice;
use crate::gpu::headless::{sort_pose, stereo_sort_pose};
use crate::gpu::probe::ProbePass;
// The view target the resolve is compiled for (Q-91).
use crate::gpu::passes::{ENGINE_NEAR, RESOLVE_TARGET_FORMAT};
use crate::gpu::{
    Capacity, ColorTransform, FrameBuffers, FrameStats, GpuModel, ModelDraw, OcclusionInput,
    Overlays, QualityTier, RenderError, RenderMode, SortMode, SortPose, SplatPasses, ViewParams,
};

/// The layer's format (§1.4 stage 5, D-35): 16-bit float RGBA.
const LAYER_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// What the render world sends the main world's `SplatStats` and observers.
#[derive(Clone, Copy, Debug)]
enum StatsMessage {
    /// The statistics to publish (`published_stats`).
    Publish(FrameStats),
    /// A unit's frame dropped records (D-34).
    Overflow { visible: u32, capacity: u32 },
}

/// Statistics on their way from the render world to `SplatStats`, in arrival order.
#[derive(Resource, Clone, Default)]
pub(crate) struct StatsInbox(Arc<Mutex<Vec<StatsMessage>>>);

impl StatsInbox {
    fn push(&self, message: StatsMessage) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(message);
    }

    fn drain(&self) -> Vec<StatsMessage> {
        std::mem::take(&mut *self.0.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

/// Main world: the published statistics into `SplatStats` (the first unit's counts, a
/// stereo pair counting once, Q-125; the units' GPU times summed, Q-403), and
/// `SplatEvent::Overflow` for each unit's frame that dropped records (D-34).
pub(crate) fn publish_stats(
    inbox: Res<StatsInbox>,
    mut stats: ResMut<SplatStats>,
    mut commands: Commands,
) {
    for message in inbox.drain() {
        match message {
            // Q-125, Q-403
            StatsMessage::Publish(s) => stats.0 = Some(s),
            StatsMessage::Overflow { visible, capacity } => {
                commands.trigger(SplatEvent::Overflow { visible, capacity });
            }
        }
    }
}

/// The running tier, in the render world: the plugin's at first, then `SplatTierState.tier`
/// as extracted (Q-239).
#[derive(Resource)]
struct RenderTier(QualityTier);

/// One instance as extracted: its model on the GPU and its frame inputs (§1.4 stage 1).
#[derive(Clone)]
struct Instance {
    model: Arc<GpuModel>,
    /// The degree drawn: the smaller of the loaded degree and the governor's cap (E16,
    /// §1.2 load option 1), so a cap change acts without a reload.
    sh_degree: u8,
    pose: Pose,
    tau: f64,
    mode: RenderMode,
    transform: ColorTransform,
    overlays: Overlays,
    /// E17's trim; off without a `SplatTrim`.
    trim: Trim,
}

/// The warm-up threads by passes key (Q-325), each holding the device until it ends.
/// Dropping them joins every thread still running (Q-381): with the render world, or at a
/// new device or tier (Q-239), so no thread outlives the state that started it and
/// releases the device on its own, during the process's exit (#650).
#[derive(Default)]
struct Warming(HashMap<(TextureFormat, bool), JoinHandle<Result<SplatPasses, RenderError>>>);

impl Warming {
    fn insert(
        &mut self,
        key: (TextureFormat, bool),
        handle: JoinHandle<Result<SplatPasses, RenderError>>,
    ) {
        self.0.insert(key, handle);
    }

    /// The passes `key`'s thread compiled, waiting for it if it is not done; none when no
    /// thread compiles `key` or its thread panicked.
    fn take(&mut self, key: &(TextureFormat, bool)) -> Option<Result<SplatPasses, RenderError>> {
        self.0.remove(key).and_then(|h| h.join().ok())
    }
}

// Q-381
impl Drop for Warming {
    fn drop(&mut self) {
        for (_, handle) in self.0.drain() {
            // The passes go with the thread; a panic in it was its own.
            let _ = handle.join();
        }
    }
}

/// What the frame's extraction found: the device, the visible instances whose models are
/// on the GPU (in entity order), and the splat cameras by render entity.
#[derive(Resource, Default)]
struct Extracted {
    dev: Option<Arc<SplatDevice>>,
    instances: Vec<Instance>,
    cameras: HashMap<Entity, SplatCamera>,
    /// Each splat camera's order, by render entity.
    orders: HashMap<Entity, isize>,
    /// The governor's render scale (§1.12, E16).
    render_scale: f32,
    /// The running tier (Q-239).
    tier: Option<QualityTier>,
}

/// What stages 1–4 depend on (§1.4): when it equals the previous frame's, they are
/// skipped and the previous frame's buffers are drawn again. A unit's key holds every
/// member's view, so the skip applies to a shared-center group as a whole (Q-100).
#[derive(Clone, PartialEq)]
struct PrepareKey {
    views: Vec<ViewParams>,
    instances: Vec<(usize, Pose, u64, RenderMode, ColorTransform, u8, Trim)>,
}

/// One splat camera's state: its passes, the layer (`LinearOver`), its view and its place
/// in the frame's prepare units.
struct ViewState {
    camera: SplatCamera,
    passes: Arc<SplatPasses>,
    layer: Option<(wgpu::TextureView, UVec2)>,
    /// The view's depth copied at the layer's size, below render scale 1 (Q-164).
    layer_depth: Option<(wgpu::TextureView, UVec2)>,
    view: ViewParams,
    /// The view's E6 camera, for the unit's sort pose.
    core_camera: ftgsxr_core::Camera,
    /// The view's place among the live splat cameras in camera order (Q-125): the index
    /// of its frame in `SplatOcclusion.frames`.
    index: usize,
    /// The view's unit in `Views.units`.
    unit: usize,
}

/// One prepare unit (Q-399): a per-view camera, or the shared-center group, whose first
/// prepare node to run in the frame runs stages 1–4 for every member, at the members'
/// sort pose with each member a cull view, into the unit's one set of frame buffers, which
/// every member then draws. Its statistics are one block.
struct Unit {
    /// The members' render entities, in camera order.
    members: Vec<Entity>,
    /// The passes of stages 1–4: the first member's.
    passes: Arc<SplatPasses>,
    frame: FrameBuffers,
    /// The members' views, in camera order: the cull views.
    views: Vec<ViewParams>,
    sort_pose: SortPose,
    instances: Vec<Instance>,
    key: Option<PrepareKey>,
    /// Whether stages 1–4 run this frame.
    prepare: bool,
    /// The newest statistics block polled (Q-T722.2).
    last_stats: Option<FrameStats>,
}

/// The render-world resource of §2.4 Frame loop hooks: the frame's number, advanced at
/// extraction, and per unit the frame whose shared prepare ran and the frame and count of
/// its members' draws so far, so the members after the first only draw, the first draw
/// writes the draw's begin timestamp and the last its end (Q-403) and copies the unit's
/// statistics block once. A value recorded for an earlier frame number counts as unset,
/// which resets it at extraction.
#[derive(Resource, Default)]
struct SharedFrame {
    frame: u64,
    prepared: Vec<u64>,
    drawn: Vec<(u64, usize)>,
}

impl SharedFrame {
    /// Whether unit `u` is still to prepare this frame; marks it prepared.
    fn take_prepare(&mut self, u: usize) -> bool {
        let frame = self.frame;
        let Some(prepared) = self.prepared.get_mut(u) else {
            return false;
        };
        let first = *prepared != frame;
        *prepared = frame;
        first
    }

    /// How many of unit `u`'s members drew before this one this frame; counts this one.
    fn take_draw(&mut self, u: usize) -> usize {
        let frame = self.frame;
        let Some(drawn) = self.drawn.get_mut(u) else {
            return 0;
        };
        if drawn.0 != frame {
            *drawn = (frame, 0);
        }
        drawn.1 += 1;
        drawn.1 - 1
    }

    /// Room for `units` units, keeping this frame's marks.
    fn resize(&mut self, units: usize) {
        self.prepared.resize(units, u64::MAX);
        self.drawn.resize(units, (u64::MAX, 0));
    }
}

/// The splat cameras' states by render entity, and the passes by (layer format, shared
/// center), made on the device they belong to.
#[derive(Resource, Default)]
struct Views {
    dev: Option<Arc<SplatDevice>>,
    passes: HashMap<(TextureFormat, bool), Arc<SplatPasses>>,
    /// The `LinearOver` passes compiling ahead on their own threads, by the same key
    /// (Q-325).
    warming: Warming,
    /// Whether this device's and tier's warm-up has started (Q-325).
    warm_started: bool,
    /// The layer probe's pass (E14's exposure match), compiled with the device's views.
    probe: Option<ProbePass>,
    /// The depth copy's pipeline (Q-164), made on first use.
    depth_copy: Option<Arc<DepthCopy>>,
    states: HashMap<Entity, ViewState>,
    /// The frame's prepare units, in their first members' camera order (Q-399).
    units: Vec<Unit>,
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
        .init_resource::<SharedFrame>()
        .init_resource::<ProbeTarget>()
        .add_systems(ExtractSchedule, (extract, extract_target))
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

/// The degree an instance is drawn at: the smallest of its loaded degree, the governor's
/// cap (E16) and a replayed request's limit (Q-235), `None` adding no cap.
// Q-233
fn drawn_degree(loaded: u8, governor_cap: u8, limit: Option<u8>) -> u8 {
    loaded.min(governor_cap).min(limit.unwrap_or(u8::MAX))
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
        Option<&'static SplatTrim>,
    ),
>;

#[allow(clippy::too_many_arguments)]
fn extract(
    mut out: ResMut<Extracted>,
    mut shared: ResMut<SharedFrame>,
    gpu: Extract<Option<Res<SplatGpu>>>,
    models: Extract<Res<GpuModels>>,
    instances: Extract<InstanceQuery>,
    tier_state: Extract<Res<SplatTierState>>,
    sh_limit: Extract<Option<Res<SplatShLimit>>>,
    cameras: Extract<Query<(&RenderEntity, &SplatCamera, Option<&::bevy::camera::Camera>)>>,
) {
    // §2.4 Frame loop hooks: a new frame, so no unit has prepared or drawn in it yet.
    shared.frame = shared.frame.wrapping_add(1);
    out.dev = gpu.as_ref().map(|g| Arc::clone(&g.0));
    // Q-239: the running tier; the device made for it resets the views.
    out.tier = Some(tier_state.tier.clone());
    let sh_cap = tier_state.sh_cap;
    let sh_limit = sh_limit.as_deref().and_then(|l| l.0);
    out.render_scale = tier_state.render_scale;
    let mut found: Vec<(Entity, Instance)> = instances
        .iter()
        .filter(|(.., vis, _, _, _, _, _)| vis.is_none_or(|v| v.get()))
        .filter_map(
            |(entity, model, transform, _, time, mode, color, overlays, trim)| {
                let asset = model.0.id();
                let gpu_model = models.ready.get(&asset)?;
                Some((
                    entity,
                    Instance {
                        model: Arc::clone(gpu_model),
                        sh_degree: drawn_degree(gpu_model.sh_degree, sh_cap, sh_limit),
                        pose: pose_of(transform),
                        // Q-99: without a time, the model's τ₀.
                        tau: time.map_or(gpu_model.tau0, |t| t.tau),
                        mode: mode.map_or(RenderMode::Splats, |m| m.0),
                        transform: color.map_or(ColorTransform::IDENTITY, |c| c.0),
                        overlays: overlays.map_or(Overlays::default(), |o| o.0),
                        trim: trim.map_or(Trim::default(), |t| t.0),
                    },
                ))
            },
        )
        .collect();
    found.sort_by_key(|(e, _)| *e);
    out.instances = found.into_iter().map(|(_, i)| i).collect();
    out.cameras = cameras.iter().map(|(r, c, _)| (r.id(), *c)).collect();
    out.orders = cameras
        .iter()
        .map(|(r, _, cam)| (r.id(), cam.map_or(0, |c| c.order)))
        .collect();
}

/// Each splat camera's passes, frame buffers, layer and inputs, and whether stages 1–4
/// run this frame.
fn prepare_views(
    extracted: Res<Extracted>,
    mut tier: ResMut<RenderTier>,
    mut views: ResMut<Views>,
    mut shared: ResMut<SharedFrame>,
    cameras: Query<(&ExtractedView, &ExtractedCamera, Option<&Msaa>)>,
) {
    let Some(dev) = extracted.dev.clone() else {
        return;
    };
    // Q-239: a new tier takes new passes and frame buffers.
    let new_tier = match &extracted.tier {
        Some(t) if *t != tier.0 => {
            tier.0 = t.clone();
            true
        }
        _ => false,
    };
    if new_tier || views.dev.as_ref().is_none_or(|d| !Arc::ptr_eq(d, &dev)) {
        // A new device: everything made on the old one goes.
        // The layer probe is compiled on every device, requested or not (E14).
        let probe = ProbePass::new(dev.device())
            .inspect_err(|e| error!("the layer probe does not compile: {e}"))
            .ok();
        *views = Views {
            dev: Some(Arc::clone(&dev)),
            probe,
            ..Views::default()
        };
    }
    if !views.warm_started {
        views.warm_started = true;
        warm_up(&mut views, &dev, &tier.0);
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
            extracted.render_scale,
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
    let ordered = camera_order(views.states.keys().copied(), &extracted.orders);
    for (index, entity) in ordered.iter().enumerate() {
        if let Some(state) = views.states.get_mut(entity) {
            state.index = index;
        }
    }
    let sorts: Vec<(Entity, SortMode)> = ordered
        .iter()
        .map(|e| (*e, views.states[e].camera.sort))
        .collect();
    let units = prepare_units(&sorts);
    for members in units.iter().filter(|m| m.len() == 1) {
        if views.states[&members[0]].camera.sort == SortMode::SharedCenter
            && units.iter().any(|m| m.len() == 2)
        {
            // Q-T722.1
            views.warn_once(
                members[0],
                "a third shared-center camera prepares alone; a group holds two (§1.4 Stereo)",
            );
        }
    }
    let mut old = std::mem::take(&mut views.units);
    let mut made = Vec::with_capacity(units.len());
    for (u, members) in units.into_iter().enumerate() {
        made.push(prepare_unit(
            &mut views,
            &mut old,
            &dev,
            &tier.0,
            &extracted.instances,
            members,
            u,
        ));
    }
    views.units = made;
    shared.resize(views.units.len());
}

/// Unit `u`'s frame buffers, cull views, sort pose and inputs, and whether stages 1–4
/// run: the frame buffers are those of the previous frame's unit of the same members while
/// its first member's passes and the capacity hold (§2.4 rule 4), and stages 1–4 run when
/// any member's view or any instance changed (§1.4, Q-100).
fn prepare_unit(
    views: &mut Views,
    old: &mut Vec<Unit>,
    dev: &Arc<SplatDevice>,
    tier: &QualityTier,
    instances: &[Instance],
    members: Vec<Entity>,
    u: usize,
) -> Unit {
    for m in &members {
        if let Some(state) = views.states.get_mut(m) {
            state.unit = u;
        }
    }
    let first = &views.states[&members[0]];
    let passes = Arc::clone(&first.passes);
    let mut unit_tier = tier.clone();
    unit_tier.sort_mode = first.camera.sort;
    let c = capacity(&unit_tier, instances);
    let member_views: Vec<ViewParams> = members.iter().map(|m| views.states[m].view).collect();
    let cameras: Vec<ftgsxr_core::Camera> = members
        .iter()
        .map(|m| views.states[m].core_camera)
        .collect();
    // Q-399: a group's sort pose is the mean of the eyes' centers along the normalized sum
    // of their forward axes; a unit of one prepares at its own pose.
    let sort_pose = match cameras.as_slice() {
        [one] => sort_pose(one),
        eyes => stereo_sort_pose(eyes),
    };
    let new_key = PrepareKey {
        views: member_views.clone(),
        instances: instances
            .iter()
            .map(|i| {
                (
                    Arc::as_ptr(&i.model) as usize,
                    i.pose,
                    i.tau.to_bits(),
                    i.mode,
                    i.transform,
                    i.sh_degree,
                    i.trim,
                )
            })
            .collect(),
    };
    // §2.4 rule 4: the frame buffers are sized once per passes (layer format and sort
    // mode) and capacity, so a change of the output alone (a `LinearOver` background)
    // keeps them, and the key below decides stages 1–4 (#201).
    let reused = old
        .iter()
        .position(|o| {
            o.members == members && Arc::ptr_eq(&o.passes, &passes) && o.frame.capacity() == c
        })
        .map(|i| old.swap_remove(i));
    let mut unit = reused.unwrap_or_else(|| Unit {
        members: members.clone(),
        passes: Arc::clone(&passes),
        frame: FrameBuffers::new(dev, &unit_tier, c),
        views: Vec::new(),
        sort_pose,
        instances: Vec::new(),
        key: None,
        prepare: true,
        last_stats: None,
    });
    // §1.4, §2.4 rule 2, Q-100: stages 1–4 only when an input changed.
    // The key names GPU models by address: safe because `unit.instances` still holds
    // the previous frame's `Arc`s here, so no address can have been reused yet.
    unit.prepare = unit.key.as_ref() != Some(&new_key);
    unit.key = Some(new_key);
    unit.views = member_views;
    unit.sort_pose = sort_pose;
    unit.instances = instances.to_vec();
    unit
}

/// The live splat cameras in camera order: by `Camera.order`, ties to the lower render
/// entity (Q-125; `ftgsxr_xr` orders the eyes by index, so the left eye comes first).
fn camera_order(
    entities: impl Iterator<Item = Entity>,
    orders: &HashMap<Entity, isize>,
) -> Vec<Entity> {
    let mut entities: Vec<Entity> = entities.collect();
    entities.sort_by_key(|e| (orders.get(e).copied().unwrap_or(0), *e));
    entities
}

/// The frame's prepare units, from the live splat cameras in camera order with their sort
/// modes: each per-view camera is a unit of its own, and the shared-center cameras form
/// one group (Q-399), at most two, since a prepare culls for one or two views (§1.4
/// Stereo); each further shared-center camera prepares alone at its own pose (Q-T722.1).
// Q-399, Q-T722.1
fn prepare_units(ordered: &[(Entity, SortMode)]) -> Vec<Vec<Entity>> {
    let mut units: Vec<Vec<Entity>> = Vec::new();
    let mut group: Option<usize> = None;
    for &(entity, sort) in ordered {
        match (sort, group) {
            (SortMode::SharedCenter, Some(g)) if units[g].len() < 2 => units[g].push(entity),
            (SortMode::SharedCenter, None) => {
                group = Some(units.len());
                units.push(vec![entity]);
            }
            _ => units.push(vec![entity]),
        }
    }
    units
}

/// What `SplatStats` publishes from the first unit's block `first` and the latest block
/// of each other unit of the frame, `others` (Q-403, Q-T722.2): the first unit's counts,
/// C and flags (Q-125: a primitive counts once), and the GPU times summed over the units,
/// a shared-center group's one block counted once.
// Q-125, Q-403, Q-T722.2
fn published_stats(first: FrameStats, others: &[FrameStats]) -> FrameStats {
    let mut published = first;
    for other in others {
        let (t, o) = (&mut published.gpu_times_ms, &other.gpu_times_ms);
        t.cull += o.cull;
        t.sort += o.sort;
        t.draw += o.draw;
        t.total += o.total;
    }
    published
}

/// The depth frame the view at `index` in camera order draws against (PRD #565, "Which
/// frame an eye draws against"): `frames[index]`, or none when the running tier's
/// `occlusion` flag is false (Q-338) or no frame has arrived for the view.
// Q-338
fn view_frame(tier_occlusion: bool, frames: &[DepthFrame], index: usize) -> Option<&DepthFrame> {
    if tier_occlusion {
        frames.get(index)
    } else {
        None
    }
}

/// The passes cache's key: the layer format and whether the sort is shared center.
fn pass_key(layer_format: TextureFormat, sort: SortMode) -> (TextureFormat, bool) {
    (layer_format, sort == SortMode::SharedCenter)
}

/// A view's passes: `tier` with the camera's sort mode, at the layer format, over the
/// core pipeline's depth.
fn compile(
    dev: &SplatDevice,
    layer_format: TextureFormat,
    tier: &QualityTier,
    sort: SortMode,
) -> Result<SplatPasses, RenderError> {
    let mut view_tier = tier.clone();
    view_tier.sort_mode = sort;
    SplatPasses::new(dev, layer_format, Some(CORE_3D_DEPTH_FORMAT), &view_tier)
}

/// Starts compiling the `LinearOver` passes of both sort modes, each on a thread of its
/// own rather than a shared pool a model load could hold (Q-325): the 0.3 s of pipeline
/// compiles on the Quest 3 then run while the session starts, not in the first frame a
/// splat camera draws, nor in the frame the governor first switches the sort mode.
/// `prepare_view` takes a key's result, waiting for it if it is not ready, so the first
/// frame still draws; a thread that cannot start leaves its key to compile there. A
/// thread not taken by then is joined when `Views` drops or resets (Q-381).
// Q-325, Q-381
fn warm_up(views: &mut Views, dev: &Arc<SplatDevice>, tier: &QualityTier) {
    for sort in [SortMode::PerView, SortMode::SharedCenter] {
        let dev = Arc::clone(dev);
        let tier = tier.clone();
        // `warm_up_teardown`'s test finds the threads by this name.
        let spawned = thread::Builder::new()
            .name("ftgsxr splat passes".into())
            .spawn(move || compile(&dev, LAYER_FORMAT, &tier, sort));
        match spawned {
            Ok(handle) => {
                views.warming.insert(pass_key(LAYER_FORMAT, sort), handle);
            }
            Err(e) => warn!("the splat passes' warm-up thread did not start: {e}"),
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

/// The splat layer's size at render scale `scale` (§1.12): the view's size times the
/// scale, rounded, at least one pixel; §1.7 Window: the scale applies to the layer only.
fn layer_size(view: UVec2, scale: f32) -> UVec2 {
    let side = |n: u32| ((n as f32 * scale).round() as u32).max(1);
    UVec2::new(side(view.x), side(view.y))
}

#[expect(
    clippy::too_many_arguments,
    reason = "one view's inputs, gathered by the system above"
)]
fn prepare_view(
    views: &mut Views,
    dev: &Arc<SplatDevice>,
    tier: &QualityTier,
    render_scale: f32,
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
    // §1.12, §1.7 Window: the render scale sizes the splat layer only (`LinearOver`);
    // `FileEncoded` draws into the view target, at its size.
    let draw_size = match camera.output {
        OutputSpace::FileEncoded => size,
        OutputSpace::LinearOver { .. } => layer_size(size, render_scale),
    };
    let layer_format = match camera.output {
        OutputSpace::FileEncoded => {
            if let Some(warning) = file_encoded_rule(view.target_format)? {
                views.warn_once(entity, warning);
            }
            view.target_format
        }
        OutputSpace::LinearOver { .. } => {
            if view.target_format != RESOLVE_TARGET_FORMAT {
                return Err("LinearOver needs an Rgba8UnormSrgb view target (Q-91)");
            }
            LAYER_FORMAT
        }
    };
    let key = pass_key(layer_format, camera.sort);
    let passes = match views.passes.get(&key) {
        Some(p) => Arc::clone(p),
        None => {
            // Q-325: the warm-up's passes when it compiled this key.
            // A warm-up thread that panicked leaves the key to compile here.
            let warmed = views.warming.take(&key);
            let made = warmed.unwrap_or_else(|| compile(dev, layer_format, tier, camera.sort));
            let p = made.map_err(|e| {
                error!("splat passes: {}: {}", e.code(), e.user_text());
                "the splat passes did not compile"
            })?;
            let p = Arc::new(p);
            views.passes.insert(key, Arc::clone(&p));
            p
        }
    };
    let params = ViewParams {
        camera: core_camera,
        target_size: draw_size,
    };
    let mut state = views.states.remove(&entity).unwrap_or_else(|| ViewState {
        camera,
        passes: Arc::clone(&passes),
        layer: None,
        layer_depth: None,
        view: params,
        core_camera,
        index: 0,
        unit: 0,
    });
    state.camera = camera;
    state.passes = passes;
    state.view = params;
    state.core_camera = core_camera;
    if matches!(camera.output, OutputSpace::LinearOver { .. })
        && state.layer.as_ref().is_none_or(|(_, s)| *s != draw_size)
    {
        let texture = dev.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("ftgsxr splat layer"),
            size: wgpu::Extent3d {
                width: draw_size.x,
                height: draw_size.y,
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
            draw_size,
        ));
    }
    // Q-164: a layer smaller than the view is drawn against the view's depth copied at
    // its size, since a pass's attachments share one size.
    if state.layer.is_none() || draw_size == size {
        state.layer_depth = None;
    } else if state
        .layer_depth
        .as_ref()
        .is_none_or(|(_, s)| *s != draw_size)
    {
        let texture = dev.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("ftgsxr splat layer depth"),
            size: wgpu::Extent3d {
                width: draw_size.x,
                height: draw_size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CORE_3D_DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        state.layer_depth = Some((
            texture.create_view(&wgpu::TextureViewDescriptor::default()),
            draw_size,
        ));
    }
    if state.layer_depth.is_some() && views.depth_copy.is_none() {
        views.depth_copy = Some(Arc::new(DepthCopy::new(dev.device(), CORE_3D_DEPTH_FORMAT)));
    }
    views.states.insert(entity, state);
    Ok(())
}

/// The instances of a view as `ModelDraw`s, at their drawn degrees.
fn draws(instances: &[Instance]) -> Vec<ModelDraw<'_>> {
    instances
        .iter()
        .map(|i| ModelDraw {
            model: &i.model,
            pose: i.pose,
            tau: i.tau,
            mode: i.mode,
            transform: i.transform,
            sh_degree: i.sh_degree,
            overlays: i.overlays,
            trim: i.trim,
        })
        .collect()
}

/// The prepare node (§2.4 Frame loop hooks): stages 1–4 for the view's unit, before the
/// opaque pass, unless nothing changed since the last frame or another member of its
/// shared-center group prepared it this frame (Q-399): every member is a cull view, at the
/// group's sort pose. Bevy runs the cameras one after another and submits their commands
/// in that order, so the first member to run prepares before every member draws.
fn prepare_node(
    view: ViewQuery<&ExtractedView>,
    views: Res<Views>,
    mut shared: ResMut<SharedFrame>,
    mut ctx: RenderContext,
) {
    let Some(state) = views.states.get(&view.entity()) else {
        return;
    };
    let Some(unit) = views.units.get(state.unit) else {
        return;
    };
    // §2.4 Frame loop hooks, Q-399: once per unit and frame.
    if !shared.take_prepare(state.unit) || !unit.prepare {
        return;
    }
    let draws = draws(&unit.instances);
    unit.passes.encode_prepare(
        ctx.command_encoder(),
        &unit.frame,
        &draws,
        &unit.sort_pose,
        &unit.views,
    );
}

/// The draw node (§2.4 Frame loop hooks): stages 5 and 6 after the opaque pass, tested
/// against its depth, then the resolve (`LinearOver`) or none (`FileEncoded`), then stage 8.
#[allow(clippy::too_many_arguments)]
fn draw_node(
    view: ViewQuery<(&ExtractedCamera, &ViewTarget, &ViewDepthTexture)>,
    views: Res<Views>,
    mut shared: ResMut<SharedFrame>,
    tier: Res<RenderTier>,
    occlusion: Option<Res<SplatOcclusion>>,
    images: Res<RenderAssets<GpuImage>>,
    probe_target: Res<ProbeTarget>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    mut ctx: RenderContext,
) {
    let Some(state) = views.states.get(&view.entity()) else {
        return;
    };
    let Some(unit) = views.units.get(state.unit) else {
        return;
    };
    // Q-403: the unit's k-th draw of n this frame; the last copies its statistics.
    let (k, n) = (shared.take_draw(state.unit), unit.members.len());
    let last = k + 1 == n;
    // E13's input of this view (Q-339: `SplatOcclusion` lives in the render world): its
    // frame's image as prepared by Bevy, with the frame's camera and planes.
    let occlusion = occlusion.as_deref().and_then(|o| {
        let frame = view_frame(tier.0.occlusion, &o.frames, state.index)?;
        let image = images.get(&frame.image)?;
        Some(OcclusionInput {
            depth: &image.texture_view,
            camera: frame.camera,
            near: frame.near,
            far: frame.far,
            soft: o.soft,
            delta_m: o.delta_m,
        })
    });
    let frame = &unit.frame;
    if matches!(state.camera.output, OutputSpace::LinearOver { .. }) && state.layer.is_none() {
        // No pass, so no draw's timestamps marked for a later `encode_stats` (Q-81).
        if last {
            state.passes.encode_stats(ctx.command_encoder(), frame);
        }
        return;
    }
    let (camera, target, depth) = view.into_inner();
    // The draw's timestamps (Q-81, Q-87, Q-114): this node owns the pass; in a group, the
    // first member's draw writes the begin and the last's the end (Q-403).
    let timestamp_writes = frame.eye_draw_timestamp_writes(k, n);
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
                .encode_draw(pass.wgpu_pass(), frame, &state.view, occlusion.as_ref());
        }
        (OutputSpace::LinearOver { background }, Some((layer, _))) => {
            // Q-164: below render scale 1, the view's depth at the layer's size first.
            let layer_depth = match (&state.layer_depth, &views.depth_copy) {
                (Some((target, s)), Some(copy)) => {
                    if let Some(dev) = &views.dev {
                        copy.encode(
                            dev.device(),
                            ctx.command_encoder(),
                            depth.view(),
                            target,
                            [s.x, s.y],
                        );
                    }
                    Some(wgpu::RenderPassDepthStencilAttachment {
                        view: target,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    })
                }
                _ => depth_attachment,
            };
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
                    depth_stencil_attachment: layer_depth,
                    timestamp_writes,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                state
                    .passes
                    .encode_draw(pass.wgpu_pass(), frame, &state.view, occlusion.as_ref());
            }
            // E14's exposure match: the first splat camera's layer (the left eye, Q-125),
            // probed once, into the buffer the main world reads back.
            if state.index == 0
                && let (Some(handle), Some(probe), Some(dev)) =
                    (&probe_target.0, &views.probe, &views.dev)
                && let Some(cells) = buffers.get(handle)
            {
                probe.encode(dev.device(), ctx.command_encoder(), layer, &cells.buffer);
            }
            let color = [Some(target.get_color_attachment())];
            let mut pass = ctx.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
                label: Some("ftgsxr splat resolve"),
                color_attachments: &color,
                // Q-101: the view's depth gates the resolve's two branches.
                depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
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
        (OutputSpace::LinearOver { .. }, None) => unreachable!("returned above"),
    }
    // §1.4 stage 8: one block per unit and frame, after its last draw.
    if last {
        state.passes.encode_stats(ctx.command_encoder(), frame);
    }
}

/// Stage 8's read back, never blocking (Q-81): each unit's newest mapped statistics of two
/// or more frames ago, to the main world. When the first unit's block arrives (the unit of
/// the first splat camera, the live one of lowest order, ties to the lower render entity:
/// Q-125, `ftgsxr_xr` orders the eyes by index, so the left eye), it is published with the
/// GPU times summed over the units' newest blocks (Q-403, Q-T722.2).
fn poll_stats(mut views: ResMut<Views>, inbox: Res<StatsInbox>) {
    let Some(dev) = views.dev.clone() else {
        return;
    };
    let mut first_arrived = false;
    for (u, unit) in views.units.iter_mut().enumerate() {
        let Some(stats) = FrameStats::poll(&unit.frame, &dev) else {
            continue;
        };
        unit.last_stats = Some(stats);
        first_arrived |= u == 0;
        if stats.overflow {
            inbox.push(StatsMessage::Overflow {
                visible: stats.visible_splats + stats.visible_centers,
                capacity: stats.capacity,
            });
        }
    }
    if first_arrived && let Some((first, others)) = views.units.split_first() {
        let others: Vec<FrameStats> = others.iter().filter_map(|o| o.last_stats).collect();
        if let Some(stats) = first.last_stats {
            // Q-125, Q-403, Q-T722.2
            inbox.push(StatsMessage::Publish(published_stats(stats, &others)));
        }
    }
}

#[cfg(test)]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    // Q-233, Q-235: an instance is drawn at the smallest of its loaded degree L, the
    // governor's cap and a replayed request's limit; no limit imposes none, and a limit
    // above L keeps L.
    #[test]
    fn drawn_degree_is_the_smallest_cap() {
        assert_eq!(drawn_degree(3, 3, None), 3);
        assert_eq!(drawn_degree(3, 3, Some(1)), 1);
        assert_eq!(drawn_degree(3, 1, Some(2)), 1);
        assert_eq!(drawn_degree(2, 3, Some(3)), 2);
        assert_eq!(drawn_degree(1, 3, None), 1);
        assert_eq!(drawn_degree(3, 2, Some(0)), 0);
    }
    #[test]
    fn layer_size_is_the_scale_times_the_views_size_rounded_at_least_one_pixel() {
        assert_eq!(
            layer_size(UVec2::new(1680, 1760), 1.0),
            UVec2::new(1680, 1760)
        );
        // 0.7 × 1280 = 896, 0.7 × 721 = 504.7 → 505.
        assert_eq!(layer_size(UVec2::new(1280, 721), 0.7), UVec2::new(896, 505));
        // 0.6 × 5 = 3, 0.6 × 3 = 1.8 → 2; 0.6 × 1 = 0.6 → 1.
        assert_eq!(layer_size(UVec2::new(5, 3), 0.6), UVec2::new(3, 2));
        assert_eq!(layer_size(UVec2::new(1, 1), 0.6), UVec2::new(1, 1));
        assert_eq!(layer_size(UVec2::new(3, 1), 0.1), UVec2::new(1, 1));
    }

    // Q-338, PRD #565: the i-th view draws against frames[i], none past the end, and
    // none at all when the running tier's occlusion flag is false.
    #[test]
    fn view_frame_is_the_indexed_frame_when_the_tier_allows_it() {
        let frame = |near: f32| DepthFrame {
            image: ::bevy::asset::Handle::default(),
            camera: ftgsxr_core::Camera {
                width: 1,
                height: 1,
                k: ftgsxr_core::DMat3::IDENTITY,
                world_to_camera: ftgsxr_core::DMat4::IDENTITY,
            },
            near,
            far: 10.0,
            age_frames: 0,
        };
        let frames = [frame(0.1), frame(0.2)];
        assert_eq!(view_frame(true, &frames, 0).map(|f| f.near), Some(0.1));
        assert_eq!(view_frame(true, &frames, 1).map(|f| f.near), Some(0.2));
        assert!(view_frame(true, &frames, 2).is_none());
        assert!(view_frame(true, &[], 0).is_none());
        assert!(view_frame(false, &frames, 0).is_none());
    }

    // Q-125: camera order, ties to the lower entity (`Entity`'s own order), a camera
    // without an order at 0.
    #[test]
    fn camera_order_sorts_by_order_then_entity() {
        let e = |i: u32| Entity::from_raw_u32(i).expect("a valid index");
        let orders = HashMap::from([(e(1), 1), (e(2), 0), (e(3), 0), (e(4), -1)]);
        let sorted = camera_order([e(1), e(3), e(2), e(4)].into_iter(), &orders);
        let (lo, hi) = (e(2).min(e(3)), e(2).max(e(3)));
        assert_eq!(sorted, vec![e(4), lo, hi, e(1)]);
        let sorted = camera_order([e(1), e(5)].into_iter(), &orders);
        assert_eq!(sorted, vec![e(5), e(1)], "no order is order 0");
    }

    // Q-399, Q-T722.1: per-view cameras prepare alone; the first two shared-center
    // cameras in camera order are one group, wherever they stand among the others, and a
    // third shared-center camera is a group of one.
    #[test]
    fn prepare_units_group_the_first_two_shared_center_cameras() {
        let e = |i: u32| Entity::from_raw_u32(i).expect("a valid index");
        let (p, s) = (SortMode::PerView, SortMode::SharedCenter);
        assert_eq!(prepare_units(&[]), Vec::<Vec<Entity>>::new());
        assert_eq!(prepare_units(&[(e(1), s)]), vec![vec![e(1)]]);
        assert_eq!(
            prepare_units(&[(e(1), p), (e(2), p)]),
            vec![vec![e(1)], vec![e(2)]]
        );
        assert_eq!(
            prepare_units(&[(e(5), s), (e(3), s)]),
            vec![vec![e(5), e(3)]],
            "camera order, not entity order"
        );
        assert_eq!(
            prepare_units(&[(e(1), p), (e(2), s), (e(3), p), (e(4), s), (e(6), s)]),
            vec![vec![e(1)], vec![e(2), e(4)], vec![e(3)], vec![e(6)]],
            "the group stands at its first member's place"
        );
    }

    // Q-125, Q-403, Q-T722.2: the first unit's counts; each time the sum over the units.
    #[test]
    fn published_stats_keep_the_first_counts_and_sum_the_times() {
        use crate::gpu::GpuTimes;
        let block = |visible: u32, non_finite: u32, t: [f32; 4]| FrameStats {
            visible_splats: visible,
            visible_centers: visible / 10,
            capacity: 1000,
            overflow: visible > 500,
            non_finite,
            trimmed: visible / 2,
            gpu_times_ms: GpuTimes {
                cull: t[0],
                sort: t[1],
                draw: t[2],
                total: t[3],
            },
        };
        let first = block(400, 12, [0.25, 0.5, 2.0, 3.0]);
        assert_eq!(published_stats(first, &[]), first, "one unit: its block");
        let second = block(600, 6, [0.125, 0.25, 1.0, 1.5]);
        let third = block(10, 0, [0.0, 0.0, 0.5, 0.5]);
        let published = published_stats(first, &[second, third]);
        assert_eq!(
            published,
            FrameStats {
                gpu_times_ms: GpuTimes {
                    cull: 0.375,
                    sort: 0.75,
                    draw: 3.5,
                    total: 5.0,
                },
                ..first
            }
        );
    }

    #[test]
    fn file_encoded_rule_draws_unwarned_on_a_16_bit_float_target() {
        assert_eq!(file_encoded_rule(TextureFormat::Rgba16Float), Ok(None));
    }

    #[test]
    fn file_encoded_rule_draws_with_the_rounding_warning_on_an_8_bit_linear_target() {
        assert!(matches!(
            file_encoded_rule(TextureFormat::Rgba8Unorm),
            Ok(Some(w)) if w.contains("rounds the layer")
        ));
    }

    #[test]
    fn file_encoded_rule_refuses_srgb_targets() {
        for format in [TextureFormat::Rgba8UnormSrgb, TextureFormat::Bgra8UnormSrgb] {
            assert!(file_encoded_rule(format).is_err(), "{format:?}");
        }
    }
}
