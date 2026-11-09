// SPDX-License-Identifier: Apache-2.0

//! Picking, selection, grabs and the pose store (§2.1 `ftgsxr_app::manipulate`, E10,
//! E11, §1.5 Poses, §1.7 Selection and manipulation, §1.8 Rays, selection and grabbing,
//! D-11, D-42, D-43). One system reads the intents in order and keeps each hand's last
//! ray and pose; the hovered and selected boxes and the desktop gizmo are drawn with
//! Bevy's gizmo lines (§1.4 Selection and debug views). Writing `poses.json` is
//! `persist`'s (`PersistCommand::SavePose`): this module changes `Poses` in memory only.
//! On the headset a first load out of D-11's size range is fitted (Q-284).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::app::{App, Plugin, Update};
use bevy::asset::Assets;
use bevy::camera::visibility::Visibility;
use bevy::camera::{Camera, RenderTarget};
use bevy::color::Color;
use bevy::ecs::hierarchy::ChildOf;
use bevy::ecs::prelude::*;
use bevy::gizmos::AppGizmoBuilder;
use bevy::gizmos::config::{GizmoConfig, GizmoConfigGroup, GizmoConfigStore};
use bevy::gizmos::gizmos::Gizmos;
use bevy::math::{Quat, Vec3};
use bevy::reflect::Reflect;
use bevy::time::Time;
use bevy::transform::components::{GlobalTransform, Transform};
use ftgsxr_core::{
    Aabb, Grab, ModelKind, Pose, Preset, ScaleLimits, SceneManifest, Trim, TrimHandle, TrimMode,
    TwoHandGrab, fit_pose, ray_hit, read_scene_manifest, snap_to_floor,
};
use ftgsxr_io::{ModelRef, PoseStore, SavedPose};
use ftgsxr_render::{SplatAsset, SplatBounds, SplatMode, SplatModel, SplatTrim};

use crate::intent::{GizmoMode, Hand, InstanceAction, Intent};
use crate::library::{LibraryCommand, LoadedModel};
use crate::platform::Headset;
use crate::playback::StepTarget;

mod trim_handles;
use trim_handles::TrimDrag;

/// What the grabs and rays obey (§1.5, §1.8, E10, E11).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct ManipulateSettings {
    /// E11's scale limits.
    pub limits: ScaleLimits,
    /// E10's t_max in metres.
    pub ray_max_m: f32,
    /// Grabs, one-hand and two-hand, turn about stage y only (E11, D-43); `persist` keeps
    /// it equal to the settings' `[xr] keep_upright`, both ways (Q-330).
    pub keep_upright: bool,
    /// The push and pull speed per metre of distance, in m/s (§1.8).
    pub push_pull_mps: f32,
    /// The fit box (§1.5 Poses): E10's content box B_c when true, its B_m when false.
    /// The fit, the headset's automatic fit and snap to floor take it; every other use of
    /// the model's box is B_m's. `persist` keeps it equal to the settings'
    /// `[general] fit_to`, both ways (Q-371).
    pub fit_to_content: bool,
}

// Q-185
impl Default for ManipulateSettings {
    fn default() -> Self {
        ManipulateSettings {
            limits: ScaleLimits {
                min: 0.01,
                max: 100.0,
            },
            ray_max_m: 10.0,
            keep_upright: true,
            push_pull_mps: 1.0,
            fit_to_content: true, // Q-371
        }
    }
}

/// An instance held by one hand or two (E11).
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub enum Grabbed {
    /// A one-hand grab.
    One {
        /// The hand.
        hand: Hand,
        /// The grab.
        grab: Grab,
    },
    /// A two-hand grab.
    Two {
        /// The grab.
        grab: TwoHandGrab,
    },
}

/// The selected instance (one at most).
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selected;

/// An instance a ray hovers: the hand and E10's entry distance.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Hovered {
    /// The hand whose ray hits it.
    pub hand: Hand,
    /// max(t_in, 0) in metres.
    pub distance: f32,
}

/// E18's trim handles of an instance that a hand touches: the handle each hand's ray
/// hovers, and the handle each hand drags; absent when none. `audio` and the XR haptics
/// read its edges (§1.8, §1.10).
// Q-378
#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct TrimHandles {
    /// The handles hovered, one per hand at most, in hand order.
    pub hovered: Vec<(Hand, TrimHandle)>,
    /// The handles dragged, one per hand at most, in the order the drags began.
    pub dragged: Vec<(Hand, TrimHandle)>,
}

/// The pose store in the world (§1.5 Poses); `persist` loads and writes the file.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Poses(pub PoseStore);

/// E10 over every instance: the nearest hit, the first of equally near ones.
pub fn pick_instance(
    origin: Vec3,
    dir: Vec3,
    instances: &[(Entity, Pose, Aabb)],
    max_m: f32,
) -> Option<(Entity, f32)> {
    let mut best: Option<(Entity, f32)> = None;
    for (e, pose, bounds) in instances {
        if let Some(t) = ray_hit(origin, dir, bounds, pose, max_m)
            && best.is_none_or(|(_, b)| t < b)
        {
            best = Some((*e, t));
        }
    }
    best
}

/// The module's systems and observers, added by `AppPlugin`.
pub(crate) struct ManipulatePlugin;

impl Plugin for ManipulatePlugin {
    fn build(&self, app: &mut App) {
        // Q-185: `persist` may have inserted them first.
        if !app.world().contains_resource::<ManipulateSettings>() {
            app.insert_resource(ManipulateSettings::default());
        }
        if !app.world().contains_resource::<Poses>() {
            app.insert_resource(Poses::default());
        }
        app.add_message::<Intent>()
            .add_message::<LibraryCommand>()
            .init_resource::<HandStates>()
            .init_resource::<GizmoState>()
            .init_resource::<TrimSends>()
            .init_resource::<HighContrast>()
            .add_observer(on_loaded)
            .add_message::<SetPoseExtras>()
            .add_systems(
                Update,
                (
                    set_pose_extras,
                    run_intents,
                    send_trim_drags,
                    step_target_follows_selection,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                draw.after(run_intents)
                    .run_if(resource_exists::<GizmoConfigStore>),
            );
        // E18's handles are drawn on top of the splats, in a gizmo group of their own;
        // only where the engine draws gizmos (Q-378).
        if app.world().contains_resource::<GizmoConfigStore>() {
            app.insert_gizmo_config(
                TrimHandleGizmos,
                GizmoConfig {
                    depth_bias: -1.0,
                    ..Default::default()
                },
            )
            .add_systems(Update, draw_trim_handles.after(run_intents));
        }
    }
}

/// The gizmo group of E18's handles: no depth test (a depth bias of −1), so a handle
/// inside or behind the content stays visible.
#[derive(Default, Reflect, GizmoConfigGroup)]
struct TrimHandleGizmos;

/// playback's `StepTarget` follows the selection (Q-175): the selected instance, else
/// none (playback's fallback).
// Q-175
fn step_target_follows_selection(
    selected: Query<Entity, With<Selected>>,
    added: Query<(), Added<Selected>>,
    mut removed: RemovedComponents<Selected>,
    mut target: ResMut<StepTarget>,
) {
    if added.is_empty() && removed.read().count() == 0 {
        return;
    }
    let next = selected.iter().next();
    if target.0 != next {
        target.0 = next;
    }
}

/// What the fit and the saved pose carry beside the pose (§1.5): the orientation preset
/// and L_fit, from the hints when the instance appears, else from its saved pose.
// Q-181
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PoseExtras {
    pub(crate) preset: Preset,
    pub(crate) fit_size_m: f32,
}

/// The Scene tab's orientation preset and L_fit for an instance (§1.5 Poses); L_fit is
/// taken into [0.25 m, 10⁸ m]. The next fit and saved pose use them.
// Q-181
#[derive(Message, Clone, Copy, Debug, PartialEq)]
pub(crate) struct SetPoseExtras {
    pub(crate) instance: Entity,
    pub(crate) preset: Preset,
    pub(crate) fit_size_m: f32,
}

/// §1.5's least L_fit in metres.
pub(crate) const FIT_MIN_M: f32 = 0.25;

/// §1.5's greatest L_fit in metres, which holds a planet at its real size.
pub(crate) const FIT_MAX_M: f32 = 1e8;

/// §1.5's default L_fit in metres.
pub(crate) const FIT_DEFAULT_M: f32 = 1.0;

/// A fit size as §1.5 takes it: the nearer bound of [0.25 m, 10⁸ m] when outside it,
/// ∞ included, none when it isn't a number.
// D-11
fn fit_size(l: f32) -> Option<f32> {
    (!l.is_nan()).then(|| l.clamp(FIT_MIN_M, FIT_MAX_M))
}

// Q-181, D-11
fn set_pose_extras(
    mut reader: MessageReader<SetPoseExtras>,
    current: Query<&PoseExtras>,
    mut commands: Commands,
) {
    // The sizes set this frame, which the deferred inserts don't show `current` yet.
    let mut set: HashMap<Entity, f32> = HashMap::new();
    for m in reader.read() {
        // A size that isn't a number leaves L_fit as it was (1 m before the instance has
        // one, §1.5's default).
        let kept = || {
            set.get(&m.instance).copied().unwrap_or_else(|| {
                current
                    .get(m.instance)
                    .map_or(FIT_DEFAULT_M, |x| x.fit_size_m)
            })
        };
        let fit_size_m = fit_size(m.fit_size_m).unwrap_or_else(kept);
        set.insert(m.instance, fit_size_m);
        if let Ok(mut e) = commands.get_entity(m.instance) {
            e.try_insert(PoseExtras {
                preset: m.preset,
                fit_size_m,
            });
        }
    }
}

/// Each hand's last ray and pose, and what it holds.
#[derive(Default)]
struct HandState {
    ray: Option<(Vec3, Vec3)>,
    pose: Option<Pose>,
    /// The held instance and the grab's distance along the ray.
    held: Option<(Entity, f32)>,
    /// E11's grab point while it holds: the point of the instance where its ray took
    /// hold, in the instance's frame (revision 34).
    point: Vec3,
    /// During a two-hand grab, the grab point in the hand's frame.
    offset: Vec3,
}

/// E11's fallback: two-hand grab points that start closer than this (metres) are
/// replaced by the hands' aim positions for that grab (revision 34).
const GRAB_POINTS_MIN_M: f32 = 0.05;

/// E11's two-hand grab points at the start of a two-hand grab: each hand's point of the
/// instance (`points`, in its frame) under the instance's pose `model`, as an offset in
/// that hand's frame, so it moves fixed in the hand from here on; both zero, the hands'
/// aim positions, when the points lie closer than `GRAB_POINTS_MIN_M`.
fn grab_offsets(model: Pose, hands: [Pose; 2], points: [Vec3; 2]) -> [Vec3; 2] {
    let stage = points.map(|p| model.transform_point(p));
    if stage[0].distance(stage[1]) < GRAB_POINTS_MIN_M {
        return [Vec3::ZERO; 2];
    }
    [
        hands[0].inverse().transform_point(stage[0]),
        hands[1].inverse().transform_point(stage[1]),
    ]
}

/// A grab point's pose, `TwoHandGrab`'s a or b: the hand's pose moved to the point
/// `offset` gives in its frame.
fn at_offset(hand: Pose, offset: Vec3) -> Pose {
    Pose {
        translation: hand.transform_point(offset),
        ..hand
    }
}

/// Starts (or restarts) the two-hand grab of the instance at `model` by hands `a` and
/// `b`, from their poses and grab points (E11, revision 34); `None` while either pose is
/// unknown.
fn begin_two(hs: &mut HandStates, (a, b): (Hand, Hand), model: Pose) -> Option<TwoHandGrab> {
    let (ha, hb) = (hs.hands.get(&a)?, hs.hands.get(&b)?);
    let hands = [ha.pose?, hb.pose?];
    let offsets = grab_offsets(model, hands, [ha.point, hb.point]);
    for (hand, offset) in [(a, offsets[0]), (b, offsets[1])] {
        hs.hands.entry(hand).or_default().offset = offset;
    }
    Some(TwoHandGrab::begin(
        at_offset(hands[0], offsets[0]),
        at_offset(hands[1], offsets[1]),
        model,
    ))
}

/// The grabs (authoritative; `Grabbed` mirrors them) and the hands.
#[derive(Resource, Default)]
struct HandStates {
    hands: HashMap<Hand, HandState>,
    grabs: HashMap<Entity, (Grabbed, Option<(Hand, Hand)>)>,
    /// E18's drags under way, one per hand at most, in the order they began.
    trim_drags: Vec<TrimDrag>,
    /// The trim last sent for each dragged instance, so an unchanged shape is not sent
    /// again while the `SetTrim` is on its way.
    trim_sent: HashMap<Entity, Trim>,
}

/// The `SetTrim` intents a frame's drags send (E18), written by `send_trim_drags`:
/// `run_intents` reads the intents, so it cannot write them too.
#[derive(Resource, Default)]
struct TrimSends(Vec<Intent>);

/// Writes the frame's `SetTrim` intents of E18's drags.
fn send_trim_drags(mut sends: ResMut<TrimSends>, mut intents: MessageWriter<Intent>) {
    for i in sends.0.drain(..) {
        intents.write(i);
    }
}

impl HandStates {
    fn trim_drag(&self, hand: Hand) -> Option<&TrimDrag> {
        self.trim_drags.iter().find(|d| d.hand == hand)
    }

    fn end_trim_drag(&mut self, hand: Hand) {
        self.trim_drags.retain(|d| d.hand != hand);
    }

    fn trim_dragged(&self, e: Entity) -> bool {
        self.trim_drags.iter().any(|d| d.entity == e)
    }
}

/// The desktop gizmo: its handle set (translate until `Intent::Gizmo` chooses) and the
/// drag under way.
// Q-186
#[derive(Resource)]
struct GizmoState {
    mode: GizmoMode,
    drag: Option<Drag>,
}

impl Default for GizmoState {
    fn default() -> Self {
        GizmoState {
            mode: GizmoMode::Translate,
            drag: None,
        }
    }
}

struct Drag {
    entity: Entity,
    handle: Handle,
    center: Vec3,
    size: f32,
    start: Pose,
    start_ray: (Vec3, Vec3),
}

/// A gizmo handle: an axis's arrow, ring, or the uniform-scale cube.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Handle {
    Arrow(Vec3),
    Ring(Vec3),
    Cube,
}

const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

/// An instance's or the rig's local transform as a pose (scale from x), shared by
/// `rig` and `persist`.
pub(crate) fn pose_of(t: &Transform) -> Pose {
    Pose {
        translation: t.translation,
        rotation: t.rotation,
        scale: t.scale.x,
    }
}

/// A pose as a transform with uniform scale.
pub(crate) fn transform_of(p: Pose) -> Transform {
    Transform {
        translation: p.translation,
        rotation: p.rotation,
        scale: Vec3::splat(p.scale),
    }
}

/// The distance ahead of the head that the fit (§1.5 Poses), the headset's auto-fit and
/// `rig`'s recenter (§1.8 Session and spaces) place the content at, metres.
pub(crate) const AHEAD_M: f32 = 1.5;

/// Whether a camera may be the head (Q-180, Q-310): active, and rendering to a window
/// or the headset's views (texture views). An image target (the panels' cameras, a
/// bench's target), a camera without a color target and any other target are not.
// Q-310
pub(crate) fn head_candidate(camera: &Camera, target: &RenderTarget) -> bool {
    camera.is_active
        && matches!(
            target,
            RenderTarget::Window(_) | RenderTarget::TextureView(_)
        )
}

/// The cameras the head is picked from.
pub(crate) type HeadCameras<'w, 's> = Query<
    'w,
    's,
    (
        &'static Camera,
        &'static RenderTarget,
        &'static GlobalTransform,
    ),
>;

/// The head (Q-180, Q-310): the candidate camera of the lowest order (rotation and
/// translation, scale 1); `None` without one. The fit, the auto-fit, locomotion and
/// recenter share it.
// Q-180, Q-310
pub(crate) fn head_pose(cameras: &HeadCameras) -> Option<Pose> {
    cameras
        .iter()
        .filter(|(c, t, _)| head_candidate(c, t))
        .min_by_key(|(c, ..)| c.order)
        .map(|(.., g)| {
            let (_, rotation, translation) = g.to_scale_rotation_translation();
            Pose {
                translation,
                rotation,
                scale: 1.0,
            }
        })
}

/// §1.5's hints for a local model: the preset from the first `ftgs-scene` manifest in
/// its directory (a checkpoint's parent), by file name, and L_fit = 2S from a
/// checkpoint's spatial scale, taken into [0.25 m, 10⁸ m]; else Y-up and 1 m.
// Q-181, D-11
fn hints(source: &ModelRef, kind: ModelKind) -> PoseExtras {
    let fit_size_m = match kind {
        ModelKind::Checkpoint { spatial_scale, .. } => fit_size((2.0 * spatial_scale) as f32),
        _ => None,
    }
    .unwrap_or(FIT_DEFAULT_M);
    let preset = match source {
        ModelRef::File(p) => p.parent().and_then(manifest_preset).unwrap_or(Preset::YUp),
        _ => Preset::YUp,
    };
    PoseExtras { preset, fit_size_m }
}

/// The first `ftgs-scene` manifest beside a model, in name order: any `*.json` of `dir`
/// that reads as one (§1.5, §1.7 Training cameras and requests).
// Q-219
pub(crate) fn manifest_beside(dir: &Path) -> Option<(PathBuf, SceneManifest)> {
    let mut jsons: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "json"))
        .collect();
    jsons.sort();
    jsons.into_iter().find_map(|p| {
        let m = read_scene_manifest(&std::fs::read(&p).ok()?).ok()?;
        Some((p, m))
    })
}

fn manifest_preset(dir: &Path) -> Option<Preset> {
    let (_, m) = manifest_beside(dir)?;
    // §1.5: each camera's up is −R_wᵀ (0, 1, 0), minus R_w's second row.
    let ups: Vec<Vec3> = m
        .cameras
        .iter()
        .map(|(_, c)| {
            let w = c.world_to_camera;
            Vec3::new(-w.x_axis.y as f32, -w.y_axis.y as f32, -w.z_axis.y as f32)
        })
        .collect();
    Some(Preset::from_camera_ups(&ups))
}

/// D-11: an instance appears at its saved pose, else at the identity, with its hints; on
/// the headset (the `Headset` marker, Q-266), one without a saved pose whose fit box's
/// longest side lies outside [0.1 m, 5 m] is fitted at once (§1.5 Poses), relative to
/// the content group (Q-247). A reload keeps its `LoadedModel` (Q-238), so only a first
/// load fits; a scene's pose, applied after, wins (Q-284).
// Q-266, Q-284
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn on_loaded(
    add: On<Add, LoadedModel>,
    mut q: Query<(
        &LoadedModel,
        &mut Transform,
        Option<&SplatModel>,
        Option<&ChildOf>,
    )>,
    groups: GroupTransforms,
    poses: Res<Poses>,
    headset: Option<Res<Headset>>,
    assets: Option<Res<Assets<SplatAsset>>>,
    settings: Option<Res<ManipulateSettings>>,
    cameras: HeadCameras,
    mut commands: Commands,
) {
    let Ok((m, mut t, model, parent)) = q.get_mut(add.entity) else {
        return;
    };
    let extras = match poses.0.get(&m.identity) {
        Some(saved) => {
            *t = transform_of(saved.pose);
            // §1.5 Poses: the trim saved with the pose loads with it (E17). Q-374
            if let Some(trim) = saved.trim {
                commands.entity(add.entity).insert(SplatTrim(trim));
            }
            PoseExtras {
                preset: saved.preset,
                // §1.5's range also holds for a size saved before it had an upper bound.
                fit_size_m: fit_size(saved.fit_size_m).unwrap_or(FIT_DEFAULT_M),
            }
        }
        None => {
            let x = hints(&m.source, m.info.kind);
            // §1.5: the fit box, content by default.
            let content = settings.is_none_or(|s| s.fit_to_content);
            let bounds = model
                .zip(assets.as_ref())
                .and_then(|(model, a)| a.get(&model.0))
                .map(|a| asset_fit_box(a, content));
            if headset.is_some()
                && let Some(b) = bounds
                && !(AUTO_FIT_RANGE_M.0..=AUTO_FIT_RANGE_M.1).contains(&b.size().max_element())
            {
                let fitted = fit_pose(
                    &b,
                    x.preset,
                    x.fit_size_m,
                    head_pose(&cameras).unwrap_or(Pose::IDENTITY),
                    AHEAD_M,
                );
                *t = transform_of(fitted.then(&group_pose(parent, &groups).inverse()));
            }
            x
        }
    };
    commands.entity(add.entity).insert(extras);
}

/// §1.5 Poses' range of the fit box's longest side, in metres, inside which the headset does not
/// fit a model on its first load (D-11).
const AUTO_FIT_RANGE_M: (f32, f32) = (0.1, 5.0);

/// `saved_at`: RFC 3339 in UTC with nine fractional digits (Q-52).
pub(crate) fn now_rfc3339() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    rfc3339(d.as_secs(), d.subsec_nanos())
}

fn rfc3339(secs: u64, nanos: u32) -> String {
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Days since 1970-01-01 to the civil date (the proleptic Gregorian calendar): Howard
    // Hinnant's `civil_from_days`, from general knowledge (source-log rows 48, 195).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:09}Z",
        rem / 3_600,
        rem / 60 % 60,
        rem % 60,
        nanos
    )
}

type Instances<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static mut Transform,
        &'static LoadedModel,
        Option<&'static SplatBounds>,
        Option<&'static PoseExtras>,
        Option<&'static Hovered>,
        Option<&'static Grabbed>,
        Option<&'static ChildOf>,
    ),
>;

/// The instances' groups: the content group (§2 `ContentGroup`, every instance's
/// parent), a root entity whose local transform is its stage pose.
type GroupTransforms<'w, 's> = Query<'w, 's, &'static Transform, Without<LoadedModel>>;

/// The transform §1.14's content group applies to an instance's pose: its parent's
/// pose, the identity without one.
fn group_pose(parent: Option<&ChildOf>, groups: &GroupTransforms) -> Pose {
    parent
        .and_then(|c| groups.get(c.parent()).ok())
        .map(pose_of)
        .unwrap_or(Pose::IDENTITY)
}

/// A stage pose: the pose relative to the content group, `t`, followed by the group's
/// transform (§1.14). Rays, the head and the floor are in stage space.
fn staged(t: &Transform, parent: Option<&ChildOf>, groups: &GroupTransforms) -> Pose {
    pose_of(t).then(&group_pose(parent, groups))
}

/// An instance's stage pose.
fn stage_pose(q: &Instances, groups: &GroupTransforms, e: Entity) -> Option<Pose> {
    let (_, t, .., parent) = q.get(e).ok()?;
    Some(staged(t, parent, groups))
}

/// Places an instance at a stage pose, written back relative to the content group
/// through the group's inverse (a similarity with uniform scale, E5).
fn set_stage_pose(q: &mut Instances, groups: &GroupTransforms, e: Entity, pose: Pose) {
    if let Ok((_, mut t, .., parent)) = q.get_mut(e) {
        let group = group_pose(parent, groups);
        *t = transform_of(pose.then(&group.inverse()));
    }
}

/// Every instance's (entity, stage pose, B_m), B_m its `SplatBounds`, trimmed while a
/// trim is on (E17); an instance whose boxes are not known yet has no box.
fn boxes(q: &Instances, groups: &GroupTransforms) -> Vec<(Entity, Pose, Aabb)> {
    q.iter()
        .filter_map(|(e, t, _, b, .., parent)| Some((e, staged(t, parent, groups), b?.aabb)))
        .collect()
}

fn bounds_of(q: &Instances, e: Entity) -> Option<Aabb> {
    let (_, _, _, b, ..) = q.get(e).ok()?;
    Some(b?.aabb)
}

/// §1.5's fit box: B_c under "Fit to: content", else B_m, both trimmed while a trim is on
/// (E17).
fn fit_box(bounds: &SplatBounds, content: bool) -> Aabb {
    if content { bounds.content } else { bounds.aabb }
}

/// §1.5's fit box of an untrimmed asset, for the automatic fit at its load, where an
/// instance without a saved pose has no trim (Q-374).
fn asset_fit_box(asset: &SplatAsset, content: bool) -> Aabb {
    let untrimmed = SplatBounds {
        aabb: asset.bounds,
        content: asset.content_bounds,
        kept_none: false,
    };
    fit_box(&untrimmed, content)
}

fn fit_box_of(q: &Instances, e: Entity, content: bool) -> Option<Aabb> {
    let (_, _, _, b, ..) = q.get(e).ok()?;
    Some(fit_box(b?, content))
}

/// After a pose changed outside a grab, the grab takes hold again from the new pose.
// Q-183
fn rebase(hs: &mut HandStates, e: Entity, pose: Pose) {
    let Some(&(g, hands)) = hs.grabs.get(&e) else {
        return;
    };
    let next = match (g, hands) {
        (Grabbed::One { hand, .. }, _) => hs.hands.get(&hand).and_then(|h| {
            Some(Grabbed::One {
                hand,
                grab: Grab::begin(h.pose?, pose, pose.transform_point(h.point)),
            })
        }),
        // Each grab point at the same place on the instance (E11, revision 34).
        (Grabbed::Two { .. }, Some(pair)) => {
            begin_two(hs, pair, pose).map(|grab| Grabbed::Two { grab })
        }
        _ => None,
    };
    if let (Some(next), Some((g, _))) = (next, hs.grabs.get_mut(&e)) {
        *g = next;
    }
}

/// The posed box's center in stage space.
fn posed_center(bounds: &Aabb, pose: &Pose) -> Vec3 {
    pose.transform_point(bounds.center())
}

/// The gizmo's size: a fifth of the distance from the viewer, at least 5 cm.
// Q-186
fn gizmo_size(center: Vec3, eye: Vec3) -> f32 {
    (0.2 * center.distance(eye)).max(0.05)
}

/// An instance's trim as it stands within the frame: one set this frame, else its
/// `SplatTrim`.
fn trim_of(now: &HashMap<Entity, Trim>, trims: &Query<&SplatTrim>, e: Entity) -> Option<Trim> {
    now.get(&e)
        .copied()
        .or_else(|| trims.get(e).ok().map(|t| t.0))
}

/// The instance whose trim handles a hand may touch (E18, §1.7): the selected one, while
/// its trim is on and it is not grabbed, the mouse only in the gizmo's trim mode; with its
/// trim and stage pose.
#[allow(clippy::too_many_arguments)]
fn trim_target(
    sel: Option<Entity>,
    hand: Hand,
    mode: GizmoMode,
    now: &HashMap<Entity, Trim>,
    trims: &Query<&SplatTrim>,
    hs: &HandStates,
    q: &Instances,
    groups: &GroupTransforms,
) -> Option<(Entity, Trim, Pose)> {
    let e = sel?;
    if hand == Hand::Mouse && mode != GizmoMode::Trim {
        return None;
    }
    let trim = trim_of(now, trims, e)?;
    if trim.mode == TrimMode::Off || hs.grabs.contains_key(&e) {
        return None;
    }
    Some((e, trim, stage_pose(q, groups, e)?))
}

/// E18's ρ for an instance: §1.7's radius floor, 10⁻³ of its untrimmed B_m's longest side.
fn trim_floor(assets: &Assets<SplatAsset>, models: &Query<&SplatModel>, e: Entity) -> f32 {
    models
        .get(e)
        .ok()
        .and_then(|m| assets.get(&m.0))
        .map_or(0.0, |a| 1e-3 * a.bounds.longest_side())
}

/// Starts a drag of the trim handle the hand's ray hits (E18), when `target` has one
/// under it; true when the action is the handles' (a drag started, or a handle another
/// hand drags, which it cannot take), so it neither picks nor grabs.
fn begin_trim_drag(
    hs: &mut HandStates,
    hand: Hand,
    by_select: bool,
    target: Option<(Entity, Trim, Pose)>,
    eye: Option<Vec3>,
    max_m: f32,
    floor: f32,
) -> bool {
    let Some((e, trim, pose)) = target else {
        return false;
    };
    let Some(ray) = hs.hands.get(&hand).and_then(|h| h.ray) else {
        return false;
    };
    // L's eye: the head, else the ray's origin. Q-378
    let Some(size) = trim_handles::handle_size(&trim, &pose, eye.unwrap_or(ray.0)) else {
        return false;
    };
    let Some((handle, t0)) = trim_handles::handle_hit(&trim, &pose, size, ray, max_m) else {
        return false;
    };
    // A handle another hand drags is the handles' action: no pick, no grab. Q-378
    if hs
        .trim_drags
        .iter()
        .any(|d| d.entity == e && d.handle == handle)
    {
        return true;
    }
    let Some(anchor) = trim
        .handles()
        .into_iter()
        .find_map(|(h, p)| (h == handle).then_some(p))
    else {
        return true;
    };
    // A mouse ray along the handle's line or plane gives no drag point: no drag.
    let Some(p) = trim_handles::drag_point(hand, handle, anchor, ray, t0, &pose, ray) else {
        return true;
    };
    hs.trim_drags.push(TrimDrag {
        entity: e,
        hand,
        handle,
        start: trim,
        p0: pose.inverse().transform_point(p),
        anchor,
        start_ray: ray,
        t0,
        by_select,
        floor,
        delta: Vec3::ZERO,
    });
    true
}

#[allow(clippy::too_many_arguments)]
fn run_intents(
    mut intents: MessageReader<Intent>,
    mut q: Instances,
    groups: GroupTransforms,
    selected: Query<Entity, With<Selected>>,
    (assets, models, trims): (
        Res<Assets<SplatAsset>>,
        Query<&SplatModel>,
        Query<&SplatTrim>,
    ),
    cameras: HeadCameras,
    settings: Res<ManipulateSettings>,
    mut poses: ResMut<Poses>,
    mut hs: ResMut<HandStates>,
    mut gizmo: ResMut<GizmoState>,
    time: Res<Time>,
    mut library: MessageWriter<LibraryCommand>,
    touched: Query<(Entity, &TrimHandles)>,
    mut sends: ResMut<TrimSends>,
    mut commands: Commands,
) {
    let s = *settings;
    let eye = head_pose(&cameras).map(|p| p.translation);
    // The trims set within the frame, which the deferred inserts don't show `trims` yet.
    let mut trim_now: HashMap<Entity, Trim> = HashMap::new();
    // The selection as it stands within the frame.
    let mut sel: Option<Entity> = selected.iter().next();
    let mut sel_changed = false;
    // Instances gone since the last frame let go of their grabs.
    hs.grabs.retain(|e, _| q.contains(*e));
    for h in hs.hands.values_mut() {
        if h.held.is_some_and(|(e, _)| !q.contains(e)) {
            h.held = None;
        }
    }
    if gizmo.drag.as_ref().is_some_and(|d| !q.contains(d.entity)) {
        gizmo.drag = None;
    }
    hs.trim_drags.retain(|d| q.contains(d.entity));
    let pick = |q: &Instances, ray: Option<(Vec3, Vec3)>| {
        let (o, d) = ray?;
        pick_instance(o, d, &boxes(q, &groups), s.ray_max_m)
    };
    // The hands with a `Pointer` this frame: only they hover (Q-263).
    let mut pointed: Vec<Hand> = Vec::new();
    for intent in intents.read() {
        match *intent {
            Intent::Pointer { hand, origin, dir } => {
                hs.hands.entry(hand).or_default().ray = Some((origin, dir));
                if !pointed.contains(&hand) {
                    pointed.push(hand);
                }
                if hand == Hand::Mouse
                    && let Some(d) = &gizmo.drag
                {
                    // The cube's scale limits are on the stage scale.
                    // Q-249
                    let p = drag_pose(d, (origin, dir), &s.limits);
                    set_stage_pose(&mut q, &groups, d.entity, p);
                }
            }
            Intent::Select { hand, pressed } => {
                if !pressed {
                    if hand == Hand::Mouse {
                        gizmo.drag = None;
                    }
                    // E18: releasing the action that started a drag ends it.
                    if hs.trim_drag(hand).is_some_and(|d| d.by_select) {
                        hs.end_trim_drag(hand);
                    }
                    continue;
                }
                // E18: a hand drags one handle; another action on it does nothing.
                if hs.trim_drag(hand).is_some() {
                    continue;
                }
                let ray = hs.hands.get(&hand).and_then(|h| h.ray);
                // The desktop gizmo's handles come before the instances behind them.
                if hand == Hand::Mouse
                    && let (mode, Some(e), Some(ray)) = (gizmo.mode, sel, ray)
                    && let Some(bounds) = bounds_of(&q, e)
                    && let Some(pose) = stage_pose(&q, &groups, e)
                {
                    let center = posed_center(&bounds, &pose);
                    let size = gizmo_size(center, ray.0);
                    if let Some(handle) = handle_hit(mode, center, size, ray) {
                        gizmo.drag = Some(Drag {
                            entity: e,
                            handle,
                            center,
                            size,
                            start: pose,
                            start_ray: ray,
                        });
                        continue;
                    }
                }
                // E18: the trim handles come before the instances.
                let target =
                    trim_target(sel, hand, gizmo.mode, &trim_now, &trims, &hs, &q, &groups);
                let floor = target.map_or(0.0, |(e, ..)| trim_floor(&assets, &models, e));
                if begin_trim_drag(&mut hs, hand, true, target, eye, s.ray_max_m, floor) {
                    continue;
                }
                sel = pick(&q, ray).map(|(e, _)| e);
                sel_changed = true;
            }
            Intent::GrabBegin { hand, pose } => {
                let ray = {
                    let h = hs.hands.entry(hand).or_default();
                    h.pose = Some(pose);
                    if h.held.is_some() {
                        continue;
                    }
                    h.ray
                };
                // E18: a hand drags one handle; another action on it does nothing.
                if hs.trim_drag(hand).is_some() {
                    continue;
                }
                let target =
                    trim_target(sel, hand, gizmo.mode, &trim_now, &trims, &hs, &q, &groups);
                let floor = target.map_or(0.0, |(e, ..)| trim_floor(&assets, &models, e));
                if begin_trim_drag(&mut hs, hand, false, target, eye, s.ray_max_m, floor) {
                    continue;
                }
                let (Some((e, dist)), Some((origin, dir))) = (pick(&q, ray), ray) else {
                    continue;
                };
                // E18: an instance whose handle is dragged cannot be grabbed.
                if hs.trim_dragged(e) {
                    continue;
                }
                let Some(model) = stage_pose(&q, &groups, e) else {
                    continue;
                };
                // E11's grab point: where the ray took hold, E10's hit, or the hand's aim
                // position from inside the box (E10 gives 0), kept in the instance's
                // frame (revision 34).
                let hit = if dist > 0.0 {
                    origin + dist * dir
                } else {
                    pose.translation
                };
                hs.hands.entry(hand).or_default().point = model.inverse().transform_point(hit);
                match hs.grabs.get(&e).copied() {
                    None => {
                        hs.grabs.insert(
                            e,
                            (
                                Grabbed::One {
                                    hand,
                                    grab: Grab::begin(pose, model, hit),
                                },
                                None,
                            ),
                        );
                    }
                    Some((Grabbed::One { hand: other, .. }, _)) if other != hand => {
                        let Some(grab) = begin_two(&mut hs, (other, hand), model) else {
                            continue;
                        };
                        hs.grabs
                            .insert(e, (Grabbed::Two { grab }, Some((other, hand))));
                    }
                    _ => continue,
                }
                hs.hands.entry(hand).or_default().held = Some((e, dist));
            }
            Intent::GrabMove { hand, pose } => {
                hs.hands.entry(hand).or_default().pose = Some(pose);
                let Some((e, _)) = hs.hands[&hand].held else {
                    continue;
                };
                let Some(&(g, hands)) = hs.grabs.get(&e) else {
                    continue;
                };
                let new = match (g, hands) {
                    // Yaw only about the grab point under keep upright, about stage +y:
                    // the grab runs on the stage pose (E11, spec revision 37).
                    // D-43, Q-248
                    (Grabbed::One { grab, .. }, _) => grab.update(pose, !s.keep_upright),
                    (Grabbed::Two { grab }, Some((a, b))) => {
                        // The grab points, fixed in their hands (E11, revision 34).
                        let grab_point = |h: Hand| {
                            hs.hands
                                .get(&h)
                                .and_then(|s| Some(at_offset(s.pose?, s.offset)))
                        };
                        let (Some(pa), Some(pb)) = (grab_point(a), grab_point(b)) else {
                            continue;
                        };
                        // Upright about stage +y, and the scale limits on the stage
                        // scale: the grab runs on the stage pose.
                        // Q-248, Q-249
                        grab.update(pa, pb, &s.limits, !s.keep_upright)
                    }
                    _ => continue,
                };
                set_stage_pose(&mut q, &groups, e, new);
            }
            Intent::GrabEnd { hand } => {
                if hs.trim_drag(hand).is_some_and(|d| !d.by_select) {
                    hs.end_trim_drag(hand);
                    continue;
                }
                let Some((e, _)) = hs.hands.get_mut(&hand).and_then(|h| h.held.take()) else {
                    continue;
                };
                match hs.grabs.get(&e).copied() {
                    // Releasing one hand of two returns to a one-hand grab by the other.
                    Some((Grabbed::Two { .. }, Some((a, b)))) => {
                        let other = if a == hand { b } else { a };
                        let other_hold =
                            hs.hands.get(&other).and_then(|h| Some((h.pose?, h.point)));
                        match (other_hold, stage_pose(&q, &groups, e)) {
                            (Some((po, point)), Some(model)) => {
                                // The other hand's grab point stays its point (E11).
                                let grab = Grab::begin(po, model, model.transform_point(point));
                                hs.grabs
                                    .insert(e, (Grabbed::One { hand: other, grab }, None));
                            }
                            _ => {
                                hs.grabs.remove(&e);
                            }
                        }
                    }
                    _ => {
                        hs.grabs.remove(&e);
                    }
                }
            }
            Intent::PushPull { hand, axis } => {
                // Q-183
                let Some(h) = hs.hands.get(&hand) else {
                    continue;
                };
                let (Some((e, dist)), Some((_, dir))) = (h.held, h.ray) else {
                    continue;
                };
                if !matches!(hs.grabs.get(&e), Some((Grabbed::One { .. }, _))) {
                    continue;
                }
                let step = axis * s.push_pull_mps * dist * time.delta_secs();
                let Some(mut p) = stage_pose(&q, &groups, e) else {
                    continue;
                };
                p.translation += step * dir;
                set_stage_pose(&mut q, &groups, e, p);
                if let Some(h) = hs.hands.get_mut(&hand) {
                    h.held = Some((e, (dist + step).max(0.0)));
                }
                rebase(&mut hs, e, p);
            }
            Intent::Gizmo(mode) => {
                // §1.7: C only while the selected instance's trim is on.
                if mode == GizmoMode::Trim
                    && !sel
                        .and_then(|e| trim_of(&trim_now, &trims, e))
                        .is_some_and(|t| t.mode != TrimMode::Off)
                {
                    continue;
                }
                // E18: a gizmo mode change ends the mouse's drag; the same mode again is
                // no change.
                if mode != gizmo.mode {
                    hs.end_trim_drag(Hand::Mouse);
                }
                gizmo.mode = mode;
                gizmo.drag = None;
            }
            Intent::Nudge { instance, delta } => {
                if let Some(mut p) = stage_pose(&q, &groups, instance) {
                    p.translation += delta;
                    set_stage_pose(&mut q, &groups, instance, p);
                    rebase(&mut hs, instance, p);
                }
            }
            Intent::Turn { instance, yaw_deg } => {
                // Q-184
                if let Some(mut p) = stage_pose(&q, &groups, instance) {
                    p.rotation = Quat::from_rotation_y(yaw_deg.to_radians()) * p.rotation;
                    set_stage_pose(&mut q, &groups, instance, p);
                    rebase(&mut hs, instance, p);
                }
            }
            Intent::Instance { instance, action } => match action {
                InstanceAction::Select => {
                    if q.contains(instance) {
                        sel = Some(instance);
                        sel_changed = true;
                    }
                }
                InstanceAction::Deselect => {
                    if sel == Some(instance) {
                        sel = None;
                        sel_changed = true;
                    }
                }
                InstanceAction::Unload => {
                    library.write(LibraryCommand::Unload(instance));
                }
                InstanceAction::SetMode(mode) => {
                    if q.contains(instance) {
                        commands.entity(instance).insert(SplatMode(mode));
                    }
                }
                InstanceAction::SetVisible(v) => {
                    if q.contains(instance) {
                        let vis = if v {
                            Visibility::Visible
                        } else {
                            Visibility::Hidden
                        };
                        commands.entity(instance).insert(vis);
                    }
                }
                InstanceAction::ForgetPose => {
                    if let Ok((_, _, m, ..)) = q.get(instance) {
                        poses.0.forget(&m.identity);
                    }
                }
                InstanceAction::SavePose => {
                    // The pose relative to the content group.
                    // Q-247
                    if let Ok((_, t, m, _, x, ..)) = q.get(instance) {
                        let x = x.copied().unwrap_or(PoseExtras {
                            preset: Preset::YUp,
                            fit_size_m: FIT_DEFAULT_M,
                        });
                        let saved = SavedPose {
                            pose: pose_of(t),
                            preset: x.preset,
                            fit_size_m: x.fit_size_m,
                            saved_at: now_rfc3339(),
                            // §1.5 Poses: the trim too, as it stands (E17). Q-374
                            trim: trims.get(instance).ok().map(|tr| tr.0),
                        };
                        let id = m.identity.clone();
                        poses.0.set(&id, saved);
                    }
                }
                InstanceAction::ResetPose => {
                    // The saved pose is relative to the content group.
                    // Q-247
                    let Ok((_, mut t, m, ..)) = q.get_mut(instance) else {
                        continue;
                    };
                    let saved = poses.0.get(&m.identity).map(|sp| sp.pose);
                    *t = transform_of(saved.unwrap_or(Pose::IDENTITY));
                    if let Some(p) = stage_pose(&q, &groups, instance) {
                        rebase(&mut hs, instance, p);
                    }
                }
                InstanceAction::SetTrim(trim) => {
                    // Q-374: well-formed shapes whatever the mode (E17), so a trim the
                    // instance holds can always be saved (Q-372).
                    if q.contains(instance) && trim.is_valid() {
                        commands.entity(instance).insert(SplatTrim(trim));
                        trim_now.insert(instance, trim);
                    }
                }
                InstanceAction::ResetTrim => {
                    // E17: off, with the initial shapes of the untrimmed B_m. Q-374
                    if q.contains(instance)
                        && let Some(asset) =
                            models.get(instance).ok().and_then(|sm| assets.get(&sm.0))
                    {
                        let trim = Trim::initial(&asset.bounds);
                        commands.entity(instance).insert(SplatTrim(trim));
                        trim_now.insert(instance, trim);
                    }
                }
                InstanceAction::Fit | InstanceAction::SnapToFloor => {
                    // The fit's head and the floor are in stage space.
                    // §1.5: both on the fit box.
                    let (Some(b), Some(current), Ok((.., x, _, _, _))) = (
                        fit_box_of(&q, instance, s.fit_to_content),
                        stage_pose(&q, &groups, instance),
                        q.get(instance),
                    ) else {
                        continue;
                    };
                    let new = if action == InstanceAction::Fit {
                        let x = x.copied().unwrap_or(PoseExtras {
                            preset: Preset::YUp,
                            fit_size_m: FIT_DEFAULT_M,
                        });
                        fit_pose(
                            &b,
                            x.preset,
                            x.fit_size_m,
                            head_pose(&cameras).unwrap_or(Pose::IDENTITY),
                            AHEAD_M,
                        )
                    } else {
                        snap_to_floor(&b, current, 0.0)
                    };
                    set_stage_pose(&mut q, &groups, instance, new);
                    rebase(&mut hs, instance, new);
                }
            },
            _ => {}
        }
    }
    // Selection: one `Selected` at most.
    if sel_changed {
        for e in &selected {
            if Some(e) != sel {
                commands.entity(e).remove::<Selected>();
            }
        }
        if let Some(e) = sel
            && !selected.contains(e)
        {
            commands.entity(e).insert(Selected);
        }
    }
    // §1.7: the gizmo's trim mode returns to translate when the selected instance's trim
    // is off (Q-378).
    // Q-378
    let sel_trim_on = |now: &HashMap<Entity, Trim>, e: Entity| {
        trim_of(now, &trims, e).is_some_and(|t| t.mode != TrimMode::Off)
    };
    if gizmo.mode == GizmoMode::Trim
        && let Some(e) = sel
        && !sel_trim_on(&trim_now, e)
    {
        gizmo.mode = GizmoMode::Translate;
    }
    // E18's ends: deselected or gone, the trim turned off, reset or switched between box
    // and sphere, the instance grabbed, and the mouse's out of the gizmo's trim mode.
    let mode = gizmo.mode;
    hs.trim_drags.retain(|d| {
        sel == Some(d.entity)
            && trim_of(&trim_now, &trims, d.entity).is_some_and(|t| t.mode == d.start.mode)
            && (d.hand != Hand::Mouse || mode == GizmoMode::Trim)
    });
    // E18's drags: each drag point from its hand's last ray under the current pose, then
    // each dragged instance's trim, sent as `InstanceAction::SetTrim` when it changes;
    // `send_trim_drags` writes it after this system, and the next frame applies it.
    let rays: HashMap<Hand, (Vec3, Vec3)> = hs
        .hands
        .iter()
        .filter_map(|(h, st)| Some((*h, st.ray?)))
        .collect();
    for d in &mut hs.trim_drags {
        let (Some(pose), Some(&ray)) = (stage_pose(&q, &groups, d.entity), rays.get(&d.hand))
        else {
            continue;
        };
        if let Some(p) =
            trim_handles::drag_point(d.hand, d.handle, d.anchor, d.start_ray, d.t0, &pose, ray)
        {
            d.delta = pose.inverse().transform_point(p) - d.p0;
        }
    }
    let mut dragged: Vec<Entity> = hs.trim_drags.iter().map(|d| d.entity).collect();
    dragged.sort_unstable();
    dragged.dedup();
    hs.trim_sent.retain(|e, _| dragged.contains(e));
    for e in dragged {
        let drags: Vec<&TrimDrag> = hs.trim_drags.iter().filter(|d| d.entity == e).collect();
        let Some(t) = trim_handles::dragged_trim(&drags) else {
            continue;
        };
        let sent = hs
            .trim_sent
            .get(&e)
            .copied()
            .or(trim_of(&trim_now, &trims, e));
        if t.is_valid() && sent != Some(t) {
            sends.0.push(Intent::Instance {
                instance: e,
                action: InstanceAction::SetTrim(t),
            });
            hs.trim_sent.insert(e, t);
        }
    }
    // Hovered: recomputed each frame from that frame's `Pointer` intents, each hand's
    // last ray of the frame over the instances as they now stand, a hand without one
    // hovering nothing (Q-263, rewording Q-182); an instance two rays hit carries the
    // nearer (Q-182).
    let all = boxes(&q, &groups);
    let mut hovered: HashMap<Entity, Hovered> = HashMap::new();
    let mut touches: HashMap<Entity, TrimHandles> = HashMap::new();
    for d in &hs.trim_drags {
        touches
            .entry(d.entity)
            .or_default()
            .dragged
            .push((d.hand, d.handle));
    }
    pointed.sort_by_key(|h| *h as u8);
    for &hand in &pointed {
        let Some((o, d)) = hs.hands.get(&hand).and_then(|h| h.ray) else {
            continue;
        };
        // A dragging hand hovers nothing; a hand over a trim handle hovers the handle,
        // not the instance (E18). `pointed` is in hand order, so `hovered` is. Q-378
        if hs.trim_drag(hand).is_some() {
            continue;
        }
        if let Some((e, trim, pose)) =
            trim_target(sel, hand, gizmo.mode, &trim_now, &trims, &hs, &q, &groups)
            && let Some(size) = trim_handles::handle_size(&trim, &pose, eye.unwrap_or(o))
            && let Some((handle, _)) =
                trim_handles::handle_hit(&trim, &pose, size, (o, d), s.ray_max_m)
        {
            touches.entry(e).or_default().hovered.push((hand, handle));
            continue;
        }
        if let Some((e, distance)) = pick_instance(o, d, &all, s.ray_max_m)
            && hovered.get(&e).is_none_or(|v| distance < v.distance)
        {
            hovered.insert(e, Hovered { hand, distance });
        }
    }
    // Written only when they change, so `Changed` and the insert observers mean a change.
    for (e, _, _, _, _, h, g, _) in &q {
        let mut c = commands.entity(e);
        match (hovered.get(&e), h) {
            (Some(v), Some(old)) if v == old => {}
            (Some(v), _) => {
                c.insert(*v);
            }
            (None, Some(_)) => {
                c.remove::<Hovered>();
            }
            (None, None) => {}
        }
        match (hs.grabs.get(&e).map(|(g, _)| g), g) {
            (Some(new), Some(old)) if new == old => {}
            (Some(new), _) => {
                c.insert(*new);
            }
            (None, Some(_)) => {
                c.remove::<Grabbed>();
            }
            (None, None) => {}
        }
    }
    for (e, old) in &touched {
        if !touches.contains_key(&e) {
            commands.entity(e).remove::<TrimHandles>();
        } else if touches.get(&e) == Some(old) {
            touches.remove(&e);
        }
    }
    for (e, new) in touches {
        if let Ok(mut c) = commands.get_entity(e) {
            c.insert(new);
        }
    }
}

/// The ray's entry distance into a box centered at `c` of half-size `h`, if it hits.
fn ray_box(ray: (Vec3, Vec3), c: Vec3, h: f32) -> Option<f32> {
    let b = Aabb {
        min: c - Vec3::splat(h),
        max: c + Vec3::splat(h),
    };
    ray_hit(ray.0, ray.1, &b, &Pose::IDENTITY, f32::INFINITY)
}

/// The parameters (t on the ray, s on the line c + s a) of the closest points of a ray
/// and a line through `c` along the unit `a`; None when they are parallel.
fn closest_on_line(ray: (Vec3, Vec3), c: Vec3, a: Vec3) -> Option<(f32, f32)> {
    let (o, d) = ray;
    let w = o - c;
    let b = d.dot(a);
    let den = 1.0 - b * b;
    if den.abs() < 1e-6 {
        return None;
    }
    let (dw, aw) = (d.dot(w), a.dot(w));
    Some(((b * aw - dw) / den, (aw - b * dw) / den))
}

/// Where the ray crosses the plane through `c` with normal `n`, ahead of its origin.
fn on_plane(ray: (Vec3, Vec3), c: Vec3, n: Vec3) -> Option<Vec3> {
    let (o, d) = ray;
    let den = d.dot(n);
    if den.abs() < 1e-6 {
        return None;
    }
    let t = (c - o).dot(n) / den;
    (t >= 0.0).then(|| o + t * d)
}

/// The handle of `mode`'s set the ray hits first: arrows of length `size` along stage
/// x, y and z from the center, rings of radius `size` about them, or the cube of
/// half-size 0.1 `size` at the center; the pick tolerance is 0.08 `size`.
// Q-186
fn handle_hit(mode: GizmoMode, c: Vec3, size: f32, ray: (Vec3, Vec3)) -> Option<Handle> {
    let tol = 0.08 * size;
    let mut best: Option<(f32, Handle)> = None;
    let mut take = |t: f32, h: Handle| {
        if t >= 0.0 && best.is_none_or(|(b, _)| t < b) {
            best = Some((t, h));
        }
    };
    match mode {
        GizmoMode::Translate => {
            for a in AXES {
                if let Some((t, s)) = closest_on_line(ray, c, a)
                    && (0.0..=size).contains(&s)
                    && (ray.0 + t * ray.1).distance(c + s * a) <= tol
                {
                    take(t, Handle::Arrow(a));
                }
            }
        }
        GizmoMode::Rotate => {
            for a in AXES {
                if let Some(p) = on_plane(ray, c, a)
                    && ((p - c).length() - size).abs() <= tol
                {
                    take((p - ray.0).length(), Handle::Ring(a));
                }
            }
        }
        GizmoMode::Scale => {
            if let Some(t) = ray_box(ray, c, 0.1 * size + tol) {
                take(t, Handle::Cube);
            }
        }
        // E18's handles are `trim_handles`'.
        GizmoMode::Trim => {}
    }
    best.map(|(_, h)| h)
}

/// The instance's pose while a handle is dragged from `d.start_ray` to `ray`: along the
/// arrow's axis by the closest points' difference; about the ring's axis through the
/// center by the angle swept on its plane; scaled about the center by 1 + the change of
/// distance from the center on the plane facing the start ray, over `size`, within the
/// scale limits. A ray that gives no answer keeps the start pose.
// Q-186
fn drag_pose(d: &Drag, ray: (Vec3, Vec3), limits: &ScaleLimits) -> Pose {
    let (c, p0) = (d.center, d.start);
    match d.handle {
        Handle::Arrow(a) => {
            let (Some((_, s0)), Some((_, s))) = (
                closest_on_line(d.start_ray, c, a),
                closest_on_line(ray, c, a),
            ) else {
                return p0;
            };
            Pose {
                translation: p0.translation + (s - s0) * a,
                ..p0
            }
        }
        Handle::Ring(a) => {
            let (Some(u), Some(v)) = (on_plane(d.start_ray, c, a), on_plane(ray, c, a)) else {
                return p0;
            };
            let (u, v) = (u - c, v - c);
            let angle = f32::atan2(a.dot(u.cross(v)), u.dot(v));
            let r = Quat::from_axis_angle(a, angle);
            Pose {
                translation: c + r * (p0.translation - c),
                rotation: r * p0.rotation,
                scale: p0.scale,
            }
        }
        Handle::Cube => {
            let n = d.start_ray.1;
            let (Some(u), Some(v)) = (on_plane(d.start_ray, c, n), on_plane(ray, c, n)) else {
                return p0;
            };
            let f = 1.0 + ((v - c).length() - (u - c).length()) / d.size;
            let s = (p0.scale * f).clamp(limits.min, limits.max);
            let f = s / p0.scale;
            Pose {
                translation: c + f * (p0.translation - c),
                rotation: p0.rotation,
                scale: s,
            }
        }
    }
}

/// The posed box's twelve edges.
fn box_lines(gizmos: &mut Gizmos, b: &Aabb, pose: &Pose, color: Color) {
    let corner = |k: usize| {
        let pick = |bit: usize, lo: f32, hi: f32| if k & bit == 0 { lo } else { hi };
        pose.transform_point(Vec3::new(
            pick(1, b.min.x, b.max.x),
            pick(2, b.min.y, b.max.y),
            pick(4, b.min.z, b.max.z),
        ))
    };
    for k in 0..8 {
        for bit in [1, 2, 4] {
            if k & bit == 0 {
                gizmos.line(corner(k), corner(k | bit), color);
            }
        }
    }
}

const HOVER: Color = Color::srgb(0.85, 0.85, 0.85);
const ACCENT: Color = Color::srgb(1.0, 0.6, 0.1);

/// `[display] high_contrast` (Q-278) as `manipulate` reads it: `persist` sets it from the
/// settings, since `manipulate` comes before `persist` in §2.1's order.
// Q-278, Q-312
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HighContrast(pub(crate) bool);

/// The hovered and the selected boxes' colors (§1.4, §1.8 Comfort and accessibility):
/// light grey and the orange accent, or under high contrast the ray's yellow (Q-304) and
/// cyan, both fully saturated.
// Q-278, Q-312
pub(crate) fn highlight_colors(high_contrast: bool) -> (Color, Color) {
    if high_contrast {
        (Color::srgb(1.0, 1.0, 0.0), Color::srgb(0.0, 1.0, 1.0))
    } else {
        (HOVER, ACCENT)
    }
}

/// The trim shape's color (§1.4 Selection and debug views: distinct from the accent):
/// magenta, apart from the hover, accent and high-contrast colors. Q-374
const TRIM_COLOR: Color = Color::srgb(1.0, 0.2, 0.85);

/// The segments drawn per great circle of a sphere trim.
const CIRCLE_SEGMENTS: u32 = 64;

/// E17's shape of the selected instance in thin lines, through its pose: the box's edges,
/// or the sphere's three great circles in the model's axis planes; nothing when off.
fn trim_lines(gizmos: &mut Gizmos, trim: &Trim, pose: &Pose) {
    match trim.mode {
        TrimMode::Off => {}
        TrimMode::Box => {
            if let Some(b) = trim.shape_bounds() {
                box_lines(gizmos, &b, pose, TRIM_COLOR);
            }
        }
        TrimMode::Sphere => {
            for (u, v) in [(Vec3::X, Vec3::Y), (Vec3::Y, Vec3::Z), (Vec3::Z, Vec3::X)] {
                let at = |k: u32| {
                    let th = k as f32 * std::f32::consts::TAU / CIRCLE_SEGMENTS as f32;
                    pose.transform_point(trim.center + trim.radius * (th.cos() * u + th.sin() * v))
                };
                for k in 0..CIRCLE_SEGMENTS {
                    gizmos.line(at(k), at(k + 1), TRIM_COLOR);
                }
            }
        }
    }
}

type TrimHandleItem = (
    &'static Transform,
    &'static SplatTrim,
    Option<&'static TrimHandles>,
    Has<Grabbed>,
    Option<&'static ChildOf>,
);

/// E18's handles of the selected instance while its trim is on and it is not grabbed:
/// spheres of radius 0.1 L in the trim's color, the hovered or dragged ones in the hover
/// color; on the desktop only in the gizmo's trim mode, in XR whatever the mode.
// Q-378
fn draw_trim_handles(
    mut gizmos: Gizmos<TrimHandleGizmos>,
    q: Query<TrimHandleItem, (With<LoadedModel>, With<Selected>)>,
    groups: GroupTransforms,
    cameras: HeadCameras,
    gizmo: Res<GizmoState>,
    headset: Option<Res<Headset>>,
    contrast: Res<HighContrast>,
) {
    if headset.is_none() && gizmo.mode != GizmoMode::Trim {
        return;
    }
    let (hover, _) = highlight_colors(contrast.0);
    let eye = head_pose(&cameras).unwrap_or(Pose::IDENTITY).translation;
    for (t, trim, touched, grabbed, parent) in &q {
        let pose = staged(t, parent, &groups);
        let Some(size) = trim_handles::handle_size(&trim.0, &pose, eye) else {
            continue;
        };
        if grabbed {
            continue;
        }
        let lit = |h: TrimHandle| {
            touched.is_some_and(|x| x.hovered.iter().chain(&x.dragged).any(|(_, k)| *k == h))
        };
        for (h, p) in trim.0.handles() {
            let color = if lit(h) { hover } else { TRIM_COLOR };
            gizmos.sphere(
                pose.transform_point(p),
                trim_handles::HANDLE_RADIUS * size,
                color,
            );
        }
    }
}

const AXIS_COLORS: [Color; 3] = [
    Color::srgb(0.9, 0.2, 0.2),
    Color::srgb(0.2, 0.9, 0.2),
    Color::srgb(0.2, 0.4, 0.95),
];

type DrawItem = (
    &'static Transform,
    &'static SplatBounds,
    Option<&'static SplatTrim>,
    Has<Hovered>,
    Has<Selected>,
    Option<&'static ChildOf>,
);

/// §1.4: the hovered box in thin lines, the selected one in the accent color, the selected
/// instance's trim shape while it is on (E17), and its gizmo handles. The boxes are B_m,
/// trimmed while a trim is on.
fn draw(
    mut gizmos: Gizmos,
    q: Query<DrawItem, With<LoadedModel>>,
    groups: GroupTransforms,
    cameras: HeadCameras,
    gizmo: Res<GizmoState>,
    contrast: Res<HighContrast>,
) {
    let (hover, accent) = highlight_colors(contrast.0);
    let eye = head_pose(&cameras).unwrap_or(Pose::IDENTITY).translation;
    for (t, bounds, trim, hovered, selected, parent) in &q {
        let pose = staged(t, parent, &groups);
        if selected {
            box_lines(&mut gizmos, &bounds.aabb, &pose, accent);
        } else if hovered {
            box_lines(&mut gizmos, &bounds.aabb, &pose, hover);
        }
        if !selected {
            continue;
        }
        if let Some(trim) = trim {
            trim_lines(&mut gizmos, &trim.0, &pose);
        }
        let mode = gizmo.mode;
        let c = posed_center(&bounds.aabb, &pose);
        let size = gizmo_size(c, eye);
        for (a, color) in AXES.into_iter().zip(AXIS_COLORS) {
            match mode {
                GizmoMode::Translate => {
                    gizmos.arrow(c, c + size * a, color);
                }
                GizmoMode::Rotate => {
                    let (u, v) = a.any_orthonormal_pair();
                    let at = |k: u32| {
                        let th = k as f32 * std::f32::consts::TAU / 48.0;
                        c + size * (th.cos() * u + th.sin() * v)
                    };
                    for k in 0..48 {
                        gizmos.line(at(k), at(k + 1), color);
                    }
                }
                GizmoMode::Scale | GizmoMode::Trim => {}
            }
        }
        if mode == GizmoMode::Scale {
            let h = 0.1 * size;
            let cube = Aabb {
                min: c - Vec3::splat(h),
                max: c + Vec3::splat(h),
            };
            box_lines(&mut gizmos, &cube, &Pose::IDENTITY, accent);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;

    /// Q-310 (#530): a camera without a color target (`RenderTarget::None`) is never the
    /// head, even at a lower order than the window camera, which is the head.
    #[test]
    fn head_pose_skips_a_camera_without_a_color_target() {
        let mut world = World::new();
        world.spawn((
            Camera {
                order: -2,
                ..Default::default()
            },
            RenderTarget::None {
                size: bevy::math::UVec2::ONE,
            },
            GlobalTransform::from_xyz(5.0, 0.0, 0.0),
        ));
        world.spawn((
            Camera::default(),
            RenderTarget::Window(Default::default()),
            GlobalTransform::from_xyz(0.0, 1.6, 2.0),
        ));
        let head = world
            .run_system_once(|q: HeadCameras| head_pose(&q))
            .expect("the system runs");
        assert_eq!(head.map(|p| p.translation), Some(Vec3::new(0.0, 1.6, 2.0)));
    }

    /// Q-278, Q-312 (§1.8 Comfort: "the ray and highlight colors have a high-contrast
    /// alternative"): under high contrast the hovered and selected boxes are fully
    /// saturated yellow and cyan, distinct from each other and from the defaults.
    #[test]
    fn highlight_colors_have_a_high_contrast_alternative() {
        assert_eq!(highlight_colors(false), (HOVER, ACCENT));
        let (hover, selected) = highlight_colors(true);
        assert_eq!(hover, Color::srgb(1.0, 1.0, 0.0));
        assert_eq!(selected, Color::srgb(0.0, 1.0, 1.0));
        assert_ne!(hover, selected);
        assert_ne!((hover, selected), highlight_colors(false));
    }

    fn ray(o: Vec3, d: Vec3) -> (Vec3, Vec3) {
        (o, d.normalize())
    }

    #[test]
    fn arrows_hit_along_their_axis_and_miss_beside() {
        let c = Vec3::ZERO;
        let r = ray(Vec3::new(0.5, 0.0, 5.0), -Vec3::Z);
        assert_eq!(
            handle_hit(GizmoMode::Translate, c, 1.0, r),
            Some(Handle::Arrow(Vec3::X))
        );
        let miss = ray(Vec3::new(0.5, 0.5, 5.0), -Vec3::Z);
        assert_eq!(handle_hit(GizmoMode::Translate, c, 1.0, miss), None);
        let beyond = ray(Vec3::new(1.5, 0.0, 5.0), -Vec3::Z);
        assert_eq!(handle_hit(GizmoMode::Translate, c, 1.0, beyond), None);
    }

    #[test]
    fn rings_and_the_cube_hit() {
        let c = Vec3::ZERO;
        // Looking down −z at the point (1, 0) of the ring about z.
        let r = ray(Vec3::new(1.0, 0.0, 5.0), -Vec3::Z);
        assert_eq!(
            handle_hit(GizmoMode::Rotate, c, 1.0, r),
            Some(Handle::Ring(Vec3::Z))
        );
        let center = ray(Vec3::new(0.0, 0.0, 5.0), -Vec3::Z);
        assert_eq!(handle_hit(GizmoMode::Rotate, c, 1.0, center), None);
        assert_eq!(
            handle_hit(GizmoMode::Scale, c, 1.0, center),
            Some(Handle::Cube)
        );
    }

    fn drag(handle: Handle, start_ray: (Vec3, Vec3)) -> Drag {
        Drag {
            entity: Entity::PLACEHOLDER,
            handle,
            center: Vec3::ZERO,
            size: 1.0,
            start: Pose::IDENTITY,
            start_ray,
        }
    }

    const LIMITS: ScaleLimits = ScaleLimits {
        min: 0.01,
        max: 100.0,
    };

    #[test]
    fn an_arrow_drag_moves_along_its_axis_only() {
        let d = drag(
            Handle::Arrow(Vec3::X),
            ray(Vec3::new(0.5, 0.0, 5.0), -Vec3::Z),
        );
        let p = drag_pose(&d, ray(Vec3::new(0.8, 0.3, 5.0), -Vec3::Z), &LIMITS);
        assert!((p.translation - Vec3::new(0.3, 0.0, 0.0)).length() < 1e-5);
        assert_eq!(p.rotation, Quat::IDENTITY);
    }

    #[test]
    fn a_ring_drag_turns_by_the_swept_angle() {
        let d = drag(
            Handle::Ring(Vec3::Z),
            ray(Vec3::new(1.0, 0.0, 5.0), -Vec3::Z),
        );
        let p = drag_pose(&d, ray(Vec3::new(0.0, 1.0, 5.0), -Vec3::Z), &LIMITS);
        let want = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        assert!((p.rotation * Vec3::X - want * Vec3::X).length() < 1e-5);
    }

    #[test]
    fn a_cube_drag_scales_within_the_limits() {
        let d = drag(Handle::Cube, ray(Vec3::new(0.0, 0.0, 5.0), -Vec3::Z));
        let p = drag_pose(&d, ray(Vec3::new(0.5, 0.0, 5.0), -Vec3::Z), &LIMITS);
        assert!((p.scale - 1.5).abs() < 1e-5);
        let p = drag_pose(&d, ray(Vec3::new(500.0, 0.0, 5.0), -Vec3::Z), &LIMITS);
        assert_eq!(p.scale, 100.0);
    }

    #[test]
    fn saved_at_matches_known_dates() {
        // The pairs `ftgsxr_io`'s own formatter is tested with.
        assert_eq!(rfc3339(0, 0), "1970-01-01T00:00:00.000000000Z");
        assert_eq!(rfc3339(951_782_400, 5), "2000-02-29T00:00:00.000000005Z");
        assert_eq!(
            rfc3339(1_793_275_200, 123_456_789),
            "2026-10-29T12:00:00.123456789Z"
        );
        assert_eq!(rfc3339(4_107_542_399, 0), "2100-02-28T23:59:59.000000000Z");
    }

    #[test]
    fn saved_at_has_the_fixed_width_form() {
        let s = now_rfc3339();
        assert_eq!(s.len(), 30, "{s}");
        assert!(s.ends_with('Z') && s.as_bytes()[10] == b'T' && s.as_bytes()[19] == b'.');
    }
}

#[cfg(test)]
mod extras_tests {
    use super::*;

    #[test]
    fn set_pose_extras_raises_l_fit_to_its_minimum_and_sets_the_preset() {
        // §1.5 (revision 43): L_fit is at least 0.25 m, with no upper bound (#621).
        let mut app = App::new();
        app.add_message::<SetPoseExtras>()
            .add_systems(Update, set_pose_extras);
        let e = app.world_mut().spawn_empty().id();
        for (asked, kept) in [(20.0, 20.0), (0.1, 0.25), (2.0, 2.0)] {
            app.world_mut().write_message(SetPoseExtras {
                instance: e,
                preset: Preset::ZUp,
                fit_size_m: asked,
            });
            app.update();
            let x = *app.world().get::<PoseExtras>(e).unwrap();
            assert_eq!((x.preset, x.fit_size_m), (Preset::ZUp, kept));
        }
    }

    #[test]
    fn set_pose_extras_a_20_m_fit_size_scales_the_longest_side_to_20_m() {
        // §1.5 (revision 43): a room-scale fit, the box's longest side 2 m → 20 m (#621).
        let mut app = App::new();
        app.add_message::<SetPoseExtras>()
            .add_systems(Update, set_pose_extras);
        let e = app.world_mut().spawn_empty().id();
        app.world_mut().write_message(SetPoseExtras {
            instance: e,
            preset: Preset::YUp,
            fit_size_m: 20.0,
        });
        app.update();
        let x = *app.world().get::<PoseExtras>(e).unwrap();
        let b = Aabb {
            min: Vec3::new(-1.0, 0.0, -0.5),
            max: Vec3::new(1.0, 1.0, 0.5),
        };
        let p = fit_pose(&b, x.preset, x.fit_size_m, Pose::IDENTITY, AHEAD_M);
        assert!((p.scale - 10.0).abs() < 1e-5, "{}", p.scale);
    }

    #[test]
    fn set_pose_extras_takes_a_size_into_the_range() {
        // §1.5 (revision 46): a size outside [0.25 m, 10⁸ m], a typed "inf" included, is
        // taken as the nearer bound; a NaN leaves L_fit as it was; the preset is taken
        // either way (D-11, #630).
        let mut app = App::new();
        app.add_message::<SetPoseExtras>()
            .add_systems(Update, set_pose_extras);
        let e = app.world_mut().spawn_empty().id();
        for (preset, asked, kept) in [
            (Preset::YUp, 3.0, 3.0),
            (Preset::ZUp, f32::INFINITY, 1e8),
            (Preset::NegYUp, f32::NAN, 1e8),
            (Preset::YUp, 1.3e7, 1.3e7),
            (Preset::ZUp, 1e9, 1e8),
            (Preset::YUp, 0.1, 0.25),
            (Preset::NegYUp, f32::NEG_INFINITY, 0.25),
            (Preset::ZUp, f32::NAN, 0.25),
        ] {
            app.world_mut().write_message(SetPoseExtras {
                instance: e,
                preset,
                fit_size_m: asked,
            });
            app.update();
            let x = *app.world().get::<PoseExtras>(e).unwrap();
            assert_eq!((x.preset, x.fit_size_m), (preset, kept), "{asked}");
        }
        // Before the instance has an L_fit, §1.5's default of 1 m.
        let fresh = app.world_mut().spawn_empty().id();
        app.world_mut().write_message(SetPoseExtras {
            instance: fresh,
            preset: Preset::YUp,
            fit_size_m: f32::NAN,
        });
        app.update();
        assert_eq!(
            app.world().get::<PoseExtras>(fresh).unwrap().fit_size_m,
            1.0
        );
        // Two messages in one frame: the second keeps the first's size.
        for asked in [5.0, f32::NAN] {
            app.world_mut().write_message(SetPoseExtras {
                instance: e,
                preset: Preset::YUp,
                fit_size_m: asked,
            });
        }
        app.update();
        assert_eq!(app.world().get::<PoseExtras>(e).unwrap().fit_size_m, 5.0);
    }

    #[test]
    fn hints_a_checkpoint_suggests_2s_in_the_range() {
        // §1.5 (revision 46): L_fit = 2S taken into [0.25 m, 10⁸ m]; S = 5 → 10 m,
        // S = 0.05 → 0.25 m, S = 10⁸ → 10⁸ m, and 2S beyond f32 (∞) → 10⁸ m; 2S that
        // isn't a number gives the 1 m default (D-11, #630).
        let source = ModelRef::File(PathBuf::from("/nonexistent/ckpt"));
        for (s, l) in [
            (5.0, 10.0),
            (0.05, 0.25),
            (1.0, 2.0),
            (1e8, 1e8),
            (1e39, 1e8),
            (f64::NAN, 1.0),
        ] {
            let kind = ModelKind::Checkpoint {
                spatial_scale: s,
                frame_dt: 0.01,
            };
            assert_eq!(hints(&source, kind).fit_size_m, l, "S = {s}");
        }
    }
}
