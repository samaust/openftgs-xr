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
use bevy::gizmos::config::GizmoConfigStore;
use bevy::gizmos::gizmos::Gizmos;
use bevy::math::{Quat, Vec3};
use bevy::time::Time;
use bevy::transform::components::{GlobalTransform, Transform};
use ftgsxr_core::{
    Aabb, Grab, ModelKind, Pose, Preset, ScaleLimits, SceneManifest, TwoHandGrab, fit_pose,
    ray_hit, read_scene_manifest, snap_to_floor,
};
use ftgsxr_io::{ModelRef, PoseStore, SavedPose};
use ftgsxr_render::{SplatAsset, SplatMode, SplatModel};

use crate::intent::{GizmoMode, Hand, InstanceAction, Intent};
use crate::library::{LibraryCommand, LoadedModel};
use crate::platform::Headset;
use crate::playback::StepTarget;

/// What the grabs and rays obey (§1.5, §1.8, E10, E11).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct ManipulateSettings {
    /// E11's scale limits.
    pub limits: ScaleLimits,
    /// E10's t_max in metres.
    pub ray_max_m: f32,
    /// Two-hand grabs turn about y only (E11).
    pub keep_upright: bool,
    /// The push and pull speed per metre of distance, in m/s (§1.8).
    pub push_pull_mps: f32,
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
            push_pull_mps: 0.5,
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
            .init_resource::<HighContrast>()
            .add_observer(on_loaded)
            .add_message::<SetPoseExtras>()
            .add_systems(
                Update,
                (set_pose_extras, run_intents, step_target_follows_selection).chain(),
            )
            .add_systems(
                Update,
                draw.after(run_intents)
                    .run_if(resource_exists::<GizmoConfigStore>),
            );
    }
}

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
/// clamped to 0.25–4 m. The next fit and saved pose use them.
// Q-181
#[derive(Message, Clone, Copy, Debug, PartialEq)]
pub(crate) struct SetPoseExtras {
    pub(crate) instance: Entity,
    pub(crate) preset: Preset,
    pub(crate) fit_size_m: f32,
}

/// §1.5's L_fit range in metres.
pub(crate) const FIT_RANGE_M: (f32, f32) = (0.25, 4.0);

fn set_pose_extras(mut reader: MessageReader<SetPoseExtras>, mut commands: Commands) {
    for m in reader.read() {
        if let Ok(mut e) = commands.get_entity(m.instance) {
            e.try_insert(PoseExtras {
                preset: m.preset,
                fit_size_m: m.fit_size_m.clamp(FIT_RANGE_M.0, FIT_RANGE_M.1),
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
/// checkpoint's spatial scale, clamped to 0.25–4 m; else Y-up and 1 m.
// Q-181
fn hints(source: &ModelRef, kind: ModelKind) -> PoseExtras {
    let fit_size_m = match kind {
        ModelKind::Checkpoint { spatial_scale, .. } if spatial_scale.is_finite() => {
            ((2.0 * spatial_scale) as f32).clamp(0.25, 4.0)
        }
        _ => 1.0,
    };
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
/// the headset (the `Headset` marker, Q-266), one without a saved pose whose B_m's
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
    cameras: HeadCameras,
    mut commands: Commands,
) {
    let Ok((m, mut t, model, parent)) = q.get_mut(add.entity) else {
        return;
    };
    let extras = match poses.0.get(&m.identity) {
        Some(saved) => {
            *t = transform_of(saved.pose);
            PoseExtras {
                preset: saved.preset,
                fit_size_m: saved.fit_size_m,
            }
        }
        None => {
            let x = hints(&m.source, m.info.kind);
            let bounds = model
                .zip(assets.as_ref())
                .and_then(|(model, a)| a.get(&model.0))
                .map(|a| a.bounds);
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

/// §1.5 Poses' range of B_m's longest side, in metres, inside which the headset does not
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
        Option<&'static SplatModel>,
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

/// Every instance's (entity, stage pose, B_m); an instance whose asset is gone has no
/// box.
fn boxes(
    q: &Instances,
    groups: &GroupTransforms,
    assets: &Assets<SplatAsset>,
) -> Vec<(Entity, Pose, Aabb)> {
    q.iter()
        .filter_map(|(e, t, _, m, .., parent)| {
            Some((e, staged(t, parent, groups), assets.get(&m?.0)?.bounds))
        })
        .collect()
}

fn bounds_of(q: &Instances, assets: &Assets<SplatAsset>, e: Entity) -> Option<Aabb> {
    let (_, _, _, m, ..) = q.get(e).ok()?;
    Some(assets.get(&m?.0)?.bounds)
}

/// After a pose changed outside a grab, the grab takes hold again from the new pose.
// Q-183
fn rebase(hs: &mut HandStates, e: Entity, pose: Pose) {
    let Some(&(g, hands)) = hs.grabs.get(&e) else {
        return;
    };
    let next = match (g, hands) {
        (Grabbed::One { hand, .. }, _) => {
            hs.hands
                .get(&hand)
                .and_then(|h| h.pose)
                .map(|h| Grabbed::One {
                    hand,
                    grab: Grab::begin(h, pose),
                })
        }
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

#[allow(clippy::too_many_arguments)]
fn run_intents(
    mut intents: MessageReader<Intent>,
    mut q: Instances,
    groups: GroupTransforms,
    selected: Query<Entity, With<Selected>>,
    assets: Res<Assets<SplatAsset>>,
    cameras: HeadCameras,
    settings: Res<ManipulateSettings>,
    mut poses: ResMut<Poses>,
    mut hs: ResMut<HandStates>,
    mut gizmo: ResMut<GizmoState>,
    time: Res<Time>,
    mut library: MessageWriter<LibraryCommand>,
    mut commands: Commands,
) {
    let s = *settings;
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
    let pick = |q: &Instances, ray: Option<(Vec3, Vec3)>| {
        let (o, d) = ray?;
        pick_instance(o, d, &boxes(q, &groups, &assets), s.ray_max_m)
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
                    continue;
                }
                let ray = hs.hands.get(&hand).and_then(|h| h.ray);
                // The desktop gizmo's handles come before the instances behind them.
                if hand == Hand::Mouse
                    && let (mode, Some(e), Some(ray)) = (gizmo.mode, sel, ray)
                    && let Some(bounds) = bounds_of(&q, &assets, e)
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
                let (Some((e, dist)), Some((origin, dir))) = (pick(&q, ray), ray) else {
                    continue;
                };
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
                                    grab: Grab::begin(pose, model),
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
                    (Grabbed::One { grab, .. }, _) => grab.update(pose),
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
                let Some((e, _)) = hs.hands.get_mut(&hand).and_then(|h| h.held.take()) else {
                    continue;
                };
                match hs.grabs.get(&e).copied() {
                    // Releasing one hand of two returns to a one-hand grab by the other.
                    Some((Grabbed::Two { .. }, Some((a, b)))) => {
                        let other = if a == hand { b } else { a };
                        let po = hs.hands.get(&other).and_then(|h| h.pose);
                        match (po, stage_pose(&q, &groups, e)) {
                            (Some(po), Some(model)) => {
                                let grab = Grab::begin(po, model);
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
                            fit_size_m: 1.0,
                        });
                        let saved = SavedPose {
                            pose: pose_of(t),
                            preset: x.preset,
                            fit_size_m: x.fit_size_m,
                            saved_at: now_rfc3339(),
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
                InstanceAction::Fit | InstanceAction::SnapToFloor => {
                    // The fit's head and the floor are in stage space.
                    let (Some(b), Some(current), Ok((.., x, _, _, _))) = (
                        bounds_of(&q, &assets, instance),
                        stage_pose(&q, &groups, instance),
                        q.get(instance),
                    ) else {
                        continue;
                    };
                    let new = if action == InstanceAction::Fit {
                        let x = x.copied().unwrap_or(PoseExtras {
                            preset: Preset::YUp,
                            fit_size_m: 1.0,
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
    // Hovered: recomputed each frame from that frame's `Pointer` intents, each hand's
    // last ray of the frame over the instances as they now stand, a hand without one
    // hovering nothing (Q-263, rewording Q-182); an instance two rays hit carries the
    // nearer (Q-182).
    let all = boxes(&q, &groups, &assets);
    let mut hovered: HashMap<Entity, Hovered> = HashMap::new();
    for &hand in &pointed {
        let Some((o, d)) = hs.hands.get(&hand).and_then(|h| h.ray) else {
            continue;
        };
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
const AXIS_COLORS: [Color; 3] = [
    Color::srgb(0.9, 0.2, 0.2),
    Color::srgb(0.2, 0.9, 0.2),
    Color::srgb(0.2, 0.4, 0.95),
];

type DrawItem = (
    &'static Transform,
    &'static SplatModel,
    Has<Hovered>,
    Has<Selected>,
    Option<&'static ChildOf>,
);

/// §1.4: the hovered box in thin lines, the selected one in the accent color, and the
/// selected instance's gizmo handles.
fn draw(
    mut gizmos: Gizmos,
    q: Query<DrawItem, With<LoadedModel>>,
    groups: GroupTransforms,
    assets: Res<Assets<SplatAsset>>,
    cameras: HeadCameras,
    gizmo: Res<GizmoState>,
    contrast: Res<HighContrast>,
) {
    let (hover, accent) = highlight_colors(contrast.0);
    let eye = head_pose(&cameras).unwrap_or(Pose::IDENTITY).translation;
    for (t, m, hovered, selected, parent) in &q {
        let Some(asset) = assets.get(&m.0) else {
            continue;
        };
        let pose = staged(t, parent, &groups);
        if selected {
            box_lines(&mut gizmos, &asset.bounds, &pose, accent);
        } else if hovered {
            box_lines(&mut gizmos, &asset.bounds, &pose, hover);
        }
        if !selected {
            continue;
        }
        let mode = gizmo.mode;
        let c = posed_center(&asset.bounds, &pose);
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
                GizmoMode::Scale => {}
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
    fn set_pose_extras_clamps_l_fit_and_sets_the_preset() {
        let mut app = App::new();
        app.add_message::<SetPoseExtras>()
            .add_systems(Update, set_pose_extras);
        let e = app.world_mut().spawn_empty().id();
        for (asked, kept) in [(10.0, 4.0), (0.1, 0.25), (2.0, 2.0)] {
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
}
