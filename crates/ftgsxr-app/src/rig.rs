// SPDX-License-Identifier: Apache-2.0

//! The player rig, the content group, locomotion and the desktop camera (§2.1
//! `ftgsxr_app::rig`, E12, §1.7 Cameras, §1.8 Locomotion, §1.14 Frames, D-17, D-44).
//! `PlayerRig` and `ContentGroup` exist in every app; the desktop camera, a child of the
//! rig, only where a primary window exists (Q-194). The camera controller reads
//! `Intent::Camera`; `Locomote` and `SnapTurn` move the rig through `apply_locomotion`.
//! The floor grid (§1.4) is an opaque unlit line mesh at y = 0 beside the camera.

use bevy::app::{App, Plugin, Startup, Update};
use bevy::asset::{Assets, RenderAssetUsages};
use bevy::camera::visibility::Visibility;
use bevy::camera::{Camera, Camera3d, PerspectiveProjection, Projection};
use bevy::color::Color;
use bevy::ecs::prelude::*;
use bevy::input::ButtonInput;
use bevy::input::mouse::MouseButton;
use bevy::math::{Quat, Vec2, Vec3};
use bevy::mesh::{Mesh, Mesh3d, PrimitiveTopology};
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::time::Time;
use bevy::transform::components::{GlobalTransform, Transform};
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use ftgsxr_core::{Aabb, Pose};
use ftgsxr_io::{CameraState, LocomotionSettings};
use ftgsxr_render::{OutputSpace, SortMode, SplatAsset, SplatCamera, SplatLoad, SplatModel};

use crate::intent::{CameraCommand, Intent};
use crate::manipulate::Selected;
use crate::persist::SettingsRes;

/// The user's origin: the desktop camera's parent, the XR tracking space (§2.1).
#[derive(Component)]
pub struct PlayerRig;

/// Every instance's parent (§1.14).
#[derive(Component)]
pub struct ContentGroup;

/// The locomotion settings (§1.8, E12; §2.1).
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct Locomotion {
    /// E12's v_max in m/s.
    pub speed_mps: f32,
    /// Snap rather than smooth turn.
    pub snap: bool,
    /// E12's θ_snap in degrees.
    pub snap_deg: f32,
    /// E12's ω in degrees per second.
    pub smooth_dps: f32,
    /// The vignette during smooth motion.
    pub vignette: bool,
    /// Teleport instead of the smooth move.
    pub teleport: bool,
    /// The seated height offset in metres.
    pub height_offset_m: f32,
    /// Stick locomotion on; false in passthrough mode (D-17).
    pub enabled: bool,
}

// Q-195: D-44's defaults, and `enabled` from the environment's mode.
impl Default for Locomotion {
    fn default() -> Self {
        Locomotion::from_settings(&LocomotionSettings::default(), true)
    }
}

impl Locomotion {
    pub(crate) fn from_settings(s: &LocomotionSettings, enabled: bool) -> Locomotion {
        Locomotion {
            speed_mps: s.speed_mps,
            snap: s.snap_turn,
            snap_deg: s.snap_deg,
            smooth_dps: s.smooth_dps,
            vignette: s.vignette,
            teleport: s.teleport,
            height_offset_m: s.height_offset_m,
            enabled,
        }
    }
}

/// E12's response curve, pure: the dead zone and the squared response on a raw stick
/// value, per axis: r(x) = sign(x)·((|x| − d)/(1 − d))² for |x| > d, else 0.
pub fn stick_response(raw: Vec2, dead_zone: f32) -> Vec2 {
    let r = |x: f32| {
        if x.abs() <= dead_zone {
            0.0
        } else {
            let u = ((x.abs() - dead_zone) / (1.0 - dead_zone)).min(1.0);
            x.signum() * u * u
        }
    };
    Vec2::new(r(raw.x), r(raw.y))
}

/// E12's snap-turn trigger, pure: one step per crossing of 0.7 with a return below 0.3
/// between crossings. `Default` is armed (Q-195).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnapTurnState {
    armed: bool,
}

impl Default for SnapTurnState {
    fn default() -> Self {
        SnapTurnState { armed: true }
    }
}

impl SnapTurnState {
    /// The steps this turn-axis value gives: ±1 on a crossing of 0.7 while armed
    /// (positive to the right), else 0; a value below 0.3 in magnitude re-arms.
    pub fn step(&mut self, turn_axis: f32) -> i32 {
        if self.armed && turn_axis.abs() > 0.7 {
            self.armed = false;
            if turn_axis > 0.0 { 1 } else { -1 }
        } else {
            if turn_axis.abs() < 0.3 {
                self.armed = true;
            }
            0
        }
    }
}

/// E12, pure: the rig's new pose from the stick intents; `enabled` is false in
/// passthrough mode (D-17). `rig` and `head` are stage-space poses; turns are positive
/// to the right and keep the head's position; `Locomote.turn` turns only when `snap` is
/// off, and every other intent returns `rig` (Q-195).
// Q-195
pub fn apply_locomotion(
    rig: Pose,
    head: Pose,
    intent: &Intent,
    loco: &Locomotion,
    dt: f32,
) -> Pose {
    if !loco.enabled {
        return rig;
    }
    let turn_right_deg = match *intent {
        Intent::Locomote { move_xz, turn } => {
            let f = head.rotation * Vec3::NEG_Z;
            let yaw = (-f.x).atan2(-f.z);
            let v = loco.speed_mps
                * (Quat::from_rotation_y(yaw) * Vec3::new(move_xz.x, 0.0, -move_xz.y));
            let moved = Pose {
                translation: rig.translation + v * dt,
                ..rig
            };
            let deg = if loco.snap {
                0.0
            } else {
                loco.smooth_dps * turn * dt
            };
            return turn_about(moved, head.translation + v * dt, deg);
        }
        Intent::SnapTurn { steps } => steps as f32 * loco.snap_deg,
        _ => return rig,
    };
    turn_about(rig, head.translation, turn_right_deg)
}

/// The pose turned to the right by `deg` about the vertical through `p`.
fn turn_about(pose: Pose, p: Vec3, deg: f32) -> Pose {
    if deg == 0.0 {
        return pose;
    }
    let q = Quat::from_rotation_y(-deg.to_radians());
    Pose {
        translation: p + q * (pose.translation - p),
        rotation: (q * pose.rotation).normalize(),
        scale: pose.scale,
    }
}

// Q-196: the controller's units and constants.
/// Radians of look or orbit per logical pixel of mouse motion.
const LOOK_RAD_PER_PX: f32 = 0.003;
/// The pitch limit, short of straight up or down.
const PITCH_LIMIT: f32 = 89.0 * std::f32::consts::PI / 180.0;
/// The base speed's factor per wheel step, and its range (§1.7).
const SPEED_STEP: f32 = 1.25;
const SPEED_RANGE: (f32, f32) = (0.1, 20.0);
/// The orbit distance's factor per wheel step (zooming in for positive steps), and its
/// range (§1.7).
const ZOOM_STEP: f32 = 0.85;
const DISTANCE_RANGE: (f32, f32) = (0.1, 100.0);
/// The pan per logical pixel, as a fraction of the orbit distance.
const PAN_PER_PX: f32 = 0.002;
/// The engine's near and far planes (§1.7, Q-87).
const NEAR_M: f32 = 0.01;
const FAR_M: f32 = 1000.0;

/// The rig's and the content group's entities, spawned when the plugin builds.
#[derive(Resource, Clone, Copy)]
pub(crate) struct RigEntities {
    pub(crate) rig: Entity,
    pub(crate) group: Entity,
}

/// The desktop camera's entity.
#[derive(Component)]
pub(crate) struct DesktopCamera;

/// The desktop camera controller's state (§1.7 Cameras): the mode, the orbit's target
/// and distance, and the nine bookmarks the scene file keeps (`persist`, #285). The base
/// speed and the field of view are `SettingsRes.desktop`'s.
#[derive(Resource, Clone, Debug)]
pub(crate) struct CameraController {
    pub(crate) orbit: bool,
    pub(crate) target: Vec3,
    pub(crate) distance: f32,
    pub(crate) bookmarks: [Option<CameraState>; 9],
}

impl Default for CameraController {
    fn default() -> Self {
        let reset = reset_pose();
        CameraController {
            orbit: false,
            target: Vec3::ZERO,
            distance: reset.translation.length(),
            bookmarks: Default::default(),
        }
    }
}

/// §1.7's reset view: 1.6 m above the origin, 2 m back, looking at the origin.
pub(crate) fn reset_pose() -> Pose {
    let t = Transform::from_xyz(0.0, 1.6, 2.0).looking_at(Vec3::ZERO, Vec3::Y);
    Pose {
        translation: t.translation,
        rotation: t.rotation,
        scale: 1.0,
    }
}

/// The distance at which a sphere of radius `r` fills the narrower of the vertical
/// field of view `fov_y` and the horizontal one at `aspect` (Q-196).
pub(crate) fn frame_distance(r: f32, fov_y: f32, aspect: f32) -> f32 {
    let half_y = 0.5 * fov_y;
    let half_x = (half_y.tan() * aspect).atan();
    r / half_y.min(half_x).sin()
}

pub(crate) struct RigPlugin;

impl Plugin for RigPlugin {
    fn build(&self, app: &mut App) {
        let rig = app
            .world_mut()
            .spawn((PlayerRig, Transform::IDENTITY, Visibility::Visible))
            .id();
        let group = app
            .world_mut()
            .spawn((ContentGroup, Transform::IDENTITY, Visibility::Visible))
            .id();
        // Q-195: `persist` may have inserted it first.
        if !app.world().contains_resource::<Locomotion>() {
            let s = app
                .world()
                .get_resource::<SettingsRes>()
                .map(|s| s.0.clone());
            let s = s.unwrap_or_default();
            let enabled = s.environment.mode != "passthrough";
            app.insert_resource(Locomotion::from_settings(&s.locomotion, enabled));
        }
        app.insert_resource(RigEntities { rig, group })
            .init_resource::<CameraController>()
            .add_message::<Intent>()
            .add_observer(parent_instance)
            .add_systems(Startup, (spawn_desktop_camera, spawn_grid))
            .add_systems(
                Update,
                (run_camera, run_locomotion, follow_settings).chain(),
            )
            .add_systems(
                Update,
                capture_cursor.run_if(resource_exists::<ButtonInput<MouseButton>>),
            );
    }
}

/// Every instance under the content group (§1.14).
fn parent_instance(add: On<Add, SplatLoad>, rig: Res<RigEntities>, mut commands: Commands) {
    if let Ok(mut e) = commands.get_entity(add.entity) {
        e.try_insert(ChildOf(rig.group));
    }
}

fn camera_projection(fov_deg: f32) -> Projection {
    Projection::Perspective(PerspectiveProjection {
        fov: fov_deg.to_radians(),
        near: NEAR_M,
        far: FAR_M,
        ..Default::default()
    })
}

/// The desktop camera, where a primary window exists (Q-194).
// Q-194
fn spawn_desktop_camera(
    windows: Query<(), With<PrimaryWindow>>,
    rig: Res<RigEntities>,
    settings: Option<Res<SettingsRes>>,
    mut commands: Commands,
) {
    if windows.is_empty() {
        return;
    }
    let s = settings.map(|s| s.0.clone()).unwrap_or_default();
    let reset = reset_pose();
    commands.spawn((
        DesktopCamera,
        Camera3d::default(),
        camera_projection(s.desktop.fov_deg),
        SplatCamera {
            sort: SortMode::PerView,
            output: OutputSpace::LinearOver {
                background: Some(s.environment.background),
            },
        },
        Transform {
            translation: reset.translation,
            rotation: reset.rotation,
            scale: Vec3::ONE,
        },
        ChildOf(rig.rig),
    ));
}

/// The floor grid's half extent and line spacing in metres, and its color (Q-194).
const GRID_HALF_M: i32 = 10;
const GRID_STEP_M: f32 = 1.0;
const GRID_COLOR: Color = Color::srgb(0.3, 0.3, 0.3);

/// The floor grid at y = 0 (§1.4).
#[derive(Component)]
pub(crate) struct FloorGrid;

/// The grid's lines along x and z, 2·GRID_HALF_M + 1 of each.
pub(crate) fn grid_mesh() -> Mesh {
    let h = GRID_HALF_M as f32 * GRID_STEP_M;
    let mut points = Vec::new();
    for i in -GRID_HALF_M..=GRID_HALF_M {
        let c = i as f32 * GRID_STEP_M;
        points.extend([[c, 0.0, -h], [c, 0.0, h], [-h, 0.0, c], [h, 0.0, c]]);
    }
    Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, points)
}

/// The floor grid with the desktop camera, where the mesh and material stores exist
/// (Q-194); its visibility follows `display.grid`.
// Q-194
fn spawn_grid(
    windows: Query<(), With<PrimaryWindow>>,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
    settings: Option<Res<SettingsRes>>,
    mut commands: Commands,
) {
    let (Some(mut meshes), Some(mut materials)) = (meshes, materials) else {
        return;
    };
    if windows.is_empty() {
        return;
    }
    let shown = settings.map(|s| s.0.display.grid).unwrap_or(true);
    commands.spawn((
        FloorGrid,
        Mesh3d(meshes.add(grid_mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: GRID_COLOR,
            unlit: true,
            ..Default::default()
        })),
        Transform::IDENTITY,
        if shown {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        },
    ));
}

fn pose_of(t: &Transform) -> Pose {
    Pose {
        translation: t.translation,
        rotation: t.rotation,
        scale: t.scale.x,
    }
}

fn transform_of(p: Pose) -> Transform {
    Transform {
        translation: p.translation,
        rotation: p.rotation,
        scale: Vec3::splat(p.scale),
    }
}

/// The yaw (positive counterclockwise from above) and pitch of a camera rotation.
fn yaw_pitch(q: Quat) -> (f32, f32) {
    let f = q * Vec3::NEG_Z;
    ((-f.x).atan2(-f.z), f.y.clamp(-1.0, 1.0).asin())
}

fn from_yaw_pitch(yaw: f32, pitch: f32) -> Quat {
    Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT))
}

fn look_at(eye: Vec3, target: Vec3) -> Quat {
    if (target - eye).length_squared() < 1e-12 {
        return Quat::IDENTITY;
    }
    Transform::from_translation(eye)
        .looking_at(target, Vec3::Y)
        .rotation
}

type Instances<'w, 's> = Query<
    'w,
    's,
    (&'static Transform, &'static SplatModel),
    (With<Selected>, Without<DesktopCamera>, Without<PlayerRig>),
>;

/// The content group's transform, disjoint from the camera, rig and instance queries.
type Groups<'w, 's> = Query<
    'w,
    's,
    &'static Transform,
    (
        With<ContentGroup>,
        Without<DesktopCamera>,
        Without<PlayerRig>,
        Without<Selected>,
    ),
>;

/// The selected instance's posed box (B_m) as its eight stage-space corners.
fn selected_corners(
    selected: &Instances,
    groups: &Groups<'_, '_>,
    assets: &Assets<SplatAsset>,
) -> Option<[Vec3; 8]> {
    let (t, m) = selected.iter().next()?;
    let b: Aabb = assets.get(&m.0)?.bounds;
    let group = groups.iter().next().map(pose_of).unwrap_or(Pose::IDENTITY);
    let pose = pose_of(t).then(&group);
    let mut out = [Vec3::ZERO; 8];
    for (i, c) in out.iter_mut().enumerate() {
        let p = Vec3::new(
            if i & 1 == 0 { b.min.x } else { b.max.x },
            if i & 2 == 0 { b.min.y } else { b.max.y },
            if i & 4 == 0 { b.min.z } else { b.max.z },
        );
        *c = pose.transform_point(p);
    }
    Some(out)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_camera(
    mut intents: MessageReader<Intent>,
    mut ctl: ResMut<CameraController>,
    mut settings: Option<ResMut<SettingsRes>>,
    mut cameras: Query<
        (&mut Transform, &mut Projection),
        (With<DesktopCamera>, Without<PlayerRig>),
    >,
    rigs: Query<&Transform, (With<PlayerRig>, Without<DesktopCamera>)>,
    selected: Instances,
    groups: Groups<'_, '_>,
    assets: Option<Res<Assets<SplatAsset>>>,
    time: Res<Time>,
) {
    let Ok((mut local, mut projection)) = cameras.single_mut() else {
        intents.clear();
        return;
    };
    let rig = rigs.iter().next().map(pose_of).unwrap_or(Pose::IDENTITY);
    let mut pose = pose_of(&local).then(&rig);
    pose.scale = 1.0;
    let dt = time.delta_secs();
    let (fov_y, aspect) = match &*projection {
        Projection::Perspective(p) => (p.fov, p.aspect_ratio),
        _ => (60f32.to_radians(), 1.0),
    };
    for intent in intents.read() {
        let Intent::Camera(cmd) = intent else {
            continue;
        };
        match *cmd {
            CameraCommand::Move(v) if !ctl.orbit => {
                let speed = settings.as_ref().map_or(2.0, |s| s.0.desktop.fly_speed_mps);
                let (yaw, _) = yaw_pitch(pose.rotation);
                let d = Quat::from_rotation_y(yaw) * Vec3::new(v.x, 0.0, v.z) + Vec3::Y * v.y;
                pose.translation += d * speed * dt;
            }
            CameraCommand::Look(d) if !ctl.orbit => {
                let (yaw, pitch) = yaw_pitch(pose.rotation);
                pose.rotation =
                    from_yaw_pitch(yaw - d.x * LOOK_RAD_PER_PX, pitch - d.y * LOOK_RAD_PER_PX);
            }
            CameraCommand::Speed(steps) => {
                if let Some(s) = settings.as_mut() {
                    let v = s.0.desktop.fly_speed_mps * SPEED_STEP.powf(steps);
                    s.0.desktop.fly_speed_mps = v.clamp(SPEED_RANGE.0, SPEED_RANGE.1);
                }
            }
            CameraCommand::ToggleOrbit => {
                ctl.orbit = !ctl.orbit;
                if ctl.orbit {
                    ctl.target = assets
                        .as_ref()
                        .and_then(|a| selected_corners(&selected, &groups, a))
                        .map_or(Vec3::ZERO, |c| c.iter().sum::<Vec3>() / 8.0);
                    ctl.distance = pose
                        .translation
                        .distance(ctl.target)
                        .clamp(DISTANCE_RANGE.0, DISTANCE_RANGE.1);
                    pose.rotation = look_at(pose.translation, ctl.target);
                    pose.translation = ctl.target - pose.rotation * Vec3::NEG_Z * ctl.distance;
                }
            }
            CameraCommand::Orbit(d) if ctl.orbit => {
                let (yaw, pitch) = yaw_pitch(pose.rotation);
                pose.rotation =
                    from_yaw_pitch(yaw - d.x * LOOK_RAD_PER_PX, pitch - d.y * LOOK_RAD_PER_PX);
                pose.translation = ctl.target - pose.rotation * Vec3::NEG_Z * ctl.distance;
            }
            CameraCommand::Pan(d) if ctl.orbit => {
                let k = PAN_PER_PX * ctl.distance;
                let shift = pose.rotation * Vec3::new(-d.x * k, d.y * k, 0.0);
                ctl.target += shift;
                pose.translation += shift;
            }
            CameraCommand::Zoom(steps) if ctl.orbit => {
                ctl.distance = (ctl.distance * ZOOM_STEP.powf(steps))
                    .clamp(DISTANCE_RANGE.0, DISTANCE_RANGE.1);
                pose.translation = ctl.target - pose.rotation * Vec3::NEG_Z * ctl.distance;
            }
            CameraCommand::Frame => {
                let Some(c) = assets
                    .as_ref()
                    .and_then(|a| selected_corners(&selected, &groups, a))
                else {
                    continue;
                };
                let center = c.iter().sum::<Vec3>() / 8.0;
                let r = c
                    .iter()
                    .map(|p| p.distance(center))
                    .fold(0.0, f32::max)
                    .max(1e-3);
                let d = frame_distance(r, fov_y, aspect);
                pose.translation = center - pose.rotation * Vec3::NEG_Z * d;
                ctl.target = center;
                ctl.distance = d.clamp(DISTANCE_RANGE.0, DISTANCE_RANGE.1);
            }
            CameraCommand::ResetView => {
                pose = reset_pose();
                ctl.target = Vec3::ZERO;
                ctl.distance = pose.translation.length();
            }
            CameraCommand::SaveBookmark(n) if (1..=9).contains(&n) => {
                ctl.bookmarks[n as usize - 1] = Some(CameraState {
                    pose,
                    fov_deg: fov_y.to_degrees(),
                    orbit: ctl.orbit,
                });
            }
            CameraCommand::RecallBookmark(n) if (1..=9).contains(&n) => {
                let Some(b) = ctl.bookmarks[n as usize - 1].clone() else {
                    continue;
                };
                pose = Pose {
                    scale: 1.0,
                    ..b.pose
                };
                ctl.orbit = b.orbit;
                if ctl.orbit {
                    ctl.target = pose.translation + pose.rotation * Vec3::NEG_Z * ctl.distance;
                }
                if let Some(s) = settings.as_mut() {
                    s.0.desktop.fov_deg = b.fov_deg;
                }
                *projection = camera_projection(b.fov_deg);
                if let Projection::Perspective(p) = &mut *projection {
                    p.aspect_ratio = aspect;
                }
            }
            _ => {}
        }
    }
    let new_local = transform_of(pose.then(&rig.inverse()));
    if new_local != *local {
        *local = Transform {
            scale: Vec3::ONE,
            ..new_local
        };
    }
}

/// `Locomote` and `SnapTurn` move the rig (E12); the head is the active camera of the
/// lowest order, as manipulate's fit (Q-180), else the rig itself.
fn run_locomotion(
    mut intents: MessageReader<Intent>,
    loco: Res<Locomotion>,
    mut rigs: Query<&mut Transform, With<PlayerRig>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    time: Res<Time>,
) {
    let Ok(mut rig_t) = rigs.single_mut() else {
        intents.clear();
        return;
    };
    let dt = time.delta_secs();
    let mut rig = pose_of(&rig_t);
    let head0 = cameras
        .iter()
        .filter(|(c, _)| c.is_active)
        .min_by_key(|(c, _)| c.order)
        .map(|(_, g)| {
            let (_, r, t) = g.to_scale_rotation_translation();
            (
                rig.inverse(),
                Pose {
                    translation: t,
                    rotation: r,
                    scale: 1.0,
                },
            )
        });
    for intent in intents.read() {
        if !matches!(intent, Intent::Locomote { .. } | Intent::SnapTurn { .. }) {
            continue;
        }
        // The head follows the rig within the frame: its rig-local pose is fixed.
        let head = match head0 {
            Some((inv0, h)) => h.then(&inv0).then(&rig),
            None => rig,
        };
        rig = apply_locomotion(rig, head, intent, &loco, dt);
    }
    let t = transform_of(rig);
    if t != *rig_t {
        *rig_t = t;
    }
}

/// The settings' field of view and background on the desktop camera.
fn follow_settings(
    settings: Option<Res<SettingsRes>>,
    mut cameras: Query<(&mut Projection, &mut SplatCamera), With<DesktopCamera>>,
    mut grids: Query<&mut Visibility, With<FloorGrid>>,
) {
    let Some(s) = settings else { return };
    if !s.is_changed() {
        return;
    }
    let shown = if s.0.display.grid {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut v in &mut grids {
        if *v != shown {
            *v = shown;
        }
    }
    for (mut projection, mut camera) in &mut cameras {
        if let Projection::Perspective(p) = &mut *projection {
            let fov = s.0.desktop.fov_deg.to_radians();
            if p.fov != fov {
                p.fov = fov;
            }
        }
        let output = OutputSpace::LinearOver {
            background: Some(s.0.environment.background),
        };
        if camera.output != output {
            camera.output = output;
        }
    }
}

/// The cursor is captured while the right button holds the fly look (§1.7).
fn capture_cursor(
    buttons: Res<ButtonInput<MouseButton>>,
    ctl: Res<CameraController>,
    mut cursors: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let Ok(mut c) = cursors.single_mut() else {
        return;
    };
    if buttons.just_pressed(MouseButton::Right) && !ctl.orbit {
        c.grab_mode = CursorGrabMode::Locked;
        c.visible = false;
    } else if buttons.just_released(MouseButton::Right) {
        c.grab_mode = CursorGrabMode::None;
        c.visible = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_distance_fits_the_narrower_angle() {
        let fov = 60f32.to_radians();
        assert!((frame_distance(1.0, fov, 2.0) - 2.0).abs() < 1e-5);
        assert!(frame_distance(1.0, fov, 0.5) > 2.0);
    }

    #[test]
    fn grid_mesh_is_lines_at_y0() {
        let m = grid_mesh();
        assert_eq!(m.primitive_topology(), PrimitiveTopology::LineList);
        let n = 4 * (2 * GRID_HALF_M as usize + 1);
        assert_eq!(m.count_vertices(), n);
    }
}
