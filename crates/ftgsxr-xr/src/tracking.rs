// SPDX-License-Identifier: Apache-2.0

//! Controllers to `XrHands` (§1.8 Controllers, Appendix D.2, §2.1). The action set
//! `player` with every Appendix D.2 action, created once per instance with the subaction
//! paths of both hands (a set cannot grow after attach); the three profiles' suggested
//! bindings, once per instance; the set attached at each session and synced each frame;
//! the aim and grip poses as action spaces, located each frame by the crate's spatial
//! plugin. Thresholds, haptic pulses and intents are `xr_input`'s. Hands (M-5, #459):
//! the permission, one tracker per side, one raw locate per hand per frame with the aim
//! state chained (`hands`), the per-side ownership rule, the frequency hint, and the hand
//! capsules and controller meshes in VR mode (`meshes`), and the depth-only gauntlets in
//! passthrough mode (`gauntlet`, §1.9).

// Dv-8's device side (#463, Q-308), built for the device and the host tests.
#[cfg(any(test, target_os = "android"))]
pub(crate) mod dv8;
// "Occlude with tracked hands" (§1.9, #608).
mod gauntlet;
pub(crate) mod hands;
mod meshes;

pub(crate) use gauntlet::GauntletsDrawn;
#[cfg(test)]
pub(crate) use gauntlet::{GauntletMaterial, add_material as add_gauntlet_material};
pub(crate) use hands::FREQUENCY_HINT_EXT;
pub(crate) use meshes::DevicesDrawn;

/// The set of the system that places the hands' and controllers' meshes, which
/// `passthrough`'s mode check runs before.
#[derive(bevy::prelude::SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DrawDevices;

use std::borrow::Cow;
use std::collections::VecDeque;

use bevy::app::{App, Plugin, PostUpdate, PreUpdate, Update};
use bevy::ecs::message::MessageWriter;
use bevy::log::info;
use bevy::math::{Isometry3d, Vec2};
use bevy::prelude::{
    Commands, Component, Entity, GlobalTransform, IntoScheduleConfigs, Local, Query, Res, ResMut,
    Resource, Transform, TransformSystems, With, Without,
};
use bevy_mod_openxr::action_binding::{OxrSendActionBindings, OxrSuggestActionBinding};
use bevy_mod_openxr::action_set_attaching::OxrAttachActionSet;
use bevy_mod_openxr::action_set_syncing::{OxrActionSetSyncSet, OxrSyncActionSet};
use bevy_mod_openxr::exts::OxrEnabledExtensions;
use bevy_mod_openxr::openxr_session_running;
use bevy_mod_openxr::render::locate_views;
use bevy_mod_openxr::resources::{OxrInstance, OxrViews};
use bevy_mod_openxr::session::OxrSession;
use bevy_mod_xr::session::{XrPreDestroySession, XrSessionCreated, XrTrackingRoot};
use bevy_mod_xr::spaces::XrSpaceLocationFlags;
use ftgsxr_app::{HudData, HudHandSource, SettingsRes};
use ftgsxr_core::Pose;
use openxr::{Action, ActionSet, Haptic, Path, Posef};

use crate::XrError;
use crate::error::log_error;

/// Where a hand's state comes from (§2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandSource {
    /// A controller through the action set `player` (Appendix D.2).
    Controller,
    /// Hand tracking (Appendix D.3, M-5).
    Hand,
}

/// One hand's state for a frame (§2.1, Q-128 for the controllers' fields).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackedHand {
    /// The aim pose (the ray), in stage space.
    pub aim: Pose,
    /// The grip pose, in stage space.
    pub grip: Pose,
    /// Index pinch strength (hands, M-5; 0 for a controller).
    pub pinch: f32,
    /// Middle-finger pinch strength (hands, M-5; 0 for a controller).
    pub pinch_middle: f32,
    /// The squeeze value (`grab`).
    pub squeeze: f32,
    /// The trigger value (`select`).
    pub trigger: f32,
    /// The thumbstick.
    pub stick: Vec2,
    /// A on the right hand, X on the left.
    pub primary: bool,
    /// B on the right hand, Y on the left.
    pub secondary: bool,
    /// The menu button (left hand).
    pub menu: bool,
    /// Where the state comes from.
    pub source: HandSource,
    /// Whether the aim pose's position and orientation are tracked.
    pub tracked: bool,
}

/// Both hands and the head, written every frame (§2.1).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct XrHands {
    /// The left hand, `None` while its controller is inactive.
    pub left: Option<TrackedHand>,
    /// The right hand, `None` while its controller is inactive.
    pub right: Option<TrackedHand>,
    /// The head, from the located views, in stage space.
    pub head: Pose,
    /// `[locomotion] left_handed` (Q-49).
    pub left_handed: bool,
}

impl Default for XrHands {
    fn default() -> Self {
        XrHands {
            left: None,
            right: None,
            head: Pose::IDENTITY,
            left_handed: false,
        }
    }
}

/// The set `player` and its actions (Appendix D.2), once per instance.
#[derive(Resource)]
pub(crate) struct Actions {
    set: ActionSet,
    select: Action<f32>,
    grab: Action<f32>,
    move_: Action<openxr::Vector2f>,
    turn: Action<f32>,
    push_pull: Action<f32>,
    menu: Action<bool>,
    play_pause: Action<bool>,
    reset_pose: Action<bool>,
    fit: Action<bool>,
    mode: Action<bool>,
    aim: Action<Posef>,
    grip: Action<Posef>,
    /// `xr_input`'s pulses (D-44) go through it.
    haptic: Action<Haptic>,
    /// With XR\_META\_detached\_controllers, a grip pose bound only on the detached
    /// paths: its bindings give the runtime a profile to move a set-down controller to,
    /// off `/user/hand/<side>` (D-55). Never located or read.
    // D-55
    detached: Option<Action<Posef>>,
    /// `DETACHED` as paths, with `detached`.
    detached_paths: Option<[Path; 2]>,
    hands: [Path; 2],
    suggested: bool,
}

const HANDS: [&str; 2] = ["/user/hand/left", "/user/hand/right"];

/// Appendix D.2's `khr/simple_controller` profile, which the tracked hand also drives on
/// the hand path on the Quest 3 (D-55).
const SIMPLE_CONTROLLER: &str = "/interaction_profiles/khr/simple_controller";

/// XR\_META\_detached\_controllers's top-level paths (D-55).
// D-55
const DETACHED: [&str; 2] = [
    "/user/detached_controller_meta/left",
    "/user/detached_controller_meta/right",
];

/// Which hand and which pose an action space entity locates.
#[derive(Component, Clone, Copy)]
pub(crate) struct HandSpace {
    hand: usize,
    aim: bool,
}

/// `tracking`'s systems; `plugin` adds it after the crate's action plugins. `hands` is
/// `XrPlugin.hands`.
pub(crate) struct TrackingPlugin {
    pub(crate) hands: bool,
}

impl Plugin for TrackingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<XrHands>()
            .init_resource::<HapticLog>()
            .init_resource::<DevicesDrawn>()
            .init_resource::<GauntletsDrawn>()
            .insert_resource(hands::HandsWanted(self.hands))
            .init_resource::<hands::HandsPermission>()
            .init_resource::<hands::LocatedHands>()
            .init_resource::<hands::Ownership>()
            .init_resource::<Profiles>()
            .init_resource::<ftgsxr_app::DeniedPermissions>()
            .add_message::<ftgsxr_app::AskPermissionAgain>();
        let Some(instance) = app.world().get_resource::<OxrInstance>() else {
            return;
        };
        // D-55: the detached paths only with their extension, since an unenabled
        // extension's paths are an error in `xrCreateAction` and the bindings.
        let detached = app
            .world()
            .get_resource::<OxrEnabledExtensions>()
            .is_some_and(|e| e.meta_detached_controllers);
        let actions = match create_actions(instance, detached) {
            Ok(a) => a,
            Err(e) => {
                log_error(&XrError::Runtime(format!("the action set `player`: {e}")));
                return;
            }
        };
        // §1.9: the gauntlets' material, where Bevy's `PbrPlugin` is in the group.
        let gauntlets = gauntlet::add_material(app);
        app.insert_resource(actions)
            .add_systems(OxrSendActionBindings, suggest_bindings)
            .add_systems(
                XrSessionCreated,
                (attach, spawn_spaces, hands::on_session_created),
            )
            .add_systems(
                XrPreDestroySession,
                (despawn_spaces, hands::destroy_trackers, clear_hud),
            )
            .add_systems(
                Update,
                (hands::poll_permission, hands::write_feature).chain(),
            )
            .add_systems(
                PreUpdate,
                sync.before(OxrActionSetSyncSet)
                    .run_if(openxr_session_running),
            )
            .add_systems(
                PostUpdate,
                // The spaces are located in PreUpdate's `XrSpaceSyncSet`; the views and the
                // hands here; the root's global transform (the rig binding, #461) after
                // propagation.
                (
                    hands::locate_hands,
                    read_profiles,
                    write_hands,
                    hands::send_hint,
                    meshes::draw_hands_and_controllers.in_set(DrawDevices),
                )
                    .chain()
                    .after(locate_views)
                    .after(TransformSystems::Propagate)
                    .run_if(openxr_session_running),
            );
        if gauntlets {
            app.add_systems(
                PostUpdate,
                gauntlet::draw_gauntlets
                    .in_set(DrawDevices)
                    .after(hands::locate_hands)
                    .after(TransformSystems::Propagate)
                    .run_if(openxr_session_running),
            );
        }
        // Dv-8 (#463): inert until the quest runner's entity names it (Q-308).
        #[cfg(target_os = "android")]
        dv8::add(app);
    }
}

impl Actions {
    /// One amplitude pulse on a hand's `haptic` output (§1.8 Controllers, D-44), side 0
    /// the left hand, at the runtime's frequency.
    pub(crate) fn pulse(
        &self,
        session: &OxrSession,
        side: usize,
        duration_ms: u32,
        amplitude: f32,
    ) -> openxr::Result<()> {
        let event = openxr::HapticVibration::new()
            .duration(openxr::Duration::from_nanos(
                i64::from(duration_ms) * 1_000_000,
            ))
            .frequency(openxr::FREQUENCY_UNSPECIFIED)
            .amplitude(amplitude);
        self.haptic
            .apply_feedback(session, self.hands[side], &event)
    }
}

/// One of D-44's pulses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Pulse {
    pub(crate) duration_ms: u32,
    pub(crate) amplitude: f32,
}

/// §1.8 Controllers, D-44: hover enter, a grab's start and end, a snap turn and a UI
/// click. `xr_input`'s haptics apply them through `Actions::pulse`.
pub(crate) const HOVER: Pulse = Pulse {
    duration_ms: 10,
    amplitude: 0.3,
};
pub(crate) const GRAB: Pulse = Pulse {
    duration_ms: 20,
    amplitude: 0.6,
};
pub(crate) const TICK: Pulse = Pulse {
    duration_ms: 15,
    amplitude: 0.5,
};

#[cfg(test)]
mod hud_tests {
    use super::*;
    use ftgsxr_core::Pose;

    /// D-55 (#540 on the Quest 3): a controller is held while its aim is active on the
    /// hand path and no profile is current on its detached path; a detached controller is
    /// not held even with the aim active, which the runtime keeps through
    /// `khr/simple_controller` driven by the tracked hand.
    #[test]
    fn controller_held_needs_the_aim_active_and_nothing_detached() {
        let touch = Path::from_raw(7);
        for hand_driven_profile in [None, Some(Path::from_raw(9))] {
            assert!(controller_held(
                true,
                touch,
                Path::NULL,
                hand_driven_profile
            ));
            assert!(!controller_held(true, touch, touch, hand_driven_profile));
            assert!(!controller_held(
                false,
                touch,
                Path::NULL,
                hand_driven_profile
            ));
            assert!(!controller_held(false, touch, touch, hand_driven_profile));
        }
    }

    /// D-55 (revision 33, #548 on the Quest 3): with the hand extensions enabled, a
    /// controller asleep or without its battery never reaches the detached path while the
    /// tracked hand drives `khr/simple_controller` on the hand path, so that profile is
    /// not a held controller; without the hand extensions every profile counts.
    #[test]
    fn controller_held_rejects_a_hand_driven_simple_controller() {
        let touch = Path::from_raw(7);
        let simple = Path::from_raw(9);
        assert!(!controller_held(true, simple, Path::NULL, Some(simple)));
        assert!(controller_held(true, touch, Path::NULL, Some(simple)));
        assert!(controller_held(true, simple, Path::NULL, None));
        assert!(controller_held(true, touch, Path::NULL, None));
    }

    /// Q-274: a side's HUD line is its device and whether its aim is tracked; none
    /// without an active device.
    #[test]
    fn hud_hand_maps_the_device_and_its_tracking() {
        let t = TrackedHand {
            aim: Pose::IDENTITY,
            grip: Pose::IDENTITY,
            pinch: 0.0,
            pinch_middle: 0.0,
            squeeze: 0.0,
            trigger: 0.0,
            stick: Vec2::ZERO,
            primary: false,
            secondary: false,
            menu: false,
            source: HandSource::Hand,
            tracked: false,
        };
        assert_eq!(hud_hand(Some(&t)), Some((HudHandSource::Hand, false)));
        let c = TrackedHand {
            source: HandSource::Controller,
            tracked: true,
            ..t
        };
        assert_eq!(hud_hand(Some(&c)), Some((HudHandSource::Controller, true)));
        assert_eq!(hud_hand(None), None);
    }
}

/// The haptics' record (D-44), which Dv-8 reads (Q-272): the pulses `xr_input` wanted on
/// a controller-owned side this frame, and the last calls of `Actions::pulse` it made,
/// numbered, with their outcomes.
#[derive(Resource, Default, Debug)]
pub(crate) struct HapticLog {
    /// This frame's pulses, by side (0 the left hand), before they are applied.
    pub(crate) wanted: Vec<(usize, Pulse)>,
    /// The next call's number.
    pub(crate) next: u64,
    /// The last `HAPTIC_LOG_LEN` calls: (number, side, pulse, outcome).
    pub(crate) applied: VecDeque<(u64, usize, Pulse, Result<(), String>)>,
}

/// How many applied pulses `HapticLog` keeps.
const HAPTIC_LOG_LEN: usize = 32;

impl HapticLog {
    /// Records one call of `Actions::pulse`.
    pub(crate) fn record(&mut self, side: usize, pulse: Pulse, result: Result<(), String>) {
        if self.applied.len() == HAPTIC_LOG_LEN {
            self.applied.pop_front();
        }
        self.applied.push_back((self.next, side, pulse, result));
        self.next += 1;
    }
}

fn create_actions(instance: &OxrInstance, with_detached: bool) -> openxr::Result<Actions> {
    let paths = |p: [&str; 2]| -> openxr::Result<[Path; 2]> {
        Ok([
            instance.string_to_path(p[0])?,
            instance.string_to_path(p[1])?,
        ])
    };
    let hands = paths(HANDS)?;
    let set = instance.create_action_set("player", "Player", 0)?;
    // D-55
    let detached_paths = with_detached.then(|| paths(DETACHED)).transpose()?;
    let detached = match &detached_paths {
        Some(p) => Some(set.create_action("detached_grip", "Detached controller pose", p)?),
        None => None,
    };
    Ok(Actions {
        detached,
        detached_paths,
        select: set.create_action("select", "Select", &hands)?,
        grab: set.create_action("grab", "Grab", &hands)?,
        move_: set.create_action("move", "Move", &hands)?,
        turn: set.create_action("turn", "Turn", &hands)?,
        push_pull: set.create_action("push_pull", "Push or pull", &hands)?,
        menu: set.create_action("menu", "Menu", &hands)?,
        play_pause: set.create_action("play_pause", "Play or pause", &hands)?,
        reset_pose: set.create_action("reset_pose", "Reset pose", &hands)?,
        fit: set.create_action("fit", "Fit", &hands)?,
        mode: set.create_action("mode", "Render mode", &hands)?,
        aim: set.create_action("aim", "Aim pose", &hands)?,
        grip: set.create_action("grip", "Grip pose", &hands)?,
        haptic: set.create_action("haptic", "Haptic", &hands)?,
        set,
        hands,
        suggested: false,
    })
}

/// The left hand's index in the bindings' hand lists.
const L: &[usize] = &[0];
/// Both hands.
const B: &[usize] = &[0, 1];

/// Appendix D.2's bindings of one Touch profile: (action, hands, path under the hand).
fn touch_bindings(a: &Actions) -> Vec<(openxr::sys::Action, &'static [usize], &'static str)> {
    const R: &[usize] = &[1];
    vec![
        (a.select.as_raw(), B, "input/trigger/value"),
        (a.grab.as_raw(), B, "input/squeeze/value"),
        (a.move_.as_raw(), L, "input/thumbstick"),
        (a.turn.as_raw(), R, "input/thumbstick/x"),
        (a.push_pull.as_raw(), B, "input/thumbstick/y"),
        (a.menu.as_raw(), L, "input/menu/click"),
        (a.play_pause.as_raw(), R, "input/a/click"),
        (a.reset_pose.as_raw(), R, "input/b/click"),
        (a.fit.as_raw(), L, "input/x/click"),
        (a.mode.as_raw(), L, "input/y/click"),
        (a.aim.as_raw(), B, "input/aim/pose"),
        (a.grip.as_raw(), B, "input/grip/pose"),
        (a.haptic.as_raw(), B, "output/haptic"),
    ]
}

/// Appendix D.2's `khr/simple_controller` column (menu on the left hand, as the Touch
/// columns; select, the poses and the haptic output on both).
fn simple_bindings(a: &Actions) -> Vec<(openxr::sys::Action, &'static [usize], &'static str)> {
    vec![
        (a.select.as_raw(), B, "input/select/click"),
        (a.menu.as_raw(), L, "input/menu/click"),
        (a.aim.as_raw(), B, "input/aim/pose"),
        (a.grip.as_raw(), B, "input/grip/pose"),
        (a.haptic.as_raw(), B, "output/haptic"),
    ]
}

/// Suggests the bindings once per instance: the crate runs this schedule at every
/// session creation, and suggesting after the first attach is an OpenXR error.
/// `meta/touch_controller_plus` only with XR\_META\_touch\_controller\_plus enabled
/// (Appendix E.1).
fn suggest_bindings(
    mut actions: ResMut<Actions>,
    exts: Option<Res<OxrEnabledExtensions>>,
    mut out: MessageWriter<OxrSuggestActionBinding>,
) {
    if actions.suggested {
        return;
    }
    actions.suggested = true;
    let mut profiles: Vec<(&'static str, _)> = Vec::new();
    if exts.is_some_and(|e| e.meta_touch_controller_plus) {
        profiles.push((
            "/interaction_profiles/meta/touch_controller_plus",
            touch_bindings(&actions),
        ));
    }
    profiles.push((
        "/interaction_profiles/oculus/touch_controller",
        touch_bindings(&actions),
    ));
    profiles.push((SIMPLE_CONTROLLER, simple_bindings(&actions)));
    for (profile, bindings) in profiles {
        // D-55: every profile here binds the grip pose, so the runtime accepts it on the
        // detached paths too.
        if let Some(detached) = &actions.detached {
            out.write(OxrSuggestActionBinding {
                action: detached.as_raw(),
                interaction_profile: Cow::Borrowed(profile),
                bindings: DETACHED
                    .iter()
                    .map(|d| Cow::Owned(format!("{d}/input/grip/pose")))
                    .collect(),
            });
        }
        for (action, hands, path) in bindings {
            out.write(OxrSuggestActionBinding {
                action,
                interaction_profile: Cow::Borrowed(profile),
                bindings: hands
                    .iter()
                    .map(|&h| Cow::Owned(format!("{}/{path}", HANDS[h])))
                    .collect(),
            });
        }
    }
}

/// Each side's current interaction profiles, read once a frame before `write_hands`
/// (profiles change only at `xrSyncActions`, in PreUpdate): on `/user/hand/<side>`, and on
/// `/user/detached_controller_meta/<side>` (`NULL` without the extension); and
/// `khr/simple_controller` as a path while the hand extensions are enabled, the profile
/// the tracked hand drives on the hand path (D-55).
#[derive(Resource, Clone, Copy, Debug, Default)]
struct Profiles {
    hand: [Path; 2],
    detached: [Path; 2],
    hand_driven_profile: Option<Path>,
}

/// Whether a side's controller is held (D-55): its aim action active on
/// `/user/hand/<side>`, no profile current on its detached path and, with the hand
/// extensions enabled (`hand_driven_profile` is then `khr/simple_controller`), the hand path's
/// profile not the one the tracked hand drives. On the Quest 3 a set-down controller's
/// Touch profile moves to the detached path while the tracked hand drives
/// `khr/simple_controller` on the hand path, so the aim stays active (#540); a controller
/// asleep or without its battery never reaches the detached path at all (#548).
// D-55
fn controller_held(
    aim_active: bool,
    hand_profile: Path,
    detached_profile: Path,
    hand_driven_profile: Option<Path>,
) -> bool {
    aim_active && detached_profile == Path::NULL && hand_driven_profile != Some(hand_profile)
}

/// Reads each side's current interaction profile on `/user/hand/<side>` and, with
/// XR\_META\_detached\_controllers, on `/user/detached_controller_meta/<side>`, keeps both
/// in `Profiles` with the hand-driven `khr/simple_controller` path while the hand
/// extensions are enabled, and logs both at the app's first frame and whenever either
/// changes, so a device run (A-11) shows where the runtime puts a set-down controller
/// (D-55). A failed read counts as no profile: on the detached path the controller then
/// stays held as without the extension, on the hand path it is not the hand-driven
/// profile; the log names the error.
// D-55
fn read_profiles(
    session: Res<OxrSession>,
    instance: Res<OxrInstance>,
    exts: Option<Res<OxrEnabledExtensions>>,
    actions: Res<Actions>,
    mut profiles: ResMut<Profiles>,
    mut last: Local<[Option<(Path, Path)>; 2]>,
    mut simple_controller_path: Local<Option<Option<Path>>>,
) {
    // Resolved once; a failure is logged once and leaves every profile counting.
    profiles.hand_driven_profile = if exts.as_deref().is_some_and(hands::hands_available) {
        *simple_controller_path.get_or_insert_with(|| {
            instance
                .string_to_path(SIMPLE_CONTROLLER)
                .map_err(|e| log_error(&XrError::Runtime(format!("{SIMPLE_CONTROLLER}: {e}"))))
                .ok()
        })
    } else {
        None
    };
    let name = |r: &openxr::Result<Path>| match r {
        Ok(p) if *p == Path::NULL => "none".to_owned(),
        Ok(p) => instance
            .path_to_string(*p)
            .unwrap_or_else(|e| format!("{e}")),
        Err(e) => format!("error {e}"),
    };
    for (side, label) in ["left", "right"].into_iter().enumerate() {
        let hand = session.current_interaction_profile(actions.hands[side]);
        let det = actions.detached_paths.map_or(Ok(Path::NULL), |d| {
            session.current_interaction_profile(d[side])
        });
        let now = (
            *hand.as_ref().unwrap_or(&Path::NULL),
            *det.as_ref().unwrap_or(&Path::NULL),
        );
        if last[side] != Some(now) {
            last[side] = Some(now);
            info!(
                "{label} interaction profile: hand {}, detached {}",
                name(&hand),
                name(&det)
            );
        }
        profiles.hand[side] = now.0;
        profiles.detached[side] = now.1;
    }
}

/// Attaches the set to each new session (attachment is per session).
fn attach(actions: Res<Actions>, mut out: MessageWriter<OxrAttachActionSet>) {
    out.write(OxrAttachActionSet(actions.set.clone()));
}

fn sync(actions: Res<Actions>, mut out: MessageWriter<OxrSyncActionSet>) {
    out.write(OxrSyncActionSet(actions.set.clone()));
}

/// The aim and grip action spaces of both hands, per session.
fn spawn_spaces(session: Option<Res<OxrSession>>, actions: Res<Actions>, mut commands: Commands) {
    let Some(session) = session else { return };
    for (hand, (&path, name)) in actions.hands.iter().zip(HANDS).enumerate() {
        for aim in [true, false] {
            let action = if aim { &actions.aim } else { &actions.grip };
            match session.create_action_space(action, path, Isometry3d::IDENTITY) {
                Ok(space) => {
                    commands.spawn((space, Transform::default(), HandSpace { hand, aim }));
                }
                Err(e) => log_error(&XrError::Runtime(format!(
                    "the {name} pose space (aim {aim}): {e}"
                ))),
            }
        }
    }
}

/// The action spaces go with their session.
fn despawn_spaces(spaces: Query<Entity, With<HandSpace>>, mut commands: Commands) {
    for e in &spaces {
        commands.entity(e).despawn();
    }
}

/// A located transform through the tracking root's global transform (the eyes' frame:
/// stage space, the rig the root is bound to and the LOCAL fallback's lift included;
/// Q-128, #461).
fn to_pose(root: &GlobalTransform, t: &Transform) -> Pose {
    let (_, rotation, translation) = root.mul_transform(*t).to_scale_rotation_translation();
    Pose {
        translation,
        rotation,
        scale: 1.0,
    }
}

/// A transform's translation and rotation as a `Pose` of scale 1 (a quad's scale is its
/// size, a located space's 1); the crate's one conversion.
pub(crate) fn pose_of(t: &Transform) -> Pose {
    Pose {
        translation: t.translation,
        rotation: t.rotation,
        scale: 1.0,
    }
}

/// A located hand's poses taken into the tracking root's frame, as the controllers'.
fn rooted(root: &GlobalTransform, mut hand: TrackedHand) -> TrackedHand {
    let t = |p: &Pose| Transform::from_translation(p.translation).with_rotation(p.rotation);
    hand.aim = to_pose(root, &t(&hand.aim));
    hand.grip = to_pose(root, &t(&hand.grip));
    hand
}

/// `HudData`'s XR fields without a session (Q-274): the head untracked, no hands.
fn clear_hud(hud: Option<ResMut<HudData>>) {
    if let Some(mut hud) = hud {
        hud.head_tracked = false;
        hud.left_hand = None;
        hud.right_hand = None;
    }
}

/// A side's HUD line (Q-274): its device and whether its aim is tracked.
// Q-274
pub(crate) fn hud_hand(hand: Option<&TrackedHand>) -> Option<(HudHandSource, bool)> {
    hand.map(|t| {
        let source = match t.source {
            HandSource::Controller => HudHandSource::Controller,
            HandSource::Hand => HudHandSource::Hand,
        };
        (source, t.tracked)
    })
}

/// Writes `XrHands` from the synced actions, the located spaces, the located hands and
/// the views: per side, the device that owns it (Q-262); and `HudData`'s head field.
#[allow(clippy::too_many_arguments)]
fn write_hands(
    session: Res<OxrSession>,
    actions: Res<Actions>,
    spaces: Query<(&HandSpace, &Transform, &XrSpaceLocationFlags), Without<XrTrackingRoot>>,
    root: Query<&GlobalTransform, With<XrTrackingRoot>>,
    views: Option<Res<OxrViews>>,
    settings: Option<Res<SettingsRes>>,
    located: Res<hands::LocatedHands>,
    profiles: Res<Profiles>,
    mut ownership: ResMut<hands::Ownership>,
    hud: Option<ResMut<HudData>>,
    mut hands: ResMut<XrHands>,
) {
    let root = root.iter().next().copied().unwrap_or_default();
    let mut out = [None, None];
    for (h, slot) in out.iter_mut().enumerate() {
        let path = actions.hands[h];
        // Q-128: a hand is `Some` while its aim pose action is active; D-55: and while
        // the controller is held, neither detached nor the tracked hand's profile.
        let aim_active = actions.aim.is_active(&session, path).unwrap_or(false);
        if !controller_held(
            aim_active,
            profiles.hand[h],
            profiles.detached[h],
            profiles.hand_driven_profile,
        ) {
            continue;
        }
        let f32_of = |a: &Action<f32>| a.state(&session, path).map_or(0.0, |s| s.current_state);
        let bool_of = |a: &Action<bool>| a.state(&session, path).is_ok_and(|s| s.current_state);
        let mut aim = Pose::IDENTITY;
        let mut grip = Pose::IDENTITY;
        let mut tracked = false;
        for (space, t, flags) in &spaces {
            if space.hand != h {
                continue;
            }
            if space.aim {
                aim = to_pose(&root, t);
                tracked = flags.position_tracked && flags.rotation_tracked;
            } else {
                grip = to_pose(&root, t);
            }
        }
        // Q-128: the left stick is `move`, the right one `turn` and `push_pull`.
        let stick = if h == 0 {
            actions.move_.state(&session, path).map_or(Vec2::ZERO, |s| {
                Vec2::new(s.current_state.x, s.current_state.y)
            })
        } else {
            Vec2::new(f32_of(&actions.turn), f32_of(&actions.push_pull))
        };
        let (primary, secondary) = if h == 0 {
            (bool_of(&actions.fit), bool_of(&actions.mode))
        } else {
            (bool_of(&actions.play_pause), bool_of(&actions.reset_pose))
        };
        *slot = Some(TrackedHand {
            aim,
            grip,
            pinch: 0.0,
            pinch_middle: 0.0,
            squeeze: f32_of(&actions.grab),
            trigger: f32_of(&actions.select),
            stick,
            primary,
            secondary,
            menu: bool_of(&actions.menu),
            source: HandSource::Controller,
            tracked,
        });
    }
    // Q-262, D-55: per side, the held controller, else the hand.
    for (h, slot) in out.iter_mut().enumerate() {
        let hand = located.raw[h].map(|raw| rooted(&root, hands::hand_state(&raw)));
        let owner = hands::side_owner(slot.as_ref(), hand.as_ref());
        ownership.owners[h] = owner;
        if owner == Some(HandSource::Hand) {
            *slot = hand;
        }
    }
    let [left, right] = out;
    hands.left = left;
    hands.right = right;
    // The head: the views' midpoint and mean orientation.
    let views_located = views.as_ref().is_some_and(|v| v.len() >= 2);
    if let Some(views) = views.filter(|v| v.len() >= 2) {
        let p = |v: &openxr::View| {
            let o = v.pose.orientation;
            Transform {
                translation: bevy::math::Vec3::new(
                    v.pose.position.x,
                    v.pose.position.y,
                    v.pose.position.z,
                ),
                rotation: bevy::math::Quat::from_xyzw(o.x, o.y, o.z, o.w),
                scale: bevy::math::Vec3::ONE,
            }
        };
        let (a, b) = (p(&views[0]), p(&views[1]));
        let mid = Transform {
            translation: (a.translation + b.translation) * 0.5,
            rotation: a.rotation.slerp(b.rotation, 0.5),
            scale: bevy::math::Vec3::ONE,
        };
        hands.head = to_pose(&root, &mid);
    }
    // Q-274, Q-289: the head counts as tracked once the views are located (the crate keeps the
    // last views and drops the view-state flags, source-log 273).
    hands.left_handed = settings.is_some_and(|s| s.0.locomotion.left_handed);
    if let Some(mut hud) = hud {
        let now = (
            views_located,
            hud_hand(hands.left.as_ref()),
            hud_hand(hands.right.as_ref()),
        );
        if (hud.head_tracked, hud.left_hand, hud.right_hand) != now {
            (hud.head_tracked, hud.left_hand, hud.right_hand) = now;
        }
    }
}
