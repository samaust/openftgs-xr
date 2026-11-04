// SPDX-License-Identifier: Apache-2.0

//! Controllers to `XrHands` (§1.8 Controllers, Appendix D.2, §2.1). The action set
//! `player` with every Appendix D.2 action, created once per instance with the subaction
//! paths of both hands (a set cannot grow after attach); the three profiles' suggested
//! bindings, once per instance; the set attached at each session and synced each frame;
//! the aim and grip poses as action spaces, located each frame by the crate's spatial
//! plugin. Thresholds, haptic pulses and intents are `xr_input`'s. Hands (M-5, #459):
//! the permission, one tracker per side, one raw locate per hand per frame with the aim
//! state chained (`hands`), the per-side ownership rule, the frequency hint, and the hand
//! capsules and controller meshes in VR mode (`meshes`).

mod hands;
mod meshes;

pub(crate) use hands::FREQUENCY_HINT_EXT;

use std::borrow::Cow;

use bevy::app::{App, Plugin, PostUpdate, PreUpdate, Update};
use bevy::ecs::message::MessageWriter;
use bevy::math::{Isometry3d, Vec2};
use bevy::prelude::{
    Commands, Component, Entity, GlobalTransform, IntoScheduleConfigs, Query, Res, ResMut,
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
use ftgsxr_app::{HudData, SettingsRes};
use ftgsxr_core::{Pose, Quat, Vec3};
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
struct Actions {
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
    // Created now since the set cannot grow after attach; pulses are M-5's.
    #[allow(dead_code)]
    haptic: Action<Haptic>,
    hands: [Path; 2],
    suggested: bool,
}

const HANDS: [&str; 2] = ["/user/hand/left", "/user/hand/right"];

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
            .insert_resource(hands::HandsWanted(self.hands))
            .init_resource::<hands::HandsPermission>()
            .init_resource::<hands::LocatedHands>()
            .init_resource::<hands::Ownership>()
            .add_message::<hands::AskHandsAgain>();
        let Some(instance) = app.world().get_resource::<OxrInstance>() else {
            return;
        };
        let actions = match create_actions(instance) {
            Ok(a) => a,
            Err(e) => {
                log_error(&XrError::Runtime(format!("the action set `player`: {e}")));
                return;
            }
        };
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
            .add_systems(Update, hands::poll_permission)
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
                    write_hands,
                    hands::send_hint,
                    meshes::draw_hands_and_controllers,
                )
                    .chain()
                    .after(locate_views)
                    .after(TransformSystems::Propagate)
                    .run_if(openxr_session_running),
            );
    }
}

fn create_actions(instance: &OxrInstance) -> openxr::Result<Actions> {
    let hands = [
        instance.string_to_path(HANDS[0])?,
        instance.string_to_path(HANDS[1])?,
    ];
    let set = instance.create_action_set("player", "Player", 0)?;
    Ok(Actions {
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
    profiles.push((
        "/interaction_profiles/khr/simple_controller",
        simple_bindings(&actions),
    ));
    for (profile, bindings) in profiles {
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
    let (_, r, p) = root.mul_transform(*t).to_scale_rotation_translation();
    Pose {
        translation: Vec3::new(p.x, p.y, p.z),
        rotation: Quat::from_xyzw(r.x, r.y, r.z, r.w),
        scale: 1.0,
    }
}

/// A located hand's poses taken into the tracking root's frame, as the controllers'.
fn rooted(root: &GlobalTransform, mut hand: TrackedHand) -> TrackedHand {
    let t = |p: &Pose| Transform {
        translation: bevy::math::Vec3::new(p.translation.x, p.translation.y, p.translation.z),
        rotation: bevy::math::Quat::from_xyzw(
            p.rotation.x,
            p.rotation.y,
            p.rotation.z,
            p.rotation.w,
        ),
        scale: bevy::math::Vec3::ONE,
    };
    hand.aim = to_pose(root, &t(&hand.aim));
    hand.grip = to_pose(root, &t(&hand.grip));
    hand
}

/// `HudData`'s head field is false without a session (Q-274).
fn clear_hud(hud: Option<ResMut<HudData>>) {
    if let Some(mut hud) = hud {
        hud.head_tracked = false;
    }
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
    mut ownership: ResMut<hands::Ownership>,
    hud: Option<ResMut<HudData>>,
    mut hands: ResMut<XrHands>,
) {
    let root = root.iter().next().copied().unwrap_or_default();
    let mut out = [None, None];
    for (h, slot) in out.iter_mut().enumerate() {
        let path = actions.hands[h];
        // Q-128: a hand is `Some` while its aim pose action is active.
        if !actions.aim.is_active(&session, path).unwrap_or(false) {
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
    // Q-262: per side, the controller or the hand that owns it.
    let ownership = &mut *ownership;
    for (h, slot) in out.iter_mut().enumerate() {
        let hand = located.raw[h].map(|raw| rooted(&root, hands::hand_state(&raw)));
        let owner = ownership.sides[h].update(*slot, hand);
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
    if let Some(mut hud) = hud {
        hud.head_tracked = views_located;
    }
    hands.left_handed = settings.is_some_and(|s| s.0.locomotion.left_handed);
}
