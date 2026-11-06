// SPDX-License-Identifier: Apache-2.0

//! Hands to `TrackedHand` (§1.8 Hands, Appendix D.3, D-25, Q-262): the joints and the
//! aim state of one `xrLocateHandJointsEXT` call per hand.

use std::ptr;
use std::task::Poll;

use bevy::ecs::message::MessageReader;
use bevy::log::{info, warn};
use bevy::math::Vec2;
use bevy::prelude::{Commands, Res, ResMut, Resource, Transform};
use bevy_mod_openxr::exts::OxrEnabledExtensions;
use bevy_mod_openxr::resources::{OxrFrameState, Pipelined};
use bevy_mod_openxr::session::OxrSession;
use bevy_mod_openxr::spaces::OxrSpaceExt;
use bevy_mod_xr::spaces::XrPrimaryReferenceSpace;
use ftgsxr_app::{
    AskPermissionAgain, DeniedPermissions, Permission, PermissionHandle, PlatformRes, Toast,
    ToastLevel,
};
use openxr::sys;

use super::{HandSource, TrackedHand, pose_of};
use crate::XrError;

/// The joint count of XR\_EXT\_hand\_tracking's default set (source-log 276).
pub(crate) const JOINTS: usize = openxr::HAND_JOINT_COUNT;

/// XR\_HAND\_JOINT\_WRIST\_EXT (source-log 276).
pub(crate) const WRIST: usize = 1;

/// The fields of `XrHandTrackingAimStateFB` the player reads (source-log 277), without
/// its chain pointer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AimState {
    pub(crate) status: sys::HandTrackingAimFlagsFB,
    pub(crate) pose: sys::Posef,
    pub(crate) index: f32,
    pub(crate) middle: f32,
}

/// One hand's locate: its joints and, with XR\_FB\_hand\_tracking\_aim, the aim state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RawHand {
    pub(crate) joints: [sys::HandJointLocationEXT; JOINTS],
    pub(crate) aim: Option<AimState>,
}

/// An OpenXR pose as a transform in the reference space.
pub(crate) fn transform_of(p: &sys::Posef) -> Transform {
    let o = p.orientation;
    Transform {
        translation: bevy::math::Vec3::new(p.position.x, p.position.y, p.position.z),
        rotation: bevy::math::Quat::from_xyzw(o.x, o.y, o.z, o.w),
        scale: bevy::math::Vec3::ONE,
    }
}

/// Whether a joint's position and orientation are both valid.
pub(crate) fn located(j: &sys::HandJointLocationEXT) -> bool {
    j.location_flags.contains(
        sys::SpaceLocationFlags::POSITION_VALID | sys::SpaceLocationFlags::ORIENTATION_VALID,
    )
}

/// A located hand as a `TrackedHand` in the reference space (Q-262): `aim` the aim state's
/// pose, `grip` the wrist joint's, `pinch` and `pinch_middle` the index and middle
/// strengths, `menu` MENU\_PRESSED, `tracked` VALID with the wrist located, the controller
/// fields zero, `source: Hand`. Without the aim state (a runtime without
/// XR\_FB\_hand\_tracking\_aim, Monado) `aim` is the wrist's pose, the strengths 0 and
/// `tracked` false.
// Q-262, Q-285
pub(crate) fn hand_state(raw: &RawHand) -> TrackedHand {
    let wrist = &raw.joints[WRIST];
    let grip = pose_of(&transform_of(&wrist.pose));
    let (aim, pinch, pinch_middle, menu, valid) = match &raw.aim {
        Some(a) => (
            pose_of(&transform_of(&a.pose)),
            a.index,
            a.middle,
            a.status.contains(sys::HandTrackingAimFlagsFB::MENU_PRESSED),
            a.status.contains(sys::HandTrackingAimFlagsFB::VALID),
        ),
        None => (grip, 0.0, 0.0, false, false),
    };
    TrackedHand {
        aim,
        grip,
        pinch,
        pinch_middle,
        squeeze: 0.0,
        trigger: 0.0,
        stick: Vec2::ZERO,
        primary: false,
        secondary: false,
        menu,
        source: HandSource::Hand,
        tracked: valid && located(wrist),
    }
}

/// The device that owns a side this frame (§1.8 Hands, Q-262 as #540 revised it), given
/// its controller and its hand (`None` when inactive): a held controller has priority, so
/// it owns the side whether or not the hand is tracked; without one the hand owns it.
/// "Held" is the controller's aim action being active on `/user/hand/<side>`, which
/// XR\_META\_detached\_controllers turns off for a controller the runtime sees put down
/// (D-55); without that extension an active controller counts as held.
// Q-262, D-55
pub(crate) fn side_owner(
    controller: Option<&TrackedHand>,
    hand: Option<&TrackedHand>,
) -> Option<HandSource> {
    match (controller, hand) {
        (Some(_), _) => Some(HandSource::Controller),
        (None, Some(_)) => Some(HandSource::Hand),
        (None, None) => None,
    }
}

/// `XrHandTrackingFrequencyHintMETA` (source-log 275): a C enum, so an `i32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct FrequencyHint(pub(crate) i32);

impl FrequencyHint {
    /// XR\_HAND\_TRACKING\_FREQUENCY\_HINT\_DEFAULT\_META.
    pub(crate) const DEFAULT: FrequencyHint = FrequencyHint(1);
    /// XR\_HAND\_TRACKING\_FREQUENCY\_HINT\_HIGH\_META.
    pub(crate) const HIGH: FrequencyHint = FrequencyHint(2);
}

/// The hint to send this frame, if it differs from the last one sent (§1.8 Hands: high
/// while hands are the active input, Q-275: while a hand owns a side; the default
/// otherwise). The first frame always sends, since the manifest's HIGH meta-data may have
/// started the runtime high.
// Q-275
pub(crate) fn hint_to_send(
    sent: Option<FrequencyHint>,
    hand_owns_a_side: bool,
) -> Option<FrequencyHint> {
    let want = if hand_owns_a_side {
        FrequencyHint::HIGH
    } else {
        FrequencyHint::DEFAULT
    };
    (sent != Some(want)).then_some(want)
}

/// XR\_META\_hand\_tracking\_frequency\_hint's name, NUL-terminated as `ExtensionSet.other`
/// holds names (source-log 271); enabled by name, since no pinned openxr release
/// carries it (Appendix E.1, Q-275).
// Q-275
pub(crate) const FREQUENCY_HINT_EXT: &[u8] = b"XR_META_hand_tracking_frequency_hint\0";

/// `xrSetHandTrackingFrequencyHintMETA(XrSession, XrHandTrackingFrequencyHintMETA)`
/// (source-log 274).
type SetHintFn = unsafe extern "system" fn(sys::Session, FrequencyHint) -> sys::Result;

/// Whether `XrPlugin.hands` asked for hands (PRD #452).
#[derive(Resource, Clone, Copy, Debug, Default)]
pub(crate) struct HandsWanted(pub(crate) bool);

/// The hand-tracking permission's state, kept across sessions so a re-created session
/// (E-XR-LOST, Q-124) does not ask again (Q-276).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Flow {
    /// Not asked yet.
    #[default]
    Idle,
    /// Asked; polled each frame.
    Asking(PermissionHandle),
    /// Granted: the trackers are created with each session.
    Granted,
    /// Denied: hands stay off until asked again.
    Denied,
}

/// The permission flow (Q-276).
// Q-287
#[derive(Resource, Clone, Copy, Debug, Default)]
pub(crate) struct HandsPermission {
    pub(crate) flow: Flow,
}

/// E-XR-PERMISSION's user text with the feature that asked (Q-261): "Permission p denied;
/// feature Y unavailable".
// Q-261
pub(crate) fn denied_text(p: Permission, feature: &str) -> String {
    let e = XrError::Permission(p);
    format!(
        "{}: Permission {p:?} denied; feature {feature} unavailable",
        e.code()
    )
}

/// What became of the simultaneous hands-and-controllers resume at the trackers'
/// creation (spike check 5, which Dv-8's runner reports, #463).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(any(test, target_os = "android")), allow(dead_code))]
pub(crate) enum Resume {
    /// XR\_META\_simultaneous\_hands\_and\_controllers is not enabled.
    Absent,
    /// The call succeeded.
    Resumed,
    /// The call failed with this result.
    Failed(sys::Result),
}

/// The session's hand trackers, the hint's loaded function and the last hint sent.
#[derive(Resource)]
pub(crate) struct HandTrackers {
    pub(crate) trackers: [Option<openxr::HandTracker>; 2],
    /// Whether the aim state is chained (XR\_FB\_hand\_tracking\_aim enabled).
    pub(crate) aim: bool,
    set_hint: Option<SetHintFn>,
    pub(crate) sent: Option<FrequencyHint>,
    #[cfg_attr(not(any(test, target_os = "android")), allow(dead_code))]
    pub(crate) resume: Resume,
}

impl HandTrackers {
    /// Whether the frequency hint's function is loaded (Q-275).
    #[cfg_attr(not(any(test, target_os = "android")), allow(dead_code))]
    pub(crate) fn has_hint(&self) -> bool {
        self.set_hint.is_some()
    }
}

/// This frame's locates, in the primary reference space (`None`: inactive or no tracker).
#[derive(Resource, Clone, Copy, Debug, Default)]
pub(crate) struct LocatedHands {
    pub(crate) raw: [Option<RawHand>; 2],
    /// This frame's locate calls and the failed ones among them (Dv-8's runner, #463).
    #[cfg_attr(not(any(test, target_os = "android")), allow(dead_code))]
    pub(crate) attempts: u32,
    #[cfg_attr(not(any(test, target_os = "android")), allow(dead_code))]
    pub(crate) failures: u32,
    warned: bool,
}

/// This frame's owner of each side (`side_owner`).
#[derive(Resource, Clone, Copy, Debug, Default)]
pub(crate) struct Ownership {
    pub(crate) owners: [Option<HandSource>; 2],
}

/// Asks for the permission through the platform; without one (a test app without
/// `AppPlugin`) the permission counts as granted, as `LinuxPlatform` grants every one.
// Q-276, Q-287
fn ask(platform: Option<&PlatformRes>) -> Flow {
    match platform {
        Some(p) => Flow::Asking(p.0.request_permission(Permission::HandTracking)),
        None => Flow::Granted,
    }
}

/// Whether the runtime offers hands as §1.8 uses them: XR\_EXT\_hand\_tracking for the
/// joints and XR\_FB\_hand\_tracking\_aim for the aim pose and the pinches. Without the
/// second (Monado, `play --xr`) hands are unavailable (§1.1): no permission is asked
/// and no tracker created, the missing extension's E-XR-UNSUPPORTED warning at plugin
/// build being the report (Q-285).
// Q-285
pub(crate) fn hands_available(exts: &OxrEnabledExtensions) -> bool {
    exts.ext_hand_tracking && exts.fb_hand_tracking_aim
}

/// At session creation, with `hands` and the hand extensions (`hands_available`): asks
/// for the permission the first time, creates the trackers once it is granted (Q-276).
// Q-285
pub(crate) fn on_session_created(
    wanted: Res<HandsWanted>,
    exts: Option<Res<OxrEnabledExtensions>>,
    session: Option<Res<OxrSession>>,
    platform: Option<Res<PlatformRes>>,
    mut permission: ResMut<HandsPermission>,
    mut commands: Commands,
) {
    if !wanted.0 || !exts.as_deref().is_some_and(hands_available) {
        return;
    }
    if permission.flow == Flow::Idle {
        permission.flow = ask(platform.as_deref());
    }
    if permission.flow == Flow::Granted
        && let (Some(session), Some(exts)) = (session, exts)
    {
        commands.insert_resource(create_trackers(&session, &exts));
    }
}

/// Polls a pending request each frame: a grant creates the trackers in the running
/// session, a denial leaves hands off with the warning and the toast naming "hands"
/// (Q-261, Q-221), and the permission listed in `DeniedPermissions` until it is asked
/// again; an `AskPermissionAgain(HandTracking)` after a denial (the Comfort tab's "ask
/// again") asks again (Q-276, Q-322).
// Q-276, Q-322
#[allow(clippy::too_many_arguments)]
pub(crate) fn poll_permission(
    exts: Option<Res<OxrEnabledExtensions>>,
    session: Option<Res<OxrSession>>,
    platform: Option<Res<PlatformRes>>,
    trackers: Option<Res<HandTrackers>>,
    mut again: MessageReader<AskPermissionAgain>,
    mut permission: ResMut<HandsPermission>,
    mut denied: Option<ResMut<DeniedPermissions>>,
    mut commands: Commands,
) {
    let mut listed = |on: bool| {
        if let Some(d) = denied.as_deref_mut() {
            let has = d.0.contains(&Permission::HandTracking);
            if on && !has {
                d.0.push(Permission::HandTracking);
            } else if !on && has {
                d.0.retain(|p| *p != Permission::HandTracking);
            }
        }
    };
    let asked_again = again
        .read()
        .filter(|a| a.0 == Permission::HandTracking)
        .count()
        > 0;
    if asked_again && permission.flow == Flow::Denied {
        permission.flow = ask(platform.as_deref());
        listed(false);
    }
    if let Flow::Asking(h) = permission.flow {
        let Some(platform) = platform.as_deref() else {
            return;
        };
        match platform.0.poll_permission(h) {
            Poll::Pending => return,
            Poll::Ready(true) => permission.flow = Flow::Granted,
            Poll::Ready(false) => {
                permission.flow = Flow::Denied;
                listed(true);
                let text = denied_text(Permission::HandTracking, "hands");
                warn!("{text}");
                commands.trigger(Toast {
                    text,
                    level: ToastLevel::Warning,
                });
                return;
            }
        }
    }
    if permission.flow == Flow::Granted
        && trackers.is_none()
        && let (Some(session), Some(exts)) = (session, exts)
    {
        commands.insert_resource(create_trackers(&session, &exts));
    }
}

/// The trackers go with their session.
pub(crate) fn destroy_trackers(mut commands: Commands, mut located: ResMut<LocatedHands>) {
    commands.remove_resource::<HandTrackers>();
    *located = LocatedHands::default();
}

/// One tracker per side; the simultaneous hands-and-controllers mode resumed when its
/// extension is enabled (spike check 5; else the runtime's own switching stands); the
/// frequency hint's function loaded when its extension is (Q-275).
fn create_trackers(session: &OxrSession, exts: &OxrEnabledExtensions) -> HandTrackers {
    let mut trackers = [None, None];
    for (slot, (hand, name)) in trackers
        .iter_mut()
        .zip([(openxr::Hand::LEFT, "left"), (openxr::Hand::RIGHT, "right")])
    {
        match session.create_hand_tracker(hand) {
            Ok(t) => *slot = Some(t),
            Err(e) => warn!("the {name} hand tracker could not be created: {e}"),
        }
    }
    info!("hand trackers created");
    let instance = session.instance();
    let mut resume_result = Resume::Absent;
    if let Some(fp) = instance
        .exts()
        .meta_simultaneous_hands_and_controllers
        .as_ref()
    {
        let resume = sys::SimultaneousHandsAndControllersTrackingResumeInfoMETA {
            ty: sys::SimultaneousHandsAndControllersTrackingResumeInfoMETA::TYPE,
            next: ptr::null(),
        };
        // SAFETY: the table was loaded for this instance with the extension enabled, the
        // session is this instance's and alive, and `resume` is a valid, unchained struct
        // that outlives the call.
        let r = unsafe {
            (fp.resume_simultaneous_hands_and_controllers_tracking)(session.as_raw(), &resume)
        };
        if r.into_raw() < 0 {
            warn!("simultaneous hands and controllers could not be resumed: {r}");
            resume_result = Resume::Failed(r);
        } else {
            info!("simultaneous hands and controllers resumed");
            resume_result = Resume::Resumed;
        }
    }
    let set_hint = if exts
        .0
        .raw()
        .other
        .iter()
        .any(|n| n.as_slice() == FREQUENCY_HINT_EXT)
    {
        load_set_hint(instance)
    } else {
        None
    };
    HandTrackers {
        trackers,
        aim: exts.fb_hand_tracking_aim,
        set_hint,
        sent: None,
        resume: resume_result,
    }
}

/// Loads `xrSetHandTrackingFrequencyHintMETA` through `xrGetInstanceProcAddr` (Q-275).
fn load_set_hint(instance: &openxr::Instance) -> Option<SetHintFn> {
    let mut f: Option<sys::pfn::VoidFunction> = None;
    // SAFETY: `get_instance_proc_addr` is the loader's, `instance` is a live instance, the
    // name is NUL-terminated and `f` is a valid out pointer.
    let r = unsafe {
        (instance.fp().get_instance_proc_addr)(
            instance.as_raw(),
            c"xrSetHandTrackingFrequencyHintMETA".as_ptr(),
            &mut f,
        )
    };
    if r.into_raw() < 0 {
        warn!("xrSetHandTrackingFrequencyHintMETA could not be loaded: {r}");
        return None;
    }
    // SAFETY: the runtime returned this pointer for xrSetHandTrackingFrequencyHintMETA,
    // whose prototype is `XrResult (XrSession, XrHandTrackingFrequencyHintMETA)`
    // (source-log 274); `SetHintFn` is that prototype, the enum as a transparent `i32`.
    f.map(|f| unsafe { std::mem::transmute::<sys::pfn::VoidFunction, SetHintFn>(f) })
}

/// One `xrLocateHandJointsEXT` call for `tracker` at `time` in `base`, the aim state
/// chained into the locations when `aim` (XR\_FB\_hand\_tracking\_aim enabled), so one
/// call gives the 26 joints and the aim state (PRD #452); `None` while the hand is
/// inactive. Shared with Dv-8's runner (#463).
pub(crate) fn locate_raw(
    instance: &openxr::Instance,
    tracker: &openxr::HandTracker,
    base: sys::Space,
    time: openxr::Time,
    aim: bool,
) -> openxr::Result<Option<RawHand>> {
    let Some(fp) = instance.exts().ext_hand_tracking.as_ref() else {
        return Err(sys::Result::ERROR_EXTENSION_NOT_PRESENT);
    };
    let mut joints = [sys::HandJointLocationEXT::default(); JOINTS];
    let mut state = sys::HandTrackingAimStateFB {
        ty: sys::HandTrackingAimStateFB::TYPE,
        next: ptr::null_mut(),
        status: sys::HandTrackingAimFlagsFB::EMPTY,
        aim_pose: sys::Posef::IDENTITY,
        pinch_strength_index: 0.0,
        pinch_strength_middle: 0.0,
        pinch_strength_ring: 0.0,
        pinch_strength_little: 0.0,
    };
    let next = if aim {
        (&raw mut state).cast::<std::ffi::c_void>()
    } else {
        ptr::null_mut()
    };
    let mut locations = sys::HandJointLocationsEXT {
        ty: sys::HandJointLocationsEXT::TYPE,
        next,
        is_active: false.into(),
        joint_count: JOINTS as u32,
        joint_locations: joints.as_mut_ptr(),
    };
    let info = sys::HandJointsLocateInfoEXT {
        ty: sys::HandJointsLocateInfoEXT::TYPE,
        next: ptr::null(),
        base_space: base,
        time,
    };
    // SAFETY: the table was loaded for this instance with XR_EXT_hand_tracking enabled;
    // the tracker was created by a session of this instance and is alive (borrowed);
    // `base` is that session's primary reference space; `locations` points at 26
    // writable joints, the default set's count the tracker was created with, and its
    // chain is null or the one aim state, which XR_FB_hand_tracking_aim (enabled when
    // `aim`) defines for this struct; every pointer outlives the call.
    let r = unsafe { (fp.locate_hand_joints)(tracker.as_raw(), &info, &mut locations) };
    if r.into_raw() < 0 {
        return Err(r);
    }
    let active: bool = locations.is_active.into();
    Ok(active.then_some(RawHand {
        joints,
        aim: aim.then_some(AimState {
            status: state.status,
            pose: state.aim_pose,
            index: state.pinch_strength_index,
            middle: state.pinch_strength_middle,
        }),
    }))
}

/// Each frame, one locate per hand at the crate's frame time: the predicted display time
/// plus one period when rendering is pipelined (source-log 273), in the primary
/// reference space. A failure is a warning, once until a locate succeeds again (§1.13's
/// rate limit).
pub(crate) fn locate_hands(
    session: Res<OxrSession>,
    trackers: Option<Res<HandTrackers>>,
    space: Option<Res<XrPrimaryReferenceSpace>>,
    frame: Option<Res<OxrFrameState>>,
    pipelined: Option<Res<Pipelined>>,
    mut located: ResMut<LocatedHands>,
) {
    located.raw = [None, None];
    located.attempts = 0;
    located.failures = 0;
    let (Some(trackers), Some(space), Some(frame)) = (trackers, space, frame) else {
        return;
    };
    let time = if pipelined.is_some() {
        openxr::Time::from_nanos(
            frame.predicted_display_time.as_nanos() + frame.predicted_display_period.as_nanos(),
        )
    } else {
        frame.predicted_display_time
    };
    let base = space.0.0.as_raw_openxr_space();
    for (h, tracker) in trackers.trackers.iter().enumerate() {
        let Some(tracker) = tracker else { continue };
        located.attempts += 1;
        match locate_raw(session.instance(), tracker, base, time, trackers.aim) {
            Ok(raw) => {
                located.raw[h] = raw;
                located.warned = false;
            }
            Err(e) => {
                located.failures += 1;
                if !located.warned {
                    located.warned = true;
                    warn!("the hand joints could not be located: {e}");
                }
            }
        }
    }
}

/// Sends the frequency hint when it changes (Q-275); nothing without the extension.
pub(crate) fn send_hint(
    session: Res<OxrSession>,
    ownership: Res<Ownership>,
    trackers: Option<ResMut<HandTrackers>>,
) {
    let Some(mut trackers) = trackers else { return };
    let Some(set_hint) = trackers.set_hint else {
        return;
    };
    let hand_owns = ownership.owners.contains(&Some(HandSource::Hand));
    let Some(hint) = hint_to_send(trackers.sent, hand_owns) else {
        return;
    };
    // SAFETY: `set_hint` was loaded from this session's instance with the extension
    // enabled (its valid usage), the session is alive, and `hint` is one of the enum's
    // two values (source-log 275).
    let r = unsafe { set_hint(session.as_raw(), hint) };
    if r.into_raw() < 0 {
        warn!("the hand-tracking frequency hint could not be set: {r}");
        // Not retried every frame.
    }
    trackers.sent = Some(hint);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn posef(x: f32, y: f32, z: f32) -> sys::Posef {
        sys::Posef {
            orientation: sys::Quaternionf::IDENTITY,
            position: sys::Vector3f { x, y, z },
        }
    }

    fn joints(flags: sys::SpaceLocationFlags) -> [sys::HandJointLocationEXT; JOINTS] {
        let mut j = [sys::HandJointLocationEXT::default(); JOINTS];
        for (i, joint) in j.iter_mut().enumerate() {
            joint.location_flags = flags;
            joint.pose = posef(i as f32, 0.0, 0.0);
            joint.radius = 0.01;
        }
        j
    }

    fn aim(status: sys::HandTrackingAimFlagsFB) -> AimState {
        AimState {
            status,
            pose: posef(0.1, 1.2, -0.3),
            index: 0.8,
            middle: 0.25,
        }
    }

    fn located_flags() -> sys::SpaceLocationFlags {
        sys::SpaceLocationFlags::POSITION_VALID | sys::SpaceLocationFlags::ORIENTATION_VALID
    }

    /// Q-262's mapping: the aim pose, the wrist joint as `grip`, the index and middle
    /// strengths, MENU\_PRESSED as `menu`, VALID with the wrist located as `tracked`.
    #[test]
    fn hand_state_maps_the_aim_state_and_the_wrist() {
        use sys::HandTrackingAimFlagsFB as F;
        let raw = RawHand {
            joints: joints(located_flags()),
            aim: Some(aim(F::COMPUTED | F::VALID | F::MENU_PRESSED)),
        };
        let h = hand_state(&raw);
        assert_eq!(h.source, HandSource::Hand);
        assert_eq!(h.aim.translation, ftgsxr_core::Vec3::new(0.1, 1.2, -0.3));
        // The wrist is joint 1 (XR_HAND_JOINT_WRIST_EXT), at x = 1 here.
        assert_eq!(h.grip.translation, ftgsxr_core::Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(h.pinch, 0.8);
        assert_eq!(h.pinch_middle, 0.25);
        assert!(h.menu);
        assert!(h.tracked);
        assert_eq!((h.trigger, h.squeeze, h.stick), (0.0, 0.0, Vec2::ZERO));
        assert!(!h.primary && !h.secondary);
    }

    /// `tracked` needs both VALID and a located wrist; `menu` follows MENU\_PRESSED only.
    #[test]
    fn hand_state_is_untracked_without_valid_or_a_located_wrist() {
        use sys::HandTrackingAimFlagsFB as F;
        let no_valid = RawHand {
            joints: joints(located_flags()),
            aim: Some(aim(F::COMPUTED | F::SYSTEM_GESTURE)),
        };
        let h = hand_state(&no_valid);
        assert!(!h.tracked);
        assert!(!h.menu, "SYSTEM_GESTURE alone is not MENU_PRESSED");
        let no_wrist = RawHand {
            joints: joints(sys::SpaceLocationFlags::POSITION_VALID),
            aim: Some(aim(F::VALID)),
        };
        assert!(!hand_state(&no_wrist).tracked);
    }

    /// Q-285: without the aim extension the aim is the wrist, untracked, no pinch.
    #[test]
    fn hand_state_without_the_aim_state_is_the_wrist_untracked() {
        let raw = RawHand {
            joints: joints(located_flags()),
            aim: None,
        };
        let h = hand_state(&raw);
        assert_eq!(h.aim, h.grip);
        assert_eq!((h.pinch, h.pinch_middle), (0.0, 0.0));
        assert!(!h.tracked && !h.menu);
    }

    fn controller() -> TrackedHand {
        TrackedHand {
            source: HandSource::Controller,
            tracked: true,
            ..hand_state(&RawHand {
                joints: joints(located_flags()),
                aim: None,
            })
        }
    }

    fn hand() -> TrackedHand {
        hand_state(&RawHand {
            joints: joints(located_flags()),
            aim: Some(aim(sys::HandTrackingAimFlagsFB::VALID)),
        })
    }

    /// Q-262 (#540): a held controller owns its side, whether or not the hand is
    /// tracked and whatever either does; without one the hand owns it; with neither, no
    /// one. A set-down controller is `None` here, its aim inactive under XR\_META\_detached\_controllers
    /// (D-55).
    #[test]
    fn side_owner_gives_a_held_controller_priority() {
        let (c, h) = (controller(), hand());
        assert_eq!(side_owner(None, None), None);
        assert_eq!(side_owner(Some(&c), None), Some(HandSource::Controller));
        assert_eq!(side_owner(None, Some(&h)), Some(HandSource::Hand));
        assert_eq!(side_owner(Some(&c), Some(&h)), Some(HandSource::Controller));
        // The hand's input does not take the side from a held controller: a full pinch,
        // the system gesture and a 10 cm aim move (Q-262's former inputs, #540's flip).
        let mut busy = h;
        busy.pinch = 1.0;
        busy.pinch_middle = 1.0;
        busy.menu = true;
        busy.aim.translation.x += 0.1;
        assert_eq!(
            side_owner(Some(&c), Some(&busy)),
            Some(HandSource::Controller)
        );
        // Nor does an idle, untracked controller lose it while still held.
        let idle = TrackedHand {
            tracked: false,
            ..c
        };
        assert_eq!(
            side_owner(Some(&idle), Some(&busy)),
            Some(HandSource::Controller)
        );
    }

    /// Q-275's state: high while a hand owns a side, the default otherwise, sent only on a
    /// change (the first frame included).
    #[test]
    fn hint_to_send_is_high_while_hands_own_a_side() {
        use FrequencyHint as H;
        assert_eq!(hint_to_send(None, false), Some(H::DEFAULT));
        assert_eq!(hint_to_send(None, true), Some(H::HIGH));
        assert_eq!(hint_to_send(Some(H::DEFAULT), false), None);
        assert_eq!(hint_to_send(Some(H::DEFAULT), true), Some(H::HIGH));
        assert_eq!(hint_to_send(Some(H::HIGH), true), None);
        assert_eq!(hint_to_send(Some(H::HIGH), false), Some(H::DEFAULT));
        // The registry's values (source-log 275).
        assert_eq!((H::DEFAULT.0, H::HIGH.0), (1, 2));
    }

    /// Q-261: the denied toast names the permission and the feature, "hands".
    #[test]
    fn denied_text_names_the_feature() {
        assert_eq!(
            denied_text(Permission::HandTracking, "hands"),
            "E-XR-PERMISSION: Permission HandTracking denied; feature hands unavailable"
        );
    }

    /// A platform that answers every permission request with `grant`, counting requests.
    struct Answers {
        paths: ftgsxr_app::AppPaths,
        grant: bool,
        asked: std::sync::Mutex<Vec<Permission>>,
    }

    impl ftgsxr_app::Platform for Answers {
        fn paths(&self) -> &ftgsxr_app::AppPaths {
            &self.paths
        }
        fn pick(&self, _: ftgsxr_app::PickKind) -> ftgsxr_app::PickHandle {
            ftgsxr_app::PickHandle(0)
        }
        fn poll_pick(
            &self,
            _: ftgsxr_app::PickHandle,
        ) -> Poll<Result<Vec<ftgsxr_app::ModelRef>, String>> {
            Poll::Pending
        }
        fn request_permission(&self, p: Permission) -> PermissionHandle {
            let mut asked = self.asked.lock().unwrap();
            asked.push(p);
            PermissionHandle(asked.len() as u32)
        }
        fn poll_permission(&self, _: PermissionHandle) -> Poll<bool> {
            Poll::Ready(self.grant)
        }
        fn document_provider(&self) -> Option<std::sync::Arc<dyn ftgsxr_app::DocumentProvider>> {
            None
        }
        fn camera_frames(&self) -> Option<std::sync::Arc<dyn ftgsxr_app::CameraFrames>> {
            None
        }
    }

    #[derive(Resource, Default)]
    struct Toasts(Vec<String>);

    fn flow_app(grant: bool) -> (bevy::app::App, std::sync::Arc<Answers>) {
        use bevy::prelude::On;
        let root = std::path::PathBuf::from("/nonexistent");
        let platform = std::sync::Arc::new(Answers {
            paths: ftgsxr_app::AppPaths {
                config: root.clone(),
                cache: root.clone(),
                models: root.clone(),
                scenes: root.clone(),
                screenshots: root.clone(),
                logs: root,
            },
            grant,
            asked: Default::default(),
        });
        let mut app = bevy::app::App::new();
        app.insert_resource(PlatformRes(platform.clone()))
            .insert_resource(HandsPermission {
                flow: Flow::Asking(PermissionHandle(1)),
            })
            .init_resource::<Toasts>()
            .init_resource::<DeniedPermissions>()
            .add_message::<AskPermissionAgain>()
            .add_observer(|t: On<Toast>, mut toasts: ResMut<Toasts>| {
                toasts.0.push(t.event().text.clone());
            })
            .add_systems(bevy::app::Update, poll_permission);
        (app, platform)
    }

    /// Q-276, Q-261: a denial leaves hands off with one toast naming "hands"; "ask again"
    /// asks the platform once more; a later grant moves the flow on.
    #[test]
    fn poll_permission_denied_toasts_hands_and_asks_again() {
        let (mut app, platform) = flow_app(false);
        app.update();
        assert_eq!(app.world().resource::<HandsPermission>().flow, Flow::Denied);
        assert_eq!(
            app.world().resource::<Toasts>().0,
            vec!["E-XR-PERMISSION: Permission HandTracking denied; feature hands unavailable"]
        );
        app.update();
        assert_eq!(
            app.world().resource::<Toasts>().0.len(),
            1,
            "one toast per denial"
        );
        assert!(platform.asked.lock().unwrap().is_empty());
        // Q-322: the denial is listed for the Comfort tab's "ask again".
        assert_eq!(
            app.world().resource::<DeniedPermissions>().0,
            [Permission::HandTracking]
        );
        // Another permission's "ask again" is not this one's.
        app.world_mut()
            .write_message(AskPermissionAgain(Permission::Camera));
        app.update();
        assert!(platform.asked.lock().unwrap().is_empty());
        app.world_mut()
            .write_message(AskPermissionAgain(Permission::HandTracking));
        app.update();
        assert_eq!(
            *platform.asked.lock().unwrap(),
            vec![Permission::HandTracking]
        );
        assert_eq!(app.world().resource::<HandsPermission>().flow, Flow::Denied);
        assert_eq!(app.world().resource::<Toasts>().0.len(), 2, "denied again");
    }

    /// A grant needs no toast; without a session no tracker is made yet (the session's
    /// creation makes them).
    #[test]
    fn poll_permission_granted_is_quiet() {
        let (mut app, _) = flow_app(true);
        app.update();
        assert_eq!(
            app.world().resource::<HandsPermission>().flow,
            Flow::Granted
        );
        assert!(app.world().resource::<Toasts>().0.is_empty());
        assert!(app.world().get_resource::<HandTrackers>().is_none());
    }

    /// Q-287: without a platform the permission counts as granted.
    #[test]
    fn ask_without_a_platform_is_granted() {
        assert_eq!(ask(None), Flow::Granted);
    }
}
