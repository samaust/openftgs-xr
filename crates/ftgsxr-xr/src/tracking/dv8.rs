// SPDX-License-Identifier: Apache-2.0

//! Dv-8 on the device (§3.4, §1.8 Hands and Controllers, D-25, D-44, Appendix D.3), as
//! Q-272 reads it without a person. It lives here, not in `ftgsxr-quest`, because it
//! needs `tracking`'s crate-private items (the trackers, the raw locates, the haptic
//! output); the quest runner starts it by spawning an entity named [`TRIGGER`]
//! (Q-308), and it prints the verdict line `ftgsxr-test: dv8 PASS|FAIL <detail>`
//! itself (§3.4, Q-122).
//!
//! The clauses, in order: (1) the joints: both trackers exist and every locate of a 5 s
//! window succeeds, reporting whether any frame saw an active hand, the aim state's
//! COMPUTED and VALID flags seen (PRD #452's spike check 2), the simultaneous
//! hands-and-controllers resume (spike check 5) and the wrist's rate seen under each
//! frequency hint (Q-275, Q-309); (2) the pulse: a grab start on the active
//! controller's side (a `Grabbed` for that hand, the event `xr_input`'s haptics read),
//! which must reach D-44's 20 ms grab pulse on its haptic output with the call
//! succeeding, "no controller" when none is active; (3) the pinch: a synthetic index pinch of 0.8 on the right hand, written into
//! `XrHands` after `tracking` wrote it, makes `xr_input`'s system write `Select`.

use std::time::{Duration, Instant};

use bevy::app::{App, PostUpdate};
use bevy::ecs::message::MessageReader;
use bevy::log::info;
use bevy::prelude::{
    Commands, Entity, IntoScheduleConfigs, Name, Query, Res, ResMut, Resource, not, resource_exists,
};
use bevy_mod_openxr::exts::OxrEnabledExtensions;
use ftgsxr_app::{Grabbed, Hand, Intent, PanelPointer};
use ftgsxr_core::{Grab, Pose, Quat, Vec3};
use openxr::sys;

use super::hands::{
    Flow, FrequencyHint, HandTrackers, HandsPermission, HandsWanted, LocatedHands, RawHand, Resume,
    WRIST, located,
};
use super::{GRAB, HandSource, HapticLog, TrackedHand, XrHands};

/// The name of the entity the quest runner spawns to start the test (Q-308).
// Q-308
pub(crate) const TRIGGER: &str = "ftgsxr-test: dv8";
/// The test's name in the verdict line.
const NAME: &str = "dv8";
/// The joints clause's window (Q-272).
const WINDOW: Duration = Duration::from_secs(5);
/// How long the trackers may take to appear (the permission, the session).
const TRACKERS_TIMEOUT: Duration = Duration::from_secs(60);
/// The synthetic index pinch (Q-272), above Appendix D.3's 0.7.
const PINCH: f32 = 0.8;
/// How many frames the pinch may take to produce `Select`.
const PINCH_FRAMES: u32 = 30;
/// How many frames the injected grab may take to reach a pulse.
const PULSE_FRAMES: u32 = 30;

/// The joints clause's tally over its window.
#[derive(Clone, Debug, Default)]
pub(crate) struct JointsWindow {
    frames: u32,
    secs: f32,
    attempts: u32,
    failures: u32,
    /// Frames each hand was active.
    active: [u32; 2],
    /// The aim state's flags seen on an active hand, together.
    flags: sys::HandTrackingAimFlagsFB,
    /// Each hand's last located wrist position.
    last_wrist: [Option<[f32; 3]>; 2],
    /// Per hint state (none sent, default, high): its seconds and each hand's wrist
    /// changes (Q-309).
    by_hint: [(f32, [u32; 2]); 3],
}

impl JointsWindow {
    /// One frame of `dt` seconds: its locates (`attempts`, `failures`), the hint sent
    /// before it and the located hands.
    pub(crate) fn frame(
        &mut self,
        dt: f32,
        hint: Option<FrequencyHint>,
        raw: &[Option<RawHand>; 2],
        attempts: u32,
        failures: u32,
    ) {
        self.frames += 1;
        self.secs += dt;
        self.attempts += attempts;
        self.failures += failures;
        let k = match hint {
            None => 0,
            Some(h) if h == FrequencyHint::HIGH => 2,
            Some(_) => 1,
        };
        self.by_hint[k].0 += dt;
        for (h, hand) in raw.iter().enumerate() {
            let Some(hand) = hand else {
                self.last_wrist[h] = None;
                continue;
            };
            self.active[h] += 1;
            if let Some(aim) = &hand.aim {
                self.flags |= aim.status;
            }
            let wrist = &hand.joints[WRIST];
            if !located(wrist) {
                continue;
            }
            let p = wrist.pose.position;
            let p = [p.x, p.y, p.z];
            // Q-309: a new wrist position counts as one tracking sample.
            if self.last_wrist[h].is_some_and(|q| q != p) {
                self.by_hint[k].1[h] += 1;
            }
            self.last_wrist[h] = Some(p);
        }
    }

    /// Q-272: every locate succeeded, and there was one.
    pub(crate) fn pass(&self) -> bool {
        self.attempts > 0 && self.failures == 0
    }

    /// Whether any frame saw an active hand.
    pub(crate) fn hand_seen(&self) -> bool {
        self.active.iter().any(|&n| n > 0)
    }

    /// The clause's report: the locates, the active frames, the rates under each hint
    /// state, and the aim state's flags (spike check 2).
    pub(crate) fn detail(&self, aim: bool) -> String {
        let mut parts = vec![format!(
            "joints: {} locates, {} failed, {} frames in {:.1} s ({:.0}/s), active frames \
             left {} right {}",
            self.attempts,
            self.failures,
            self.frames,
            self.secs,
            rate(self.frames, self.secs),
            self.active[0],
            self.active[1],
        )];
        if !self.hand_seen() {
            parts.push("no active hand seen".into());
        }
        for (name, (secs, n)) in ["none sent", "default", "high"].iter().zip(self.by_hint) {
            if secs > 0.0 {
                parts.push(format!(
                    "wrist rate with hint {name}: left {:.0}/s right {:.0}/s over {secs:.1} s",
                    rate(n[0], secs),
                    rate(n[1], secs)
                ));
            }
        }
        parts.push(if !aim {
            "aim state: XR_FB_hand_tracking_aim absent".into()
        } else if !self.hand_seen() {
            "aim state: flags not seen (no active hand)".into()
        } else {
            let yes = |f| if self.flags.contains(f) { "yes" } else { "no" };
            format!(
                "aim state: COMPUTED {}, VALID {}",
                yes(sys::HandTrackingAimFlagsFB::COMPUTED),
                yes(sys::HandTrackingAimFlagsFB::VALID)
            )
        });
        parts.join(", ")
    }
}

fn rate(n: u32, secs: f32) -> f32 {
    if secs > 0.0 { n as f32 / secs } else { 0.0 }
}

/// Spike check 5's line.
pub(crate) fn resume_detail(r: Resume) -> String {
    match r {
        Resume::Absent => {
            "simultaneous resume: XR_META_simultaneous_hands_and_controllers absent".into()
        }
        Resume::Resumed => "simultaneous resume: resumed".into(),
        Resume::Failed(e) => format!("simultaneous resume: failed {e}"),
    }
}

/// The synthetic right hand of the pinch clause: tracked, pinching `pinch`, its aim far
/// above the rig and pointing up, so its ray meets no panel quad (Q-263) and reaches
/// `Select` rather than `PanelPointer`.
pub(crate) fn injected(pinch: f32) -> TrackedHand {
    let aim = Pose {
        translation: Vec3::new(0.0, 100.0, 0.0),
        // −z turned to +y.
        rotation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        scale: 1.0,
    };
    TrackedHand {
        aim,
        grip: aim,
        pinch,
        pinch_middle: 0.0,
        squeeze: 0.0,
        trigger: 0.0,
        stick: bevy::math::Vec2::ZERO,
        primary: false,
        secondary: false,
        menu: false,
        source: HandSource::Hand,
        tracked: true,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PinchStep {
    /// The hand open, so the pinch is a rising edge.
    Open,
    /// Pinching; waiting for `Select`.
    Pressed(u32),
    /// Open again, releasing.
    Released,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Phase {
    Trackers,
    Joints {
        started: Instant,
    },
    Pulse,
    /// The injected grab (`entity`) on `side`'s controller, waiting for its grab pulse
    /// among the haptics' calls numbered from `from`; `wanted` once the haptics asked
    /// for it.
    PulseWait {
        entity: Entity,
        side: usize,
        from: u64,
        frames: u32,
        wanted: bool,
    },
    Pinch(PinchStep),
    Done,
}

/// The test's state.
#[derive(Resource, Debug)]
pub(crate) struct Dv8 {
    started: Instant,
    last: Option<Instant>,
    pub(crate) phase: Phase,
    window: JointsWindow,
    parts: Vec<String>,
    pass: bool,
    /// The verdict, once printed.
    pub(crate) verdict: Option<(bool, String)>,
}

impl Dv8 {
    pub(crate) fn new(phase: Phase) -> Dv8 {
        Dv8 {
            started: Instant::now(),
            last: None,
            phase,
            window: JointsWindow::default(),
            parts: Vec::new(),
            pass: true,
            verdict: None,
        }
    }

    fn fail(&mut self, detail: String) {
        self.pass = false;
        self.parts.push(detail);
    }

    fn finish(&mut self) {
        let detail = self.parts.join("; ");
        let word = if self.pass { "PASS" } else { "FAIL" };
        // §3.4, Q-122: the verdict line, as `ftgsxr-quest`'s runner prints it.
        info!("ftgsxr-test: {NAME} {word} {detail}");
        self.verdict = Some((self.pass, detail));
        self.phase = Phase::Done;
    }
}

/// Adds the test's systems: inert until an entity named [`TRIGGER`] exists.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn add(app: &mut App) {
    app.add_systems(
        PostUpdate,
        (
            detect.run_if(not(resource_exists::<Dv8>)),
            run.run_if(resource_exists::<Dv8>)
                .after(super::write_hands)
                .before(super::hands::send_hint),
        ),
    );
}

/// Starts the test when the quest runner's entity appears (Q-308).
#[cfg_attr(test, allow(dead_code))]
fn detect(names: Query<&Name>, mut commands: Commands) {
    if names.iter().any(|n| n.as_str() == TRIGGER) {
        info!("dv8: started");
        commands.insert_resource(Dv8::new(Phase::Trackers));
    }
}

/// One frame of the test, after `tracking` wrote `XrHands` and before `xr_input` reads
/// it in the next frame's `PreUpdate`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    mut test: ResMut<Dv8>,
    trackers: Option<Res<HandTrackers>>,
    located: Option<Res<LocatedHands>>,
    permission: Option<Res<HandsPermission>>,
    wanted: Option<Res<HandsWanted>>,
    exts: Option<Res<OxrEnabledExtensions>>,
    log: Option<Res<HapticLog>>,
    mut hands: ResMut<XrHands>,
    mut intents: MessageReader<Intent>,
    mut pointers: MessageReader<PanelPointer>,
    mut commands: Commands,
) {
    let selected = intents.read().any(|i| {
        matches!(
            i,
            Intent::Select {
                hand: Hand::Right,
                pressed: true
            }
        )
    });
    let on_panel = pointers.read().any(|p| p.hand == Hand::Right);
    let now = Instant::now();
    let dt = test.last.map_or(0.0, |l| (now - l).as_secs_f32());
    test.last = Some(now);
    match test.phase.clone() {
        Phase::Done => {}
        Phase::Trackers => {
            // The window opens at the first frame with a locate: the trackers can exist
            // some frames before the session runs and `locate_hands` with it.
            let locating = located.as_deref().is_some_and(|l| l.attempts > 0);
            if let Some(t) = trackers.as_deref().filter(|_| locating) {
                let missing: Vec<&str> = ["left", "right"]
                    .into_iter()
                    .zip(&t.trackers)
                    .filter(|(_, t)| t.is_none())
                    .map(|(n, _)| n)
                    .collect();
                if !missing.is_empty() {
                    test.fail(format!("joints: no {} hand tracker", missing.join(" or ")));
                    test.phase = Phase::Pulse;
                } else {
                    test.parts.push(resume_detail(t.resume));
                    let hint = if t.has_hint() {
                        "frequency hint: loaded"
                    } else {
                        "frequency hint: XR_META_hand_tracking_frequency_hint absent"
                    };
                    test.parts.push(hint.into());
                    // D-55: whether the runtime offers the held-controller signal.
                    let detached = if exts.as_deref().is_some_and(|e| e.meta_detached_controllers) {
                        "detached controllers: enabled"
                    } else {
                        "detached controllers: XR_META_detached_controllers absent"
                    };
                    test.parts.push(detached.into());
                    test.phase = Phase::Joints { started: now };
                }
            } else if test.started.elapsed() > TRACKERS_TIMEOUT {
                let flow = permission.map(|p| p.flow);
                let detail = format!(
                    "joints: {} after {} s (hands wanted {}, XR_EXT_hand_tracking {}, \
                     permission {})",
                    if trackers.is_some() {
                        "no locate yet"
                    } else {
                        "no hand trackers"
                    },
                    TRACKERS_TIMEOUT.as_secs(),
                    wanted.is_some_and(|w| w.0),
                    exts.is_some_and(|e| e.ext_hand_tracking),
                    match flow {
                        Some(Flow::Idle) | None => "not asked",
                        Some(Flow::Asking(_)) => "asking",
                        Some(Flow::Granted) => "granted",
                        Some(Flow::Denied) => "denied",
                    }
                );
                test.fail(detail);
                test.phase = Phase::Pulse;
            }
        }
        Phase::Joints { started } => {
            if let (Some(t), Some(l)) = (trackers.as_deref(), located.as_deref()) {
                test.window
                    .frame(dt, t.sent, &l.raw, l.attempts, l.failures);
            }
            if now - started >= WINDOW {
                let aim = trackers.as_deref().is_some_and(|t| t.aim);
                let detail = test.window.detail(aim);
                if test.window.pass() {
                    test.parts.push(detail);
                } else {
                    test.fail(detail);
                }
                test.phase = Phase::Pulse;
            }
        }
        Phase::Pulse => {
            // The active controller: the right one first, as `XrHands` has it now.
            let side = [1usize, 0].into_iter().find(|&s| {
                let slot = if s == 0 { hands.left } else { hands.right };
                slot.is_some_and(|t| t.source == HandSource::Controller)
            });
            match side {
                None => {
                    test.parts.push("pulse: no controller".into());
                    test.phase = Phase::Pinch(PinchStep::Open);
                }
                Some(side) => {
                    // A grab start on that side, through the haptics (Q-270's event).
                    let hand = if side == 0 { Hand::Left } else { Hand::Right };
                    let entity = commands
                        .spawn((
                            Name::new("ftgsxr-test: dv8 grab"),
                            Grabbed::One {
                                hand,
                                grab: Grab::begin(Pose::IDENTITY, Pose::IDENTITY, Vec3::ZERO),
                            },
                        ))
                        .id();
                    test.phase = Phase::PulseWait {
                        entity,
                        side,
                        from: log.as_deref().map_or(0, |l| l.next),
                        frames: 0,
                        wanted: false,
                    };
                }
            }
        }
        Phase::PulseWait {
            entity,
            side,
            from,
            frames,
            wanted,
        } => {
            let name = if side == 0 { "left" } else { "right" };
            let wanted = wanted
                || log
                    .as_deref()
                    .is_some_and(|l| l.wanted.contains(&(side, GRAB)));
            let applied = log.as_deref().and_then(|l| {
                l.applied
                    .iter()
                    .find(|(n, s, p, _)| *n >= from && *s == side && *p == GRAB)
                    .map(|(.., r)| r.clone())
            });
            let done = match applied {
                Some(Ok(())) => {
                    test.parts.push(format!(
                        "pulse: a grab start gave {} ms at {} on the {name} controller, applied",
                        GRAB.duration_ms, GRAB.amplitude
                    ));
                    true
                }
                Some(Err(e)) => {
                    test.fail(format!("pulse: the {name} controller: {e}"));
                    true
                }
                None if frames >= PULSE_FRAMES => {
                    test.fail(format!(
                        "pulse: a grab start on the {name} controller {} after {PULSE_FRAMES} frames",
                        if wanted {
                            "reached the haptics but no pulse was applied (no running \
                             session)"
                        } else {
                            "reached no pulse"
                        }
                    ));
                    true
                }
                None => false,
            };
            if done {
                commands.entity(entity).despawn();
                test.phase = Phase::Pinch(PinchStep::Open);
            } else {
                test.phase = Phase::PulseWait {
                    entity,
                    side,
                    from,
                    frames: frames + 1,
                    wanted,
                };
            }
        }
        Phase::Pinch(step) => match step {
            PinchStep::Open => {
                hands.right = Some(injected(0.0));
                test.phase = Phase::Pinch(PinchStep::Pressed(0));
            }
            PinchStep::Pressed(n) => {
                if selected {
                    test.parts
                        .push(format!("pinch: {PINCH} wrote Select on the right hand"));
                    hands.right = Some(injected(0.0));
                    test.phase = Phase::Pinch(PinchStep::Released);
                } else if on_panel {
                    test.fail("pinch: the ray went to a panel quad, not Select".into());
                    test.finish();
                } else if n >= PINCH_FRAMES {
                    test.fail(format!("pinch: no Select after {PINCH_FRAMES} frames"));
                    test.finish();
                } else {
                    hands.right = Some(injected(PINCH));
                    test.phase = Phase::Pinch(PinchStep::Pressed(n + 1));
                }
            }
            PinchStep::Released => test.finish(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::hands::AimState;

    fn raw(x: f32, status: sys::HandTrackingAimFlagsFB) -> RawHand {
        let mut joints = [sys::HandJointLocationEXT::default(); super::super::hands::JOINTS];
        joints[WRIST].location_flags =
            sys::SpaceLocationFlags::POSITION_VALID | sys::SpaceLocationFlags::ORIENTATION_VALID;
        joints[WRIST].pose.position.x = x;
        RawHand {
            joints,
            aim: Some(AimState {
                status,
                pose: sys::Posef::IDENTITY,
                index: 0.0,
                middle: 0.0,
            }),
        }
    }

    /// The window counts the locates and their failures, the active frames, the flags
    /// seen together, and a new wrist position per hint state (Q-309).
    #[test]
    fn joints_window_tallies_locates_flags_and_wrist_changes() {
        use sys::HandTrackingAimFlagsFB as F;
        let mut w = JointsWindow::default();
        // Before any hint: the left wrist at 0, then 0 again (no new sample).
        w.frame(0.5, None, &[Some(raw(0.0, F::COMPUTED)), None], 2, 0);
        w.frame(0.5, None, &[Some(raw(0.0, F::VALID)), None], 2, 0);
        assert!(w.pass());
        // Under the high hint: two new positions in 1 s.
        let high = Some(FrequencyHint::HIGH);
        w.frame(0.5, high, &[Some(raw(0.1, F::EMPTY)), None], 2, 0);
        w.frame(0.5, high, &[Some(raw(0.2, F::EMPTY)), None], 2, 1);
        assert!(!w.pass(), "one failed locate fails the clause");
        assert_eq!(w.attempts, 8);
        assert_eq!(w.active, [4, 0]);
        assert!(w.flags.contains(F::COMPUTED | F::VALID));
        assert_eq!(w.by_hint[0], (1.0, [0, 0]));
        assert_eq!(w.by_hint[2], (1.0, [2, 0]));
        let d = w.detail(true);
        assert!(d.contains("COMPUTED yes, VALID yes"), "{d}");
        assert!(d.contains("hint high: left 2/s right 0/s"), "{d}");
        assert!(!d.contains("no active hand"), "{d}");
    }

    /// Without an active hand the clause still passes on its locates and says so.
    #[test]
    fn joints_window_without_a_hand_passes_and_reports_it() {
        let mut w = JointsWindow::default();
        w.frame(0.1, None, &[None, None], 2, 0);
        assert!(w.pass() && !w.hand_seen());
        let d = w.detail(true);
        assert!(d.contains("no active hand seen"), "{d}");
        assert!(d.contains("flags not seen"), "{d}");
        assert!(JointsWindow::default().detail(false).contains("absent"));
        assert!(!JointsWindow::default().pass(), "no locate is no pass");
    }
}
