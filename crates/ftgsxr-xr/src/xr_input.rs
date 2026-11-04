// SPDX-License-Identifier: Apache-2.0

//! Appendix D.2 and D.3 to intents (§1.8, E10–E12, §2.1 `ftgsxr_xr::xr_input`): the pure
//! `hands_to_intents` with its per-hand state (the thresholds' hysteresis, the snap-turn
//! arming, the button and menu edges), and the module's systems (`system`): the ray
//! arbitration between the panels and the instances (Q-263), the rays, the haptic pulses
//! (D-44, Q-270) and the teleport arc (Q-269).

mod system;

#[allow(unused_imports)] // `panel` (a later ticket) inserts it on both quads (Q-265).
pub(crate) use system::PanelQuad;
pub(crate) use system::XrInputPlugin;
// Dv-8's pulse clause (#463).
#[cfg(any(test, target_os = "android"))]
pub(crate) use system::GRAB;

use bevy::ecs::entity::Entity;
use bevy::math::{Vec2, Vec3};
use ftgsxr_app::{
    Feature, Hand, InstanceAction, Intent, PlaybackCommand, SnapTurnState, stick_response,
};
use ftgsxr_core::Pose;
use ftgsxr_render::RenderMode;

use crate::tracking::{HandSource, TrackedHand, XrHands};

/// The thresholds and switches `hands_to_intents` reads (§2.1): the analog controls'
/// begin and end values (§1.8 Controllers: 0.6 and 0.4, the trigger's `select` and the
/// squeeze's `grab` alike, Q-302), the index pinch's (Appendix D.3: ≥ 0.7 begins,
/// < 0.4 ends), E12's dead zone, snap or smooth turn, and whether stick locomotion is on
/// (`Locomotion.enabled`, false in passthrough mode, D-17).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XrInputConfig {
    /// A controller's squeeze (and trigger) begins above this value.
    pub grab_begin: f32,
    /// A controller's squeeze (and trigger) ends below this value.
    pub grab_end: f32,
    /// A hand's pinch begins at or above this strength.
    pub pinch_begin: f32,
    /// A hand's pinch ends below this strength.
    pub pinch_end: f32,
    /// E12's dead zone d.
    pub stick_deadzone: f32,
    /// Snap turn rather than smooth turn.
    pub snap: bool,
    /// Stick locomotion on (D-17).
    pub locomotion_enabled: bool,
}

/// §1.8's and Appendix D.3's thresholds, D-44's dead zone, snap turn, locomotion on.
impl Default for XrInputConfig {
    fn default() -> Self {
        XrInputConfig {
            grab_begin: 0.6,
            grab_end: 0.4,
            pinch_begin: 0.7,
            pinch_end: 0.4,
            stick_deadzone: 0.15,
            snap: true,
            locomotion_enabled: true,
        }
    }
}

/// One hand's edges and hysteresis.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct HandEdges {
    /// `select` held (the trigger, or the index pinch for a tracked hand).
    selecting: bool,
    /// `grab` held (the squeeze, or the index pinch for a tracked hand).
    grabbing: bool,
    /// The index pinch held (a tracked hand).
    pinching: bool,
    /// The middle-finger pinch held (a tracked hand).
    middle: bool,
    /// The menu button or MENU\_PRESSED held.
    menu_down: bool,
    /// A or X held.
    primary_down: bool,
    /// B or Y held.
    secondary_down: bool,
}

/// The per-hand state the thresholds need (§2.1): the hysteresis of select, grab and
/// pinch, the snap-turn arming, the menu and button edges, and the selected instance
/// with the render mode its `mode` button sets (Q-302).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct XrInputState {
    hands: [HandEdges; 2],
    snap: SnapTurnState,
    selected: Option<(Entity, RenderMode)>,
}

impl XrInputState {
    /// The selected instance and the render mode its `mode` button (Y) sets, the next of
    /// Q-198's cycle; `None` when nothing is selected, so B, X and Y do nothing.
    // Q-302
    pub fn set_selected(&mut self, selected: Option<(Entity, RenderMode)>) {
        self.selected = selected;
    }
}

/// A controller's analog hysteresis (§1.8): begins above `begin`, ends below `end`.
fn analog(held: bool, value: f32, begin: f32, end: f32) -> bool {
    if held { value >= end } else { value > begin }
}

/// A pinch's hysteresis (Appendix D.3): begins at or above `begin`, ends below `end`.
fn pinch(held: bool, value: f32, begin: f32, end: f32) -> bool {
    if held { value >= end } else { value >= begin }
}

/// The ray of an aim pose: its position and its −z (OpenXR's aim convention, source-log
/// 287).
pub(crate) fn aim_ray(aim: &Pose) -> (Vec3, Vec3) {
    (aim.translation, (aim.rotation * Vec3::NEG_Z).normalize())
}

const SIDES: [Hand; 2] = [Hand::Left, Hand::Right];

/// Appendix D.2 and D.3 to intents, pure given the state (§2.1): per hand in the order
/// left, right, its `Pointer` (while tracked), its `Select` edge, its grab (`GrabBegin`
/// on the rising edge, `GrabMove` while held, `GrabEnd` on release; `manipulate` makes
/// the one-to-two-hand transitions from them), its `PushPull`; then `Locomote` and
/// `SnapTurn` (none when `locomotion_enabled` is false, D-17); then the toggles and the
/// face buttons. The grab pose is the aim pose (Q-302). A hand that goes away
/// releases what it held.
pub fn hands_to_intents(
    state: &mut XrInputState,
    now: &XrHands,
    cfg: &XrInputConfig,
) -> Vec<Intent> {
    let mut out = Vec::new();
    let hands = [now.left, now.right];
    for (i, (hand, slot)) in SIDES.into_iter().zip(hands).enumerate() {
        let e = &mut state.hands[i];
        let Some(t) = slot else {
            if e.selecting {
                out.push(Intent::Select {
                    hand,
                    pressed: false,
                });
            }
            if e.grabbing {
                out.push(Intent::GrabEnd { hand });
            }
            *e = HandEdges::default();
            continue;
        };
        let (sel, grab) = match t.source {
            HandSource::Controller => {
                e.pinching = false;
                (
                    analog(e.selecting, t.trigger, cfg.grab_begin, cfg.grab_end),
                    analog(e.grabbing, t.squeeze, cfg.grab_begin, cfg.grab_end),
                )
            }
            HandSource::Hand => {
                // Appendix D.3: the index pinch is `select` and `grab` at once.
                e.pinching = pinch(e.pinching, t.pinch, cfg.pinch_begin, cfg.pinch_end);
                (e.pinching, e.pinching)
            }
        };
        // Q-302, Q-285: an untracked hand begins nothing and casts no ray; it may
        // still release.
        let sel = sel && (e.selecting || t.tracked);
        let grab = grab && (e.grabbing || t.tracked);
        if t.source == HandSource::Hand {
            e.pinching = sel;
        }
        let (origin, dir) = aim_ray(&t.aim);
        if t.tracked {
            out.push(Intent::Pointer { hand, origin, dir });
        }
        if sel != e.selecting {
            out.push(Intent::Select { hand, pressed: sel });
            e.selecting = sel;
        }
        match (e.grabbing, grab) {
            (false, true) => out.push(Intent::GrabBegin { hand, pose: t.aim }),
            (true, true) if t.tracked => out.push(Intent::GrabMove { hand, pose: t.aim }),
            (true, false) => out.push(Intent::GrabEnd { hand }),
            _ => {}
        }
        e.grabbing = grab;
        // §1.8: while a hand grabs, its stick's y is `push_pull` and its x does nothing.
        if grab && t.source == HandSource::Controller {
            let y = stick_response(t.stick, cfg.stick_deadzone).y;
            if y != 0.0 {
                out.push(Intent::PushPull { hand, axis: y });
            }
        }
    }

    // E12; §1.8 Comfort: left-handed mode swaps the sticks' roles (Q-128: both sticks
    // are full thumbsticks).
    let (move_i, turn_i) = if now.left_handed { (1, 0) } else { (0, 1) };
    let stick = |i: usize| -> Vec2 {
        match hands[i] {
            Some(t) if t.source == HandSource::Controller && !state.hands[i].grabbing => {
                stick_response(t.stick, cfg.stick_deadzone)
            }
            _ => Vec2::ZERO,
        }
    };
    if cfg.locomotion_enabled {
        let move_xz = stick(move_i);
        let x_t = stick(turn_i).x;
        // Q-195: `Locomote.turn` turns only with snap off; the snap trigger reads E12's
        // x_t after the response (Q-302).
        let (turn, steps) = if cfg.snap {
            (0.0, state.snap.step(x_t))
        } else {
            (x_t, 0)
        };
        if move_xz != Vec2::ZERO || turn != 0.0 {
            out.push(Intent::Locomote { move_xz, turn });
        }
        if steps != 0 {
            out.push(Intent::SnapTurn { steps });
        }
    }

    // The toggles and the face buttons (Appendix D.2: A and B on the right hand, X and Y
    // on the left, whatever the handedness; D.3: the middle pinch and MENU_PRESSED).
    let selected = state.selected;
    for (i, slot) in hands.iter().enumerate() {
        let e = &mut state.hands[i];
        let Some(t): Option<&TrackedHand> = slot.as_ref() else {
            continue;
        };
        let rose = |was: &mut bool, now: bool| {
            let r = now && !*was;
            *was = now;
            r
        };
        if rose(&mut e.menu_down, t.menu) {
            out.push(Intent::Toggle(Feature::Panel));
        }
        let middle = t.source == HandSource::Hand
            && pinch(e.middle, t.pinch_middle, cfg.pinch_begin, cfg.pinch_end);
        if rose(&mut e.middle, middle) {
            out.push(Intent::Playback(PlaybackCommand::Toggle));
        }
        let primary = rose(&mut e.primary_down, t.primary);
        let secondary = rose(&mut e.secondary_down, t.secondary);
        let act = |action| selected.map(|(instance, _)| Intent::Instance { instance, action });
        if i == 1 {
            if primary {
                out.push(Intent::Playback(PlaybackCommand::Toggle));
            }
            if secondary {
                out.extend(act(InstanceAction::ResetPose));
            }
        } else {
            if primary {
                out.extend(act(InstanceAction::Fit));
            }
            if secondary {
                out.extend(
                    selected
                        .map(|(_, mode)| mode)
                        .and_then(|m| act(InstanceAction::SetMode(m))),
                );
            }
        }
    }
    out
}
