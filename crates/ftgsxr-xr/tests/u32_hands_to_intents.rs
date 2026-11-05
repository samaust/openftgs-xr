// SPDX-License-Identifier: Apache-2.0

//! U-32 (`cpu`, §3.1), the `hands_to_intents` clauses, through `XrInputState`,
//! `XrInputConfig` and `XrHands` (source of truth: U-32's row, §1.8 Controllers, Hands,
//! Rays, selection and grabbing, Locomotion, Comfort and accessibility, Appendix D.2 and
//! D.3, D-17): the pinch and squeeze thresholds with their hysteresis (a squeeze of 0.5
//! continues a grab but starts none), grab begin and end, the one-to-two-hand
//! transitions, the menu toggle on the MENU\_PRESSED edge, no locomotion intents when
//! locomotion is disabled, left-handed mode swapping the sticks' roles. `pick_instance`
//! and `panel_hit` are `ftgsxr-app`'s (`u32_pick_instance.rs`, `u32_panel_hit.rs`).

use bevy::ecs::entity::Entity;
use bevy::math::{Vec2, Vec3};
use ftgsxr_app::{Feature, Hand, InstanceAction, Intent, PlaybackCommand};
use ftgsxr_core::Pose;
use ftgsxr_render::RenderMode;
use ftgsxr_xr::{HandSource, TrackedHand, XrHands, XrInputConfig, XrInputState, hands_to_intents};

fn controller() -> TrackedHand {
    TrackedHand {
        aim: Pose {
            translation: Vec3::new(0.1, 1.2, -0.2),
            ..Pose::IDENTITY
        },
        grip: Pose::IDENTITY,
        pinch: 0.0,
        pinch_middle: 0.0,
        squeeze: 0.0,
        trigger: 0.0,
        stick: Vec2::ZERO,
        primary: false,
        secondary: false,
        menu: false,
        source: HandSource::Controller,
        tracked: true,
    }
}

fn hand() -> TrackedHand {
    TrackedHand {
        source: HandSource::Hand,
        ..controller()
    }
}

fn hands(left: Option<TrackedHand>, right: Option<TrackedHand>) -> XrHands {
    XrHands {
        left,
        right,
        head: Pose::IDENTITY,
        left_handed: false,
    }
}

fn cfg() -> XrInputConfig {
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

fn grabs(out: &[Intent]) -> Vec<String> {
    out.iter()
        .filter_map(|i| match i {
            Intent::GrabBegin { hand, .. } => Some(format!("begin {hand:?}")),
            Intent::GrabMove { hand, .. } => Some(format!("move {hand:?}")),
            Intent::GrabEnd { hand } => Some(format!("end {hand:?}")),
            _ => None,
        })
        .collect()
}

fn selects(out: &[Intent]) -> Vec<(Hand, bool)> {
    out.iter()
        .filter_map(|i| match *i {
            Intent::Select { hand, pressed } => Some((hand, pressed)),
            _ => None,
        })
        .collect()
}

fn squeeze(s: f32) -> Option<TrackedHand> {
    Some(TrackedHand {
        squeeze: s,
        ..controller()
    })
}

#[test]
fn u32_squeeze_hysteresis_a_half_continues_a_grab_but_starts_none() {
    let (mut st, c) = (XrInputState::default(), cfg());
    // 0.5 from rest starts nothing, nor does 0.6 itself (begins above 0.6).
    assert!(grabs(&hands_to_intents(&mut st, &hands(None, squeeze(0.5)), &c)).is_empty());
    assert!(grabs(&hands_to_intents(&mut st, &hands(None, squeeze(0.6)), &c)).is_empty());
    assert_eq!(
        grabs(&hands_to_intents(&mut st, &hands(None, squeeze(0.61)), &c)),
        ["begin Right"]
    );
    // 0.5 continues the grab, so does 0.4; below 0.4 ends it.
    assert_eq!(
        grabs(&hands_to_intents(&mut st, &hands(None, squeeze(0.5)), &c)),
        ["move Right"]
    );
    assert_eq!(
        grabs(&hands_to_intents(&mut st, &hands(None, squeeze(0.4)), &c)),
        ["move Right"]
    );
    assert_eq!(
        grabs(&hands_to_intents(&mut st, &hands(None, squeeze(0.39)), &c)),
        ["end Right"]
    );
    assert!(grabs(&hands_to_intents(&mut st, &hands(None, squeeze(0.5)), &c)).is_empty());
}

#[test]
fn u32_trigger_select_edges_at_0_6_and_0_4() {
    let (mut st, c) = (XrInputState::default(), cfg());
    let trig = |v: f32| {
        Some(TrackedHand {
            trigger: v,
            ..controller()
        })
    };
    assert!(selects(&hands_to_intents(&mut st, &hands(trig(0.6), None), &c)).is_empty());
    assert_eq!(
        selects(&hands_to_intents(&mut st, &hands(trig(0.7), None), &c)),
        [(Hand::Left, true)]
    );
    assert!(selects(&hands_to_intents(&mut st, &hands(trig(0.45), None), &c)).is_empty());
    assert_eq!(
        selects(&hands_to_intents(&mut st, &hands(trig(0.3), None), &c)),
        [(Hand::Left, false)]
    );
}

#[test]
fn u32_pinch_hysteresis_selects_and_grabs_at_once() {
    let (mut st, c) = (XrInputState::default(), cfg());
    let pinch = |v: f32| Some(TrackedHand { pinch: v, ..hand() });
    let out = hands_to_intents(&mut st, &hands(pinch(0.69), None), &c);
    assert!(selects(&out).is_empty() && grabs(&out).is_empty());
    // ≥ 0.7 begins: `select` and `grab` at once, the select first.
    let out = hands_to_intents(&mut st, &hands(pinch(0.7), None), &c);
    assert_eq!(selects(&out), [(Hand::Left, true)]);
    assert_eq!(grabs(&out), ["begin Left"]);
    let s = out
        .iter()
        .position(|i| matches!(i, Intent::Select { .. }))
        .unwrap();
    let g = out
        .iter()
        .position(|i| matches!(i, Intent::GrabBegin { .. }))
        .unwrap();
    assert!(s < g, "selects before it grabs");
    // 0.4 holds; < 0.4 ends both.
    let out = hands_to_intents(&mut st, &hands(pinch(0.4), None), &c);
    assert!(selects(&out).is_empty());
    assert_eq!(grabs(&out), ["move Left"]);
    let out = hands_to_intents(&mut st, &hands(pinch(0.39), None), &c);
    assert_eq!(selects(&out), [(Hand::Left, false)]);
    assert_eq!(grabs(&out), ["end Left"]);
}

#[test]
fn u32_grab_begin_and_end_carry_the_hand_and_its_pose() {
    let (mut st, c) = (XrInputState::default(), cfg());
    let h = squeeze(0.9).unwrap();
    let out = hands_to_intents(&mut st, &hands(Some(h), None), &c);
    assert!(out.contains(&Intent::GrabBegin {
        hand: Hand::Left,
        pose: h.aim
    }));
    // The ray comes first, so `manipulate` picks from this frame's ray.
    assert!(matches!(
        out[0],
        Intent::Pointer {
            hand: Hand::Left,
            ..
        }
    ));
    let moved = TrackedHand {
        aim: Pose {
            translation: Vec3::new(0.3, 1.0, -0.4),
            ..Pose::IDENTITY
        },
        ..h
    };
    let out = hands_to_intents(&mut st, &hands(Some(moved), None), &c);
    assert!(out.contains(&Intent::GrabMove {
        hand: Hand::Left,
        pose: moved.aim
    }));
    let out = hands_to_intents(&mut st, &hands(squeeze(0.0), None), &c);
    assert_eq!(grabs(&out), ["end Left"]);
}

#[test]
fn u32_one_to_two_hand_transitions() {
    let (mut st, c) = (XrInputState::default(), cfg());
    // The left hand grabs, then the right while the left holds.
    assert_eq!(
        grabs(&hands_to_intents(
            &mut st,
            &hands(squeeze(0.9), squeeze(0.0)),
            &c
        )),
        ["begin Left"]
    );
    assert_eq!(
        grabs(&hands_to_intents(
            &mut st,
            &hands(squeeze(0.9), squeeze(0.9)),
            &c
        )),
        ["move Left", "begin Right"]
    );
    assert_eq!(
        grabs(&hands_to_intents(
            &mut st,
            &hands(squeeze(0.9), squeeze(0.9)),
            &c
        )),
        ["move Left", "move Right"]
    );
    // Releasing either hand: the other keeps moving (a one-hand grab in `manipulate`).
    assert_eq!(
        grabs(&hands_to_intents(
            &mut st,
            &hands(squeeze(0.1), squeeze(0.9)),
            &c
        )),
        ["end Left", "move Right"]
    );
    assert_eq!(
        grabs(&hands_to_intents(
            &mut st,
            &hands(squeeze(0.1), squeeze(0.9)),
            &c
        )),
        ["move Right"]
    );
    // And the other way round.
    hands_to_intents(&mut st, &hands(squeeze(0.9), squeeze(0.9)), &c);
    assert_eq!(
        grabs(&hands_to_intents(
            &mut st,
            &hands(squeeze(0.9), squeeze(0.2)),
            &c
        )),
        ["move Left", "end Right"]
    );
}

#[test]
fn u32_push_pull_is_the_grabbing_hands_stick_y() {
    let (mut st, c) = (XrInputState::default(), cfg());
    let h = TrackedHand {
        squeeze: 0.9,
        stick: Vec2::new(1.0, 1.0),
        ..controller()
    };
    let out = hands_to_intents(&mut st, &hands(None, Some(h)), &c);
    assert!(out.contains(&Intent::PushPull {
        hand: Hand::Right,
        axis: 1.0
    }));
    // Its x does nothing: the right stick turns nothing while it grabs.
    assert!(
        !out.iter()
            .any(|i| matches!(i, Intent::SnapTurn { .. } | Intent::Locomote { .. }))
    );
}

#[test]
fn u32_menu_toggles_the_panel_on_the_menu_pressed_edge() {
    let (mut st, c) = (XrInputState::default(), cfg());
    let menu = |m: bool| Some(TrackedHand { menu: m, ..hand() });
    let toggles = |out: &[Intent]| {
        out.iter()
            .filter(|i| **i == Intent::Toggle(Feature::Panel))
            .count()
    };
    assert_eq!(
        toggles(&hands_to_intents(&mut st, &hands(None, menu(true)), &c)),
        1
    );
    assert_eq!(
        toggles(&hands_to_intents(&mut st, &hands(None, menu(true)), &c)),
        0
    );
    assert_eq!(
        toggles(&hands_to_intents(&mut st, &hands(None, menu(false)), &c)),
        0
    );
    assert_eq!(
        toggles(&hands_to_intents(&mut st, &hands(None, menu(true)), &c)),
        1
    );
    // The controller's menu button too.
    let button = Some(TrackedHand {
        menu: true,
        ..controller()
    });
    assert_eq!(
        toggles(&hands_to_intents(&mut st, &hands(button, None), &c)),
        1
    );
}

fn sticks(left: Vec2, right: Vec2, left_handed: bool) -> XrHands {
    XrHands {
        left_handed,
        ..hands(
            Some(TrackedHand {
                stick: left,
                ..controller()
            }),
            Some(TrackedHand {
                stick: right,
                ..controller()
            }),
        )
    }
}

fn locomotion(out: &[Intent]) -> Vec<Intent> {
    out.iter()
        .filter(|i| matches!(i, Intent::Locomote { .. } | Intent::SnapTurn { .. }))
        .copied()
        .collect()
}

#[test]
fn u32_no_locomotion_intents_when_disabled() {
    let mut st = XrInputState::default();
    let off = XrInputConfig {
        locomotion_enabled: false,
        ..cfg()
    };
    let smooth_off = XrInputConfig { snap: false, ..off };
    for c in [off, smooth_off] {
        let out = hands_to_intents(
            &mut st,
            &sticks(Vec2::new(0.3, 1.0), Vec2::new(1.0, 0.0), false),
            &c,
        );
        assert!(locomotion(&out).is_empty(), "{out:?}");
    }
    // Enabled, the same sticks move and turn.
    let out = hands_to_intents(
        &mut XrInputState::default(),
        &sticks(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0), false),
        &cfg(),
    );
    assert_eq!(
        locomotion(&out),
        [
            Intent::Locomote {
                move_xz: Vec2::new(0.0, 1.0),
                turn: 0.0
            },
            Intent::SnapTurn { steps: 1 }
        ]
    );
}

#[test]
fn u32_left_handed_mode_swaps_the_sticks_roles() {
    let smooth = XrInputConfig {
        snap: false,
        ..cfg()
    };
    // Right-handed: the left stick moves, the right one turns.
    let out = hands_to_intents(
        &mut XrInputState::default(),
        &sticks(Vec2::new(0.0, 1.0), Vec2::new(-1.0, 0.0), false),
        &smooth,
    );
    assert_eq!(
        locomotion(&out),
        [Intent::Locomote {
            move_xz: Vec2::new(0.0, 1.0),
            turn: -1.0
        }]
    );
    // Left-handed: `move` on the right stick, `turn` on the left.
    let out = hands_to_intents(
        &mut XrInputState::default(),
        &sticks(Vec2::new(0.0, 1.0), Vec2::new(-1.0, 0.0), true),
        &smooth,
    );
    assert_eq!(
        locomotion(&out),
        [Intent::Locomote {
            move_xz: Vec2::new(-1.0, 0.0),
            turn: 0.0
        }]
    );
    // Snap turn follows the turn stick: the left one in left-handed mode.
    let out = hands_to_intents(
        &mut XrInputState::default(),
        &sticks(Vec2::new(1.0, 0.0), Vec2::ZERO, true),
        &cfg(),
    );
    assert_eq!(locomotion(&out), [Intent::SnapTurn { steps: 1 }]);
}

#[test]
fn u32_face_buttons_keep_their_sides_in_left_handed_mode() {
    let e = Entity::from_raw_u32(7).unwrap();
    let mode = RenderMode::Splats;
    for left_handed in [false, true] {
        let mut st = XrInputState::default();
        st.set_selected(Some((e, mode)));
        let now = XrHands {
            left_handed,
            ..hands(
                Some(TrackedHand {
                    primary: true,
                    secondary: true,
                    ..controller()
                }),
                Some(TrackedHand {
                    primary: true,
                    secondary: true,
                    ..controller()
                }),
            )
        };
        let out = hands_to_intents(&mut st, &now, &cfg());
        for want in [
            Intent::Instance {
                instance: e,
                action: InstanceAction::Fit,
            },
            Intent::Instance {
                instance: e,
                action: InstanceAction::SetMode(mode),
            },
            Intent::Playback(PlaybackCommand::Toggle),
            Intent::Instance {
                instance: e,
                action: InstanceAction::ResetPose,
            },
        ] {
            assert!(out.contains(&want), "{want:?} in {out:?}");
        }
        // Held, they fire once.
        let again = hands_to_intents(&mut st, &now, &cfg());
        assert!(
            !again
                .iter()
                .any(|i| matches!(i, Intent::Instance { .. } | Intent::Playback(_)))
        );
    }
}
