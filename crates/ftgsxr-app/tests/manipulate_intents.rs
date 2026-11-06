// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr_app::manipulate` through its intents (`cpu`, #282; source of truth §1.7
//! Selection and manipulation, §1.8 Rays, selection and grabbing, E10, E11): a pointer
//! hovers the instance it hits, `Select` on it selects and on nothing deselects; a
//! one-hand grab keeps the instance's place in the hand; a second hand makes it a
//! two-hand grab that resizes as a hand moves, and releasing one returns to the
//! other; push and pull move along the ray; `Unload` reaches the library.

use crate::support;

use bevy::prelude::*;
use ftgsxr_app::{
    Grabbed, Hand, Hovered, InstanceAction, Intent, LibraryCommand, LoadedModel, ModelRef, Selected,
};
use ftgsxr_core::Pose;
use ftgsxr_render::{SplatAsset, SplatModel};
use ftgsxr_testkit::fixtures_dir;
use support::TestApp;

fn loaded(t: &mut TestApp) -> Entity {
    let path = fixtures_dir().join("probes/P1.safetensors");
    t.command(LibraryCommand::LoadRef(ModelRef::File(path)));
    assert!(
        t.until(600, |app| app
            .world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1),
        "P1 loads"
    );
    t.frames(1);
    let mut q = t
        .app
        .world_mut()
        .query_filtered::<Entity, With<LoadedModel>>();
    q.single(t.app.world()).unwrap()
}

fn send(t: &mut TestApp, intents: &[Intent]) {
    for i in intents {
        t.app.world_mut().write_message(*i);
    }
    t.frames(1);
}

/// A ray from 3 m in front of the stage origin toward it: through P1, whatever its box.
fn at_it(hand: Hand) -> Intent {
    Intent::Pointer {
        hand,
        origin: Vec3::new(0.0, 0.0, 3.0),
        dir: -Vec3::Z,
    }
}

fn away(hand: Hand) -> Intent {
    Intent::Pointer {
        hand,
        origin: Vec3::new(0.0, 0.0, 3.0),
        dir: Vec3::Z,
    }
}

fn hand_at(x: f32) -> Pose {
    Pose {
        translation: Vec3::new(x, 0.0, 2.0),
        ..Pose::IDENTITY
    }
}

fn pose(t: &TestApp, e: Entity) -> Pose {
    let tr = t.app.world().get::<Transform>(e).unwrap();
    Pose {
        translation: tr.translation,
        rotation: tr.rotation,
        scale: tr.scale.x,
    }
}

#[test]
fn hover_select_and_deselect() {
    let mut t = TestApp::new("manip_select");
    let e = loaded(&mut t);
    send(&mut t, &[at_it(Hand::Mouse)]);
    let h = *t.app.world().get::<Hovered>(e).expect("hovered");
    assert_eq!(h.hand, Hand::Mouse);
    assert!(h.distance > 0.0 && h.distance < 3.0);
    // The same ray again: the component is not written again.
    send(&mut t, &[at_it(Hand::Mouse)]);
    assert!(
        !t.app
            .world()
            .entity(e)
            .get_ref::<Hovered>()
            .unwrap()
            .is_changed()
    );
    send(
        &mut t,
        &[Intent::Select {
            hand: Hand::Mouse,
            pressed: true,
        }],
    );
    assert!(t.app.world().get::<Selected>(e).is_some());
    // On nothing: deselects, and the hover goes.
    send(
        &mut t,
        &[
            away(Hand::Mouse),
            Intent::Select {
                hand: Hand::Mouse,
                pressed: true,
            },
        ],
    );
    assert!(t.app.world().get::<Selected>(e).is_none());
    assert!(t.app.world().get::<Hovered>(e).is_none());
    send(
        &mut t,
        &[Intent::Instance {
            instance: e,
            action: InstanceAction::Select,
        }],
    );
    assert!(t.app.world().get::<Selected>(e).is_some());
    send(
        &mut t,
        &[Intent::Instance {
            instance: e,
            action: InstanceAction::Deselect,
        }],
    );
    assert!(t.app.world().get::<Selected>(e).is_none());
}

/// `Hovered` is recomputed each frame from that frame's `Pointer` intents (Q-263, which
/// rewords Q-182): a hand whose ray hits the instance hovers it, and in a frame without
/// that hand's `Pointer` (its ray on a panel, or no ray) the hover goes; another hand's
/// ray that frame keeps its own hover.
// Q-263, Q-182
#[test]
fn a_hand_s_hover_clears_in_a_frame_without_its_pointer() {
    let mut t = TestApp::new("manip_hover_frame");
    let e = loaded(&mut t);
    send(&mut t, &[at_it(Hand::Right)]);
    assert_eq!(
        t.app.world().get::<Hovered>(e).map(|h| h.hand),
        Some(Hand::Right)
    );
    send(&mut t, &[]);
    assert!(
        t.app.world().get::<Hovered>(e).is_none(),
        "no Pointer, no hover"
    );
    // The left hand's ray alone: the hover is the left's, the right staying cleared.
    send(&mut t, &[at_it(Hand::Right)]);
    send(&mut t, &[at_it(Hand::Left)]);
    assert_eq!(
        t.app.world().get::<Hovered>(e).map(|h| h.hand),
        Some(Hand::Left)
    );
}

#[test]
fn one_and_two_hand_grabs() {
    let mut t = TestApp::new("manip_grab");
    let e = loaded(&mut t);
    send(
        &mut t,
        &[
            at_it(Hand::Left),
            at_it(Hand::Right),
            Intent::GrabBegin {
                hand: Hand::Left,
                pose: hand_at(-0.1),
            },
        ],
    );
    assert!(matches!(
        t.app.world().get::<Grabbed>(e),
        Some(Grabbed::One {
            hand: Hand::Left,
            ..
        })
    ));
    // The instance follows the hand: 0.5 m to the right and 30° about y.
    let moved = Pose {
        translation: Vec3::new(0.4, 0.0, 2.0),
        rotation: Quat::from_rotation_y(0.5),
        scale: 1.0,
    };
    send(
        &mut t,
        &[Intent::GrabMove {
            hand: Hand::Left,
            pose: moved,
        }],
    );
    let p = pose(&t, e);
    let in_hand = p.then(&moved.inverse());
    let start_in_hand = Pose::IDENTITY.then(&hand_at(-0.1).inverse());
    assert!((in_hand.translation - start_in_hand.translation).length() < 1e-5);
    // The second hand: a two-hand grab; moving a hand resizes the instance (by the grab
    // points' distance, which `manipulate_grab_points` checks: E11, revision 34).
    // The right ray aimed again at the moved instance's box center.
    let h = &t.app.world().get::<SplatModel>(e).unwrap().0;
    let b = t
        .app
        .world()
        .resource::<Assets<SplatAsset>>()
        .get(h)
        .unwrap()
        .bounds;
    let c = pose(&t, e).transform_point(b.center());
    let aim = Intent::Pointer {
        hand: Hand::Right,
        origin: c + Vec3::new(0.0, 0.0, 3.0),
        dir: -Vec3::Z,
    };
    send(
        &mut t,
        &[
            aim,
            Intent::GrabBegin {
                hand: Hand::Right,
                pose: hand_at(0.6),
            },
        ],
    );
    assert!(matches!(
        t.app.world().get::<Grabbed>(e),
        Some(Grabbed::Two { .. })
    ));
    let s0 = pose(&t, e).scale;
    send(
        &mut t,
        &[Intent::GrabMove {
            hand: Hand::Right,
            pose: hand_at(0.8),
        }],
    );
    assert!((pose(&t, e).scale - s0).abs() > 0.01 * s0, "resizes");
    // Releasing the left returns to a one-hand grab by the right.
    send(&mut t, &[Intent::GrabEnd { hand: Hand::Left }]);
    assert!(matches!(
        t.app.world().get::<Grabbed>(e),
        Some(Grabbed::One {
            hand: Hand::Right,
            ..
        })
    ));
    let before = pose(&t, e);
    send(
        &mut t,
        &[Intent::GrabMove {
            hand: Hand::Right,
            pose: hand_at(1.0),
        }],
    );
    let after = pose(&t, e);
    assert!((after.translation - before.translation - Vec3::new(0.2, 0.0, 0.0)).length() < 1e-5);
    send(&mut t, &[Intent::GrabEnd { hand: Hand::Right }]);
    assert!(t.app.world().get::<Grabbed>(e).is_none());
}

#[test]
fn push_pull_moves_along_the_ray() {
    let mut t = TestApp::new("manip_push");
    let e = loaded(&mut t);
    send(
        &mut t,
        &[
            at_it(Hand::Right),
            Intent::GrabBegin {
                hand: Hand::Right,
                pose: hand_at(0.0),
            },
        ],
    );
    let d = t.app.world().get::<Hovered>(e).unwrap().distance;
    let z0 = pose(&t, e).translation.z;
    send(
        &mut t,
        &[Intent::PushPull {
            hand: Hand::Right,
            axis: 1.0,
        }],
    );
    // §1.8 (revision 35): 1 m/s times the distance in metres, over one 60 Hz frame,
    // away along −z; within U-32's 10⁻⁵.
    let dt = support::FRAME.as_secs_f32();
    let dz = pose(&t, e).translation.z - z0;
    assert!((dz + 1.0 * d * dt).abs() < 1e-5, "{dz}");
    // The grab holds the new place.
    send(
        &mut t,
        &[Intent::GrabMove {
            hand: Hand::Right,
            pose: hand_at(0.0),
        }],
    );
    assert!((pose(&t, e).translation.z - z0 - dz).abs() < 1e-5);
}

#[test]
fn unload_reaches_the_library() {
    let mut t = TestApp::new("manip_unload");
    let e = loaded(&mut t);
    send(
        &mut t,
        &[Intent::Instance {
            instance: e,
            action: InstanceAction::Unload,
        }],
    );
    t.frames(2);
    assert!(t.app.world().get_entity(e).is_err());
}
