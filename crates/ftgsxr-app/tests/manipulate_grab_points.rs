// SPDX-License-Identifier: Apache-2.0

//! E11's two-hand grab points through `ftgsxr_app::manipulate`'s intents (`cpu`, #551;
//! source of truth E11 and U-32 as revised in spec revision 34, §1.8 Rays, selection and
//! grabbing): each hand's grab point is the point of the instance where its ray took
//! hold (the hand's aim position from inside the box), it stays that point of the
//! instance through a one-hand push and pull, then moves fixed in its hand's frame while
//! two hands hold; grab points closer than 5 cm fall back to the hands' aim positions;
//! a pose change during a two-hand grab restarts it with each grab point in place.

use crate::support;

use bevy::prelude::*;
use ftgsxr_app::{
    Grabbed, Hand, InstanceAction, Intent, LibraryCommand, LoadedModel, ManipulateSettings,
    ModelRef,
};
use ftgsxr_core::{Aabb, Pose};
use ftgsxr_render::{SplatAsset, SplatModel};
use ftgsxr_testkit::fixtures_dir;
use support::TestApp;

/// U-16's tolerance, which the grab points inherit through `TwoHandGrab`.
const TOL: f32 = 1e-4;

/// P1 loaded and posed so that its box B_m's longest side is `side_m` long, centred at
/// `center`; returns the entity and B_m.
fn loaded_sized(t: &mut TestApp, side_m: f32, center: Vec3) -> (Entity, Aabb) {
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
    let e = t
        .app
        .world_mut()
        .query_filtered::<Entity, With<LoadedModel>>()
        .single(t.app.world())
        .unwrap();
    let h = t.app.world().get::<SplatModel>(e).unwrap().0.clone();
    let b = t
        .app
        .world()
        .resource::<Assets<SplatAsset>>()
        .get(&h)
        .unwrap()
        .bounds;
    let pose = sized(&b, side_m, center);
    *t.app.world_mut().get_mut::<Transform>(e).unwrap() = Transform {
        translation: pose.translation,
        rotation: pose.rotation,
        scale: Vec3::splat(pose.scale),
    };
    t.frames(1);
    (e, b)
}

/// The pose that makes B_m's longest side `side_m` long, centred at `center`.
fn sized(b: &Aabb, side_m: f32, center: Vec3) -> Pose {
    let k = side_m / (b.max - b.min).max_element();
    Pose {
        translation: center - k * b.center(),
        rotation: Quat::IDENTITY,
        scale: k,
    }
}

fn send(t: &mut TestApp, intents: &[Intent]) {
    for i in intents {
        t.app.world_mut().write_message(*i);
    }
    t.frames(1);
}

fn pose(t: &TestApp, e: Entity) -> Pose {
    let tr = t.app.world().get::<Transform>(e).unwrap();
    Pose {
        translation: tr.translation,
        rotation: tr.rotation,
        scale: tr.scale.x,
    }
}

/// A hand at `at`, its aim along −z: the ray's origin is the aim position.
fn hand(at: Vec3) -> Pose {
    Pose {
        translation: at,
        ..Pose::IDENTITY
    }
}

/// The hand's ray (aim along −z) and its grab press.
fn grab_at(h: Hand, at: Vec3) -> [Intent; 2] {
    [
        Intent::Pointer {
            hand: h,
            origin: at,
            dir: -Vec3::Z,
        },
        Intent::GrabBegin {
            hand: h,
            pose: hand(at),
        },
    ]
}

fn move_to(h: Hand, at: Vec3) -> Intent {
    Intent::GrabMove {
        hand: h,
        pose: hand(at),
    }
}

/// A stage point's box-normalized coordinates under the instance's pose: (0, 0, 0) at
/// B_m's min corner, (1, 1, 1) at its max.
fn normalized(b: &Aabb, p: &Pose, stage: Vec3) -> Vec3 {
    (p.inverse().transform_point(stage) - b.min) / (b.max - b.min)
}

/// E10's hit of a −z ray from `origin` on the box at `p`: the front face, z = max.
fn front_hit(b: &Aabb, p: &Pose, origin: Vec3) -> Vec3 {
    let z = p.transform_point(b.max).z;
    Vec3::new(origin.x, origin.y, z)
}

fn assert_near(a: Vec3, b: Vec3, what: &str) {
    assert!((a - b).length() < TOL, "{what}: {a} against {b}");
}

/// Two rays on a 1 m box from 3 m away grab at `la` and `lb`, then the hands move by
/// each `(da, db)` in turn. With `both`, each grab point (its hit, moved with its hand)
/// keeps its box-normalized coordinates; otherwise their midpoint does.
fn two_rays(
    name: &str,
    free_rotation: bool,
    la: Vec3,
    lb: Vec3,
    moves: &[(Vec3, Vec3)],
    both: bool,
) {
    let mut t = TestApp::new(name);
    t.app
        .world_mut()
        .resource_mut::<ManipulateSettings>()
        .keep_upright = !free_rotation;
    let (e, b) = loaded_sized(&mut t, 1.0, Vec3::ZERO);
    let p0 = pose(&t, e);
    let (ga, gb) = (front_hit(&b, &p0, la), front_hit(&b, &p0, lb));
    let (na, nb) = (normalized(&b, &p0, ga), normalized(&b, &p0, gb));
    let nm = normalized(&b, &p0, (ga + gb) * 0.5);
    let mut intents = grab_at(Hand::Left, la).to_vec();
    intents.extend(grab_at(Hand::Right, lb));
    send(&mut t, &intents);
    assert!(matches!(
        t.app.world().get::<Grabbed>(e),
        Some(Grabbed::Two { .. })
    ));
    for &(da, db) in moves {
        send(
            &mut t,
            &[move_to(Hand::Left, la + da), move_to(Hand::Right, lb + db)],
        );
        let p = pose(&t, e);
        let (a, b2) = (ga + da, gb + db);
        assert_near(
            normalized(&b, &p, (a + b2) * 0.5),
            nm,
            "the grab points' midpoint",
        );
        if both {
            assert_near(normalized(&b, &p, a), na, "the left grab point");
            assert_near(normalized(&b, &p, b2), nb, "the right grab point");
        }
    }
}

/// Keep upright, the grab points level: as the hands move apart, closer, forward, up
/// and turn, each grab point keeps its place on the instance (E11, revision 34; U-16;
/// A-3's "stay under the rays").
#[test]
fn two_hand_grab_points_stay_on_the_instance() {
    two_rays(
        "grab_points_stay",
        false,
        Vec3::new(-0.3, 0.1, 3.0),
        Vec3::new(0.25, 0.1, 3.0),
        &[
            (Vec3::new(-0.2, 0.0, 0.0), Vec3::new(0.2, 0.0, 0.0)),
            (Vec3::new(0.1, 0.0, 0.0), Vec3::new(-0.1, 0.0, 0.0)),
            (Vec3::new(-0.1, 0.0, -0.5), Vec3::new(0.15, 0.0, -0.5)),
            (Vec3::new(0.0, 0.3, 0.2), Vec3::new(0.3, 0.3, 0.2)),
            (Vec3::new(0.0, 0.0, 0.0), Vec3::new(-0.2, 0.0, -0.4)),
        ],
        true,
    );
}

/// Keep upright with the grab points at different heights: the midpoint keeps its
/// place on the instance whatever the hands do (E11, revision 34; U-16).
#[test]
fn upright_grab_points_keep_their_midpoint() {
    two_rays(
        "grab_points_upright",
        false,
        Vec3::new(-0.3, 0.1, 3.0),
        Vec3::new(0.25, -0.2, 3.0),
        &[
            (Vec3::new(-0.2, 0.0, 0.0), Vec3::new(0.2, 0.0, 0.0)),
            (Vec3::new(0.0, 0.2, -0.3), Vec3::new(0.1, -0.1, 0.2)),
        ],
        false,
    );
}

/// Free rotation, the grab points at different heights, the hands moving every way:
/// each grab point keeps its place on the instance (E11, revision 34; U-16).
#[test]
fn free_rotation_grab_points_stay_on_the_instance() {
    two_rays(
        "grab_points_free",
        true,
        Vec3::new(-0.3, 0.1, 3.0),
        Vec3::new(0.25, -0.2, 3.0),
        &[
            (Vec3::new(-0.2, 0.0, 0.0), Vec3::new(0.2, 0.0, 0.0)),
            (Vec3::new(0.0, 0.2, -0.3), Vec3::new(0.1, -0.1, 0.2)),
            (Vec3::new(0.1, -0.1, 0.4), Vec3::new(-0.2, 0.3, -0.1)),
        ],
        true,
    );
}

/// A one-hand push moves the instance along the ray; the left grab point stays the point
/// of the instance it was (U-32, revision 34), so the two-hand grab that follows holds it
/// there, not at the hand's old hit distance.
#[test]
fn a_grab_point_stays_on_the_instance_through_push_and_pull() {
    let mut t = TestApp::new("grab_points_push");
    let (e, b) = loaded_sized(&mut t, 1.0, Vec3::ZERO);
    let p0 = pose(&t, e);
    let la = Vec3::new(-0.3, 0.1, 3.0);
    let na = normalized(&b, &p0, front_hit(&b, &p0, la));
    send(&mut t, &grab_at(Hand::Left, la));
    for _ in 0..30 {
        send(
            &mut t,
            &[Intent::PushPull {
                hand: Hand::Left,
                axis: 1.0,
            }],
        );
    }
    let p1 = pose(&t, e);
    assert!(
        p1.translation.z < p0.translation.z - 0.01,
        "the push moved it"
    );
    // The left grab point now lies on the instance where it was taken.
    let ga = p1.transform_point(na * (b.max - b.min) + b.min);
    // Level with the left, so that keep upright holds both points (U-16).
    let lb = Vec3::new(0.25, 0.1, 3.0);
    let gb = front_hit(&b, &p1, lb);
    let nb = normalized(&b, &p1, gb);
    send(&mut t, &grab_at(Hand::Right, lb));
    let (da, db) = (Vec3::new(-0.2, 0.0, 0.0), Vec3::new(0.2, 0.0, 0.0));
    send(
        &mut t,
        &[move_to(Hand::Left, la + da), move_to(Hand::Right, lb + db)],
    );
    let p = pose(&t, e);
    assert_near(
        normalized(&b, &p, ga + da),
        na,
        "the pushed left grab point",
    );
    assert_near(normalized(&b, &p, gb + db), nb, "the right grab point");
}

/// Both rays on one spot: the grab points lie closer than 5 cm, so the grab uses the
/// hands' aim positions and spreading the hands still resizes (E11, revision 34; U-32).
#[test]
fn close_grab_points_fall_back_to_the_hands() {
    let mut t = TestApp::new("grab_points_close");
    let (e, _) = loaded_sized(&mut t, 1.0, Vec3::ZERO);
    let o = Vec3::new(0.0, 0.0, 3.0);
    let ray = |h: Hand| Intent::Pointer {
        hand: h,
        origin: o,
        dir: -Vec3::Z,
    };
    let (ha, hb) = (Vec3::new(-0.1, 0.0, 2.0), Vec3::new(0.1, 0.0, 2.0));
    send(
        &mut t,
        &[
            ray(Hand::Left),
            ray(Hand::Right),
            Intent::GrabBegin {
                hand: Hand::Left,
                pose: hand(ha),
            },
            Intent::GrabBegin {
                hand: Hand::Right,
                pose: hand(hb),
            },
        ],
    );
    let p0 = pose(&t, e);
    send(&mut t, &[move_to(Hand::Right, Vec3::new(0.3, 0.0, 2.0))]);
    let p = pose(&t, e);
    assert!(
        (p.scale - 2.0 * p0.scale).abs() < TOL * p0.scale,
        "doubled: {}",
        p.scale
    );
    // Scaled about the hands' midpoint, which moved to (0.1, 0, 2).
    let m0 = (ha + hb) * 0.5;
    let m = (ha + Vec3::new(0.3, 0.0, 2.0)) * 0.5;
    assert_near(
        p.translation,
        m + (p0.translation - m0) * 2.0,
        "about the hands",
    );
}

/// From inside a 4 m box, E10 gives 0, so each grab point is the hand's aim position;
/// as the hands move, each keeps its box-normalized coordinates (E11, revision 34;
/// U-32; A-3's "from inside the model").
#[test]
fn from_inside_the_box_the_grab_points_are_the_hands() {
    let mut t = TestApp::new("grab_points_inside");
    let (e, b) = loaded_sized(&mut t, 4.0, Vec3::new(0.0, 1.0, 0.0));
    let p0 = pose(&t, e);
    let (la, lb) = (Vec3::new(-0.2, 1.2, 0.3), Vec3::new(0.2, 1.2, 0.3));
    let (na, nb) = (normalized(&b, &p0, la), normalized(&b, &p0, lb));
    for v in [na, nb] {
        assert!(
            v.cmpgt(Vec3::ZERO).all() && v.cmplt(Vec3::ONE).all(),
            "inside: {v}"
        );
    }
    let mut intents = grab_at(Hand::Left, la).to_vec();
    intents.extend(grab_at(Hand::Right, lb));
    send(&mut t, &intents);
    assert!(matches!(
        t.app.world().get::<Grabbed>(e),
        Some(Grabbed::Two { .. })
    ));
    let (a, bb) = (Vec3::new(-0.4, 1.2, 0.1), Vec3::new(0.4, 1.2, 0.1));
    send(&mut t, &[move_to(Hand::Left, a), move_to(Hand::Right, bb)]);
    let p = pose(&t, e);
    assert!((p.scale - 2.0 * p0.scale).abs() < TOL * p0.scale, "doubled");
    assert_near(normalized(&b, &p, a), na, "the left hand");
    assert_near(normalized(&b, &p, bb), nb, "the right hand");
}

/// Snap to floor during a two-hand grab restarts it with each grab point at the same
/// place on the instance (E11, revision 34; Q-183's restart).
#[test]
fn a_pose_change_restarts_the_grab_with_the_points_in_place() {
    let mut t = TestApp::new("grab_points_rebase");
    let (e, b) = loaded_sized(&mut t, 1.0, Vec3::new(0.0, 1.5, 0.0));
    let p0 = pose(&t, e);
    let (la, lb) = (Vec3::new(-0.3, 1.6, 3.0), Vec3::new(0.25, 1.6, 3.0));
    let (na, nb) = (
        normalized(&b, &p0, front_hit(&b, &p0, la)),
        normalized(&b, &p0, front_hit(&b, &p0, lb)),
    );
    let mut intents = grab_at(Hand::Left, la).to_vec();
    intents.extend(grab_at(Hand::Right, lb));
    send(&mut t, &intents);
    send(
        &mut t,
        &[Intent::Instance {
            instance: e,
            action: InstanceAction::SnapToFloor,
        }],
    );
    let p1 = pose(&t, e);
    assert!(p1.translation.y < p0.translation.y - 0.1, "snapped down");
    // The grab points after the restart: the same places on the snapped instance.
    let point = |n: Vec3| p1.transform_point(n * (b.max - b.min) + b.min);
    let (ga, gb) = (point(na), point(nb));
    let (da, db) = (Vec3::new(-0.2, 0.0, 0.0), Vec3::new(0.2, 0.0, 0.0));
    send(
        &mut t,
        &[move_to(Hand::Left, la + da), move_to(Hand::Right, lb + db)],
    );
    let p = pose(&t, e);
    assert_near(normalized(&b, &p, ga + da), na, "the left grab point");
    assert_near(normalized(&b, &p, gb + db), nb, "the right grab point");
}
