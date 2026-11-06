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

/// U-32's tolerance for its two-hand grab-point clause (spec revision 36): U-16's
/// 10⁻⁵, the tolerance of `TwoHandGrab` that the grab points go through, relative for
/// scale ratios.
const TOL: f32 = 1e-5;

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
    move_pose(h, hand(at))
}

fn move_pose(h: Hand, pose: Pose) -> Intent {
    Intent::GrabMove { hand: h, pose }
}

/// A test app with keep upright on, or off for free rotation.
fn app_with(name: &str, free_rotation: bool) -> TestApp {
    let mut t = TestApp::new(name);
    t.app
        .world_mut()
        .resource_mut::<ManipulateSettings>()
        .keep_upright = !free_rotation;
    t
}

/// Both hands grab with their rays from `la` and `lb`.
fn grab_both(t: &mut TestApp, la: Vec3, lb: Vec3) {
    let mut intents = grab_at(Hand::Left, la).to_vec();
    intents.extend(grab_at(Hand::Right, lb));
    send(t, &intents);
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

/// A box-normalized point of B_m under the instance's pose, in stage space.
fn at_normalized(b: &Aabb, p: &Pose, n: Vec3) -> Vec3 {
    p.transform_point(b.min + n * (b.max - b.min))
}

fn grab_end(h: Hand) -> Intent {
    Intent::GrabEnd { hand: h }
}

fn assert_near(a: Vec3, b: Vec3, what: &str) {
    assert!((a - b).length() < TOL, "{what}: {a} against {b}");
}

/// Two rays on a 1 m box from 3 m away grab at `la` and `lb`, then the hands move by
/// each `(da, db)` in turn. With `both`, each grab point (its hit, moved with its hand)
/// keeps its box-normalized coordinates; the midpoint always does.
fn two_rays(
    name: &str,
    free_rotation: bool,
    la: Vec3,
    lb: Vec3,
    moves: &[(Vec3, Vec3)],
    both: bool,
) {
    let mut t = app_with(name, free_rotation);
    let (e, b) = loaded_sized(&mut t, 1.0, Vec3::ZERO);
    let p0 = pose(&t, e);
    let (ga, gb) = (front_hit(&b, &p0, la), front_hit(&b, &p0, lb));
    let (na, nb) = (normalized(&b, &p0, ga), normalized(&b, &p0, gb));
    let nm = normalized(&b, &p0, (ga + gb) * 0.5);
    grab_both(&mut t, la, lb);
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
        let (left, right) = (ga + da, gb + db);
        assert_near(
            normalized(&b, &p, (left + right) * 0.5),
            nm,
            "the grab points' midpoint",
        );
        if both {
            assert_near(normalized(&b, &p, left), na, "the left grab point");
            assert_near(normalized(&b, &p, right), nb, "the right grab point");
        }
    }
}

/// Keep upright, the grab points level: as the hands move apart, closer, forward, up
/// and turn, each grab point keeps its place on the instance (E11, revision 34; U-16;
/// A-3's "stay under the rays").
#[test]
fn u32_two_hand_grab_points_stay_on_the_instance() {
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
fn u32_upright_grab_points_keep_their_midpoint() {
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
fn u32_free_rotation_grab_points_stay_on_the_instance() {
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

/// Turning hands carry their grab points as rigid offsets (a = H H₀⁻¹ a₀, E11 revision
/// 34): a grab point 3 m down a ray swings with its hand's turn, and each keeps its place
/// on the instance (free rotation; U-32's "fixed in its hand's frame").
#[test]
fn u32_turning_hands_carry_their_grab_points() {
    let mut t = app_with("grab_points_turn", true);
    let (e, b) = loaded_sized(&mut t, 1.0, Vec3::ZERO);
    let p0 = pose(&t, e);
    let (la, lb) = (Vec3::new(-0.3, 0.1, 3.0), Vec3::new(0.25, -0.2, 3.0));
    let (ga, gb) = (front_hit(&b, &p0, la), front_hit(&b, &p0, lb));
    let (na, nb) = (normalized(&b, &p0, ga), normalized(&b, &p0, gb));
    grab_both(&mut t, la, lb);
    // The hands' start poses are unrotated, so each offset in its hand's frame is g − l.
    for (ra, rb, da, db) in [
        (
            Quat::from_rotation_y(0.2),
            Quat::from_rotation_x(-0.1),
            Vec3::new(-0.1, 0.0, 0.0),
            Vec3::new(0.1, 0.05, 0.0),
        ),
        (
            Quat::from_rotation_z(0.4) * Quat::from_rotation_y(-0.15),
            Quat::from_rotation_y(0.3),
            Vec3::new(0.05, -0.1, 0.2),
            Vec3::new(0.2, 0.0, -0.1),
        ),
    ] {
        let (ha, hb) = (
            Pose {
                translation: la + da,
                rotation: ra,
                scale: 1.0,
            },
            Pose {
                translation: lb + db,
                rotation: rb,
                scale: 1.0,
            },
        );
        send(
            &mut t,
            &[move_pose(Hand::Left, ha), move_pose(Hand::Right, hb)],
        );
        let p = pose(&t, e);
        let (left, right) = (la + da + ra * (ga - la), lb + db + rb * (gb - lb));
        assert_near(normalized(&b, &p, left), na, "the turned left grab point");
        assert_near(normalized(&b, &p, right), nb, "the turned right grab point");
    }
}

/// A one-hand push moves the instance along the ray; the left grab point stays the point
/// of the instance it was (U-32, revision 34), so the two-hand grab that follows holds it
/// there, not at the hand's old hit distance.
#[test]
fn u32_a_grab_point_stays_on_the_instance_through_push_and_pull() {
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
    let ga = at_normalized(&b, &p1, na);
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

/// Releasing one hand returns to a one-hand grab; when the other hand grabs again, the
/// held hand's grab point is still the point of the instance it took first (E11,
/// revision 34: "it stays that point of the instance").
#[test]
fn u32_a_regrab_keeps_the_held_hand_s_grab_point() {
    let mut t = TestApp::new("grab_points_regrab");
    let (e, b) = loaded_sized(&mut t, 1.0, Vec3::ZERO);
    let p0 = pose(&t, e);
    let (la, lb) = (Vec3::new(-0.3, 0.1, 3.0), Vec3::new(0.25, 0.1, 3.0));
    let na = normalized(&b, &p0, front_hit(&b, &p0, la));
    grab_both(&mut t, la, lb);
    send(
        &mut t,
        &[
            move_to(Hand::Left, la + Vec3::new(-0.1, 0.0, 0.0)),
            move_to(Hand::Right, lb + Vec3::new(0.2, 0.0, 0.0)),
        ],
    );
    send(&mut t, &[grab_end(Hand::Right)]);
    // The left hand alone carries the instance 0.3 m up.
    let la1 = la + Vec3::new(-0.1, 0.3, 0.0);
    send(&mut t, &[move_to(Hand::Left, la1)]);
    let p1 = pose(&t, e);
    let ga = at_normalized(&b, &p1, na);
    let lb2 = Vec3::new(0.2, 0.4, 3.0);
    let gb = front_hit(&b, &p1, lb2);
    let nb = normalized(&b, &p1, gb);
    send(&mut t, &grab_at(Hand::Right, lb2));
    assert!(matches!(
        t.app.world().get::<Grabbed>(e),
        Some(Grabbed::Two { .. })
    ));
    let (da, db) = (Vec3::new(-0.15, 0.0, 0.1), Vec3::new(0.15, 0.0, 0.1));
    send(
        &mut t,
        &[
            move_to(Hand::Left, la1 + da),
            move_to(Hand::Right, lb2 + db),
        ],
    );
    let p = pose(&t, e);
    assert_near(normalized(&b, &p, ga + da), na, "the held left grab point");
    assert_near(normalized(&b, &p, gb + db), nb, "the new right grab point");
}

/// Two rays `apart_m` apart on the box, from hands 20 cm apart, the right hand moving 20 cm
/// out: the scale ratio of the instance.
fn ratio_for_rays_apart(name: &str, apart_m: f32) -> f32 {
    let mut t = TestApp::new(name);
    let (e, _) = loaded_sized(&mut t, 1.0, Vec3::ZERO);
    let (oa, ob) = (Vec3::new(0.0, 0.0, 3.0), Vec3::new(apart_m, 0.0, 3.0));
    let ray = |h: Hand, origin: Vec3| Intent::Pointer {
        hand: h,
        origin,
        dir: -Vec3::Z,
    };
    let (ha, hb) = (Vec3::new(-0.1, 0.0, 2.0), Vec3::new(0.1, 0.0, 2.0));
    send(
        &mut t,
        &[
            ray(Hand::Left, oa),
            ray(Hand::Right, ob),
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
    let s0 = pose(&t, e).scale;
    send(
        &mut t,
        &[move_to(Hand::Right, hb + Vec3::new(0.2, 0.0, 0.0))],
    );
    pose(&t, e).scale / s0
}

/// E11's fallback (revision 34): grab points 4 cm apart are closer than 5 cm, so the
/// grab uses the hands' aim positions (20 cm → 40 cm: twice the size); 6 cm apart they
/// are the grab points (6 cm → 26 cm: 26/6 the size).
#[test]
fn u32_grab_points_closer_than_5_cm_fall_back_to_the_hands() {
    let close = ratio_for_rays_apart("grab_points_4cm", 0.04);
    assert!((close - 2.0).abs() < TOL, "4 cm apart: {close}");
    let apart = ratio_for_rays_apart("grab_points_6cm", 0.06);
    let expected = 0.26 / 0.06;
    assert!(
        (apart - expected).abs() < TOL * expected,
        "6 cm apart: {apart}"
    );
}

/// From inside a 4 m box, E10 gives 0, so each grab point is the hand's aim position;
/// as the hands move, each keeps its box-normalized coordinates (E11, revision 34;
/// U-32; A-3's "from inside the model").
#[test]
fn u32_from_inside_the_box_the_grab_points_are_the_hands() {
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
    grab_both(&mut t, la, lb);
    assert!(matches!(
        t.app.world().get::<Grabbed>(e),
        Some(Grabbed::Two { .. })
    ));
    let (a, b2) = (Vec3::new(-0.4, 1.2, 0.1), Vec3::new(0.4, 1.2, 0.1));
    send(&mut t, &[move_to(Hand::Left, a), move_to(Hand::Right, b2)]);
    let p = pose(&t, e);
    assert!((p.scale - 2.0 * p0.scale).abs() < TOL * p0.scale, "doubled");
    assert_near(normalized(&b, &p, a), na, "the left hand");
    assert_near(normalized(&b, &p, b2), nb, "the right hand");
}

/// A pose change during a two-hand grab restarts it with each grab point at the same
/// place on the instance (E11, revision 34; Q-183's restart).
fn restart_after(name: &str, action: InstanceAction) {
    let mut t = TestApp::new(name);
    let (e, b) = loaded_sized(&mut t, 1.0, Vec3::new(0.0, 1.5, 0.0));
    let p0 = pose(&t, e);
    let (la, lb) = (Vec3::new(-0.3, 1.6, 3.0), Vec3::new(0.25, 1.6, 3.0));
    let (na, nb) = (
        normalized(&b, &p0, front_hit(&b, &p0, la)),
        normalized(&b, &p0, front_hit(&b, &p0, lb)),
    );
    grab_both(&mut t, la, lb);
    send(
        &mut t,
        &[Intent::Instance {
            instance: e,
            action,
        }],
    );
    let p1 = pose(&t, e);
    assert!(
        (p1.translation - p0.translation).length() > 0.1,
        "{action:?} moved it"
    );
    // The grab points after the restart: the same places on the moved instance.
    let (ga, gb) = (at_normalized(&b, &p1, na), at_normalized(&b, &p1, nb));
    let (da, db) = (Vec3::new(-0.2, 0.0, 0.0), Vec3::new(0.2, 0.0, 0.0));
    send(
        &mut t,
        &[move_to(Hand::Left, la + da), move_to(Hand::Right, lb + db)],
    );
    let p = pose(&t, e);
    assert_near(normalized(&b, &p, ga + da), na, "the left grab point");
    assert_near(normalized(&b, &p, gb + db), nb, "the right grab point");
}

#[test]
fn u32_snap_to_floor_restarts_the_grab_with_the_points_in_place() {
    restart_after("grab_points_snap", InstanceAction::SnapToFloor);
}

#[test]
fn u32_fit_restarts_the_grab_with_the_points_in_place() {
    restart_after("grab_points_fit", InstanceAction::Fit);
}
