// SPDX-License-Identifier: Apache-2.0

//! `manipulate` under a content group that isn't the identity (#419; source of truth
//! §1.14 Frames: the content group is a transform applied to every instance's pose, the
//! rays, the head and the floor are in stage space): picking and hover (E10), fit and
//! snap to floor (§1.5, I-4's tolerance), grabs (E11), `Turn` and `Nudge` (Q-184, I-8's
//! tolerance), the gizmo's stage axes (Q-186), the saved pose relative to the group
//! (Q-245) and keep upright about stage +y (Q-246). Where an instance is drawn is read
//! from its `GlobalTransform` after Bevy's transform propagation.

use crate::support::TestApp;

use bevy::prelude::*;
use ftgsxr_app::{
    ContentGroup, Hand, Hovered, InstanceAction, Intent, LibraryCommand, LoadedModel, ModelRef,
    Selected,
};
use ftgsxr_core::{Aabb, Pose};
use ftgsxr_render::{SplatAsset, SplatModel};
use ftgsxr_testkit::fixtures_dir;

/// I-4's tolerance.
const TOL: f32 = 1e-4;

/// I-8's tolerance.
const I8_TOL: f32 = 1e-6;

/// A group moved 4 m along +x and 0.5 m up, turned 90° about y and scaled 2.
fn moved_group() -> Transform {
    Transform {
        translation: Vec3::new(4.0, 0.5, 0.0),
        rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        scale: Vec3::splat(2.0),
    }
}

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

/// A loaded P1 under a content group set to `group`, propagated.
fn under(name: &str, group: Transform) -> (TestApp, Entity) {
    let mut t = TestApp::new(name);
    let e = loaded(&mut t);
    set_group(&mut t, group);
    (t, e)
}

fn set_group(t: &mut TestApp, group: Transform) {
    let mut q = t
        .app
        .world_mut()
        .query_filtered::<&mut Transform, With<ContentGroup>>();
    *q.single_mut(t.app.world_mut()).unwrap() = group;
    t.frames(1);
}

fn send(t: &mut TestApp, intents: &[Intent]) {
    for i in intents {
        t.app.world_mut().write_message(*i);
    }
    // One frame for the intents, one for the propagation to reach `GlobalTransform`
    // when the intents ran after it.
    t.frames(2);
}

/// Where the instance is drawn: its stage pose.
fn stage(t: &TestApp, e: Entity) -> Pose {
    let (s, rotation, translation) = t
        .app
        .world()
        .get::<GlobalTransform>(e)
        .unwrap()
        .to_scale_rotation_translation();
    Pose {
        translation,
        rotation,
        scale: s.x,
    }
}

fn local(t: &TestApp, e: Entity) -> Pose {
    let tr = t.app.world().get::<Transform>(e).unwrap();
    Pose {
        translation: tr.translation,
        rotation: tr.rotation,
        scale: tr.scale.x,
    }
}

fn bounds(t: &TestApp, e: Entity) -> Aabb {
    let h = &t.app.world().get::<SplatModel>(e).unwrap().0;
    t.app
        .world()
        .resource::<Assets<SplatAsset>>()
        .get(h)
        .unwrap()
        .bounds
}

fn corners(b: &Aabb, p: &Pose) -> Vec<Vec3> {
    (0..8)
        .map(|k| {
            p.transform_point(Vec3::new(
                if k & 1 == 0 { b.min.x } else { b.max.x },
                if k & 2 == 0 { b.min.y } else { b.max.y },
                if k & 4 == 0 { b.min.z } else { b.max.z },
            ))
        })
        .collect()
}

/// A ray from 5 m along +z of `at`, toward it.
fn ray_at(hand: Hand, at: Vec3) -> Intent {
    Intent::Pointer {
        hand,
        origin: at + Vec3::new(0.0, 0.0, 5.0),
        dir: -Vec3::Z,
    }
}

#[test]
fn content_group_rays_pick_the_instance_where_it_is_drawn() {
    let (mut t, e) = under("cg_pick", moved_group());
    let b = bounds(&t, e);
    let drawn = stage(&t, e).transform_point(b.center());
    let unmoved = local(&t, e).transform_point(b.center());
    // The drawn box and the group-relative one are apart: a ray through one misses the
    // other.
    let half = 2.0 * b.size().max_element();
    assert!(
        (drawn - unmoved).abs().max_element() > half,
        "{drawn} {unmoved}"
    );
    send(&mut t, &[ray_at(Hand::Mouse, drawn)]);
    let h = *t
        .app
        .world()
        .get::<Hovered>(e)
        .expect("hovered where drawn");
    // E10's entry distance along the stage ray: the box's front face, within 5 m.
    assert!(h.distance > 0.0 && h.distance < 5.0, "{}", h.distance);
    send(
        &mut t,
        &[Intent::Select {
            hand: Hand::Mouse,
            pressed: true,
        }],
    );
    assert!(t.app.world().get::<Selected>(e).is_some());
    send(&mut t, &[ray_at(Hand::Mouse, unmoved)]);
    assert!(
        t.app.world().get::<Hovered>(e).is_none(),
        "nothing is drawn there"
    );
}

fn act(t: &mut TestApp, instance: Entity, action: InstanceAction) {
    send(t, &[Intent::Instance { instance, action }]);
}

fn near(a: Pose, b: Pose, tol: f32) -> bool {
    (a.translation - b.translation).length() < tol
        && (a.rotation * Vec3::X - b.rotation * Vec3::X).length() < tol
        && (a.rotation * Vec3::Y - b.rotation * Vec3::Y).length() < tol
        && (a.scale - b.scale).abs() < tol
}

fn at(translation: Vec3) -> Pose {
    Pose {
        translation,
        ..Pose::IDENTITY
    }
}

#[test]
fn content_group_fit_and_snap_are_in_stage_space() {
    let (mut t, e) = under("cg_fit", moved_group());
    let b = bounds(&t, e);
    act(&mut t, e, InstanceAction::Fit);
    // No camera: the head is the stage origin facing −z (Q-180).
    let p = stage(&t, e);
    let bottom = corners(&b, &p)
        .iter()
        .map(|c| c.y)
        .fold(f32::INFINITY, f32::min);
    assert!(bottom.abs() < TOL, "{bottom}");
    let c = p.transform_point(b.center());
    assert!(c.x.abs() < TOL && (c.z + 1.5).abs() < TOL, "{c}");
    assert!((p.scale * b.size().max_element() - 1.0).abs() < TOL);
    // Lifted relative to the group (0.3 m up its y, 0.6 m in stage space), then
    // snapped: the bottom back on stage y = 0, the rest kept.
    t.app
        .world_mut()
        .get_mut::<Transform>(e)
        .unwrap()
        .translation
        .y += 0.3;
    t.frames(1);
    assert!((stage(&t, e).translation.y - p.translation.y - 0.6).abs() < TOL);
    act(&mut t, e, InstanceAction::SnapToFloor);
    assert!(near(stage(&t, e), p, TOL));
}

#[test]
fn content_group_a_one_hand_grab_moves_the_stage_pose_with_the_hand() {
    let (mut t, e) = under("cg_grab", moved_group());
    let c = stage(&t, e).transform_point(bounds(&t, e).center());
    let before = stage(&t, e);
    let h0 = at(c + Vec3::new(0.0, 0.0, 2.0));
    send(
        &mut t,
        &[
            ray_at(Hand::Right, c),
            Intent::GrabBegin {
                hand: Hand::Right,
                pose: h0,
            },
        ],
    );
    // E11: P = H M, so a hand moved by Δ moves the instance's stage pose by Δ.
    let delta = Vec3::new(0.3, 0.1, -0.2);
    send(
        &mut t,
        &[Intent::GrabMove {
            hand: Hand::Right,
            pose: at(h0.translation + delta),
        }],
    );
    let want = Pose {
        translation: before.translation + delta,
        ..before
    };
    assert!(near(stage(&t, e), want, TOL), "{:?}", stage(&t, e));
}

#[test]
fn content_group_turn_and_nudge_are_in_stage_space() {
    let (mut t, e) = under("cg_nudge", moved_group());
    let p0 = stage(&t, e);
    // Q-184: `Nudge` adds its delta to the stage translation.
    let delta = Vec3::new(0.01, 0.0, 0.0);
    send(&mut t, &[Intent::Nudge { instance: e, delta }]);
    let p1 = stage(&t, e);
    let want = Pose {
        translation: p0.translation + delta,
        ..p0
    };
    assert!(near(p1, want, I8_TOL), "{p1:?}");
    // Q-184: `Turn` turns about stage +y through the instance's origin.
    send(
        &mut t,
        &[Intent::Turn {
            instance: e,
            yaw_deg: 5.0,
        }],
    );
    let want = Pose {
        rotation: Quat::from_rotation_y(5f32.to_radians()) * p1.rotation,
        ..p1
    };
    assert!(near(stage(&t, e), want, I8_TOL), "{:?}", stage(&t, e));
}

#[test]
fn content_group_a_gizmo_arrow_moves_along_stage_x() {
    let (mut t, e) = under("cg_gizmo", moved_group());
    let p0 = stage(&t, e);
    let c = p0.transform_point(bounds(&t, e).center());
    let select = |pressed| Intent::Select {
        hand: Hand::Mouse,
        pressed,
    };
    send(
        &mut t,
        &[ray_at(Hand::Mouse, c), select(true), select(false)],
    );
    assert!(t.app.world().get::<Selected>(e).is_some());
    // Q-186: the arrows run along stage x, y and z from the posed box's center, of a
    // fifth of the distance from the viewer (the stage origin without a camera).
    let size = (0.2 * c.length()).max(0.05);
    let on_arrow = c + 0.5 * size * Vec3::X;
    send(&mut t, &[ray_at(Hand::Mouse, on_arrow), select(true)]);
    send(
        &mut t,
        &[ray_at(Hand::Mouse, on_arrow + Vec3::new(0.3, 0.2, 0.0))],
    );
    let want = Pose {
        translation: p0.translation + Vec3::new(0.3, 0.0, 0.0),
        ..p0
    };
    assert!(near(stage(&t, e), want, TOL), "{:?}", stage(&t, e));
}

#[test]
fn content_group_saved_pose_is_relative_to_the_group() {
    // Q-245
    let (mut t, e) = under("cg_saved", moved_group());
    let l = Pose {
        translation: Vec3::new(0.3, 0.2, -1.0),
        rotation: Quat::from_rotation_y(0.4),
        scale: 1.5,
    };
    *t.app.world_mut().get_mut::<Transform>(e).unwrap() = Transform {
        translation: l.translation,
        rotation: l.rotation,
        scale: Vec3::splat(l.scale),
    };
    t.frames(1);
    act(&mut t, e, InstanceAction::SavePose);
    let id = t
        .app
        .world()
        .get::<LoadedModel>(e)
        .unwrap()
        .identity
        .clone();
    let saved = t
        .app
        .world()
        .resource::<ftgsxr_app::Poses>()
        .0
        .get(&id)
        .cloned()
        .expect("saved");
    assert!(near(saved.pose, l, TOL), "{:?}", saved.pose);
    // The group moves; reset returns to the saved pose relative to it.
    set_group(&mut t, Transform::from_xyz(-1.0, 0.0, 2.0));
    *t.app.world_mut().get_mut::<Transform>(e).unwrap() = Transform::IDENTITY;
    act(&mut t, e, InstanceAction::ResetPose);
    assert!(near(local(&t, e), l, TOL), "{:?}", local(&t, e));
    let want = l.then(&at(Vec3::new(-1.0, 0.0, 2.0)));
    assert!(near(stage(&t, e), want, TOL));
}

#[test]
fn content_group_upright_grab_turns_about_stage_y() {
    // Q-246: keep upright (on by default) yaws about stage +y under a tilted group.
    let tilted = Transform {
        translation: Vec3::new(0.0, 1.0, -1.0),
        rotation: Quat::from_rotation_x(30f32.to_radians()),
        scale: Vec3::ONE,
    };
    let (mut t, e) = under("cg_upright", tilted);
    let before = stage(&t, e);
    let c = before.transform_point(bounds(&t, e).center());
    let m = c + Vec3::new(0.0, 0.0, 0.5);
    let (a0, b0) = (m - Vec3::new(0.3, 0.0, 0.0), m + Vec3::new(0.3, 0.0, 0.0));
    send(
        &mut t,
        &[
            ray_at(Hand::Left, c),
            ray_at(Hand::Right, c),
            Intent::GrabBegin {
                hand: Hand::Left,
                pose: at(a0),
            },
            Intent::GrabBegin {
                hand: Hand::Right,
                pose: at(b0),
            },
        ],
    );
    // The hands' axis turned 90° about stage +y about their midpoint: (1, 0, 0) to
    // (0, 0, −1), its length kept.
    send(
        &mut t,
        &[
            Intent::GrabMove {
                hand: Hand::Left,
                pose: at(m + Vec3::new(0.0, 0.0, 0.3)),
            },
            Intent::GrabMove {
                hand: Hand::Right,
                pose: at(m - Vec3::new(0.0, 0.0, 0.3)),
            },
        ],
    );
    let after = stage(&t, e);
    let turn = after.rotation * before.rotation.inverse();
    let want = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    assert!(
        (turn * Vec3::X - want * Vec3::X).length() < TOL
            && (turn * Vec3::Y - Vec3::Y).length() < TOL,
        "{turn:?}"
    );
    assert!((after.scale - before.scale).abs() < TOL);
}
