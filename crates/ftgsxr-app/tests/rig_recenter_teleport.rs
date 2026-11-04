// SPDX-License-Identifier: Apache-2.0

//! `rig`'s consumers of `Intent::Teleport` and `Intent::Recenter` (`cpu`, #455; source
//! of truth §1.8 Locomotion: releasing teleports the rig to the arc's floor point; §1.8
//! Session and spaces: recenter moves the content group so the selected instance, or
//! the scene's center, stands 1.5 m in front of the user at its saved height; Q-254: the
//! group stays yaw-only; Q-269, Q-282, Q-283). Where things are is read from
//! `GlobalTransform` after Bevy's transform propagation.

use crate::support::TestApp;

use bevy::prelude::*;
use ftgsxr_app::{
    ContentGroup, Intent, LibraryCommand, LoadedModel, ModelRef, PlayerRig, Selected,
};
use ftgsxr_render::{SplatAsset, SplatModel};
use ftgsxr_testkit::fixtures_dir;

const TOL: f32 = 1e-4;

fn rig(t: &mut TestApp) -> Entity {
    let mut q = t
        .app
        .world_mut()
        .query_filtered::<Entity, With<PlayerRig>>();
    q.single(t.app.world()).unwrap()
}

fn global(t: &TestApp, e: Entity) -> Vec3 {
    t.app
        .world()
        .get::<GlobalTransform>(e)
        .unwrap()
        .translation()
}

fn send(t: &mut TestApp, intent: Intent) {
    t.app.world_mut().write_message(intent);
    // One frame for the intent, one for the propagation.
    t.frames(2);
}

/// A head camera under the rig, off the rig's origin, and the rig moved, raised and
/// turned, so that only the head's own floor point can land on the target.
fn rig_with_head(name: &str) -> (TestApp, Entity, Entity) {
    let mut t = TestApp::new(name);
    let r = rig(&mut t);
    *t.app.world_mut().get_mut::<Transform>(r).unwrap() = Transform {
        translation: Vec3::new(1.0, 0.2, 2.0),
        rotation: Quat::from_rotation_y(30f32.to_radians()),
        scale: Vec3::ONE,
    };
    let head = t
        .app
        .world_mut()
        .spawn((
            Camera3d::default(),
            Transform::from_xyz(0.3, 1.6, 0.2),
            ChildOf(r),
        ))
        .id();
    t.frames(2);
    (t, r, head)
}

#[test]
fn teleport_puts_the_heads_floor_point_on_the_target_height_unchanged() {
    let (mut t, r, head) = rig_with_head("teleport_head");
    let head_y = global(&t, head).y;
    let target = Vec3::new(5.0, 0.0, -3.0);
    send(&mut t, Intent::Teleport { target });
    let h = global(&t, head);
    assert!(
        (h.x - target.x).abs() < TOL && (h.z - target.z).abs() < TOL,
        "{h} vs {target}"
    );
    assert!((h.y - head_y).abs() < TOL, "{} vs {head_y}", h.y);
    assert!(
        (global(&t, r).y - 0.2).abs() < TOL,
        "the rig keeps its height"
    );
    // A second teleport to the same point leaves the rig where the first put it.
    let before = global(&t, r);
    send(&mut t, Intent::Teleport { target });
    assert!((global(&t, r) - before).length() < TOL);
}

#[test]
fn teleport_moves_nothing_in_passthrough_mode() {
    // D-17: stick locomotion is off in passthrough mode (Q-282).
    let (mut t, r, _) = rig_with_head("teleport_passthrough");
    t.app
        .world_mut()
        .resource_mut::<ftgsxr_app::Locomotion>()
        .enabled = false;
    let before = global(&t, r);
    send(
        &mut t,
        Intent::Teleport {
            target: Vec3::new(5.0, 0.0, -3.0),
        },
    );
    assert!((global(&t, r) - before).length() < TOL);
}

/// P1 and P2 loaded, P2 moved 3 m along +x and 0.5 m up relative to the group, under a
/// yaw-only group moved off the origin and raised (Q-254), with a head camera at yaw 40°.
fn scene(name: &str) -> (TestApp, [Entity; 2], Vec3, f32) {
    let mut t = TestApp::new(name);
    let yaw = 40f32.to_radians();
    let head = Vec3::new(0.7, 1.6, -0.4);
    t.app.world_mut().spawn((
        Camera3d::default(),
        Transform::from_translation(head).with_rotation(Quat::from_rotation_y(yaw)),
    ));
    for p in ["P1", "P2"] {
        let path = fixtures_dir().join(format!("probes/{p}.safetensors"));
        t.command(LibraryCommand::LoadRef(ModelRef::File(path)));
    }
    assert!(
        t.until(600, |app| app
            .world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 2),
        "P1 and P2 load"
    );
    t.frames(1);
    let mut by_name = |n: &str| {
        let mut q = t.app.world_mut().query::<(Entity, &LoadedModel)>();
        q.iter(t.app.world())
            .find(|(_, m)| m.name.contains(n))
            .map(|(e, _)| e)
            .unwrap()
    };
    let es = [by_name("P1"), by_name("P2")];
    *t.app.world_mut().get_mut::<Transform>(es[1]).unwrap() = Transform::from_xyz(3.0, 0.5, 0.0);
    let mut q = t
        .app
        .world_mut()
        .query_filtered::<&mut Transform, With<ContentGroup>>();
    *q.single_mut(t.app.world_mut()).unwrap() = Transform {
        translation: Vec3::new(-2.0, 0.3, 4.0),
        rotation: Quat::from_rotation_y(70f32.to_radians()),
        scale: Vec3::ONE,
    };
    t.frames(2);
    (t, es, head, yaw)
}

/// Where the instance's box center is drawn.
fn center(t: &TestApp, e: Entity) -> Vec3 {
    let h = &t.app.world().get::<SplatModel>(e).unwrap().0;
    let b = t
        .app
        .world()
        .resource::<Assets<SplatAsset>>()
        .get(h)
        .unwrap()
        .bounds;
    t.app
        .world()
        .get::<GlobalTransform>(e)
        .unwrap()
        .transform_point(b.center())
}

fn group(t: &mut TestApp) -> Transform {
    let mut q = t
        .app
        .world_mut()
        .query_filtered::<&Transform, With<ContentGroup>>();
    *q.single(t.app.world()).unwrap()
}

/// 1.5 m ahead of the head along its yaw, at `y`.
fn ahead(head: Vec3, yaw: f32, y: f32) -> Vec3 {
    let p = head + 1.5 * Vec3::new(-yaw.sin(), 0.0, -yaw.cos());
    Vec3::new(p.x, y, p.z)
}

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).abs().max_element() < TOL
}

#[test]
fn recenter_brings_the_selected_instance_ahead_of_the_head_at_its_height() {
    let (mut t, es, head, yaw) = scene("recenter_selected");
    t.app.world_mut().entity_mut(es[1]).insert(Selected);
    let (c0, other0) = (center(&t, es[1]), center(&t, es[0]));
    let g0 = group(&mut t);
    send(&mut t, Intent::Recenter);
    let c = center(&t, es[1]);
    assert!(
        close(c, ahead(head, yaw, c0.y)),
        "{c} vs {}",
        ahead(head, yaw, c0.y)
    );
    // By translation only: the other instance moves by the same step, the group keeps
    // its rotation and scale (Q-254).
    assert!(close(center(&t, es[0]) - other0, c - c0));
    let g = group(&mut t);
    assert!((g.rotation - g0.rotation).length() < TOL && g.scale == g0.scale);
    assert!((g.translation.y - g0.translation.y).abs() < TOL);
}

#[test]
fn recenter_without_a_selection_brings_the_scene_center() {
    // The scene's center: the mean of the instances' posed box centers (Q-283).
    let (mut t, es, head, yaw) = scene("recenter_scene");
    let mean0 = (center(&t, es[0]) + center(&t, es[1])) / 2.0;
    send(&mut t, Intent::Recenter);
    let mean = (center(&t, es[0]) + center(&t, es[1])) / 2.0;
    assert!(close(mean, ahead(head, yaw, mean0.y)), "{mean}");
}

#[test]
fn recenter_works_in_passthrough_mode() {
    // D-17: recenter brings the content to the user when the sticks move nothing.
    let (mut t, es, head, yaw) = scene("recenter_passthrough");
    t.app
        .world_mut()
        .resource_mut::<ftgsxr_app::Locomotion>()
        .enabled = false;
    t.app.world_mut().entity_mut(es[0]).insert(Selected);
    let c0 = center(&t, es[0]);
    send(&mut t, Intent::Recenter);
    assert!(close(center(&t, es[0]), ahead(head, yaw, c0.y)));
}
