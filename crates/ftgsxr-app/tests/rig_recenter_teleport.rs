// SPDX-License-Identifier: Apache-2.0

//! `rig`'s consumers of `Intent::Teleport` and `Intent::Recenter` (`cpu`, #455; source
//! of truth §1.8 Locomotion: releasing teleports the rig to the arc's floor point; §1.8
//! Session and spaces: recenter moves the content group so the selected instance, or
//! the scene's center, stands 1.5 m in front of the user at its saved height; Q-254: the
//! group stays yaw-only; Q-269, Q-T455.1, Q-T455.2). Where things are is read from
//! `GlobalTransform` after Bevy's transform propagation.

use crate::support::TestApp;

use bevy::prelude::*;
use ftgsxr_app::{Intent, PlayerRig};

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
    // D-17: stick locomotion is off in passthrough mode (Q-T455.1).
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
