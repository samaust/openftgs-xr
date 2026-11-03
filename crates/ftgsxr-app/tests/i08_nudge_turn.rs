// SPDX-License-Identifier: Apache-2.0

//! I-8's application (`cpu`, §1.7 Selection and manipulation): `Intent::Nudge` moves the
//! instance by 1 cm, and by 10 cm as the Shift nudge, and `Intent::Turn` turns it by 5°
//! about y (10⁻⁶). The keys' half (arrow keys, Shift, `[` and `]`) is `desktop_input`'s.

mod support;

use bevy::prelude::*;
use ftgsxr_app::{Intent, LibraryCommand, LoadedModel, ModelRef};
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

fn send(t: &mut TestApp, i: Intent) {
    t.app.world_mut().write_message(i);
    t.frames(1);
}

fn transform(t: &TestApp, e: Entity) -> Transform {
    *t.app.world().get::<Transform>(e).unwrap()
}

#[test]
fn i08_nudges_move_1_cm_and_10_cm() {
    let mut t = TestApp::new("i08_nudge");
    let e = loaded(&mut t);
    send(
        &mut t,
        Intent::Nudge {
            instance: e,
            delta: Vec3::new(0.01, 0.0, 0.0),
        },
    );
    assert!((transform(&t, e).translation - Vec3::new(0.01, 0.0, 0.0)).length() < 1e-6);
    send(
        &mut t,
        Intent::Nudge {
            instance: e,
            delta: Vec3::new(0.0, 0.0, -0.1),
        },
    );
    let p = transform(&t, e).translation;
    assert!((p - Vec3::new(0.01, 0.0, -0.1)).length() < 1e-6, "{p}");
    send(
        &mut t,
        Intent::Nudge {
            instance: e,
            delta: Vec3::new(0.0, 0.01, 0.0),
        },
    );
    let p = transform(&t, e).translation;
    assert!((p - Vec3::new(0.01, 0.01, -0.1)).length() < 1e-6, "{p}");
    assert_eq!(transform(&t, e).rotation, Quat::IDENTITY);
}

#[test]
fn i08_turns_are_5_degrees_about_y() {
    let mut t = TestApp::new("i08_turn");
    let e = loaded(&mut t);
    send(
        &mut t,
        Intent::Turn {
            instance: e,
            yaw_deg: 5.0,
        },
    );
    let q = transform(&t, e).rotation;
    let want = Quat::from_rotation_y(5f32.to_radians());
    assert!((q * Vec3::X - want * Vec3::X).length() < 1e-6, "{q}");
    assert!((q * Vec3::Y - Vec3::Y).length() < 1e-6);
    send(
        &mut t,
        Intent::Turn {
            instance: e,
            yaw_deg: -5.0,
        },
    );
    send(
        &mut t,
        Intent::Turn {
            instance: e,
            yaw_deg: -5.0,
        },
    );
    let q = transform(&t, e).rotation;
    let want = Quat::from_rotation_y(-5f32.to_radians());
    assert!((q * Vec3::X - want * Vec3::X).length() < 1e-6, "{q}");
    assert_eq!(transform(&t, e).translation, Vec3::ZERO);
}
