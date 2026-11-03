// SPDX-License-Identifier: Apache-2.0

//! The desktop camera controller under `Intent::Camera` (`cpu`, #283; source of truth
//! §1.7 Cameras): `ResetView` gives 1.6 m above the origin, 2 m back, looking at the
//! origin; `Frame` sees the selected instance's posed box whole from the current
//! direction; a bookmark recalls the saved view. The camera exists only with a primary
//! window (Q-194).

mod support;

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use ftgsxr_app::{CameraCommand, Hand, Intent, LibraryCommand, LoadedModel, ModelRef, PlayerRig};
use ftgsxr_render::{SplatAsset, SplatModel};
use ftgsxr_testkit::fixtures_dir;
use support::{Setup, TestApp};

fn windowed(name: &str) -> TestApp {
    TestApp::with(name, Setup::default(), |app| {
        app.world_mut().spawn((Window::default(), PrimaryWindow));
    })
}

fn camera(t: &mut TestApp) -> (Transform, GlobalTransform, Projection) {
    let mut q = t
        .app
        .world_mut()
        .query_filtered::<(&Transform, &GlobalTransform, &Projection), With<Camera3d>>();
    let (a, b, c) = q.single(t.app.world()).expect("one desktop camera");
    (*a, *b, c.clone())
}

fn send(t: &mut TestApp, intents: &[Intent]) {
    for i in intents {
        t.app.world_mut().write_message(*i);
    }
    t.frames(2);
}

fn cam(c: CameraCommand) -> Intent {
    Intent::Camera(c)
}

#[test]
fn no_window_spawns_no_camera_but_the_rig() {
    let mut t = TestApp::new("rig_no_window");
    t.frames(1);
    let w = t.app.world_mut();
    assert_eq!(w.query_filtered::<(), With<Camera3d>>().iter(w).count(), 0);
    assert_eq!(w.query_filtered::<(), With<PlayerRig>>().iter(w).count(), 1);
}

#[test]
fn reset_view_is_1_6_up_2_back_looking_at_the_origin() {
    let mut t = windowed("rig_reset");
    t.frames(1);
    send(
        &mut t,
        &[
            cam(CameraCommand::Look(Vec2::new(120.0, -40.0))),
            cam(CameraCommand::Move(Vec3::X)),
        ],
    );
    assert!(
        camera(&mut t)
            .1
            .translation()
            .distance(Vec3::new(0.0, 1.6, 2.0))
            > 1e-3
    );
    send(&mut t, &[cam(CameraCommand::ResetView)]);
    let (_, g, _) = camera(&mut t);
    assert!(g.translation().distance(Vec3::new(0.0, 1.6, 2.0)) < 1e-5);
    let fwd = g.forward().as_vec3();
    let want = (Vec3::ZERO - Vec3::new(0.0, 1.6, 2.0)).normalize();
    assert!(fwd.distance(want) < 1e-5, "{fwd:?}");
}

#[test]
fn bookmark_recalls_the_saved_view() {
    let mut t = windowed("rig_bookmark");
    t.frames(1);
    send(&mut t, &[cam(CameraCommand::Look(Vec2::new(80.0, 10.0)))]);
    let saved = camera(&mut t).1;
    send(
        &mut t,
        &[
            cam(CameraCommand::SaveBookmark(3)),
            cam(CameraCommand::ResetView),
        ],
    );
    assert!(
        camera(&mut t).1.translation().distance(saved.translation())
            + camera(&mut t).1.rotation().angle_between(saved.rotation())
            > 1e-3
    );
    send(&mut t, &[cam(CameraCommand::RecallBookmark(3))]);
    let g = camera(&mut t).1;
    assert!(g.translation().distance(saved.translation()) < 1e-5);
    assert!(g.rotation().angle_between(saved.rotation()) < 1e-5);
}

#[test]
fn frame_sees_the_selected_box_whole_from_the_current_direction() {
    let mut t = windowed("rig_frame");
    t.command(LibraryCommand::LoadRef(ModelRef::File(
        fixtures_dir().join("probes/P1.safetensors"),
    )));
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
    let e = {
        let mut q = t
            .app
            .world_mut()
            .query_filtered::<Entity, With<LoadedModel>>();
        q.single(t.app.world()).unwrap()
    };
    send(
        &mut t,
        &[
            Intent::Pointer {
                hand: Hand::Mouse,
                origin: Vec3::new(0.0, 0.0, 3.0),
                dir: -Vec3::Z,
            },
            Intent::Select {
                hand: Hand::Mouse,
                pressed: true,
            },
        ],
    );
    assert!(t.app.world().get::<ftgsxr_app::Selected>(e).is_some());
    send(&mut t, &[cam(CameraCommand::Look(Vec2::new(60.0, 30.0)))]);
    let before = camera(&mut t).1.rotation();
    send(&mut t, &[cam(CameraCommand::Frame)]);
    let (_, g, p) = camera(&mut t);
    assert!(
        g.rotation().angle_between(before) < 1e-5,
        "the direction is kept"
    );
    let Projection::Perspective(p) = p else {
        panic!("perspective")
    };
    let ty = (0.5 * p.fov).tan();
    let tx = ty * p.aspect_ratio;
    let model = t.app.world().get::<SplatModel>(e).unwrap().0.clone();
    let b = t
        .app
        .world()
        .resource::<Assets<SplatAsset>>()
        .get(&model)
        .unwrap()
        .bounds;
    let inst = *t.app.world().get::<GlobalTransform>(e).unwrap();
    let view = g.affine().inverse();
    let mid = view.transform_point3(inst.transform_point(0.5 * (b.min + b.max)));
    assert!(
        mid.x.abs() < 1e-4 && mid.y.abs() < 1e-4 && mid.z < 0.0,
        "centered: {mid:?}"
    );
    for i in 0..8 {
        let c = Vec3::new(
            if i & 1 == 0 { b.min.x } else { b.max.x },
            if i & 2 == 0 { b.min.y } else { b.max.y },
            if i & 4 == 0 { b.min.z } else { b.max.z },
        );
        let v = view.transform_point3(inst.transform_point(c));
        assert!(v.z < -p.near, "corner {i} in front: {v:?}");
        let (x, y) = (v.x / -v.z, v.y / -v.z);
        assert!(
            x.abs() <= tx + 1e-5 && y.abs() <= ty + 1e-5,
            "corner {i} inside: {x} {y}"
        );
    }
}
