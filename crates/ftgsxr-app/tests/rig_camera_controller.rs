// SPDX-License-Identifier: Apache-2.0

//! The desktop camera controller under `Intent::Camera` (`cpu`, #283; source of truth
//! §1.7 Cameras): `ResetView` gives 1.6 m above the origin, 2 m back, looking at the
//! origin; `Frame` sees the selected instance's posed box whole from the current
//! direction; a bookmark recalls its pose. The camera exists only with a primary
//! window (Q-194).

use crate::support;

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use ftgsxr_app::{
    CameraCommand, Hand, InstanceAction, Intent, LibraryCommand, LoadedModel, ModelRef, PlayerRig,
    Selected,
};
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

/// PRD #275 ("Settings as the source of truth"), owner 2026-11-03: `persist` maps the
/// settings' field of view, grid and background into the camera and the grid, and the
/// wheel's speed back into `[desktop] fly_speed_mps`; a recalled bookmark leaves the
/// settings' field of view as it is (Q-196, as Q-219's jump).
#[test]
fn the_settings_reach_the_camera_and_the_speed_returns_to_them() {
    use ftgsxr_app::{SettingsRes, SplatCamera};
    use ftgsxr_render::OutputSpace;
    let mut t = windowed("rig_settings_map");
    t.frames(1);
    {
        let mut s = t.app.world_mut().resource_mut::<SettingsRes>();
        s.0.desktop.fov_deg = 75.0;
        s.0.environment.background = [0.2, 0.3, 0.4];
    }
    t.frames(2);
    let (_, _, p) = camera(&mut t);
    let Projection::Perspective(p) = p else {
        panic!("a perspective camera")
    };
    assert!((p.fov - 75f32.to_radians()).abs() < 1e-6, "{}", p.fov);
    let w = t.app.world_mut();
    let out = w
        .query_filtered::<&SplatCamera, With<Camera3d>>()
        .single(w)
        .unwrap()
        .output;
    assert_eq!(
        out,
        OutputSpace::LinearOver {
            background: Some([0.2, 0.3, 0.4])
        }
    );
    let before = t
        .app
        .world()
        .resource::<SettingsRes>()
        .0
        .desktop
        .fly_speed_mps;
    send(&mut t, &[cam(CameraCommand::Speed(1.0))]);
    let after = t
        .app
        .world()
        .resource::<SettingsRes>()
        .0
        .desktop
        .fly_speed_mps;
    assert!((after - before * 1.25).abs() < 1e-5, "{before} → {after}");
    send(&mut t, &[cam(CameraCommand::SaveBookmark(1))]);
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .desktop
        .fov_deg = 90.0;
    t.frames(2);
    send(&mut t, &[cam(CameraCommand::RecallBookmark(1))]);
    let (_, _, p) = camera(&mut t);
    let Projection::Perspective(p) = p else {
        panic!("a perspective camera")
    };
    assert!(
        (p.fov - 75f32.to_radians()).abs() < 1e-5,
        "the bookmark's fov"
    );
    let fov = t.app.world().resource::<SettingsRes>().0.desktop.fov_deg;
    assert_eq!(fov, 90.0, "the settings' fov is left as it is");
}

/// Loads P1 and places it off the origin, at x = 3 m.
fn load_p1_off_the_origin(t: &mut TestApp) -> Entity {
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
    t.app
        .world_mut()
        .get_mut::<Transform>(e)
        .unwrap()
        .translation = Vec3::new(3.0, 0.0, 0.0);
    e
}

/// The stage-space center of the instance's posed box (E10's B_m).
fn posed_box_center(t: &TestApp, e: Entity) -> Vec3 {
    let w = t.app.world();
    let m = w.get::<SplatModel>(e).unwrap();
    let b = w.resource::<Assets<SplatAsset>>().get(&m.0).unwrap().bounds;
    let g = w.get::<GlobalTransform>(e).unwrap();
    g.transform_point((b.min + b.max) / 2.0)
}

fn select(e: Entity) -> Intent {
    Intent::Instance {
        instance: e,
        action: InstanceAction::Select,
    }
}

/// §1.7 Orbit: while orbiting, selecting an instance moves the orbit's target to its
/// box center, the distance kept.
#[test]
fn the_orbit_follows_a_new_selection() {
    let mut t = windowed("rig_orbit_follow");
    let e = load_p1_off_the_origin(&mut t);
    send(&mut t, &[cam(CameraCommand::ToggleOrbit)]);
    let before = camera(&mut t).1.translation();
    let d0 = before.length();
    send(&mut t, &[select(e)]);
    t.frames(1);
    let center = posed_box_center(&t, e);
    let g = camera(&mut t).1;
    assert!(
        (g.translation().distance(center) - d0).abs() < 1e-3,
        "the distance kept: {} vs {d0}",
        g.translation().distance(center)
    );
    let to = (center - g.translation()).normalize();
    assert!(
        g.forward().as_vec3().distance(to) < 1e-4,
        "looks at the box center"
    );
}

/// Orbits P1, selected off the origin, then sends `action` to it and waits until
/// `emptied` holds; the camera must then orbit the stage origin, at the distance it had
/// from the box center before (§1.7 Orbit, #421). The rig stays at the identity, so the
/// camera's stage position is its global translation.
fn the_orbit_returns_to_the_origin_after(
    name: &str,
    action: InstanceAction,
    emptied: impl Fn(&mut App) -> bool,
) {
    let mut t = windowed(name);
    let e = load_p1_off_the_origin(&mut t);
    send(&mut t, &[cam(CameraCommand::ToggleOrbit), select(e)]);
    t.frames(1);
    let center = posed_box_center(&t, e);
    assert!(center.length() > 1.0, "the box center is off the origin");
    let d = camera(&mut t).1.translation().distance(center);
    send(
        &mut t,
        &[Intent::Instance {
            instance: e,
            action,
        }],
    );
    assert!(t.until(600, emptied), "nothing is selected");
    // One more frame: the controller reads the removal the frame after it is made.
    t.frames(1);
    let g = camera(&mut t).1;
    assert!(
        (g.translation().length() - d).abs() < 1e-3,
        "the distance kept: {} vs {d}",
        g.translation().length()
    );
    let to = (Vec3::ZERO - g.translation()).normalize();
    assert!(
        g.forward().as_vec3().distance(to) < 1e-4,
        "looks at the stage origin: {:?} vs {to:?}",
        g.forward().as_vec3()
    );
}

/// §1.7 Orbit (#421): while orbiting, a deselect moves the orbit's target to the stage
/// origin, the distance kept.
#[test]
fn the_orbit_returns_to_the_origin_on_a_deselect() {
    the_orbit_returns_to_the_origin_after("rig_orbit_deselect", InstanceAction::Deselect, |app| {
        let w = app.world_mut();
        w.query_filtered::<(), With<Selected>>().iter(w).count() == 0
    });
}

/// §1.7 Orbit (#421): while orbiting, unloading the selected instance leaves nothing
/// selected, so the orbit's target moves to the stage origin, the distance kept.
#[test]
fn the_orbit_returns_to_the_origin_when_the_selected_instance_unloads() {
    the_orbit_returns_to_the_origin_after("rig_orbit_unload", InstanceAction::Unload, |app| {
        let w = app.world_mut();
        w.query::<&LoadedModel>().iter(w).count() == 0
    });
}
