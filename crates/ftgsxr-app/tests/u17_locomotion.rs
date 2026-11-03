// SPDX-License-Identifier: Apache-2.0

//! U-17 (`cpu`, §3.1): E12's `stick_response`, `SnapTurnState` and `apply_locomotion`,
//! to 10⁻⁶.

use ftgsxr_app::{Intent, Locomotion, SnapTurnState, apply_locomotion, stick_response};
use ftgsxr_core::{Pose, Quat, Vec2, Vec3};

const TOL: f32 = 1e-6;

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).abs().max_element() <= TOL
}

fn pose(t: Vec3, yaw_deg: f32) -> Pose {
    Pose {
        translation: t,
        rotation: Quat::from_rotation_y(yaw_deg.to_radians()),
        scale: 1.0,
    }
}

fn smooth() -> Locomotion {
    Locomotion {
        speed_mps: 2.0,
        snap: false,
        snap_deg: 45.0,
        smooth_dps: 60.0,
        ..Locomotion::default()
    }
}

#[test]
fn u17_stick_response_has_the_dead_zone_and_the_squared_curve() {
    let d = 0.15;
    assert_eq!(stick_response(Vec2::new(0.1, -0.1), d), Vec2::ZERO);
    let r = stick_response(Vec2::new(0.5, -0.5), d);
    let want = ((0.5f32 - 0.15) / 0.85).powi(2);
    assert!((want - 0.1696).abs() < 1e-4);
    assert!((r.x - want).abs() <= TOL && (r.y + want).abs() <= TOL);
    let r = stick_response(Vec2::new(1.0, -1.0), d);
    assert!((r.x - 1.0).abs() <= TOL && (r.y + 1.0).abs() <= TOL);
}

#[test]
fn u17_snap_turn_steps_once_per_crossing_with_return_below_0_3() {
    let mut s = SnapTurnState::default();
    assert_eq!(s.step(0.8), 1);
    assert_eq!(s.step(0.9), 0);
    assert_eq!(s.step(0.5), 0);
    assert_eq!(s.step(0.8), 0, "no return below 0.3 yet");
    assert_eq!(s.step(0.2), 0);
    assert_eq!(s.step(-0.75), -1);
    assert_eq!(s.step(0.0), 0);
    assert_eq!(s.step(0.71), 1);
}

#[test]
fn u17_move_goes_v_max_dt_along_the_heads_yaw() {
    let loco = smooth();
    let dt = 0.25;
    let rig = pose(Vec3::new(1.0, 0.0, 2.0), 0.0);
    // The head is turned 90° left (yaw +90° about +y): forward is −x.
    let head = pose(Vec3::new(1.5, 1.6, 2.0), 90.0);
    let out = apply_locomotion(
        rig,
        head,
        &Intent::Locomote {
            move_xz: Vec2::new(0.0, 1.0),
            turn: 0.0,
        },
        &loco,
        dt,
    );
    assert!(close(
        out.translation,
        rig.translation + Vec3::new(-2.0 * dt, 0.0, 0.0)
    ));
    assert!(out.rotation.angle_between(rig.rotation) <= TOL);
}

#[test]
fn u17_smooth_and_snap_turns_keep_the_heads_position() {
    let rig = pose(Vec3::new(0.3, 0.0, -0.7), 10.0);
    let head = pose(Vec3::new(1.0, 1.6, 0.5), 10.0);
    let head_local = head.then(&rig.inverse());
    let check = |out: Pose, deg: f32| {
        let h = head_local.then(&out);
        assert!(
            close(h.translation, head.translation),
            "{:?}",
            h.translation
        );
        let want = Quat::from_rotation_y(-deg.to_radians()) * rig.rotation;
        assert!(out.rotation.angle_between(want) <= 1e-5);
    };
    let dt = 0.5;
    let out = apply_locomotion(
        rig,
        head,
        &Intent::Locomote {
            move_xz: Vec2::ZERO,
            turn: 1.0,
        },
        &smooth(),
        dt,
    );
    check(out, 60.0 * dt);
    let snap = Locomotion {
        snap: true,
        ..smooth()
    };
    let out = apply_locomotion(rig, head, &Intent::SnapTurn { steps: -1 }, &snap, dt);
    check(out, -45.0);
}

#[test]
fn u17_disabled_returns_the_input_pose() {
    let loco = Locomotion {
        enabled: false,
        ..smooth()
    };
    let rig = pose(Vec3::new(0.3, 0.0, -0.7), 10.0);
    let head = pose(Vec3::new(1.0, 1.6, 0.5), 30.0);
    for i in [
        Intent::Locomote {
            move_xz: Vec2::new(0.4, 1.0),
            turn: 1.0,
        },
        Intent::SnapTurn { steps: 2 },
    ] {
        assert_eq!(apply_locomotion(rig, head, &i, &loco, 0.1), rig);
    }
}
