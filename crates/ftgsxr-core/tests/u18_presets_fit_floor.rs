// SPDX-License-Identifier: Apache-2.0

//! U-18, §1.5 Poses (`Preset`, `fit_pose`, `snap_to_floor`, `cpu` lane, D-11): Z-up
//! maps (0, 0, 1) to (0, 1, 0); `from_camera_ups` picks the nearest preset for
//! openftgs-synth's z-up cameras; `fit_pose` puts the longest side at the size, the
//! bottom on y = 0 and the center 1.5 m along the yaw; `snap_to_floor`. Tolerance 10⁻⁶.
//!
//! openftgs-synth's world is right-handed, in metres, with z up, and its cameras follow
//! OpenCV (x right, y down, z forward, world-to-camera; spec Appendix A row 10): the
//! z-up cameras below are built by hand in that convention.

use ftgsxr_core::{
    Aabb, Camera, DMat3, DMat4, DVec3, Pose, Preset, Quat, Vec3, fit_pose, snap_to_floor,
};
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// The row's tolerance.
const TOL: f32 = 1e-6;

fn assert_vec3(got: Vec3, want: Vec3, what: &str) {
    // 10⁻⁶ absolute, or relative for values above 1.
    let scale = want.abs().max_element().max(1.0);
    assert!(
        (got - want).abs().max_element() <= TOL * scale,
        "{what}: got {got}, want {want}"
    );
}

/// E5: x ↦ s R(q) x + t, worked here rather than through `Pose`.
fn apply(p: &Pose, x: Vec3) -> Vec3 {
    p.rotation * x * p.scale + p.translation
}

#[test]
fn u18_presets_rotate_the_model_frame_into_stage_space() {
    // Z-up is −90° about x: the model's z becomes up, its y becomes −z.
    let z_up = Preset::ZUp.rotation();
    assert_vec3(z_up * Vec3::Z, Vec3::Y, "Z-up: (0, 0, 1)");
    assert_vec3(z_up * Vec3::Y, -Vec3::Z, "Z-up: (0, 1, 0)");
    assert_vec3(z_up * Vec3::X, Vec3::X, "Z-up: (1, 0, 0)");
    // Y-up is the identity.
    for v in [Vec3::X, Vec3::Y, Vec3::Z] {
        assert_vec3(Preset::YUp.rotation() * v, v, "Y-up");
    }
    // −Y-up is 180° about x: y and z change sign.
    let neg = Preset::NegYUp.rotation();
    assert_vec3(neg * Vec3::Y, -Vec3::Y, "−Y-up: (0, 1, 0)");
    assert_vec3(neg * Vec3::Z, -Vec3::Z, "−Y-up: (0, 0, 1)");
    assert_vec3(neg * Vec3::X, Vec3::X, "−Y-up: (1, 0, 0)");
    // A custom rotation is the rotation given.
    let q = Quat::from_rotation_z(0.7);
    assert_vec3(
        Preset::Custom(q).rotation() * Vec3::X,
        q * Vec3::X,
        "custom",
    );
}

/// An OpenCV world-to-camera camera at `eye` looking at `target` in a world whose up is
/// `up`: z_c forward, x_c = z_c × up (right), y_c = z_c × x_c (down); R_w has the rows
/// x_c, y_c, z_c and t_w = −R_w eye.
fn look_at(eye: DVec3, target: DVec3, up: DVec3) -> Camera {
    let z = (target - eye).normalize();
    let x = z.cross(up).normalize();
    let y = z.cross(x);
    let r = DMat3::from_cols(x, y, z).transpose();
    let t = -(r * eye);
    let mut w2c = DMat4::from_mat3(r);
    w2c.w_axis = t.extend(1.0);
    Camera {
        width: 160,
        height: 120,
        k: DMat3::from_cols(
            DVec3::new(150.0, 0.0, 0.0),
            DVec3::new(0.0, 150.0, 0.0),
            DVec3::new(80.0, 60.0, 1.0),
        ),
        world_to_camera: w2c,
    }
}

/// §1.5's camera up: −R_wᵀ (0, 1, 0).
fn camera_up(cam: &Camera) -> Vec3 {
    let r = DMat3::from_mat4(cam.world_to_camera);
    (-(r.transpose() * DVec3::Y)).as_vec3()
}

/// Four cameras on a circle of radius 3 m, 1.5 m above the ground, looking down at a
/// point 0.5 m above the scene's center: a `tiny`-like rig.
fn ring(up: DVec3, height_axis: DVec3, a: DVec3, b: DVec3) -> Vec<Vec3> {
    (0..4)
        .map(|k| {
            let phi = f64::from(k) * std::f64::consts::FRAC_PI_2 + 0.3;
            let eye = 3.0 * (phi.cos() * a + phi.sin() * b) + 1.5 * height_axis;
            camera_up(&look_at(eye, 0.5 * height_axis, up))
        })
        .collect()
}

#[test]
fn u18_from_camera_ups_picks_z_up_for_openftgs_synth_cameras() {
    let ups = ring(DVec3::Z, DVec3::Z, DVec3::X, DVec3::Y);
    // The rig is built right: each camera's up lies in the vertical plane through its
    // view, tilted from z by the downward look (about 18.4°).
    for up in &ups {
        assert!(up.z > 0.9 && up.z < 0.99, "camera up {up}");
    }
    assert_eq!(Preset::from_camera_ups(&ups), Preset::ZUp);
    // The same rig in a y-up world, and upside down.
    let y_up = ring(DVec3::Y, DVec3::Y, DVec3::Z, DVec3::X);
    assert_eq!(Preset::from_camera_ups(&y_up), Preset::YUp);
    let down = ring(-DVec3::Y, -DVec3::Y, DVec3::X, DVec3::Z);
    assert_eq!(Preset::from_camera_ups(&down), Preset::NegYUp);
    // No cameras: the default, Y-up (Q-T11.1).
    assert_eq!(Preset::from_camera_ups(&[]), Preset::YUp);
}

/// §1.5's defaults: L_fit = 1 m, 1.5 m ahead.
const AHEAD_M: f32 = 1.5;

#[test]
fn u18_fit_pose_y_up_with_a_turned_and_pitched_head() {
    // B_m spans 4 × 1 × 1 m, longest along x; its center is (1, 0.5, 0).
    let bounds = Aabb {
        min: Vec3::new(-1.0, 0.0, -0.5),
        max: Vec3::new(3.0, 1.0, 0.5),
    };
    // The head at (2, 1.6, 3), yawed 90° (R_y(90°) takes −z to −x), pitched 30° down
    // and rolled: its yaw alone counts, so "ahead" is −x and the center lands at
    // (2 − 1.5, ·, 3).
    let head = Pose {
        translation: Vec3::new(2.0, 1.6, 3.0),
        rotation: Quat::from_rotation_y(FRAC_PI_2)
            * Quat::from_rotation_x(-PI / 6.0)
            * Quat::from_rotation_z(0.2),
        scale: 1.0,
    };
    let p = fit_pose(&bounds, Preset::YUp, 1.0, head, AHEAD_M);
    // Scale 1/4 puts the 4 m side at 1 m; the posed box is 1 × 0.25 × 0.25 m with its
    // bottom on y = 0, so its center is at y = 0.125. The rotation is the preset's
    // alone (Q-T11.2): the model's x stays along stage x.
    assert!((p.scale - 0.25).abs() <= TOL, "scale {}", p.scale);
    assert_vec3(
        apply(&p, Vec3::new(1.0, 0.5, 0.0)),
        Vec3::new(0.5, 0.125, 3.0),
        "center",
    );
    assert_vec3(
        apply(&p, Vec3::new(1.0, 0.0, 0.0)),
        Vec3::new(0.5, 0.0, 3.0),
        "bottom",
    );
    assert_vec3(
        apply(&p, Vec3::new(3.0, 0.5, 0.0)),
        Vec3::new(1.0, 0.125, 3.0),
        "+x end",
    );
    assert_vec3(
        apply(&p, Vec3::new(-1.0, 0.5, 0.0)),
        Vec3::new(0.0, 0.125, 3.0),
        "−x end",
    );
    assert_vec3(
        apply(&p, Vec3::new(1.0, 0.5, 0.5)),
        Vec3::new(0.5, 0.125, 3.125),
        "+z face",
    );

    // A head looking straight down has no yaw of its own; the top of the head points
    // along −x here, as before (Q-T11.1).
    let down = Pose {
        rotation: Quat::from_rotation_y(FRAC_PI_2) * Quat::from_rotation_x(-FRAC_PI_2),
        ..head
    };
    let p = fit_pose(&bounds, Preset::YUp, 1.0, down, AHEAD_M);
    assert_vec3(
        apply(&p, Vec3::new(1.0, 0.5, 0.0)),
        Vec3::new(0.5, 0.125, 3.0),
        "center, looking down",
    );
}

#[test]
fn u18_fit_pose_z_up_stands_the_model_on_the_floor() {
    // A z-up model 1 × 2 × 4 m, tallest along its z, its center (0.5, 1, 2).
    let bounds = Aabb {
        min: Vec3::ZERO,
        max: Vec3::new(1.0, 2.0, 4.0),
    };
    // The head at eye height, facing −z.
    let head = Pose {
        translation: Vec3::new(0.0, 1.7, 0.0),
        ..Pose::IDENTITY
    };
    let p = fit_pose(&bounds, Preset::ZUp, 2.0, head, AHEAD_M);
    // Scale 2/4; Z-up takes (x, y, z) to (x, z, −y), so the 4 m side stands 2 m tall,
    // from y = 0 to y = 2, centered 1.5 m ahead at (0, 1, −1.5).
    assert!((p.scale - 0.5).abs() <= TOL, "scale {}", p.scale);
    assert_vec3(
        apply(&p, Vec3::new(0.5, 1.0, 2.0)),
        Vec3::new(0.0, 1.0, -1.5),
        "center",
    );
    assert_vec3(
        apply(&p, Vec3::new(0.5, 1.0, 0.0)),
        Vec3::new(0.0, 0.0, -1.5),
        "bottom",
    );
    assert_vec3(
        apply(&p, Vec3::new(0.5, 1.0, 4.0)),
        Vec3::new(0.0, 2.0, -1.5),
        "top",
    );
    assert_vec3(
        apply(&p, Vec3::new(1.0, 1.0, 2.0)),
        Vec3::new(0.25, 1.0, -1.5),
        "+x face",
    );
    assert_vec3(
        apply(&p, Vec3::new(0.5, 2.0, 2.0)),
        Vec3::new(0.0, 1.0, -2.0),
        "+y face",
    );
}

#[test]
fn u18_snap_to_floor_moves_the_box_bottom_to_the_floor() {
    // B_m = [0, 1] × [0, 3] × [0, 1]; R_z(90°) takes (x, y, z) to (−y, x, z), so under
    // scale 2 the posed box spans y ∈ [2x + t_y] = [t_y, t_y + 2].
    let bounds = Aabb {
        min: Vec3::ZERO,
        max: Vec3::new(1.0, 3.0, 1.0),
    };
    let posed = |t_y: f32| Pose {
        translation: Vec3::new(1.0, t_y, 2.0),
        rotation: Quat::from_rotation_z(FRAC_PI_2),
        scale: 2.0,
    };
    // Down from y = 5 to the floor at 0.5, and up from y = −3.
    for (from, floor) in [(5.0, 0.5), (-3.0, 0.5), (0.0, 0.0)] {
        let p = snap_to_floor(&bounds, posed(from), floor);
        assert_vec3(p.translation, Vec3::new(1.0, floor, 2.0), "translation");
        assert_eq!(
            (p.rotation, p.scale),
            (posed(from).rotation, 2.0),
            "only y moves"
        );
    }
    // A box turned 45° about x rests on an edge: the corners (·, ±1, ±1) of [−1, 1]³
    // reach y = −√2 at the lowest, so snapping to y = 0 lifts the center to √2.
    let cube = Aabb {
        min: Vec3::splat(-1.0),
        max: Vec3::ONE,
    };
    let tilted = Pose {
        rotation: Quat::from_rotation_x(FRAC_PI_4),
        ..Pose::IDENTITY
    };
    let p = snap_to_floor(&cube, tilted, 0.0);
    assert_vec3(
        p.translation,
        Vec3::new(0.0, 2f32.sqrt(), 0.0),
        "tilted cube",
    );
}
