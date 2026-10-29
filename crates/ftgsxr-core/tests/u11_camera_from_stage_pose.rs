// SPDX-License-Identifier: Apache-2.0

//! U-11, the camera clauses (E6's desktop and XR rules, `cpu` lane):
//! `Camera::from_stage_pose` with `Fov::Vertical` and `Fov::Angles`. The projection
//! clauses of U-11 (`covariance::project`) are the `covariance` module's.
//!
//! The row prints f_y = 935.31, f_x = 492.4 and c_x = 413.2 rounded; the tolerance
//! (10⁻⁴ px, 10⁻⁶ relative) is applied to the same values worked by hand at full
//! precision from E6's formulas below.

use approx::assert_relative_eq;
use ftgsxr_core::{Camera, DVec3, Fov, Quat, Vec3};

/// The row's tolerances: 10⁻⁴ px absolute, 10⁻⁶ relative.
const PX: f64 = 1e-4;
const REL: f64 = 1e-6;

#[test]
fn u11_desktop_camera_60_degrees_at_1920_by_1080() {
    // E6 desktop: f_y = (H/2)/tan(φ/2) = 540/tan 30° = 540·√3 = 935.307 436…
    let cam = Camera::from_stage_pose(
        Vec3::ZERO,
        Quat::IDENTITY,
        Fov::Vertical {
            radians: 60f32.to_radians(),
        },
        1920,
        1080,
    );
    let k = cam.intrinsics();
    let fy = 540.0 * 3f64.sqrt();
    assert_relative_eq!(k.fy as f64, fy, epsilon = PX, max_relative = REL);
    assert_relative_eq!(k.fx as f64, fy, epsilon = PX, max_relative = REL);
    assert_relative_eq!(k.cx as f64, 960.0, epsilon = PX);
    assert_relative_eq!(k.cy as f64, 540.0, epsilon = PX);
    assert_eq!((k.width, k.height), (1920, 1080));
    assert_eq!((cam.width, cam.height), (1920, 1080));
}

#[test]
fn u11_xr_camera_from_four_angles_at_1000_by_1000() {
    // E6 XR: f_x = W/(tan a_R − tan a_L) = 1000/(tan 50° + tan 40°) = 492.403 876…,
    // c_x = −f_x tan a_L = f_x tan 40° = 413.175 911…; the vertical pair is symmetric
    // (±45°), so f_y = 1000/2 = 500 and c_y = f_y tan 45° = 500.
    let deg = |d: f32| d.to_radians();
    let cam = Camera::from_stage_pose(
        Vec3::ZERO,
        Quat::IDENTITY,
        Fov::Angles {
            left: deg(-40.0),
            right: deg(50.0),
            up: deg(45.0),
            down: deg(-45.0),
        },
        1000,
        1000,
    );
    let k = cam.intrinsics();
    let t40 = 40f64.to_radians().tan();
    let t50 = 50f64.to_radians().tan();
    let fx = 1000.0 / (t50 + t40);
    assert_relative_eq!(k.fx as f64, fx, epsilon = PX, max_relative = REL);
    assert_relative_eq!(k.cx as f64, fx * t40, epsilon = PX, max_relative = REL);
    assert_relative_eq!(k.fy as f64, 500.0, epsilon = PX, max_relative = REL);
    assert_relative_eq!(k.cy as f64, 500.0, epsilon = PX, max_relative = REL);
    // The row's rounded figures.
    assert!((k.fx as f64 - 492.4).abs() < 0.005);
    assert!((k.cx as f64 - 413.2).abs() < 0.05);
}

#[test]
fn u11_stage_pose_gives_opencv_world_to_camera() {
    // E6: R_w = diag(1, −1, −1) R(q)ᵀ, t_w = −R_w p, so the camera's center is p and a
    // point ahead of it along −z (stage) lies at positive camera z, with stage y up
    // mapped to camera y down.
    let p = Vec3::new(1.0, 2.0, 3.0);
    let cam = Camera::from_stage_pose(
        p,
        Quat::IDENTITY,
        Fov::Vertical {
            radians: 90f32.to_radians(),
        },
        64,
        48,
    );
    let c = cam.center();
    assert_relative_eq!(c.x, 1.0, epsilon = 1e-12);
    assert_relative_eq!(c.y, 2.0, epsilon = 1e-12);
    assert_relative_eq!(c.z, 3.0, epsilon = 1e-12);
    let ahead_above = DVec3::new(1.0, 2.5, 1.0); // 2 m ahead (−z), 0.5 m up
    let xc = cam.world_to_camera.transform_point3(ahead_above);
    assert_relative_eq!(xc.x, 0.0, epsilon = 1e-12);
    assert_relative_eq!(xc.y, -0.5, epsilon = 1e-12);
    assert_relative_eq!(xc.z, 2.0, epsilon = 1e-12);

    // A camera yawed 90° to the left (about +y) looks along stage −x.
    let yaw = Quat::from_rotation_y(90f32.to_radians());
    let cam = Camera::from_stage_pose(Vec3::ZERO, yaw, Fov::Vertical { radians: 1.0 }, 64, 48);
    let xc = cam
        .world_to_camera
        .transform_point3(DVec3::new(-3.0, 0.0, 0.0));
    assert_relative_eq!(xc.x, 0.0, epsilon = 1e-6);
    assert_relative_eq!(xc.y, 0.0, epsilon = 1e-6);
    assert_relative_eq!(xc.z, 3.0, epsilon = 1e-6);
    let c = cam.center();
    assert!(c.length() < 1e-12);
}
