// SPDX-License-Identifier: Apache-2.0

//! A Bevy view as E6's camera: Bevy's view space (x right, y up, looking along −z) and its
//! clip mapping turned into openftgs's pinhole camera (x right, y down, looking along +z;
//! intrinsics in pixels of the view's target), so the cull and the draw see the view the
//! engine's opaque pass sees.

use ::bevy::math::{DMat4, DVec4, Mat4, UVec2};
use ftgsxr_core::{Camera, DMat3};

/// openftgs's camera axes from Bevy's view axes: y and z flipped.
const FLIP_YZ: DMat4 = DMat4::from_cols(DVec4::X, DVec4::NEG_Y, DVec4::NEG_Z, DVec4::W);

/// E6's camera for a view of `size` physical pixels whose view-to-world transform is
/// `world_from_view` and whose clip mapping is `clip_from_view`.
///
/// The clip mapping must be a perspective one (w_clip = −z_view), as Bevy's
/// `PerspectiveProjection` and an XR eye's asymmetric frustum are: then
/// x_ndc = P₀₀ x_v/(−z_v) − P₀₂ and y_ndc = P₁₁ y_v/(−z_v) − P₁₂, and with the target's
/// pixel x = (x_ndc + 1) W/2, y = (1 − y_ndc) H/2 (continuous, pixel i's center at i + ½)
/// the intrinsics are f_x = P₀₀ W/2, c_x = (1 − P₀₂) W/2, f_y = P₁₁ H/2,
/// c_y = (1 + P₁₂) H/2. `None` for any other mapping (an orthographic one) or an empty
/// target.
// Q-100
pub(crate) fn view_camera(
    world_from_view: Mat4,
    clip_from_view: Mat4,
    size: UVec2,
) -> Option<Camera> {
    let p = clip_from_view;
    // A perspective mapping: w_clip = −z_view, and no x or y in w.
    let perspective = p.x_axis.w == 0.0
        && p.y_axis.w == 0.0
        && p.z_axis.w == -1.0
        && p.w_axis.w == 0.0
        && p.x_axis.x > 0.0
        && p.y_axis.y > 0.0;
    if !perspective || size.x == 0 || size.y == 0 {
        return None;
    }
    let (w, h) = (f64::from(size.x), f64::from(size.y));
    let fx = f64::from(p.x_axis.x) * w / 2.0;
    let fy = f64::from(p.y_axis.y) * h / 2.0;
    let cx = (1.0 - f64::from(p.z_axis.x)) * w / 2.0;
    let cy = (1.0 + f64::from(p.z_axis.y)) * h / 2.0;
    let k = DMat3::from_cols_array(&[fx, 0.0, 0.0, 0.0, fy, 0.0, cx, cy, 1.0]);
    let view_from_world = world_from_view.as_dmat4().inverse();
    Some(Camera {
        width: size.x,
        height: size.y,
        k,
        world_to_camera: FLIP_YZ * view_from_world,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::bevy::camera::{CameraProjection, PerspectiveProjection};
    use ::bevy::math::{Vec3, Vec4};
    use ::bevy::transform::components::Transform;

    /// P1's first request item (`tests/fixtures/probes/P1.request.json`): 64 × 48, f = 60,
    /// the principal point at the center, the camera at (0, −4, 0) looking along +y with
    /// the image's y along −z.
    fn p1_camera() -> Camera {
        Camera {
            width: 64,
            height: 48,
            k: DMat3::from_cols_array(&[60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 32.0, 24.0, 1.0]),
            world_to_camera: DMat4::from_cols_array_2d(&[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 1.0],
            ]),
        }
    }

    fn assert_close(a: &Camera, b: &Camera, tol: f64) {
        assert_eq!((a.width, a.height), (b.width, b.height));
        let ka = a.k.to_cols_array();
        let kb = b.k.to_cols_array();
        for (x, y) in ka.iter().zip(&kb) {
            assert!((x - y).abs() <= tol * 60.0, "K {ka:?} != {kb:?}");
        }
        let wa = a.world_to_camera.to_cols_array();
        let wb = b.world_to_camera.to_cols_array();
        for (x, y) in wa.iter().zip(&wb) {
            assert!(
                (x - y).abs() <= tol * 4.0,
                "world_to_camera {wa:?} != {wb:?}"
            );
        }
    }

    #[test]
    fn view_camera_of_a_bevy_perspective_camera_is_p1s_request_camera() {
        let transform = Transform::from_xyz(0.0, -4.0, 0.0).looking_at(Vec3::ZERO, Vec3::Z);
        let projection = PerspectiveProjection {
            fov: 2.0 * (24.0f32 / 60.0).atan(),
            aspect_ratio: 64.0 / 48.0,
            near: 0.01,
            ..PerspectiveProjection::default()
        };
        let camera = view_camera(
            transform.to_matrix(),
            projection.get_clip_from_view(),
            UVec2::new(64, 48),
        )
        .expect("a perspective view");
        assert_close(&camera, &p1_camera(), 1e-6);
    }

    #[test]
    fn view_camera_reads_an_off_center_principal_point() {
        // A frustum shifted right and down by (3, 2) pixels: c = (35, 26).
        let (w, h, f) = (64.0f32, 48.0f32, 60.0f32);
        let (cx, cy) = (35.0f32, 26.0f32);
        let near = 0.01;
        let p = Mat4::from_cols(
            Vec4::new(2.0 * f / w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 * f / h, 0.0, 0.0),
            Vec4::new(1.0 - 2.0 * cx / w, 2.0 * cy / h - 1.0, 0.0, -1.0),
            Vec4::new(0.0, 0.0, near, 0.0),
        );
        let camera = view_camera(Mat4::IDENTITY, p, UVec2::new(64, 48)).expect("perspective");
        let k = camera.k;
        assert!((k.z_axis.x - 35.0).abs() < 1e-5 && (k.z_axis.y - 26.0).abs() < 1e-5);
        // A point on the view's −z axis at depth 2 lands on the principal point, and one
        // 1 m up (Bevy's +y) lands f/2 pixels higher in the image (openftgs's −y).
        let project = |v: Vec3| {
            let x = camera.world_to_camera * v.as_dvec3().extend(1.0);
            (
                k.x_axis.x * x.x / x.z + k.z_axis.x,
                k.y_axis.y * x.y / x.z + k.z_axis.y,
            )
        };
        let (u, v) = project(Vec3::new(0.0, 0.0, -2.0));
        assert!((u - 35.0).abs() < 1e-5 && (v - 26.0).abs() < 1e-5);
        let (_, v_up) = project(Vec3::new(0.0, 1.0, -2.0));
        assert!((v_up - (26.0 - 30.0)).abs() < 1e-4);
    }

    #[test]
    fn view_camera_refuses_an_orthographic_mapping_and_an_empty_target() {
        let ortho = Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
        assert!(view_camera(Mat4::IDENTITY, ortho, UVec2::new(8, 8)).is_none());
        let persp = Mat4::perspective_infinite_reverse_rh(1.0, 1.0, 0.01);
        assert!(view_camera(Mat4::IDENTITY, persp, UVec2::new(0, 8)).is_none());
    }
}
