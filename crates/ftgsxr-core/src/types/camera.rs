// SPDX-License-Identifier: Apache-2.0

//! E6's camera in openftgs's conventions, built per source (§1.14 Frames).

use glam::{DMat3, DMat4, DQuat, DVec3, Quat, Vec3};

/// A pinhole camera in openftgs's conventions (E6): OpenCV axes (x right, y down, z
/// forward), world-to-camera, continuous pixels (the image spans \[0, W\] × \[0, H\]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// Image width W in pixels.
    pub width: u32,
    /// Image height H in pixels.
    pub height: u32,
    /// The intrinsic matrix K (f_x, f_y on the diagonal, c_x, c_y in the last column).
    pub k: DMat3,
    /// The world-to-camera transform \[R_w t_w; 0 1\].
    pub world_to_camera: DMat4,
}

/// A camera's pinhole parameters in pixels (E6), as the shaders take them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    /// Horizontal focal length f_x in pixels.
    pub fx: f32,
    /// Vertical focal length f_y in pixels.
    pub fy: f32,
    /// Principal point c_x in pixels, continuous convention.
    pub cx: f32,
    /// Principal point c_y in pixels, continuous convention.
    pub cy: f32,
    /// Image width W in pixels.
    pub width: u32,
    /// Image height H in pixels.
    pub height: u32,
}

/// Field of view (E6): a symmetric vertical angle (desktop) or four tangent angles (XR,
/// XrFovf), all in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fov {
    /// The desktop camera's vertical field of view φ.
    Vertical {
        /// φ in radians.
        radians: f32,
    },
    /// An OpenXR view's XrFovf angles; `left` and `down` are negative.
    Angles {
        /// a_L in radians (negative).
        left: f32,
        /// a_R in radians.
        right: f32,
        /// a_U in radians.
        up: f32,
        /// a_D in radians (negative).
        down: f32,
    },
}

impl Camera {
    /// The camera's center in world space, −R_wᵀ t_w (E4's eye of a headless render).
    pub fn center(&self) -> DVec3 {
        let r = DMat3::from_mat4(self.world_to_camera);
        let t = self.world_to_camera.w_axis.truncate();
        -(r.transpose() * t)
    }

    /// f_x, f_y, c_x, c_y read from K, with the image size (E6).
    pub fn intrinsics(&self) -> Intrinsics {
        Intrinsics {
            fx: self.k.x_axis.x as f32,
            fy: self.k.y_axis.y as f32,
            cx: self.k.z_axis.x as f32,
            cy: self.k.z_axis.y as f32,
            width: self.width,
            height: self.height,
        }
    }

    /// E6's desktop and XR rules: a stage-space pose (looking along −z with y up) and a
    /// field of view give R_w = diag(1, −1, −1) R(q)ᵀ, t_w = −R_w p and the pinhole
    /// (§1.14 Frames). Computed in float64.
    pub fn from_stage_pose(
        position: Vec3,
        rotation: Quat,
        fov: Fov,
        width: u32,
        height: u32,
    ) -> Camera {
        let w = f64::from(width);
        let h = f64::from(height);
        let (fx, fy, cx, cy) = match fov {
            Fov::Vertical { radians } => {
                let fy = (h / 2.0) / (f64::from(radians) / 2.0).tan();
                (fy, fy, w / 2.0, h / 2.0)
            }
            Fov::Angles {
                left,
                right,
                up,
                down,
            } => {
                let (tl, tr) = (f64::from(left).tan(), f64::from(right).tan());
                let (tu, td) = (f64::from(up).tan(), f64::from(down).tan());
                let fx = w / (tr - tl);
                let fy = h / (tu - td);
                (fx, fy, -fx * tl, fy * tu)
            }
        };
        let q = DQuat::from_xyzw(
            f64::from(rotation.x),
            f64::from(rotation.y),
            f64::from(rotation.z),
            f64::from(rotation.w),
        )
        .normalize();
        let flip = DMat3::from_diagonal(DVec3::new(1.0, -1.0, -1.0));
        let r_w = flip * DMat3::from_quat(q).transpose();
        let t_w = -(r_w * position.as_dvec3());
        let k = DMat3::from_cols(
            DVec3::new(fx, 0.0, 0.0),
            DVec3::new(0.0, fy, 0.0),
            DVec3::new(cx, cy, 1.0),
        );
        let world_to_camera = DMat4::from_cols(
            r_w.x_axis.extend(0.0),
            r_w.y_axis.extend(0.0),
            r_w.z_axis.extend(0.0),
            t_w.extend(1.0),
        );
        Camera {
            width,
            height,
            k,
            world_to_camera,
        }
    }
}
