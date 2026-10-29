// SPDX-License-Identifier: Apache-2.0

//! A model instance's pose P_m (E5).

use glam::{Mat4, Quat, Vec3};

/// A model instance's pose (E5): a similarity transform with a uniform scale,
/// model space → stage space, x ↦ s R(q) x + t.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// t_m, in stage metres.
    pub translation: Vec3,
    /// q_m, a unit quaternion in the engine's order (§1.14 Quaternions).
    pub rotation: Quat,
    /// s_m > 0, uniform.
    pub scale: f32,
}

impl Pose {
    /// The identity pose: no translation, no rotation, scale 1.
    pub const IDENTITY: Pose = Pose {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: 1.0,
    };

    /// The pose as a 4×4 matrix T(t) R(q) S(s).
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::splat(self.scale),
            self.rotation,
            self.translation,
        )
    }

    /// P⁻¹: x ↦ R(q)⁻¹ (x − t) / s.
    pub fn inverse(&self) -> Pose {
        let inv_rotation = self.rotation.inverse();
        let inv_scale = 1.0 / self.scale;
        Pose {
            translation: -(inv_rotation * self.translation) * inv_scale,
            rotation: inv_rotation,
            scale: inv_scale,
        }
    }

    /// `outer ∘ self`: this pose first, then `outer`.
    pub fn then(&self, outer: &Pose) -> Pose {
        Pose {
            translation: outer.transform_point(self.translation),
            rotation: outer.rotation * self.rotation,
            scale: outer.scale * self.scale,
        }
    }

    /// E5's μ^S = s R(q) p + t.
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        self.rotation * (self.scale * p) + self.translation
    }

    /// R(q) d: the rotation only, no scale or translation.
    pub fn transform_direction(&self, d: Vec3) -> Vec3 {
        self.rotation * d
    }
}
