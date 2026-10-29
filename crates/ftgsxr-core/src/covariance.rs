// SPDX-License-Identifier: Apache-2.0

//! Quaternion conversion, E5's covariance and pose, and the CPU mirrors of E6's projection,
//! E7's α and E8's sort key (§2.1 `ftgsxr_core::covariance`).

use glam::{DMat3, DQuat, Mat3, Quat, Vec3};

use crate::types::Pose;

/// File order (w, x, y, z) → the engine's quaternion, normalized (§1.14 Quaternions).
pub fn quat_from_wxyz(q: [f32; 4]) -> Quat {
    let [w, x, y, z] = q;
    Quat::from_xyzw(x, y, z, w).normalize()
}

/// The engine's quaternion → file order (w, x, y, z) with the canonical sign (§1.2).
pub fn quat_to_wxyz(q: Quat) -> [f32; 4] {
    let wxyz = [q.w, q.x, q.y, q.z];
    // §1.2: w ≥ 0, and where w = 0 the first nonzero component positive. −0.0 compares
    // equal to 0, so the leading nonzero component decides for both zeros.
    let negative = wxyz.iter().find(|c| **c != 0.0).is_some_and(|c| *c < 0.0);
    if negative { wxyz.map(|c| -c) } else { wxyz }
}

/// E5's Σ = R(q) diag(scale)² R(q)ᵀ, with q normalized first (openftgs E5).
pub fn covariance(rotation: Quat, scale: Vec3) -> Mat3 {
    let r = rotation_f64(rotation);
    let s = scale.as_dvec3();
    (r * DMat3::from_diagonal(s * s) * r.transpose()).as_mat3()
}

/// E5's Σ^S = s_m² R(q_m) Σ R(q_m)ᵀ: a covariance taken into stage space by a pose.
pub fn posed_covariance(cov: Mat3, pose: &Pose) -> Mat3 {
    let r = rotation_f64(pose.rotation);
    let s = f64::from(pose.scale);
    (r * cov.as_dmat3() * r.transpose() * (s * s)).as_mat3()
}

/// R(q/‖q‖) in float64: glam's matrix of a unit quaternion is openftgs E5's R(w, x, y, z).
fn rotation_f64(q: Quat) -> DMat3 {
    let q = DQuat::from_xyzw(
        f64::from(q.x),
        f64::from(q.y),
        f64::from(q.z),
        f64::from(q.w),
    )
    .normalize();
    DMat3::from_quat(q)
}
