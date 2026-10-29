// SPDX-License-Identifier: Apache-2.0

//! Quaternion conversion, E5's covariance and pose, and the CPU mirrors of E6's projection,
//! E7's α and E8's sort key (§2.1 `ftgsxr_core::covariance`).

use glam::{DMat3, DQuat, DVec2, DVec3, Mat3, Quat, Vec2, Vec3};

use crate::types::{Camera, Pose, RenderConventions};

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

/// One projected primitive as the cull shader writes it (E6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projected {
    /// μ′ in pixels, continuous pixel convention.
    pub mean: Vec2,
    /// The conic Σ′⁻¹ as (a, b, c): the entries (0, 0), (0, 1) and (1, 1). // Q-T8.1
    pub conic: [f32; 3],
    /// (r_x, r_y) in whole pixels, both positive; the box is μ′ ± r.
    pub radii: [u32; 2],
    /// z_c, the camera-space depth of the mean in stage metres (E8's input).
    pub depth: f32,
}

/// E6 for one primitive given its stage-space mean and covariance and σ_eff; None when culled.
pub fn project(
    mean: Vec3,
    cov: Mat3,
    sigma_eff: f32,
    view: &Camera,
    conv: &RenderConventions,
) -> Option<Projected> {
    let k = view.k;
    let (fx, fy, cx, cy) = (k.x_axis.x, k.y_axis.y, k.z_axis.x, k.z_axis.y);
    let (w, h) = (f64::from(view.width), f64::from(view.height));
    let r_w = DMat3::from_mat4(view.world_to_camera);
    let t_w = view.world_to_camera.w_axis.truncate();

    // x_c = R_w μ^S + t_w. z_c < near or z_c > far is culled, and a NaN depth.
    let x_c = r_w * mean.as_dvec3() + t_w;
    let z = x_c.z;
    if z.is_nan() || z < f64::from(conv.near) || z > f64::from(conv.far) {
        return None;
    }
    // μ′ from the unclamped ratios: the clamp applies to J only.
    let (u, v) = (x_c.x / z, x_c.y / z);
    let mean2d = DVec2::new(fx * u + cx, fy * v + cy);

    // The EWA Jacobian at x_c, its ratios clamped m·W/f_x (m·H/f_y) beyond the image
    // (source log T8.1 for J's form).
    let m = f64::from(conv.jacobian_clamp_margin);
    let (mx, my) = (m * w / fx, m * h / fy);
    let tx = u.clamp(-(cx / fx + mx), (w - cx) / fx + mx);
    let ty = v.clamp(-(cy / fy + my), (h - cy) / fy + my);
    let j0 = DVec3::new(fx / z, 0.0, -fx * tx / z);
    let j1 = DVec3::new(0.0, fy / z, -fy * ty / z);
    let sigma_c = r_w * cov.as_dmat3() * r_w.transpose();
    let eps = f64::from(conv.eps2d);
    let s00 = j0.dot(sigma_c * j0) + eps;
    let s01 = j0.dot(sigma_c * j1);
    let s11 = j1.dot(sigma_c * j1) + eps;
    let det = s00 * s11 - s01 * s01;
    // det Σ′ ≤ 0 is culled; a NaN det too.
    if det.is_nan() || det <= 0.0 {
        return None;
    }

    // The opacity-aware extent √(2 ln(σ_eff/α_skip)), capped (`opacity_aware` whatever the
    // block's rule, D-39). σ_eff < α_skip (E3: never active) or a NaN σ_eff makes it NaN,
    // culled here since `f64::min` would drop a NaN.
    let ratio = f64::from(sigma_eff) / f64::from(conv.alpha_skip);
    let extent = (2.0 * ratio.ln()).sqrt();
    if extent.is_nan() {
        return None;
    }
    let extent = extent.min(f64::from(conv.screen_extent_max));
    let rx = (extent * s00.sqrt()).ceil();
    let ry = (extent * s11.sqrt()).ceil();
    // Visible only with both radii positive (Q-T8.1).
    if rx.is_nan() || ry.is_nan() || rx <= 0.0 || ry <= 0.0 {
        return None;
    }
    // The box μ′ ± r wholly outside [0, W] × [0, H]; a box that only touches the image is off
    // it, as in openftgs's reference backend (source log T8.1): it covers no pixel center.
    if mean2d.x + rx <= 0.0 || mean2d.x - rx >= w || mean2d.y + ry <= 0.0 || mean2d.y - ry >= h {
        return None;
    }

    let projected = Projected {
        mean: mean2d.as_vec2(),
        conic: [(s11 / det) as f32, (-s01 / det) as f32, (s00 / det) as f32],
        radii: [rx as u32, ry as u32],
        depth: z as f32,
    };
    // A non-finite value anywhere in the projection culls the primitive (E6).
    let finite = projected.mean.is_finite()
        && projected.conic.iter().all(|c| c.is_finite())
        && rx.is_finite()
        && ry.is_finite()
        && rx <= f64::from(u32::MAX)
        && ry <= f64::from(u32::MAX)
        && projected.depth.is_finite();
    finite.then_some(projected)
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
