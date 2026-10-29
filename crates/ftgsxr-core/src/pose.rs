// SPDX-License-Identifier: Apache-2.0

//! Grabs, fits and orientation presets (§2.1 `ftgsxr_core::pose`): E10's ray test
//! against a posed box, E11's one- and two-hand grabs, and §1.5's orientation presets,
//! fit and snap to floor.

use glam::Vec3;

use crate::types::{Aabb, Pose};

/// E10: the entry distance of a stage-space ray into the posed box, None when it misses
/// or lies beyond `max_m`.
///
/// `dir` is a unit direction in stage space (E10's d). The ray is taken into model space,
/// o_m = P⁻¹ o and d_m = R(q)ᵀ d / s, so the slab parameter t is the same along both rays
/// and, with a unit `dir`, measures stage metres. The ray hits when t_in ≤ t_out,
/// t_out ≥ 0 and max(t_in, 0) ≤ `max_m`; the result is max(t_in, 0), 0 for an origin
/// inside the box. A zero component of d_m opens its slab for every t when the origin
/// lies in it (its planes included) and closes it otherwise, the limit (±∞) of the
/// usual slab test. Computed in float64.
pub fn ray_hit(origin: Vec3, dir: Vec3, bounds: &Aabb, pose: &Pose, max_m: f32) -> Option<f32> {
    let rotation = pose.rotation.as_dquat().normalize();
    let scale = f64::from(pose.scale);
    let inv = rotation.inverse();
    let o = inv * (origin.as_dvec3() - pose.translation.as_dvec3()) / scale;
    let d = inv * dir.as_dvec3() / scale;
    let (lo, hi) = (bounds.min.as_dvec3(), bounds.max.as_dvec3());

    let mut t_in = f64::NEG_INFINITY;
    let mut t_out = f64::INFINITY;
    for a in 0..3 {
        if d[a] == 0.0 {
            if o[a] < lo[a] || o[a] > hi[a] {
                return None;
            }
            continue;
        }
        let t1 = (lo[a] - o[a]) / d[a];
        let t2 = (hi[a] - o[a]) / d[a];
        t_in = t_in.max(t1.min(t2));
        t_out = t_out.min(t1.max(t2));
    }
    let entry = t_in.max(0.0);
    (t_in <= t_out && t_out >= 0.0 && entry <= f64::from(max_m)).then_some(entry as f32)
}
