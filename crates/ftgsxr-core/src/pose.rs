// SPDX-License-Identifier: Apache-2.0

//! Grabs, fits and orientation presets (§2.1 `ftgsxr_core::pose`): E10's ray test
//! against a posed box, E11's one- and two-hand grabs, and §1.5's orientation presets,
//! fit and snap to floor.

use glam::{Quat, Vec3};

use crate::types::{Aabb, Pose};

/// E11, one hand: the model's pose in the hand's frame, M = H₀⁻¹ P₀, taken when the
/// grab starts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grab {
    in_hand: Pose,
}

impl Grab {
    /// Starts a one-hand grab: M = H₀⁻¹ P₀, with `hand` the hand's pose H₀ (a rigid
    /// transform) and `model` the instance's pose P₀.
    pub fn begin(hand: Pose, model: Pose) -> Grab {
        Grab {
            in_hand: model.then(&hand.inverse()),
        }
    }

    /// The instance's pose for the hand's current pose: P = H M, so the instance keeps
    /// its place in the hand.
    pub fn update(&self, hand: Pose) -> Pose {
        self.in_hand.then(&hand)
    }
}

/// E11's scale limits s_min and s_max: 0.01 and 100 (D-43).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleLimits {
    /// s_min, the smallest scale a two-hand grab gives.
    pub min: f32,
    /// s_max, the largest scale a two-hand grab gives.
    pub max: f32,
}

/// E11, two hands: the hands' start positions a₀ and b₀, their midpoint m₀ and the
/// instance's start pose P₀.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoHandGrab {
    a0: Vec3,
    b0: Vec3,
    m0: Vec3,
    model0: Pose,
}

/// Below this length (in metres) a hand axis, or its horizontal projection, gives no
/// direction (Q-T11.1).
const DEGENERATE_M: f32 = 1e-6;

impl TwoHandGrab {
    /// Starts a two-hand grab from the hands' poses (only their positions count) and the
    /// instance's pose P₀.
    pub fn begin(a: Pose, b: Pose, model: Pose) -> TwoHandGrab {
        let (a0, b0) = (a.translation, b.translation);
        TwoHandGrab {
            a0,
            b0,
            m0: (a0 + b0) * 0.5,
            model0: model,
        }
    }

    /// The instance's pose for the hands' current poses:
    /// P' = T(m) R(θ) Scale(s′/s₀) T(−m₀) P₀, with s′ = clamp(s₀ ‖b − a‖/‖b₀ − a₀‖,
    /// s_min, s_max) and m = (a + b)/2. R is R_ŷ(θ), the yaw between the horizontal
    /// projections of d₀ = b₀ − a₀ and d = b − a ("keep upright"), or with
    /// `free_rotation` the shortest rotation taking d₀ to d.
    ///
    /// Degenerate hands (Q-T11.1): when ‖d₀‖ is below 1 µm the ratio is 1; when d₀ or d
    /// has no direction (or, upright, no horizontal direction) the rotation is the
    /// identity.
    pub fn update(&self, a: Pose, b: Pose, limits: &ScaleLimits, free_rotation: bool) -> Pose {
        let (a, b) = (a.translation, b.translation);
        let (d0, d) = (self.b0 - self.a0, b - a);
        let m = (a + b) * 0.5;
        let s0 = self.model0.scale;

        // Q-T11.1: hands that start together give no ratio.
        let ratio = if d0.length() < DEGENERATE_M {
            1.0
        } else {
            d.length() / d0.length()
        };
        let s1 = (s0 * ratio).max(limits.min).min(limits.max);

        let turn = if free_rotation {
            if d0.length() < DEGENERATE_M || d.length() < DEGENERATE_M {
                Quat::IDENTITY // Q-T11.1
            } else {
                Quat::from_rotation_arc(d0.normalize(), d.normalize())
            }
        } else {
            let (h0, h) = (Vec3::new(d0.x, 0.0, d0.z), Vec3::new(d.x, 0.0, d.z));
            if h0.length() < DEGENERATE_M || h.length() < DEGENERATE_M {
                Quat::IDENTITY // Q-T11.1
            } else {
                // θ = atan2((d̄₀ × d̄)·ŷ, d̄₀·d̄); atan2 needs no normalization.
                let theta = h0.cross(h).y.atan2(h0.dot(h));
                Quat::from_rotation_y(theta)
            }
        };

        let k = s1 / s0;
        Pose {
            translation: m + turn * ((self.model0.translation - self.m0) * k),
            rotation: (turn * self.model0.rotation).normalize(),
            scale: s1,
        }
    }
}

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
