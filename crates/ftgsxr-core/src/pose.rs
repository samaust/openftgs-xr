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
/// direction (Q-17).
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
    /// Degenerate hands (Q-17): when ‖d₀‖ is below 1 µm the ratio is 1; when d₀ or d
    /// has no direction (or, upright, no horizontal direction) the rotation is the
    /// identity.
    pub fn update(&self, a: Pose, b: Pose, limits: &ScaleLimits, free_rotation: bool) -> Pose {
        let (a, b) = (a.translation, b.translation);
        let (d0, d) = (self.b0 - self.a0, b - a);
        let m = (a + b) * 0.5;
        let s0 = self.model0.scale;

        // Q-17: hands that start together give no ratio.
        let ratio = if d0.length() < DEGENERATE_M {
            1.0
        } else {
            d.length() / d0.length()
        };
        let s1 = (s0 * ratio).max(limits.min).min(limits.max);

        let turn = if free_rotation {
            if d0.length() < DEGENERATE_M || d.length() < DEGENERATE_M {
                Quat::IDENTITY // Q-17
            } else {
                Quat::from_rotation_arc(d0.normalize(), d.normalize())
            }
        } else {
            let (h0, h) = (Vec3::new(d0.x, 0.0, d0.z), Vec3::new(d.x, 0.0, d.z));
            if h0.length() < DEGENERATE_M || h.length() < DEGENERATE_M {
                Quat::IDENTITY // Q-17
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

/// §1.5's orientation presets: the rotation that takes the model's frame into stage space
/// before the rest of the pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Preset {
    /// Y-up: the identity, the default.
    YUp,
    /// Z-up: −90° about x, so the model's z becomes up (openftgs-synth's scenes).
    ZUp,
    /// −Y-up: 180° about x.
    NegYUp,
    /// A custom rotation, model frame → stage frame.
    Custom(Quat),
}

impl Preset {
    /// The preset's rotation, normalized.
    pub fn rotation(&self) -> Quat {
        match self {
            Preset::YUp => Quat::IDENTITY,
            Preset::ZUp => Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            Preset::NegYUp => Quat::from_rotation_x(std::f32::consts::PI),
            Preset::Custom(q) => q.normalize(),
        }
    }

    /// §1.5's hint from a `ftgs-scene` manifest or a checkpoint: given each camera's up
    /// vector in model space, −R_wᵀ (0, 1, 0), the fixed preset whose model-space up
    /// (the vector its rotation takes to stage +y) lies nearest the mean of `ups`.
    /// No cameras, or a mean shorter than 1 µm, give Y-up; ties go to the first of Y-up,
    /// Z-up, −Y-up (Q-17).
    pub fn from_camera_ups(ups: &[Vec3]) -> Preset {
        let mean = ups.iter().copied().sum::<Vec3>() / ups.len().max(1) as f32;
        if mean.length() < DEGENERATE_M {
            return Preset::YUp; // Q-17
        }
        let mut best = (Preset::YUp, f32::NEG_INFINITY);
        for preset in [Preset::YUp, Preset::ZUp, Preset::NegYUp] {
            let model_up = preset.rotation().inverse() * Vec3::Y;
            let score = model_up.dot(mean);
            if score > best.1 {
                best = (preset, score); // Q-17: strict, so the earlier preset wins ties
            }
        }
        best.0
    }
}

/// §1.5's fit: the box's longest side scaled to `size_m`, centered `ahead_m` along the
/// head's yaw, bottom on the floor.
///
/// The pose's rotation is the preset's followed by the yaw about stage +y that takes
/// stage −z to the head's horizontal forward, the horizontal part of its −z axis (Q-17
/// when the head looks straight up or down), so the same side of the model faces the
/// user wherever they stand (Q-18). Its scale is `size_m` over B_m's longest side. Its
/// translation puts the posed box's center `ahead_m` from the head's position along
/// that forward, at the height where the posed box's lowest corner lies on the floor
/// y = 0.
pub fn fit_pose(bounds: &Aabb, preset: Preset, size_m: f32, head: Pose, ahead_m: f32) -> Pose {
    let longest = bounds.longest_side();
    let scale = if longest > 0.0 && longest.is_finite() {
        size_m / longest
    } else {
        1.0 // Q-17
    };
    let forward = yaw_forward(&head);
    // Q-18: R_y(ψ) takes −z to (−sin ψ, 0, −cos ψ), which is `forward`.
    let yaw = Quat::from_rotation_y(f32::atan2(-forward.x, -forward.z));
    let placed = Pose {
        translation: Vec3::ZERO,
        rotation: yaw * preset.rotation(),
        scale,
    };
    let center = placed.transform_point(bounds.center());
    let target = head.translation + ahead_m * forward;
    Pose {
        translation: Vec3::new(
            target.x - center.x,
            -lowest_y(bounds, &placed),
            target.z - center.z,
        ),
        ..placed
    }
}

/// §1.5's snap to floor: `pose` moved along y until the posed box's lowest corner lies on
/// `floor_y`; rotation, scale and the horizontal position are kept.
pub fn snap_to_floor(bounds: &Aabb, pose: Pose, floor_y: f32) -> Pose {
    let mut snapped = pose;
    snapped.translation.y += floor_y - lowest_y(bounds, &pose);
    snapped
}

/// The lowest stage y over the eight corners of the posed box.
fn lowest_y(bounds: &Aabb, pose: &Pose) -> f32 {
    (0..8)
        .map(|k| {
            let corner = Vec3::new(
                if k & 1 == 0 {
                    bounds.min.x
                } else {
                    bounds.max.x
                },
                if k & 2 == 0 {
                    bounds.min.y
                } else {
                    bounds.max.y
                },
                if k & 4 == 0 {
                    bounds.min.z
                } else {
                    bounds.max.z
                },
            );
            pose.transform_point(corner).y
        })
        .fold(f32::INFINITY, f32::min)
}

/// The head's horizontal forward, a unit vector: the horizontal part of its −z axis, or,
/// when the head looks straight up or down, of the direction its top points (negated
/// when it looks up) (Q-17).
fn yaw_forward(head: &Pose) -> Vec3 {
    let horizontal = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
    let look = head.rotation * Vec3::NEG_Z;
    let mut forward = horizontal(look);
    if forward.length() < DEGENERATE_M {
        // Q-17
        let top = horizontal(head.rotation * Vec3::Y);
        forward = if look.y > 0.0 { -top } else { top };
    }
    forward.try_normalize().unwrap_or(Vec3::NEG_Z)
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
