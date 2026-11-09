// SPDX-License-Identifier: Apache-2.0

//! E18's trim handles in stage space: their size, the ray's hit, the drag point and the
//! trim a frame's drags give (§1.7, §1.8, D-57). `run_intents` keeps the drags and
//! decides when they start and end; this file holds the geometry.

use bevy::math::Vec3;
use ftgsxr_core::{Aabb, Pose, Trim, TrimHandle};

use super::{closest_on_line, gizmo_size, on_plane};
use crate::intent::Hand;

/// E18's pick tolerance beyond a handle's sphere, as a fraction of L.
const PICK_TOLERANCE: f32 = 0.08;

/// E18's handle sphere radius, as a fraction of L.
pub(super) const HANDLE_RADIUS: f32 = 0.1;

/// E18's L for an instance's trim: Q-186's gizmo size at the center of the trim's shape
/// box mapped by the pose, seen from `eye`; `None` when the trim is off.
pub(super) fn handle_size(trim: &Trim, pose: &Pose, eye: Vec3) -> Option<f32> {
    let b: Aabb = trim.shape_bounds()?;
    Some(gizmo_size(pose.transform_point(b.center()), eye))
}

/// The handle a ray hits first (E18): within 0.18 L of the handle's stage point, ahead of
/// the ray's origin and within `max_m`, the nearest along the ray; with its distance t
/// along the ray (the handle point's projection on it).
pub(super) fn handle_hit(
    trim: &Trim,
    pose: &Pose,
    size: f32,
    ray: (Vec3, Vec3),
    max_m: f32,
) -> Option<(TrimHandle, f32)> {
    let (o, d) = ray;
    let reach = (HANDLE_RADIUS + PICK_TOLERANCE) * size;
    let mut best: Option<(TrimHandle, f32)> = None;
    for (h, p) in trim.handles() {
        let p = pose.transform_point(p);
        // t, which a drag keeps as t₀: the handle point's projection on the ray. Q-378
        let t = (p - o).dot(d);
        if !(0.0..=max_m).contains(&t) || (o + t * d).distance(p) > reach {
            continue;
        }
        if best.is_none_or(|(_, b)| t < b) {
            best = Some((h, t));
        }
    }
    best
}

/// A drag under way (E18): the instance, the handle, the trim at the drag's start, the
/// model-space drag point p₀ and what the drag point needs: the handle's model point at
/// the start, the start ray (the desktop center's plane) and t₀ on the ray (XR).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct TrimDrag {
    pub(super) entity: bevy::ecs::entity::Entity,
    pub(super) hand: Hand,
    pub(super) handle: TrimHandle,
    pub(super) start: Trim,
    pub(super) p0: Vec3,
    pub(super) anchor: Vec3,
    pub(super) start_ray: (Vec3, Vec3),
    pub(super) t0: f32,
    /// Whether `select` started it (else `grab`); releasing that action ends it.
    pub(super) by_select: bool,
    /// E18's ρ: §1.7's radius floor, 10⁻³ of the untrimmed B_m's longest side.
    pub(super) floor: f32,
    /// The last displacement, kept while a ray gives no drag point.
    pub(super) delta: Vec3,
}

/// The model axis a face or a radius handle moves along; `None` for the center.
fn axis_of(h: TrimHandle) -> Option<usize> {
    match h {
        TrimHandle::Face { axis, .. } | TrimHandle::Radius { axis, .. } => Some(axis),
        TrimHandle::Center => None,
    }
}

/// E18's drag point in stage space for `ray`, under the instance's current pose: the
/// mouse's closest point of the handle's line, or its crossing of the plane through the
/// center facing the start ray; a hand's point at t₀ on its ray. `None` when the ray
/// runs along the line or the plane.
pub(super) fn drag_point(
    hand: Hand,
    handle: TrimHandle,
    anchor: Vec3,
    start_ray: (Vec3, Vec3),
    t0: f32,
    pose: &Pose,
    ray: (Vec3, Vec3),
) -> Option<Vec3> {
    if hand != Hand::Mouse {
        return Some(ray.0 + t0 * ray.1);
    }
    let c = pose.transform_point(anchor);
    match axis_of(handle) {
        Some(axis) => {
            let mut e = Vec3::ZERO;
            e[axis] = 1.0;
            let a = (pose.rotation * e).normalize();
            let (_, s) = closest_on_line(ray, c, a)?;
            Some(c + s * a)
        }
        None => on_plane(ray, c, start_ray.1),
    }
}

/// The trim a frame's drags of one instance give (E18, Q-378): one drag gives its start
/// trim dragged, `start.drag_handle(handle, δ, ρ)`; with two, the first one's result,
/// then the second's own value applied to it: a face's limit through `set_box_limit`,
/// the radius or the center.
// Q-378
pub(super) fn dragged_trim(drags: &[&TrimDrag]) -> Option<Trim> {
    let (first, rest) = drags.split_first()?;
    let mut t = first
        .start
        .drag_handle(first.handle, first.delta, first.floor);
    for d in rest {
        let r = d.start.drag_handle(d.handle, d.delta, d.floor);
        match d.handle {
            TrimHandle::Face { axis, max } => {
                let v = if max {
                    r.box_max[axis]
                } else {
                    r.box_min[axis]
                };
                t.set_box_limit(axis, max, v);
            }
            TrimHandle::Radius { .. } => t.radius = r.radius,
            TrimHandle::Center => t.center = r.center,
        }
    }
    Some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::entity::Entity;
    use bevy::math::Quat;
    use ftgsxr_core::TrimMode;

    fn boxed() -> Trim {
        Trim {
            mode: TrimMode::Box,
            box_min: Vec3::new(-1.0, -1.0, -1.0),
            box_max: Vec3::ONE,
            center: Vec3::ZERO,
            radius: 1.0,
        }
    }

    #[test]
    fn handle_hit_takes_the_nearest_handle_within_its_reach() {
        let t = boxed();
        let pose = Pose::IDENTITY;
        // L = 1: reach 0.18. Down −z through x = 1 (the +x face's handle at (1, 0, 0))
        // and the +z face's handle would need x = 0.
        let ray = (Vec3::new(1.1, 0.0, 5.0), -Vec3::Z);
        assert_eq!(
            handle_hit(&t, &pose, 1.0, ray, 10.0),
            Some((TrimHandle::Face { axis: 0, max: true }, 5.0))
        );
        let miss = (Vec3::new(1.2, 0.0, 5.0), -Vec3::Z);
        assert_eq!(handle_hit(&t, &pose, 1.0, miss, 10.0), None);
        // Down −z through the middle: the +z face (t = 4) before the −z face (t = 6).
        let mid = (Vec3::new(0.0, 0.0, 5.0), -Vec3::Z);
        assert_eq!(
            handle_hit(&t, &pose, 1.0, mid, 10.0),
            Some((TrimHandle::Face { axis: 2, max: true }, 4.0))
        );
        // Beyond t_max, nothing.
        assert_eq!(handle_hit(&t, &pose, 1.0, mid, 3.0), None);
    }

    #[test]
    fn handle_hit_maps_the_handles_by_the_pose() {
        let t = boxed();
        // Scale 2, then up by 3: the +y face's handle is at (0, 5, 0).
        let pose = Pose {
            translation: Vec3::new(0.0, 3.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 2.0,
        };
        let ray = (Vec3::new(0.0, 5.0, 4.0), -Vec3::Z);
        assert_eq!(
            handle_hit(&t, &pose, 1.0, ray, 10.0),
            Some((TrimHandle::Face { axis: 1, max: true }, 4.0))
        );
    }

    #[test]
    fn drag_point_follows_the_mouse_on_the_line_and_the_hand_on_its_ray() {
        let pose = Pose {
            translation: Vec3::ZERO,
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            scale: 1.0,
        };
        // The model's x axis turned to stage −z; the +x face's handle at stage (0, 0, −1).
        let face = TrimHandle::Face { axis: 0, max: true };
        let anchor = Vec3::X;
        let start = (Vec3::new(0.0, 5.0, -1.0), -Vec3::Y);
        let ray = (Vec3::new(0.3, 5.0, -2.0), -Vec3::Y);
        let p = drag_point(Hand::Mouse, face, anchor, start, 0.0, &pose, ray).unwrap();
        assert!((p - Vec3::new(0.0, 0.0, -2.0)).length() < 1e-6, "{p}");
        // A hand's point at t₀ on its current ray, whatever the handle.
        let hand = (Vec3::new(1.0, 2.0, 3.0), Vec3::X);
        assert_eq!(
            drag_point(Hand::Left, face, anchor, start, 2.0, &pose, hand),
            Some(Vec3::new(3.0, 2.0, 3.0))
        );
    }

    fn drag(handle: TrimHandle, delta: Vec3) -> TrimDrag {
        TrimDrag {
            entity: Entity::PLACEHOLDER,
            hand: Hand::Left,
            handle,
            start: boxed(),
            p0: Vec3::ZERO,
            anchor: Vec3::ZERO,
            start_ray: (Vec3::ZERO, Vec3::Z),
            t0: 0.0,
            by_select: false,
            floor: 1e-3,
            delta,
        }
    }

    #[test]
    fn dragged_trim_applies_both_hands_faces() {
        let a = drag(
            TrimHandle::Face {
                axis: 0,
                max: false,
            },
            Vec3::new(0.5, 0.0, 0.0),
        );
        let b = drag(
            TrimHandle::Face { axis: 0, max: true },
            Vec3::new(-0.25, 0.0, 0.0),
        );
        let t = dragged_trim(&[&a, &b]).unwrap();
        assert_eq!(t.box_min.x, -0.5);
        assert_eq!(t.box_max.x, 0.75);
        assert_eq!(dragged_trim(&[]), None);
    }
}
