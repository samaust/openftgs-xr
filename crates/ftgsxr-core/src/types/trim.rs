// SPDX-License-Identifier: Apache-2.0

//! E17's trim (§2.1 `ftgsxr_core::types`): an instance's box or sphere in model space,
//! outside which its primitives are not drawn (D-56).

use glam::Vec3;

use super::model::Aabb;

/// Which shape a trim keeps (E17).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TrimMode {
    /// No trim: every primitive is kept.
    #[default]
    Off,
    /// The box \[`box_min`, `box_max`\].
    Box,
    /// The sphere of `center` and `radius`.
    Sphere,
}

/// E17's trim, in model space. `Default` is off with zero shapes, replaced by `initial`
/// when the trim is turned on. Both shapes keep their values while the mode changes.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Trim {
    /// Off, box or sphere.
    pub mode: TrimMode,
    /// The box's lower corner b_min.
    pub box_min: Vec3,
    /// The box's upper corner b_max.
    pub box_max: Vec3,
    /// The sphere's center c.
    pub center: Vec3,
    /// The sphere's radius r.
    pub radius: f32,
}

impl Trim {
    /// E17's initial shapes for an untrimmed B_m `bounds`, mode off: the box is `bounds`,
    /// the sphere is centered on its center with half its diagonal as radius, so turning
    /// either on hides nothing.
    pub fn initial(bounds: &Aabb) -> Trim {
        Trim {
            mode: TrimMode::Off,
            box_min: bounds.min,
            box_max: bounds.max,
            center: bounds.center(),
            radius: 0.5 * bounds.size().length(),
        }
    }

    /// E17's closed test: whether `p` (a primitive's center μx(τ) in model space) lies
    /// inside the shape, the boundary included; true when the trim is off. The sphere
    /// compares ‖p − c‖² with r², as the cull pass does.
    pub fn contains(&self, p: Vec3) -> bool {
        match self.mode {
            TrimMode::Off => true,
            TrimMode::Box => p.cmpge(self.box_min).all() && p.cmple(self.box_max).all(),
            TrimMode::Sphere => (p - self.center).length_squared() <= self.radius * self.radius,
        }
    }

    /// The shape's own box: \[b_min, b_max\], or c ± r on each axis; `None` when off.
    pub fn shape_bounds(&self) -> Option<Aabb> {
        match self.mode {
            TrimMode::Off => None,
            TrimMode::Box => Some(Aabb {
                min: self.box_min,
                max: self.box_max,
            }),
            TrimMode::Sphere => Some(Aabb {
                min: self.center - Vec3::splat(self.radius),
                max: self.center + Vec3::splat(self.radius),
            }),
        }
    }

    /// Sets the box's limit on `axis` (0 = x, 1 = y, 2 = z): b_max when `max`, else b_min.
    /// A value past the opposite limit moves that limit with it, so b_min ≤ b_max holds
    /// (§1.7 Trim).
    ///
    /// # Panics
    /// If `axis` is not 0, 1 or 2.
    pub fn set_box_limit(&mut self, axis: usize, max: bool, value: f32) {
        assert!(axis < 3, "axis {axis} is not 0, 1 or 2");
        if max {
            self.box_max[axis] = value;
            if self.box_min[axis] > value {
                self.box_min[axis] = value;
            }
        } else {
            self.box_min[axis] = value;
            if self.box_max[axis] < value {
                self.box_max[axis] = value;
            }
        }
    }

    /// Whether both shapes are well formed, whatever the mode: every value finite,
    /// b_min ≤ b_max on each axis and r > 0 (§1.11's rule for a `trim` read from a file).
    pub fn is_valid(&self) -> bool {
        self.box_min.is_finite()
            && self.box_max.is_finite()
            && self.center.is_finite()
            && self.radius.is_finite()
            && self.box_min.cmple(self.box_max).all()
            && self.radius > 0.0
    }
}
