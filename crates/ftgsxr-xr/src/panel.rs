// SPDX-License-Identifier: Apache-2.0

//! The main panel's and the wrist menu's placement (§1.8 User interface, §2.1
//! `ftgsxr_xr::panel`): the pure placement functions of U-37 (Q-265), and the systems
//! (`system`) that spawn the quads, summon, move and resize the main panel and place
//! the wrist menu.

mod system;

pub(crate) use system::PanelPlugin;

use bevy::math::Mat3;
use ftgsxr_core::{Pose, Quat, Vec3};

/// Below this a length counts as zero (Q-17's 1 µm).
const DEGENERATE_M: f32 = 1e-6;
/// The summon pose's distance ahead of the head and drop below the eyes (§1.8).
const SUMMON_AHEAD_M: f32 = 1.0;
const SUMMON_BELOW_M: f32 = 0.2;
/// The wrist menu's offset from the grip along the palm's outward normal (§1.8, Q-265).
const WRIST_ABOVE_M: f32 = 0.08;
/// The palm faces the head when its normal's dot with the unit vector to the head
/// exceeds this (Q-265).
const PALM_FACING_DOT: f32 = 0.5;

/// How the main panel follows the user (§2.1): `Summon` places it at the summon pose
/// each time it is shown; `WorldLocked` shows it where it was left (Q-T462.3).
// Q-T462.3
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelFollow {
    /// Shown where it was left.
    WorldLocked,
    /// Shown at the summon pose (§1.8).
    Summon,
}

/// The main panel's quad (§2.1), with `WorldPanel` and `PanelQuad`.
#[derive(bevy::prelude::Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct MainPanel {
    /// How it follows the user.
    pub follow: PanelFollow,
}

/// The head's horizontal forward, a unit vector: the horizontal part of its −z, or, when
/// the head looks straight up or down, of the direction its top points (negated when it
/// looks up), as `fit_pose` takes it (Q-17).
// Q-17
fn yaw_forward(head: &Pose) -> Vec3 {
    let horizontal = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
    let look = head.rotation * Vec3::NEG_Z;
    let mut forward = horizontal(look);
    if forward.length() < DEGENERATE_M {
        let top = horizontal(head.rotation * Vec3::Y);
        forward = if look.y > 0.0 { -top } else { top };
    }
    forward.try_normalize().unwrap_or(Vec3::NEG_Z)
}

/// The rotation whose +z is `facing` and whose +y is stage up made perpendicular to it
/// (upright); the identity when `facing` is zero.
fn upright_facing(facing: Vec3) -> Quat {
    let Some(z) = facing.try_normalize() else {
        return Quat::IDENTITY;
    };
    // Q-17: facing straight up or down, +x stays stage x.
    let x = Vec3::Y.cross(z);
    let x = if x.length() < DEGENERATE_M {
        Vec3::X
    } else {
        x.normalize()
    };
    let y = z.cross(x);
    Quat::from_mat3(&Mat3::from_cols(x, y, z)).normalize()
}

/// The summon pose (§1.8, U-37): the quad's centre 1 m from the head along its
/// horizontal forward and 0.2 m below the eyes, upright, its +z toward the head.
// Q-265
pub fn summon_pose(head: Pose) -> Pose {
    let forward = yaw_forward(&head);
    Pose {
        translation: head.translation + SUMMON_AHEAD_M * forward - SUMMON_BELOW_M * Vec3::Y,
        rotation: upright_facing(-forward),
        scale: 1.0,
    }
}

/// The wrist menu's pose (§1.8, U-37): 8 cm from the grip along the palm's outward
/// normal, the palm frame's +x (Q-T462.1), facing the head (+z toward its position) and
/// upright.
// Q-265, Q-T462.1
pub fn wrist_menu_pose(grip: Pose, head: Pose) -> Pose {
    let normal = grip.rotation * Vec3::X;
    let translation = grip.translation + WRIST_ABOVE_M * normal;
    let facing = (head.translation - translation)
        .try_normalize()
        .unwrap_or(normal);
    Pose {
        translation,
        rotation: upright_facing(facing),
        scale: 1.0,
    }
}

/// Whether the wrist menu shows (§1.8, U-37): the palm's outward normal (the palm
/// frame's +x, Q-T462.1) dotted with the unit vector from the grip to the head exceeds
/// 0.5.
// Q-265, Q-T462.1
pub fn wrist_menu_visible(grip: Pose, head: Pose) -> bool {
    let normal = grip.rotation * Vec3::X;
    (head.translation - grip.translation)
        .try_normalize()
        .is_some_and(|to_head| normal.dot(to_head) > PALM_FACING_DOT)
}
