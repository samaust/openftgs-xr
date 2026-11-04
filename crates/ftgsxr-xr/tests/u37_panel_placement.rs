// SPDX-License-Identifier: Apache-2.0

//! U-37 (`cpu`, §3.1), panel placement (source of truth: U-37's row and §1.8 User
//! interface): the summon pose 1 m ahead of the head along its yaw, 0.2 m below eye
//! height, facing the head; the wrist menu 8 cm above the grip pose, visible when the
//! palm faces the head (a dot-product test). Tolerance 10⁻⁶. The functions are Q-265's;
//! their `grip` is the palm frame of Q-305 (+x the palm's outward normal).

use ftgsxr_core::{Pose, Quat, Vec3};
use ftgsxr_xr::{summon_pose, wrist_menu_pose, wrist_menu_visible};

const TOL: f32 = 1e-6;

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).abs().max_element() <= TOL
}

fn at(t: Vec3, rotation: Quat) -> Pose {
    Pose {
        translation: t,
        rotation,
        scale: 1.0,
    }
}

/// U-37: a head at (0.5, 1.6, 2) turned 30° left and looking 20° down; the quad's centre
/// is 1 m along the yaw, (−sin 30°, 0, −cos 30°), and 0.2 m below the eyes, upright,
/// its +z toward the head.
#[test]
fn u37_summon_pose_is_1_m_ahead_along_the_yaw_and_0_2_m_below_facing_the_head() {
    let yaw = 30f32.to_radians();
    let head = at(
        Vec3::new(0.5, 1.6, 2.0),
        Quat::from_rotation_y(yaw) * Quat::from_rotation_x(-20f32.to_radians()),
    );
    let p = summon_pose(head);
    let expected = Vec3::new(0.5 - 0.5, 1.4, 2.0 - 3f32.sqrt() / 2.0);
    assert!(close(p.translation, expected), "{:?}", p.translation);
    // Facing the head: +z points back along the yaw, horizontally.
    assert!(close(
        p.rotation * Vec3::Z,
        Vec3::new(0.5, 0.0, 3f32.sqrt() / 2.0)
    ));
    // Upright: its up is stage up.
    assert!(close(p.rotation * Vec3::Y, Vec3::Y));
    assert!((p.scale - 1.0).abs() <= TOL);
}

/// U-37 with the head looking straight down: Q-17's rule takes the top of the head as
/// the forward, so the pose stays finite (no NaN reaches a pose).
#[test]
fn u37_summon_pose_with_the_head_looking_straight_down_is_finite() {
    let head = at(
        Vec3::new(0.0, 1.6, 0.0),
        Quat::from_rotation_x(-90f32.to_radians()),
    );
    let p = summon_pose(head);
    // The head's top, R(q)(0, 1, 0), points along −z.
    assert!(
        close(p.translation, Vec3::new(0.0, 1.4, -1.0)),
        "{:?}",
        p.translation
    );
    assert!(close(p.rotation * Vec3::Z, Vec3::Z));
}

/// U-37: the wrist menu is 8 cm from the grip along the palm's outward normal (+x of the
/// palm frame) and faces the head.
#[test]
fn u37_wrist_menu_is_8_cm_above_the_grip_facing_the_head() {
    // The palm frame turned −90° about y, so its outward normal is (0, 0, 1), toward a head
    // straight ahead of it.
    let grip = at(
        Vec3::new(-0.2, 1.1, -0.3),
        Quat::from_rotation_y(-90f32.to_radians()),
    );
    let head = at(Vec3::new(-0.2, 1.1, 0.5), Quat::IDENTITY);
    let p = wrist_menu_pose(grip, head);
    assert!(
        close(p.translation, Vec3::new(-0.2, 1.1, -0.22)),
        "{:?}",
        p.translation
    );
    // The head level with the menu, straight along +z: the menu faces +z, upright.
    assert!(close(p.rotation * Vec3::Z, Vec3::Z));
    assert!(close(p.rotation * Vec3::Y, Vec3::Y));
}

/// U-37: the dot-product test, the palm normal against the unit vector to the head:
/// visible above 0.5 (within 60°), hidden at 90° and with the palm turned away.
#[test]
fn u37_wrist_menu_is_visible_only_when_the_palm_faces_the_head() {
    let grip = at(Vec3::ZERO, Quat::IDENTITY); // the outward normal is +x
    let head = |dir: Vec3| at(dir * 0.5, Quat::IDENTITY);
    assert!(wrist_menu_visible(grip, head(Vec3::X)));
    // 55° off the normal: cos 55° ≈ 0.574 > 0.5.
    let a = 55f32.to_radians();
    assert!(wrist_menu_visible(
        grip,
        head(Vec3::new(a.cos(), a.sin(), 0.0))
    ));
    // 65° off: cos 65° ≈ 0.423.
    let a = 65f32.to_radians();
    assert!(!wrist_menu_visible(
        grip,
        head(Vec3::new(a.cos(), a.sin(), 0.0))
    ));
    assert!(!wrist_menu_visible(grip, head(Vec3::Z)));
    assert!(!wrist_menu_visible(grip, head(Vec3::NEG_X)));
}
