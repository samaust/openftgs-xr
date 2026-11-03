// SPDX-License-Identifier: Apache-2.0

//! U-32's `panel_hit` part (§1.8, `cpu`): the quad's corners map to (0, 0) and (W, H),
//! a ray outside misses.

use bevy::math::{Quat, UVec2, Vec2, Vec3};
use ftgsxr_app::panel_hit;
use ftgsxr_core::Pose;

const SIZE_M: Vec2 = Vec2::new(0.56, 0.42);
const SIZE_PX: UVec2 = UVec2::new(1024, 768);

/// A panel 1 m in front of the origin along −z, turned to face it, and the ray from the
/// origin through a point of the quad given in its own (x right, y up) metres.
fn ray_to(panel: Pose, local: Vec2) -> (Vec3, Vec3) {
    let target = panel.translation + panel.rotation * local.extend(0.0);
    (Vec3::ZERO, target.normalize())
}

fn panels() -> Vec<Pose> {
    vec![
        Pose {
            translation: Vec3::new(0.0, 0.0, -1.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
        },
        Pose {
            translation: Vec3::new(-1.0, 0.3, 0.0),
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            scale: 1.0,
        },
    ]
}

fn close(a: Vec2, b: Vec2) -> bool {
    // §3: exact; f32 rounding of the ray's own construction aside.
    (a - b).abs().max_element() <= 1e-4
}

#[test]
fn u32_the_corners_map_to_zero_and_w_h() {
    for p in panels() {
        let h = SIZE_M / 2.0;
        let (o, d) = ray_to(p, Vec2::new(-h.x, h.y));
        assert!(close(
            panel_hit(o, d, p, SIZE_M, SIZE_PX).unwrap(),
            Vec2::ZERO
        ));
        let (o, d) = ray_to(p, Vec2::new(h.x, -h.y));
        assert!(close(
            panel_hit(o, d, p, SIZE_M, SIZE_PX).unwrap(),
            SIZE_PX.as_vec2()
        ));
        let (o, d) = ray_to(p, Vec2::ZERO);
        assert!(close(
            panel_hit(o, d, p, SIZE_M, SIZE_PX).unwrap(),
            SIZE_PX.as_vec2() / 2.0
        ));
    }
}

#[test]
fn u32_a_ray_outside_the_quad_misses() {
    for p in panels() {
        let h = SIZE_M / 2.0;
        for local in [
            Vec2::new(h.x * 1.01, 0.0),
            Vec2::new(0.0, -h.y * 1.01),
            Vec2::new(-h.x * 1.2, h.y * 1.2),
        ] {
            let (o, d) = ray_to(p, local);
            assert_eq!(panel_hit(o, d, p, SIZE_M, SIZE_PX), None);
        }
        // Pointing away, and from behind the quad.
        let (o, d) = ray_to(p, Vec2::ZERO);
        assert_eq!(panel_hit(o, -d, p, SIZE_M, SIZE_PX), None);
        let behind = p.translation * 2.0;
        assert_eq!(panel_hit(behind, -d, p, SIZE_M, SIZE_PX), None);
    }
}
