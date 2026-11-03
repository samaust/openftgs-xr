// SPDX-License-Identifier: Apache-2.0

//! U-32's desktop part, `pick_instance` (`cpu`, §3.1): E10 over every instance returns
//! the nearest hit (D-42); misses and the ray's limit give nothing.

use bevy::ecs::world::World;
use ftgsxr_app::pick_instance;
use ftgsxr_core::{Aabb, Pose, Quat, Vec3};

fn unit() -> Aabb {
    Aabb {
        min: Vec3::splat(-0.5),
        max: Vec3::splat(0.5),
    }
}

fn at(z: f32) -> Pose {
    Pose {
        translation: Vec3::new(0.0, 0.0, z),
        ..Pose::IDENTITY
    }
}

#[test]
fn u32_pick_instance_returns_the_nearest() {
    let mut w = World::new();
    let (far, near, beside) = (
        w.spawn_empty().id(),
        w.spawn_empty().id(),
        w.spawn_empty().id(),
    );
    let beside_pose = Pose {
        translation: Vec3::new(3.0, 0.0, -1.0),
        ..Pose::IDENTITY
    };
    // The far one listed first: order does not decide.
    let instances = [
        (far, at(-6.0), unit()),
        (near, at(-3.0), unit()),
        (beside, beside_pose, unit()),
    ];
    let (e, t) = pick_instance(Vec3::ZERO, -Vec3::Z, &instances, 10.0).unwrap();
    assert_eq!(e, near);
    assert!((t - 2.5).abs() < 1e-6, "{t}");
    // The near one moved aside: the far one wins.
    let instances = [(far, at(-6.0), unit()), (beside, beside_pose, unit())];
    let (e, t) = pick_instance(Vec3::ZERO, -Vec3::Z, &instances, 10.0).unwrap();
    assert_eq!(e, far);
    assert!((t - 5.5).abs() < 1e-6, "{t}");
}

#[test]
fn u32_pick_instance_misses_and_the_limit() {
    let mut w = World::new();
    let a = w.spawn_empty().id();
    let instances = [(a, at(-12.0), unit())];
    assert_eq!(pick_instance(Vec3::ZERO, -Vec3::Z, &instances, 10.0), None);
    assert_eq!(
        pick_instance(Vec3::ZERO, Vec3::Z, &[(a, at(-3.0), unit())], 10.0),
        None
    );
    assert_eq!(pick_instance(Vec3::ZERO, -Vec3::Z, &[], 10.0), None);
    // A posed box (90° about y, scale 2): its entry at 3 − 1.
    let posed = Pose {
        translation: Vec3::new(0.0, 0.0, -3.0),
        rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        scale: 2.0,
    };
    let (e, t) = pick_instance(Vec3::ZERO, -Vec3::Z, &[(a, posed, unit())], 10.0).unwrap();
    assert_eq!(e, a);
    assert!((t - 2.0).abs() < 1e-6, "{t}");
}
