// SPDX-License-Identifier: Apache-2.0

//! U-39 (§3.1, `cpu` lane): E17's `Trim`: the closed test on a box and a sphere, off
//! keeping everything, the initial shapes of a hand-worked box, `set_box_limit` past the
//! opposite limit, and `is_valid` on each broken case. Exact.

use ftgsxr_core::{Aabb, Trim, TrimMode, Vec3};

fn boxed(min: [f32; 3], max: [f32; 3]) -> Trim {
    Trim {
        mode: TrimMode::Box,
        box_min: Vec3::from_array(min),
        box_max: Vec3::from_array(max),
        center: Vec3::ZERO,
        radius: 1.0,
    }
}

fn sphere(center: [f32; 3], radius: f32) -> Trim {
    Trim {
        mode: TrimMode::Sphere,
        box_min: Vec3::ZERO,
        box_max: Vec3::ONE,
        center: Vec3::from_array(center),
        radius,
    }
}

#[test]
fn u39_box_contains_inside_and_every_face_not_outside() {
    let t = boxed([-1.0, 0.0, 2.0], [1.0, 3.0, 4.0]);
    assert!(t.contains(Vec3::new(0.0, 1.5, 3.0)));
    // Each of the six faces, the boundary included.
    for p in [
        [-1.0, 1.0, 3.0],
        [1.0, 1.0, 3.0],
        [0.0, 0.0, 3.0],
        [0.0, 3.0, 3.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 4.0],
    ] {
        assert!(t.contains(Vec3::from_array(p)), "{p:?} on a face");
    }
    // A corner.
    assert!(t.contains(Vec3::new(1.0, 3.0, 4.0)));
    // Just outside each face, and far away.
    for p in [
        [-1.5, 1.0, 3.0],
        [1.5, 1.0, 3.0],
        [0.0, -0.5, 3.0],
        [0.0, 3.5, 3.0],
        [0.0, 1.0, 1.5],
        [0.0, 1.0, 4.5],
        [10.0, 10.0, 10.0],
    ] {
        assert!(!t.contains(Vec3::from_array(p)), "{p:?} outside");
    }
}

#[test]
fn u39_sphere_contains_inside_and_its_surface_not_outside() {
    let t = sphere([1.0, 2.0, 3.0], 2.0);
    assert!(t.contains(Vec3::new(1.0, 2.0, 3.0)));
    assert!(t.contains(Vec3::new(2.0, 2.5, 3.0)));
    // On the surface along each axis, both ways.
    for p in [
        [3.0, 2.0, 3.0],
        [-1.0, 2.0, 3.0],
        [1.0, 4.0, 3.0],
        [1.0, 0.0, 3.0],
        [1.0, 2.0, 5.0],
        [1.0, 2.0, 1.0],
    ] {
        assert!(t.contains(Vec3::from_array(p)), "{p:?} on the surface");
    }
    // Outside: past the surface, and the corner of the sphere's box.
    assert!(!t.contains(Vec3::new(3.5, 2.0, 3.0)));
    assert!(!t.contains(Vec3::new(3.0, 4.0, 5.0)));
}

#[test]
fn u39_off_contains_everything() {
    let t = Trim::default();
    assert_eq!(t.mode, TrimMode::Off);
    for p in [Vec3::ZERO, Vec3::splat(1e30), Vec3::splat(-1e30)] {
        assert!(t.contains(p));
    }
    // Off with shapes that would exclude the point still keeps it.
    let off = Trim {
        mode: TrimMode::Off,
        ..boxed([0.0; 3], [1.0; 3])
    };
    assert!(off.contains(Vec3::splat(5.0)));
    assert_eq!(off.shape_bounds(), None);
}

#[test]
fn u39_initial_of_a_hand_worked_box() {
    // A 2 × 4 × 4 box: center (1, 0, 2), diagonal √36 = 6, radius 3.
    let b = Aabb {
        min: Vec3::new(0.0, -2.0, 0.0),
        max: Vec3::new(2.0, 2.0, 4.0),
    };
    let t = Trim::initial(&b);
    assert_eq!(
        t,
        Trim {
            mode: TrimMode::Off,
            box_min: b.min,
            box_max: b.max,
            center: Vec3::new(1.0, 0.0, 2.0),
            radius: 3.0,
        }
    );
    // Turned on, either shape keeps every corner of the box.
    let boxed = Trim {
        mode: TrimMode::Box,
        ..t
    };
    let round = Trim {
        mode: TrimMode::Sphere,
        ..t
    };
    for corner in [
        b.min,
        b.max,
        Vec3::new(0.0, 2.0, 4.0),
        Vec3::new(2.0, -2.0, 0.0),
    ] {
        assert!(boxed.contains(corner) && round.contains(corner), "{corner}");
    }
}

#[test]
fn u39_shape_bounds_of_each_mode() {
    let b = boxed([-1.0, 0.0, 2.0], [1.0, 3.0, 4.0]);
    assert_eq!(
        b.shape_bounds(),
        Some(Aabb {
            min: Vec3::new(-1.0, 0.0, 2.0),
            max: Vec3::new(1.0, 3.0, 4.0)
        })
    );
    let s = sphere([1.0, 2.0, 3.0], 2.0);
    assert_eq!(
        s.shape_bounds(),
        Some(Aabb {
            min: Vec3::new(-1.0, 0.0, 1.0),
            max: Vec3::new(3.0, 4.0, 5.0)
        })
    );
}

#[test]
fn u39_set_box_limit_moves_the_opposite_limit_when_crossed() {
    let mut t = boxed([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    // Within range: only the one limit moves.
    t.set_box_limit(0, true, 1.5);
    assert_eq!((t.box_min.x, t.box_max.x), (0.0, 1.5));
    t.set_box_limit(1, false, 0.5);
    assert_eq!((t.box_min.y, t.box_max.y), (0.5, 2.0));
    // A max below the min moves the min down with it.
    t.set_box_limit(2, true, -1.0);
    assert_eq!((t.box_min.z, t.box_max.z), (-1.0, -1.0));
    // A min above the max moves the max up with it.
    t.set_box_limit(0, false, 3.0);
    assert_eq!((t.box_min.x, t.box_max.x), (3.0, 3.0));
    // The other axes are untouched.
    assert_eq!((t.box_min.y, t.box_max.y), (0.5, 2.0));
    assert!(t.is_valid());
}

#[test]
fn u39_is_valid_on_each_broken_case() {
    let good = boxed([0.0; 3], [1.0; 3]);
    assert!(good.is_valid());
    // A degenerate box (min = max) is valid; r must be > 0.
    assert!(boxed([1.0; 3], [1.0; 3]).is_valid());
    let broken = [
        Trim {
            box_min: Vec3::new(0.0, 2.0, 0.0),
            ..good
        }, // min > max on y
        Trim {
            radius: 0.0,
            ..good
        },
        Trim {
            radius: -1.0,
            ..good
        },
        Trim {
            radius: f32::NAN,
            ..good
        },
        Trim {
            box_min: Vec3::new(f32::NAN, 0.0, 0.0),
            ..good
        },
        Trim {
            box_max: Vec3::new(0.0, f32::INFINITY, 0.0),
            ..good
        },
        Trim {
            center: Vec3::new(0.0, 0.0, f32::NEG_INFINITY),
            ..good
        },
    ];
    for t in broken {
        assert!(!t.is_valid(), "{t:?}");
    }
    // The default (zero radius) is not a valid shape, whatever the mode.
    assert!(!Trim::default().is_valid());
}
