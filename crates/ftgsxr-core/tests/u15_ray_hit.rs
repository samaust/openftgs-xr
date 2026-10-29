// SPDX-License-Identifier: Apache-2.0

//! U-15, E10 (`ray_hit`, `cpu` lane, D-42): hand-worked hits and misses on the unit
//! box \[0, 1\]³; an origin inside gives 0; a zero direction component; a box posed by a
//! 90° rotation and scale 2; the nearer of two instances wins (two calls compared); the
//! 10 m limit. Tolerance 10⁻⁶.

use approx::assert_relative_eq;
use ftgsxr_core::{Aabb, Pose, Quat, Vec3, ray_hit};

/// The row's tolerance.
const TOL: f32 = 1e-6;
/// E10's t_max.
const T_MAX: f32 = 10.0;

fn unit_box() -> Aabb {
    Aabb {
        min: Vec3::ZERO,
        max: Vec3::ONE,
    }
}

fn hit(origin: Vec3, dir: Vec3, pose: &Pose) -> Option<f32> {
    ray_hit(origin, dir, &unit_box(), pose, T_MAX)
}

fn assert_hit(got: Option<f32>, want: f32) {
    let got = got.unwrap_or_else(|| panic!("expected a hit at {want}, got a miss"));
    assert_relative_eq!(got, want, epsilon = TOL, max_relative = TOL);
}

#[test]
fn u15_straight_hit_enters_the_near_face() {
    // From z = −2 along +z through the middle of the box: entry at z = 0, 2 m away.
    assert_hit(
        hit(Vec3::new(0.5, 0.5, -2.0), Vec3::Z, &Pose::IDENTITY),
        2.0,
    );
}

#[test]
fn u15_diagonal_hit_enters_at_the_edge() {
    // From (−1, −1, 0.5) along (1, 1, 0)/√2: the slabs x and y both open at t = √2,
    // the entry point (0, 0, 0.5) on the box's edge.
    let dir = Vec3::new(1.0, 1.0, 0.0).normalize();
    assert_hit(
        hit(Vec3::new(-1.0, -1.0, 0.5), dir, &Pose::IDENTITY),
        2f32.sqrt(),
    );
}

#[test]
fn u15_misses_beside_and_behind() {
    // Beside: x = 2 lies outside [0, 1] and the ray never changes x.
    assert_eq!(
        hit(Vec3::new(2.0, 0.5, -2.0), Vec3::Z, &Pose::IDENTITY),
        None
    );
    // Behind: the box lies at t ∈ [−3, −2], t_out < 0.
    assert_eq!(
        hit(Vec3::new(0.5, 0.5, 3.0), Vec3::Z, &Pose::IDENTITY),
        None
    );
    // Skew: from (−1, 0.5, −1) along (1, 0, −1)/√2 the ray moves away from the box.
    let away = Vec3::new(1.0, 0.0, -1.0).normalize();
    assert_eq!(hit(Vec3::new(-1.0, 0.5, -1.0), away, &Pose::IDENTITY), None);
    // Past a corner: from (−1, 0.1, 0.5) along (1, 1, 0)/√2, y leaves the box at
    // t = 0.9·√2 before x enters it at t = √2, so t_in > t_out.
    let diag = Vec3::new(1.0, 1.0, 0.0).normalize();
    assert_eq!(hit(Vec3::new(-1.0, 0.1, 0.5), diag, &Pose::IDENTITY), None);
}

#[test]
fn u15_origin_inside_gives_zero() {
    for dir in [Vec3::X, -Vec3::Y, Vec3::new(1.0, -2.0, 3.0).normalize()] {
        assert_hit(hit(Vec3::new(0.5, 0.25, 0.75), dir, &Pose::IDENTITY), 0.0);
    }
}

#[test]
fn u15_zero_direction_component_uses_the_infinite_limit() {
    // d = (0, 0, 1): the x and y slabs are open for all t when the origin lies in them,
    // including exactly on their planes, and closed for all t when it lies outside.
    assert_hit(
        hit(Vec3::new(0.0, 0.5, -2.0), Vec3::Z, &Pose::IDENTITY),
        2.0,
    );
    assert_hit(
        hit(Vec3::new(1.0, 1.0, -2.0), Vec3::Z, &Pose::IDENTITY),
        2.0,
    );
    assert_eq!(
        hit(Vec3::new(1.5, 0.5, -2.0), Vec3::Z, &Pose::IDENTITY),
        None
    );
    assert_eq!(
        hit(Vec3::new(0.5, -0.5, -2.0), Vec3::Z, &Pose::IDENTITY),
        None
    );
}

#[test]
fn u15_box_posed_by_a_90_degree_rotation_and_scale_2() {
    // P = (t = (5, 0, 0), q = 90° about y, s = 2). R_y(90°) maps (x, y, z) to (z, y, −x),
    // so the posed unit box is the stage box x ∈ [5, 7], y ∈ [0, 2], z ∈ [−2, 0].
    let pose = Pose {
        translation: Vec3::new(5.0, 0.0, 0.0),
        rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        scale: 2.0,
    };
    // Down −z from z = 5 at (6, 1): entry at z = 0, 5 m.
    assert_hit(hit(Vec3::new(6.0, 1.0, 5.0), -Vec3::Z, &pose), 5.0);
    // Along +x from x = 0 at (y, z) = (1, −1): entry at x = 5, 5 m.
    assert_hit(hit(Vec3::new(0.0, 1.0, -1.0), Vec3::X, &pose), 5.0);
    // Along −x from x = 9: entry at x = 7, 2 m (the far side of the unposed box would
    // be 4 m: the scale counts).
    assert_hit(hit(Vec3::new(9.0, 1.5, -0.5), -Vec3::X, &pose), 2.0);
    // Inside the posed box but outside the unposed one: 0.
    assert_hit(hit(Vec3::new(6.5, 1.5, -1.5), Vec3::Y, &pose), 0.0);
    // y = 2.5 lies above the posed box (and inside the unposed box's x and z): a miss.
    assert_eq!(hit(Vec3::new(6.0, 2.5, 5.0), -Vec3::Z, &pose), None);
    // z = 0.5 lies in the unposed box but in front of the posed one: a miss along x.
    assert_eq!(hit(Vec3::new(0.0, 0.5, 0.5), Vec3::X, &pose), None);
}

#[test]
fn u15_nearer_of_two_instances_wins() {
    // Two instances of the unit box along the same ray: A at the identity, entered at
    // 2 m; B moved to z ∈ [4, 5], entered at 6 m. The selection takes the smaller.
    let origin = Vec3::new(0.5, 0.5, -2.0);
    let near = Pose::IDENTITY;
    let far = Pose {
        translation: Vec3::new(0.0, 0.0, 4.0),
        ..Pose::IDENTITY
    };
    let a = hit(origin, Vec3::Z, &near).expect("A is hit");
    let b = hit(origin, Vec3::Z, &far).expect("B is hit");
    assert_relative_eq!(a, 2.0, epsilon = TOL);
    assert_relative_eq!(b, 6.0, epsilon = TOL, max_relative = TOL);
    assert!(a < b, "the nearer instance is A");
    // Listed the other way round, the selection is unchanged.
    let hits = [("B", b), ("A", a)];
    let nearest = hits.iter().min_by(|x, y| x.1.total_cmp(&y.1)).unwrap();
    assert_eq!(nearest.0, "A");
}

#[test]
fn u15_ten_metre_limit() {
    let origin = Vec3::new(0.5, 0.5, 0.0);
    let at = |z: f32| Pose {
        translation: Vec3::new(0.0, 0.0, z),
        ..Pose::IDENTITY
    };
    // Entry at exactly 10 m: max(t_in, 0) ≤ t_max holds.
    assert_hit(hit(origin, Vec3::Z, &at(10.0)), 10.0);
    // Entry at 9.5 m: a hit even though the exit (10.5 m) lies beyond the limit.
    assert_hit(hit(origin, Vec3::Z, &at(9.5)), 9.5);
    // Entry at 10.5 m: beyond the limit.
    assert_eq!(hit(origin, Vec3::Z, &at(10.5)), None);
    // The limit is the caller's: 11 m admits the same box.
    assert!(ray_hit(origin, Vec3::Z, &unit_box(), &at(10.5), 11.0).is_some());
}
