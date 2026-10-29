// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr_core::types` items no §3 row pins on their own (`cpu` lane), each against a
//! hand-worked literal: `Model::bounds` (E10's B_m, Q-9), `Model::cap_sh`'s shape
//! (§1.2 load option 1; its color clause is U-9's), `TimeSpan` (E9's r, D-32), `Pose`
//! (E5) and `Aabb`.

use approx::assert_relative_eq;
use ftgsxr_core::{Aabb, Model, ModelKind, Pose, Quat, RenderConventions, TimeSpan, Vec3};

fn assert_vec3(a: Vec3, b: Vec3, eps: f32) {
    assert!((a - b).abs().max_element() <= eps, "{a} vs {b}");
}

/// Primitive A moves along x at 1 unit per s through (0, 0, 0) at τ = 1, scale ≤ 0.1,
/// opacity 0.5; primitive B sits at (0, 5, 0) with scale 0.2 and opacity 0.001 < 1/255.
fn two_rows() -> Model {
    Model {
        position: vec![[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
        time_center: vec![1.0, 1.0],
        duration: vec![1.0, 1.0],
        velocity: vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        scale: vec![[0.1, 0.05, 0.02], [0.2, 0.2, 0.2]],
        rotation: vec![[1.0, 0.0, 0.0, 0.0]; 2],
        opacity: vec![0.5, 0.001],
        sh: vec![0.0; 2 * 3],
        sh_degree: 0,
        time: TimeSpan {
            tau0: 0.0,
            tau1: 2.0,
            time_unit: "s".into(),
        },
        conventions: RenderConventions::default(),
        producer: String::new(),
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    }
}

#[test]
fn bounds_cover_the_moving_visible_rows_expanded_by_three_scales() {
    // E10: A spans x ∈ [−1, 1] over τ ∈ [0, 2]; B is below α_skip and leaves the box and
    // the margin; the margin is 3 · 0.1.
    let b = two_rows().bounds(8);
    assert_vec3(b.min, Vec3::new(-1.3, -0.3, -0.3), 1e-6);
    assert_vec3(b.max, Vec3::new(1.3, 0.3, 0.3), 1e-6);
}

#[test]
fn bounds_sample_at_the_requested_times() {
    // Two samples (k = 0, 1, 2 at τ = 0, 1, 2) reach the same ends; at τ = 1.0 only.
    let b = two_rows().bounds(2);
    assert_vec3(b.min, Vec3::new(-1.3, -0.3, -0.3), 1e-6);
    let mut m = two_rows();
    m.time_center = vec![0.5, 1.0]; // A at x = −0.5 at τ₀ and 1.5 at τ₁
    let b = m.bounds(1);
    assert_vec3(b.min, Vec3::new(-0.8, -0.3, -0.3), 1e-6);
    assert_vec3(b.max, Vec3::new(1.8, 0.3, 0.3), 1e-6);
}

#[test]
fn bounds_of_a_single_timestamp_take_one_sample() {
    let mut m = two_rows();
    m.time.tau0 = 1.0;
    m.time.tau1 = 1.0;
    m.velocity[0] = [0.0; 3];
    let b = m.bounds(8);
    assert_vec3(b.min, Vec3::splat(-0.3), 1e-6);
    assert_vec3(b.max, Vec3::splat(0.3), 1e-6);
}

#[test]
fn bounds_without_a_visible_row_cover_every_row() {
    // Q-9: with every opacity ≤ α_skip, the box is over all rows.
    let mut m = two_rows();
    m.opacity = vec![0.001, 0.001];
    let b = m.bounds(8);
    assert_vec3(b.min, Vec3::new(-1.6, -0.6, -0.6), 1e-6);
    assert_vec3(b.max, Vec3::new(1.6, 5.6, 0.6), 1e-6);
}

#[test]
fn cap_sh_keeps_the_low_bands_of_every_row() {
    // Degree 2: 9 coefficients · 3 channels per row; value = 100·row + index in the row.
    let mut m = two_rows();
    m.sh_degree = 2;
    m.sh = (0..2)
        .flat_map(|r| (0..27).map(move |i| (100 * r + i) as f32))
        .collect();
    let mut same = m.clone();
    same.cap_sh(2);
    assert_eq!(same, m, "a cap at L is the identity");
    same.cap_sh(3);
    assert_eq!(same, m, "a cap above L is the identity");

    m.cap_sh(1);
    assert_eq!(m.sh_degree, 1);
    let expected: Vec<f32> = (0..2)
        .flat_map(|r| (0..12).map(move |i| (100 * r + i) as f32))
        .collect();
    assert_eq!(m.sh, expected);
    m.validate()
        .expect("a model with its SH degree capped stays valid");
    m.cap_sh(0);
    assert_eq!(m.sh, vec![0.0, 1.0, 2.0, 100.0, 101.0, 102.0]);
}

#[test]
fn len_and_single_timestamp() {
    let mut m = two_rows();
    assert_eq!(m.len(), 2);
    assert!(!m.is_single_timestamp());
    m.time.tau1 = 0.0;
    assert!(m.is_single_timestamp());
}

#[test]
fn time_span_rate_and_span() {
    let mut span = TimeSpan {
        tau0: 10.0,
        tau1: 12.5,
        time_unit: "s".into(),
    };
    assert_eq!(span.span(), 2.5);
    assert!(!span.is_single());
    assert_eq!(span.rate_per_second().unwrap(), 1.0);
    span.time_unit = "ms".into();
    assert_eq!(span.rate_per_second().unwrap(), 1000.0);
    span.time_unit = "frames".into();
    assert_eq!(span.rate_per_second().unwrap_err().code(), "E-FILE-VALUE");
    span.tau1 = 10.0;
    assert!(span.is_single());
    assert_eq!(span.span(), 0.0);
}

#[test]
fn pose_maps_points_and_directions_as_e5() {
    // t = (1, 2, 3), a 90° turn about +y (x → −z), scale 2.
    let pose = Pose {
        translation: Vec3::new(1.0, 2.0, 3.0),
        rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        scale: 2.0,
    };
    assert_vec3(
        pose.transform_point(Vec3::X),
        Vec3::new(1.0, 2.0, 1.0),
        1e-6,
    );
    assert_vec3(
        pose.transform_direction(Vec3::X),
        Vec3::new(0.0, 0.0, -1.0),
        1e-6,
    );
    assert_vec3(
        pose.matrix().transform_point3(Vec3::X),
        Vec3::new(1.0, 2.0, 1.0),
        1e-6,
    );
    let inv = pose.inverse();
    assert_vec3(inv.transform_point(Vec3::new(1.0, 2.0, 1.0)), Vec3::X, 1e-6);
    assert_relative_eq!(inv.scale, 0.5);
    let id = pose.then(&inv);
    assert_vec3(id.translation, Vec3::ZERO, 1e-6);
    assert_relative_eq!(id.scale, 1.0);
    assert!(id.rotation.dot(Quat::IDENTITY).abs() > 1.0 - 1e-6);
    assert_eq!(Pose::IDENTITY.transform_point(Vec3::ONE), Vec3::ONE);
}

#[test]
fn pose_then_applies_self_first() {
    let a = Pose {
        translation: Vec3::new(0.0, 0.0, 1.0),
        rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2), // x → y
        scale: 3.0,
    };
    let b = Pose {
        translation: Vec3::new(5.0, 0.0, 0.0),
        rotation: Quat::IDENTITY,
        scale: 0.5,
    };
    // a: (1, 0, 0) → (0, 3, 1); then b: → (5, 1.5, 0.5).
    assert_vec3(
        a.then(&b).transform_point(Vec3::X),
        Vec3::new(5.0, 1.5, 0.5),
        1e-6,
    );
}

#[test]
fn aabb_center_size_longest_side() {
    let b = Aabb {
        min: Vec3::new(-1.0, 0.0, 2.0),
        max: Vec3::new(3.0, 1.0, 2.5),
    };
    assert_eq!(b.center(), Vec3::new(1.0, 0.5, 2.25));
    assert_eq!(b.size(), Vec3::new(4.0, 1.0, 0.5));
    assert_eq!(b.longest_side(), 4.0);
}
