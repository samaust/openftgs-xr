// SPDX-License-Identifier: Apache-2.0

//! U-11, the projection clauses (E6's CPU mirror `covariance::project`, `cpu` lane): a
//! primitive at camera depth 4 with scale 0.08 through a 64 × 48 camera of focal 60 gives
//! μ′ = (32, 24), Σ′ = 1.44 + 0.3 on the diagonal and radii 5 at σ_eff = 1; the J clamp
//! engaged beyond the margin against the unclamped value; det Σ′ ≤ 0, z = 0.005 and a box
//! wholly off-image give `None`. 10⁻⁴ px, 10⁻⁶ relative. The camera clauses are in
//! `u11_camera_from_stage_pose.rs`.
//!
//! `Projected` exposes Σ′ through its conic Σ′⁻¹ = (a, b, c), the upper triangle row by
//! row (Q-21); the expected values are worked by hand from E6 below.

use approx::assert_relative_eq;
use ftgsxr_core::{
    Camera, DMat3, DMat4, DVec3, Fov, Mat3, Quat, RenderConventions, Vec3, covariance, project,
};

/// The row's tolerances: 10⁻⁴ px absolute, 10⁻⁶ relative.
const PX: f32 = 1e-4;
const REL: f32 = 1e-6;

/// The row's camera: 64 × 48, f_x = f_y = 60, (c_x, c_y) = (32, 24), at the origin looking
/// along +z in OpenCV axes (world-to-camera the identity).
fn camera() -> Camera {
    Camera {
        width: 64,
        height: 48,
        k: DMat3::from_cols(
            DVec3::new(60.0, 0.0, 0.0),
            DVec3::new(0.0, 60.0, 0.0),
            DVec3::new(32.0, 24.0, 1.0),
        ),
        world_to_camera: DMat4::IDENTITY,
    }
}

#[test]
fn u11_primitive_at_depth_4_projects_to_the_center_with_radii_5() {
    // Σ = 0.08² I; J = diag(f/z, f/z) = diag(15, 15) on the axis, so Σ′ = 225 · 0.0064 + 0.3
    // = 1.44 + 0.3 = 1.74 on the diagonal and 0 off it. e = min(3.33, √(2 ln 255)) = 3.3290…
    // and r = ⌈3.3290 · √1.74⌉ = ⌈4.391⌉ = 5.
    let cov = covariance(Quat::IDENTITY, Vec3::splat(0.08));
    let conv = RenderConventions::default();
    let p = project(Vec3::new(0.0, 0.0, 4.0), cov, 1.0, &camera(), &conv).expect("visible");
    assert_relative_eq!(p.mean.x, 32.0, epsilon = PX);
    assert_relative_eq!(p.mean.y, 24.0, epsilon = PX);
    assert_relative_eq!(p.conic[0], 1.0 / 1.74, max_relative = REL);
    assert_relative_eq!(p.conic[1], 0.0, epsilon = REL);
    assert_relative_eq!(p.conic[2], 1.0 / 1.74, max_relative = REL);
    assert_eq!(p.radii, [5, 5]);
    assert_relative_eq!(p.depth, 4.0, max_relative = REL);
}

#[test]
fn u11_jacobian_clamp_engages_beyond_the_margin_on_x() {
    // Σ = I at (4, 0, 4): x_c/z_c = 1 lies beyond the clamp's upper bound
    // (W − c_x)/f_x + m W/f_x = 32/60 + 0.15·64/60 = 41.6/60, so J's first row is
    // (15, 0, −15 · 41.6/60) = (15, 0, −10.4) and Σ′₀₀ = 225 + 10.4² + 0.3 = 333.46, against
    // 225 + 15² + 0.3 = 450.3 unclamped. μ′ keeps the unclamped ratio: (60 + 32, 24).
    // r_x = ⌈3.3290 · √333.46⌉ = ⌈60.79⌉ = 61, r_y = ⌈3.3290 · √225.3⌉ = ⌈49.97⌉ = 50; the
    // box 92 ± 61 reaches into the image.
    let conv = RenderConventions::default();
    let p = project(
        Vec3::new(4.0, 0.0, 4.0),
        Mat3::IDENTITY,
        1.0,
        &camera(),
        &conv,
    )
    .expect("visible");
    assert_relative_eq!(p.mean.x, 92.0, epsilon = PX);
    assert_relative_eq!(p.mean.y, 24.0, epsilon = PX);
    assert_relative_eq!(p.conic[0], 1.0 / 333.46, max_relative = REL);
    assert_relative_eq!(p.conic[1], 0.0, epsilon = REL);
    assert_relative_eq!(p.conic[2], 1.0 / 225.3, max_relative = REL);
    assert!(
        (p.conic[0] - 1.0 / 450.3).abs() > 1e-4,
        "the clamp changed Σ′₀₀"
    );
    assert_eq!(p.radii, [61, 50]);
}

#[test]
fn u11_jacobian_clamp_engages_beyond_the_margin_on_negative_y() {
    // Σ = I at (0, −4, 4): y_c/z_c = −1 lies below −(c_y/f_y + m H/f_y) = −(0.4 + 0.12) =
    // −0.52, so J's second row is (0, 15, 15 · 0.52) = (0, 15, 7.8) and Σ′₁₁ = 225 + 7.8² +
    // 0.3 = 286.14, against 450.3 unclamped. μ′ = (32, −60 + 24) = (32, −36);
    // r_y = ⌈3.3290 · √286.14⌉ = ⌈56.31⌉ = 57, so the box reaches y = 21.
    let conv = RenderConventions::default();
    let p = project(
        Vec3::new(0.0, -4.0, 4.0),
        Mat3::IDENTITY,
        1.0,
        &camera(),
        &conv,
    )
    .expect("visible");
    assert_relative_eq!(p.mean.x, 32.0, epsilon = PX);
    assert_relative_eq!(p.mean.y, -36.0, epsilon = PX);
    assert_relative_eq!(p.conic[0], 1.0 / 225.3, max_relative = REL);
    assert_relative_eq!(p.conic[1], 0.0, epsilon = REL);
    assert_relative_eq!(p.conic[2], 1.0 / 286.14, max_relative = REL);
    assert!(
        (p.conic[2] - 1.0 / 450.3).abs() > 1e-4,
        "the clamp changed Σ′₁₁"
    );
    assert_eq!(p.radii, [50, 57]);
}

#[test]
fn u11_off_diagonal_covariance_gives_the_inverse_as_the_conic() {
    // On the axis at depth 4, J = [[15, 0, 0], [0, 15, 0]]. With Σ's upper-left block
    // [[0.01, 0.004], [0.004, 0.02]], Σ′ = [[2.55, 0.9], [0.9, 4.8]] and det = 11.43, so
    // Σ′⁻¹ = (4.8, −0.9, 2.55)/11.43 (Q-21's order).
    let cov = Mat3::from_cols(
        Vec3::new(0.01, 0.004, 0.0),
        Vec3::new(0.004, 0.02, 0.0),
        Vec3::new(0.0, 0.0, 0.03),
    );
    let conv = RenderConventions::default();
    let p = project(Vec3::new(0.0, 0.0, 4.0), cov, 1.0, &camera(), &conv).expect("visible");
    let det = 2.55 * 4.8 - 0.9 * 0.9;
    assert_relative_eq!(p.conic[0], 4.8 / det, max_relative = REL);
    assert_relative_eq!(p.conic[1], -0.9 / det, max_relative = REL);
    assert_relative_eq!(p.conic[2], 2.55 / det, max_relative = REL);
}

#[test]
fn u11_non_positive_det_is_culled() {
    // Σ = diag(1, −1, 1) on the axis at depth 4: Σ′ = diag(225.3, −224.7), det < 0.
    let cov = Mat3::from_diagonal(Vec3::new(1.0, -1.0, 1.0));
    let conv = RenderConventions::default();
    assert_eq!(
        project(Vec3::new(0.0, 0.0, 4.0), cov, 1.0, &camera(), &conv),
        None
    );
    // det Σ′ = 0 exactly: a zero covariance with ε = 0.
    let flat = RenderConventions {
        eps2d: 0.0,
        ..RenderConventions::default()
    };
    assert_eq!(
        project(Vec3::new(0.0, 0.0, 4.0), Mat3::ZERO, 1.0, &camera(), &flat),
        None
    );
}

#[test]
fn u11_depth_outside_near_and_far_is_culled() {
    let cov = covariance(Quat::IDENTITY, Vec3::splat(0.08));
    let conv = RenderConventions::default();
    // z = 0.005 < near = 0.01.
    assert_eq!(
        project(Vec3::new(0.0, 0.0, 0.005), cov, 1.0, &camera(), &conv),
        None
    );
    // Behind the camera.
    assert_eq!(
        project(Vec3::new(0.0, 0.0, -4.0), cov, 1.0, &camera(), &conv),
        None
    );
    // z = near is kept; z beyond a model's far is culled (the constants are the model's, D-39).
    assert!(project(Vec3::new(0.0, 0.0, 0.01), cov, 1.0, &camera(), &conv).is_some());
    let short = RenderConventions {
        far: 100.0,
        ..RenderConventions::default()
    };
    assert!(project(Vec3::new(0.0, 0.0, 100.0), cov, 1.0, &camera(), &short).is_some());
    assert_eq!(
        project(Vec3::new(0.0, 0.0, 101.0), cov, 1.0, &camera(), &short),
        None
    );
}

#[test]
fn u11_box_wholly_off_image_is_culled() {
    // Scale 0.08 at (8, 0, 4): μ′ = (152, 24), r = 5 near the axis and smaller further out
    // (the clamp keeps J's ratio), so the box lies right of W = 64.
    let cov = covariance(Quat::IDENTITY, Vec3::splat(0.08));
    let conv = RenderConventions::default();
    assert_eq!(
        project(Vec3::new(8.0, 0.0, 4.0), cov, 1.0, &camera(), &conv),
        None
    );
    assert_eq!(
        project(Vec3::new(-8.0, 0.0, 4.0), cov, 1.0, &camera(), &conv),
        None
    );
    assert_eq!(
        project(Vec3::new(0.0, 8.0, 4.0), cov, 1.0, &camera(), &conv),
        None
    );
    assert_eq!(
        project(Vec3::new(0.0, -8.0, 4.0), cov, 1.0, &camera(), &conv),
        None
    );
}

#[test]
fn u11_box_that_only_touches_the_image_is_off_it() {
    // Scale 0.2 at (3, 0, 4): x_c/z_c = 0.75, μ′_x = 45 + 32 = 77; J's ratio is clamped to
    // 41.6/60, so Σ′₀₀ = 0.04 · (225 + 10.4²) + 0.3 = 13.6264 and r_x = ⌈3.3290 · 3.6914⌉ =
    // ⌈12.29⌉ = 13: the box [64, 90] meets the image only on its edge x = W, which covers no
    // pixel center, and counts as off it as in openftgs's reference backend (source log 27).
    let cov = covariance(Quat::IDENTITY, Vec3::splat(0.2));
    let conv = RenderConventions::default();
    assert_eq!(
        project(Vec3::new(3.0, 0.0, 4.0), cov, 1.0, &camera(), &conv),
        None
    );
    // A little further in, the box overlaps the image.
    assert!(project(Vec3::new(2.9, 0.0, 4.0), cov, 1.0, &camera(), &conv).is_some());
}

#[test]
fn u11_sigma_eff_at_or_below_alpha_skip_is_culled() {
    // e = √(2 ln(σ_eff/α_skip)) is 0 at σ_eff = α_skip, so both radii are 0 (not visible,
    // Q-21), and undefined below it (E3: never active).
    let cov = covariance(Quat::IDENTITY, Vec3::splat(0.08));
    let conv = RenderConventions::default();
    let at = Vec3::new(0.0, 0.0, 4.0);
    assert_eq!(project(at, cov, conv.alpha_skip, &camera(), &conv), None);
    assert_eq!(
        project(at, cov, 0.5 * conv.alpha_skip, &camera(), &conv),
        None
    );
    assert_eq!(project(at, cov, 0.0, &camera(), &conv), None);
    assert!(project(at, cov, 2.0 * conv.alpha_skip, &camera(), &conv).is_some());
}

#[test]
fn u11_non_finite_values_are_culled() {
    let cov = covariance(Quat::IDENTITY, Vec3::splat(0.08));
    let conv = RenderConventions::default();
    let at = Vec3::new(0.0, 0.0, 4.0);
    assert_eq!(
        project(Vec3::new(f32::NAN, 0.0, 4.0), cov, 1.0, &camera(), &conv),
        None
    );
    assert_eq!(
        project(Vec3::new(0.0, 0.0, f32::NAN), cov, 1.0, &camera(), &conv),
        None
    );
    assert_eq!(project(at, cov * f32::NAN, 1.0, &camera(), &conv), None);
    assert_eq!(
        project(at, cov * f32::INFINITY, 1.0, &camera(), &conv),
        None
    );
    assert_eq!(project(at, cov, f32::NAN, &camera(), &conv), None);
}

#[test]
fn u11_screen_extent_cap_and_foreign_rule() {
    // A cap of 1 standard deviation: r = ⌈√1.74⌉ = 2. A foreign rule renders as
    // `opacity_aware` (D-39): the same radii as openftgs's rule.
    let cov = covariance(Quat::IDENTITY, Vec3::splat(0.08));
    let at = Vec3::new(0.0, 0.0, 4.0);
    let capped = RenderConventions {
        screen_extent_max: 1.0,
        ..RenderConventions::default()
    };
    assert_eq!(
        project(at, cov, 1.0, &camera(), &capped).unwrap().radii,
        [2, 2]
    );
    let foreign = RenderConventions {
        screen_extent_rule: "fixed".to_owned(),
        rasterize_mode: "antialiased".to_owned(),
        ..RenderConventions::default()
    };
    let default = RenderConventions::default();
    assert_eq!(
        project(at, cov, 1.0, &camera(), &foreign),
        project(at, cov, 1.0, &camera(), &default)
    );
}

#[test]
fn u11_stage_space_camera_applies_the_world_to_camera_transform() {
    // The desktop camera at (0, 0, 4) looking along −z, with tan(φ/2) = 0.4 so that
    // f_y = 24/0.4 = 60 on the 64 × 48 image: R_w = diag(1, −1, −1), t_w = (0, 0, 4). The
    // stage origin is at x_c = (0, 0, 4); stage (1, 1, 0) is at x_c = (1, −1, 4), so
    // μ′ = (15 + 32, −15 + 24) = (47, 9). Σ^S = diag(0.08², 0.16², 0.04²) keeps its diagonal
    // under R_w, so Σ′ = diag(1.44 + 0.3, 5.76 + 0.3) at the center.
    let cam = Camera::from_stage_pose(
        Vec3::new(0.0, 0.0, 4.0),
        Quat::IDENTITY,
        Fov::Vertical {
            radians: 2.0 * 0.4f32.atan(),
        },
        64,
        48,
    );
    let cov = covariance(Quat::IDENTITY, Vec3::new(0.08, 0.16, 0.04));
    let conv = RenderConventions::default();
    let p = project(Vec3::ZERO, cov, 1.0, &cam, &conv).expect("visible");
    assert_relative_eq!(p.mean.x, 32.0, epsilon = PX);
    assert_relative_eq!(p.mean.y, 24.0, epsilon = PX);
    assert_relative_eq!(p.conic[0], 1.0 / 1.74, max_relative = REL);
    assert_relative_eq!(p.conic[2], 1.0 / 6.06, max_relative = REL);
    assert_relative_eq!(p.depth, 4.0, max_relative = REL);
    let q = project(Vec3::new(1.0, 1.0, 0.0), cov, 1.0, &cam, &conv).expect("visible");
    assert_relative_eq!(q.mean.x, 47.0, epsilon = PX);
    assert_relative_eq!(q.mean.y, 9.0, epsilon = PX);
    assert_relative_eq!(q.depth, 4.0, max_relative = REL);
}
