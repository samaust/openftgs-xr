// SPDX-License-Identifier: Apache-2.0

//! U-7's E1 and E2 clauses (`cpu` lane): `motion::moved_position` and
//! `motion::temporal_opacity` against hand-worked literals and the properties the row
//! states, with proptest variants. The rebased float32 clause is the layout ticket's.

use approx::assert_abs_diff_eq;
use ftgsxr_core::Vec3;
use ftgsxr_core::motion::{effective_opacity, moved_position, temporal_opacity};
use proptest::prelude::*;

/// U-7: μx(time_center) = position, exactly.
#[test]
fn u07_moved_position_at_time_center_is_position() {
    let position = Vec3::new(1.5, -2.25, 1.0e6);
    let velocity = Vec3::new(0.5, -3.0, 7.0);
    for time_center in [0.0, 2.5, -7.0, 1.0e9 + 0.125] {
        assert_eq!(
            moved_position(position, velocity, time_center, time_center),
            position
        );
    }
}

/// U-7: μx(τ₂) − μx(τ₁) = velocity (τ₂ − τ₁), exactly on values every step represents.
#[test]
fn u07_moved_position_moves_by_velocity_times_elapsed() {
    let position = Vec3::new(1.0, 2.0, 3.0);
    let velocity = Vec3::new(0.5, -0.25, 2.0);
    let time_center = 10.0;
    let a = moved_position(position, velocity, time_center, 12.0);
    let b = moved_position(position, velocity, time_center, 16.0);
    assert_eq!(b - a, Vec3::new(2.0, -1.0, 8.0));
    // Linear outside any span too (no clamping): 20 time units before the center.
    let c = moved_position(position, velocity, time_center, -10.0);
    assert_eq!(c, Vec3::new(-9.0, 7.0, -37.0));
}

/// e^−0.5 and e^−2, U-7's expected values, from their decimal expansions.
const E_M05: f64 = 0.606_530_659_712_633_4;
const E_M2: f64 = 0.135_335_283_236_612_7;

/// U-7: σt is 1 at the center, e^−0.5 at time_center ± duration and e^−2 at
/// time_center ± 2 duration, within 10⁻¹² in float64.
#[test]
fn u07_temporal_opacity_at_one_and_two_durations() {
    for (time_center, duration) in [(0.0, 1.0), (3.5, 0.25), (-12.0, 4.0), (1.0e9, 0.5)] {
        assert_eq!(
            temporal_opacity(time_center, time_center, duration, false),
            1.0
        );
        for sign in [-1.0, 1.0] {
            let one = temporal_opacity(time_center + sign * duration, time_center, duration, false);
            let two = temporal_opacity(
                time_center + sign * 2.0 * duration,
                time_center,
                duration,
                false,
            );
            assert_abs_diff_eq!(one, E_M05, epsilon = 1e-12);
            assert_abs_diff_eq!(two, E_M2, epsilon = 1e-12);
        }
    }
}

/// U-7, D-9: with a single timestamp σt ≡ 1 at any τ, near the center or far from it.
#[test]
fn u07_temporal_opacity_is_one_for_a_single_timestamp() {
    let time_center = 3.0;
    for tau in [3.0, 3.5, 4.0, -100.0, 1.0e9, f64::MAX] {
        assert_eq!(temporal_opacity(tau, time_center, 1.0, true), 1.0);
    }
}

/// E2: σ_eff = σt · opacity, on products every step represents exactly; the peak is the
/// opacity itself, and the effective opacity two durations out is e^−2 of it.
#[test]
fn u07_effective_opacity_is_the_product() {
    assert_eq!(effective_opacity(0.5, 0.5), 0.25);
    assert_eq!(effective_opacity(0.75, 1.0), 0.75);
    assert_eq!(effective_opacity(0.8, 0.0), 0.0);
    let sigma_t = temporal_opacity(6.0, 2.0, 2.0, false);
    assert_abs_diff_eq!(
        effective_opacity(0.5, sigma_t),
        (0.5 * E_M2) as f32,
        epsilon = 1e-7
    );
}

/// A primitive's position or velocity, each component in [−m, m].
fn vec3_within(m: f32) -> impl Strategy<Value = Vec3> {
    (-m..=m, -m..=m, -m..=m).prop_map(|(x, y, z)| Vec3::new(x, y, z))
}

proptest! {
    /// U-7's E1 clauses on random primitives (positions within 10³, velocities within 10,
    /// times within 10³): μx(time_center) = position exactly; μx(τ₂) − μx(τ₁) =
    /// velocity (τ₂ − τ₁) up to the float32 rounding of the two positions, half an ulp
    /// each (E1 returns float32).
    #[test]
    fn u07_proptest_moved_position(
        position in vec3_within(1.0e3),
        velocity in vec3_within(10.0),
        time_center in -1.0e3..1.0e3f64,
        tau1 in -1.0e3..1.0e3f64,
        tau2 in -1.0e3..1.0e3f64,
    ) {
        prop_assert_eq!(moved_position(position, velocity, time_center, time_center), position);
        let a = moved_position(position, velocity, time_center, tau1);
        let b = moved_position(position, velocity, time_center, tau2);
        let half_ulp = f64::from(f32::EPSILON) / 2.0;
        for k in 0..3 {
            let moved = f64::from(b[k]) - f64::from(a[k]);
            let expected = f64::from(velocity[k]) * (tau2 - tau1);
            let tol = half_ulp * (f64::from(a[k].abs()) + f64::from(b[k].abs())) + 1e-12;
            prop_assert!(
                (moved - expected).abs() <= tol,
                "axis {}: moved {} expected {} tolerance {}", k, moved, expected, tol
            );
        }
    }

    /// U-7's E2 clauses on random primitives (centers within 10³, durations in
    /// [0.5, 10²], so τ = time_center ± k·duration rounds by far less than 10⁻¹² of a
    /// duration): σt at ± duration and ± 2 duration is e^−0.5 and e^−2 within 10⁻¹².
    #[test]
    fn u07_proptest_temporal_opacity(
        time_center in -1.0e3..1.0e3f64,
        duration in 0.5..1.0e2f64,
    ) {
        for sign in [-1.0, 1.0] {
            let one = temporal_opacity(time_center + sign * duration, time_center, duration, false);
            let two = temporal_opacity(time_center + sign * 2.0 * duration, time_center, duration, false);
            prop_assert!((one - E_M05).abs() <= 1e-12, "σt(±d) = {}", one);
            prop_assert!((two - E_M2).abs() <= 1e-12, "σt(±2d) = {}", two);
        }
    }

    /// U-7, D-9: σt ≡ 1 for a single timestamp at any finite τ and duration.
    #[test]
    fn u07_proptest_single_timestamp(
        tau in proptest::num::f64::NORMAL | proptest::num::f64::ZERO,
        time_center in -1.0e9..1.0e9f64,
        duration in 1.0e-6..1.0e6f64,
    ) {
        prop_assert_eq!(temporal_opacity(tau, time_center, duration, true), 1.0);
    }
}
