// SPDX-License-Identifier: Apache-2.0

//! A primitive's motion and its temporal opacity in raw time (§2.1 `ftgsxr_core::motion`):
//! E1's moved position, E2's temporal and effective opacities with the single-timestamp
//! rule (D-9), and E3's half-width, the temporal window the index widens (D-10).

use glam::Vec3;

use crate::types::model::moved_position_f64;

/// E1's moved position μx(τ) = position + velocity (τ − time_center), linear at every τ
/// with no clamping, computed in float64 and rounded to float32.
pub fn moved_position(position: Vec3, velocity: Vec3, time_center: f64, tau: f64) -> Vec3 {
    moved_position_f64(position.to_array(), velocity.to_array(), time_center, tau)
}

/// E2's temporal opacity σt(τ) = exp(−½ ((τ − time_center)/duration)²), unnormalized
/// (1 at the center), in float64. With a single timestamp (`single`, τ₀ = τ₁) it is 1 at
/// every τ and the formula is not applied, as openftgs evaluates such a model at t ≡ 0
/// (D-9).
pub fn temporal_opacity(tau: f64, time_center: f64, duration: f64, single: bool) -> f64 {
    if single {
        return 1.0;
    }
    let x = (tau - time_center) / duration;
    (-0.5 * x * x).exp()
}

/// E2's effective opacity σ_eff = σt · opacity, the opacity the renderer composites,
/// multiplied in float64 and rounded to float32.
pub fn effective_opacity(opacity: f32, sigma_t: f64) -> f32 {
    (f64::from(opacity) * sigma_t) as f32
}
