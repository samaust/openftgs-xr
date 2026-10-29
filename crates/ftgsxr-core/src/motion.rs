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

/// E3's half-width w = duration·√(2 ln(opacity/α_skip)): a primitive is active at τ
/// only when |τ − time_center| ≤ w. `None` when opacity ≤ `alpha_skip` (255·opacity ≤ 1
/// for openftgs's α_skip), since such a primitive is never active. Computed in float64,
/// with `alpha_skip` promoted exactly from the model's float32 (D-39).
///
/// `widened` applies the temporal index's margins, as openftgs widens its buckets
/// (D-10): the opacity multiplied by 1 + 10⁻⁴ inside the logarithm and 10⁻⁶·T′ added to
/// w, where T′ is `span` (T_τ = τ₁ − τ₀), or 1 when `span` is not positive (a single
/// timestamp).
// Q-T7.1: None exactly when opacity ≤ alpha_skip; alpha_skip promoted from f32; T′ from span.
pub fn temporal_window(
    opacity: f32,
    duration: f64,
    alpha_skip: f32,
    span: f64,
    widened: bool,
) -> Option<f64> {
    if opacity <= alpha_skip {
        return None;
    }
    let ratio = f64::from(opacity) / f64::from(alpha_skip);
    if !widened {
        return Some(duration * (2.0 * ratio.ln()).sqrt());
    }
    let t_prime = if span > 0.0 { span } else { 1.0 };
    Some(duration * (2.0 * (ratio * OPACITY_MARGIN).ln()).sqrt() + TIME_MARGIN * t_prime)
}

/// The index's opacity margin, the factor inside E3's logarithm (D-10, openftgs D-55).
const OPACITY_MARGIN: f64 = 1.0 + 1e-4;

/// The index's time margin in units of T′, added to E3's half-width (D-10, openftgs D-55).
const TIME_MARGIN: f64 = 1e-6;
