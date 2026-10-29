// SPDX-License-Identifier: Apache-2.0

//! U-12 (E7's CPU mirror `covariance::alpha_at`, `cpu` lane): α at μ′ equals
//! min(α_max, σ_eff); α at the quadratic form 2 ln 255 is 1/255 and beyond it 0; a negative
//! form gives 0. 10⁻⁶.
//!
//! The row's last clause, "the occlusion weight multiplies", is not tested: §2.1's
//! `alpha_at` takes no weight (w_occ = 1), and the spec question on PRD #1 ("U-12 and
//! `alpha_at`") is unanswered.
//!
//! The quadratic form is (p − μ′)ᵀ Σ′⁻¹ (p − μ′) with the conic (a, b, c) of Q-21, and
//! p is the pixel center the caller passes.

use approx::assert_relative_eq;
use ftgsxr_core::{Projected, RenderConventions, Vec2, alpha_at};

/// The row's tolerance.
const TOL: f32 = 1e-6;

/// A projected primitive at `mean` with the conic (a, b, c).
fn projected(mean: Vec2, conic: [f32; 3]) -> Projected {
    Projected {
        mean,
        conic,
        radii: [8, 8],
        depth: 4.0,
    }
}

#[test]
fn u12_alpha_at_the_mean_is_sigma_eff_capped_at_alpha_max() {
    let conv = RenderConventions::default();
    let p = projected(Vec2::new(10.25, 7.5), [1.0, 0.0, 1.0]);
    assert_relative_eq!(alpha_at(p.mean, &p, 1.0, &conv), 0.999, epsilon = TOL);
    assert_relative_eq!(alpha_at(p.mean, &p, 0.5, &conv), 0.5, epsilon = TOL);
    assert_relative_eq!(alpha_at(p.mean, &p, 0.9995, &conv), 0.999, epsilon = TOL);
    // The model's α_max (D-39).
    let low = RenderConventions {
        alpha_max: 0.9,
        ..RenderConventions::default()
    };
    assert_relative_eq!(alpha_at(p.mean, &p, 0.95, &low), 0.9, epsilon = TOL);
    // σ_eff below α_skip is skipped even at the mean.
    assert_eq!(alpha_at(p.mean, &p, 0.5 / 255.0, &conv), 0.0);
}

#[test]
fn u12_alpha_follows_the_quadratic_form_of_the_conic() {
    // Conic (2, 0.5, 1), p − μ′ = (0.3, −0.4): form = 2·0.09 + 2·0.5·0.3·(−0.4) + 0.16
    // = 0.22, so α = 0.8 e^(−0.11).
    let conv = RenderConventions::default();
    let p = projected(Vec2::new(4.0, 6.0), [2.0, 0.5, 1.0]);
    let alpha = alpha_at(Vec2::new(4.3, 5.6), &p, 0.8, &conv);
    assert_relative_eq!(alpha, 0.8 * (-0.11f32).exp(), epsilon = TOL);
}

#[test]
fn u12_alpha_at_the_form_2_ln_255_is_one_over_255_and_beyond_it_0() {
    // With σ_eff = 1 and the identity conic, a pixel at distance d has form d², and
    // α = e^(−d²/2) reaches 1/255 at d² = 2 ln 255, exactly on the skip threshold. The form
    // is taken 10⁻⁴ inside it (α = (1/255)·e^(5·10⁻⁵), within 10⁻⁶ of 1/255), clear of the
    // float32 rounding of p, and 10⁻³ and 1 beyond it.
    let conv = RenderConventions::default();
    let p = projected(Vec2::ZERO, [1.0, 0.0, 1.0]);
    let bound = 2.0 * 255f64.ln();
    let at = |form: f64| Vec2::new(form.sqrt() as f32, 0.0);
    let inside = alpha_at(at(bound - 1e-4), &p, 1.0, &conv);
    assert_relative_eq!(inside, 1.0 / 255.0, epsilon = TOL);
    assert!(inside > 0.0);
    assert_eq!(alpha_at(at(bound + 1e-3), &p, 1.0, &conv), 0.0);
    assert_eq!(alpha_at(at(bound + 1.0), &p, 1.0, &conv), 0.0);
    // The same bound along y and along a diagonal.
    let r = (bound - 1e-4).sqrt() as f32;
    let diag = r / 2f32.sqrt();
    assert_relative_eq!(
        alpha_at(Vec2::new(0.0, -r), &p, 1.0, &conv),
        1.0 / 255.0,
        epsilon = TOL
    );
    assert_relative_eq!(
        alpha_at(Vec2::new(diag, diag), &p, 1.0, &conv),
        1.0 / 255.0,
        epsilon = TOL
    );
}

#[test]
fn u12_negative_quadratic_form_gives_0() {
    // A conic with a negative form away from the mean (a numerical guard): skipped, although
    // e^(−form/2) exceeds 1.
    let conv = RenderConventions::default();
    let p = projected(Vec2::new(5.0, 5.0), [-1.0, 0.0, -1.0]);
    assert_eq!(alpha_at(Vec2::new(5.5, 5.0), &p, 0.5, &conv), 0.0);
    // An indefinite conic: negative along x, positive along y.
    let q = projected(Vec2::new(5.0, 5.0), [-1.0, 0.0, 1.0]);
    assert_eq!(alpha_at(Vec2::new(6.0, 5.0), &q, 0.5, &conv), 0.0);
    assert_relative_eq!(
        alpha_at(Vec2::new(5.0, 6.0), &q, 0.5, &conv),
        0.5 * (-0.5f32).exp(),
        epsilon = TOL
    );
}
