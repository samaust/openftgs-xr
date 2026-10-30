// SPDX-License-Identifier: Apache-2.0

//! R-19, D-7's bound (§3.2, E7, Q-10): a near primitive of α = 0.95 covering a far one of
//! α = α_max = 0.999, so openftgs's transmittance stop excludes the far one where the near
//! one leaves T = 0.05. At every pixel the player's render differs from the reference by at
//! most T·c per channel and T in alpha, T the reference's remaining transmittance there
//! (1 − its alpha) and c the far primitive's color, the one the stop leaves out, plus
//! §1.12's half-float rounding allowance: one binary16 step of the larger of the two values
//! compared (Q-88). The tier is not required.

mod probe_support;

use ftgsxr_core::{Vec3, sh_color};
use ftgsxr_testkit::{layer_messages, probe};
use probe_support::render_probe;

/// One binary16 step (unit in the last place) at the magnitude of `v`: 2^(⌊log₂|v|⌋ − 10)
/// for normal halves, 2⁻²⁴ below 2⁻¹⁴; about 5 × 10⁻⁴ to 10⁻³ of the value (§1.12, D-35,
/// the record's half floats of §1.4 stage 2). // Q-88
fn half_step(v: f32) -> f32 {
    let v = v.abs();
    if v < 2f32.powi(-14) {
        return 2f32.powi(-24);
    }
    2f32.powi(v.log2().floor() as i32 - 10)
}

#[test]
#[ignore = "gpu"]
fn gpu_r19_difference_within_the_remaining_transmittance() {
    let _layer = layer_messages(); // R-16, Q-105
    let (model, _) = probe("R19_transmittance");
    assert_eq!(model.len(), 2, "the near and the far primitive");
    // The far primitive is the near-opaque one (opacity 1 − 10⁻⁷); degree 0, so its color
    // C₀c₀₀ + 0.5 holds in every direction (E4).
    let far = (0..2)
        .max_by(|&a, &b| model.opacity[a].total_cmp(&model.opacity[b]))
        .unwrap();
    assert_eq!(model.sh_degree, 0);
    let c = sh_color(0, &model.sh[3 * far..3 * far + 3], Vec3::Z);
    let mut worst = f32::NEG_INFINITY;
    let mut failures = Vec::new();
    for (id, ours, reference) in render_probe("R19_transmittance") {
        for p in 0..reference.alpha.len() {
            let t = 1.0 - reference.alpha[p];
            let (a, r) = (ours.alpha[p], reference.alpha[p]);
            // Q-88: D-7's bound plus one half-float step of the values compared.
            let bound = t + half_step(a.abs().max(r.abs()));
            let da = (a - r).abs();
            let mut ok = da <= bound;
            worst = worst.max(da - bound);
            for (ch, &c_ch) in c.iter().enumerate() {
                let (o, r) = (ours.rgb[3 * p + ch], reference.rgb[3 * p + ch]);
                let bound = t * c_ch + half_step(o.abs().max(r.abs())); // Q-88
                let d = (o - r).abs();
                worst = worst.max(d - bound);
                ok &= d <= bound;
            }
            if !ok {
                failures.push(format!(
                    "{id} pixel {p}: T {t:e}, alpha {} vs {}, rgb {:?} vs {:?}",
                    ours.alpha[p],
                    reference.alpha[p],
                    &ours.rgb[3 * p..3 * p + 3],
                    &reference.rgb[3 * p..3 * p + 3]
                ));
            }
        }
    }
    eprintln!("R19_transmittance: largest difference minus its bound {worst:e} (≤ 0 passes)");
    assert!(
        failures.is_empty(),
        "{} pixels exceed D-7's bound:\n{}",
        failures.len(),
        failures
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
