// SPDX-License-Identifier: Apache-2.0

//! R-19, D-7's bound (§3.2, E7, Q-10): a near primitive of α = 0.95 covering a far one of
//! α = α_max = 0.999, so openftgs's transmittance stop excludes the far one where the near
//! one leaves T = 0.05. At every pixel the player's render differs from the reference by at
//! most T·c per channel and T in alpha, T the reference's remaining transmittance there
//! (1 − its alpha) and c the far primitive's color, the one the stop leaves out. The tier is
//! not required.

mod probe_support;

use ftgsxr_core::{Vec3, sh_color};
use ftgsxr_testkit::probe;
use probe_support::render_probe;

#[test]
#[ignore = "gpu"]
fn gpu_r19_difference_within_the_remaining_transmittance() {
    let (model, _) = probe("R19_transmittance");
    assert_eq!(model.len(), 2, "the near and the far primitive");
    // The far primitive is the near-opaque one (opacity 1 − 10⁻⁷); degree 0, so its color
    // C₀c₀₀ + 0.5 holds in every direction (E4).
    let far = (0..2)
        .max_by(|&a, &b| model.opacity[a].total_cmp(&model.opacity[b]))
        .unwrap();
    assert_eq!(model.sh_degree, 0);
    let c = sh_color(0, &model.sh[3 * far..3 * far + 3], Vec3::Z);
    let mut worst = 0.0f32;
    let mut failures = Vec::new();
    for (id, ours, reference) in render_probe("R19_transmittance") {
        for p in 0..reference.alpha.len() {
            let t = 1.0 - reference.alpha[p];
            let da = (ours.alpha[p] - reference.alpha[p]).abs();
            let mut ok = da <= t;
            worst = worst.max(da - t);
            for (ch, &c_ch) in c.iter().enumerate() {
                let d = (ours.rgb[3 * p + ch] - reference.rgb[3 * p + ch]).abs();
                worst = worst.max(d - t * c_ch);
                ok &= d <= t * c_ch;
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
    eprintln!("R19_transmittance: largest excess over the bound {worst:e}");
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
