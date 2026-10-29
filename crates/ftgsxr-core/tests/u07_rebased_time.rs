// SPDX-License-Identifier: Apache-2.0

//! U-7's rebased clause (`cpu` lane): E2's argument as the GPU computes it, from the
//! float32 arrays `layout::pack_for_gpu` writes (time centers rebased to τ − τ₀, durations
//! float32, §1.2 load option 3) and τ − τ₀ rounded to float32 as the prepare stage writes
//! it (§1.4), against float64, for τ₀ = 10⁹ s and T_τ = 10 s. §1.2 bounds the absolute
//! error of τ − time_center by 10⁻⁶ T_τ, so the argument errs by less than
//! 10⁻⁶ T_τ/duration.

use ftgsxr_core::layout::{GpuLayoutOptions, pack_for_gpu};
use ftgsxr_core::{Model, ModelKind, RenderConventions, TimeSpan};
use proptest::prelude::*;

const TAU0: f64 = 1.0e9;
const SPAN: f64 = 10.0;

/// A model of one primitive per (time center, duration) at τ₀ = 10⁹ s, T_τ = 10 s.
fn model(rows: &[(f64, f64)]) -> Model {
    let n = rows.len();
    Model {
        position: vec![[0.0; 3]; n],
        time_center: rows.iter().map(|r| r.0).collect(),
        duration: rows.iter().map(|r| r.1).collect(),
        velocity: vec![[0.0; 3]; n],
        scale: vec![[0.01; 3]; n],
        rotation: vec![[1.0, 0.0, 0.0, 0.0]; n],
        opacity: vec![0.5; n],
        sh: vec![0.0; n * 3],
        sh_degree: 0,
        time: TimeSpan {
            tau0: TAU0,
            tau1: TAU0 + SPAN,
            time_unit: "s".to_owned(),
        },
        conventions: RenderConventions::default(),
        producer: String::new(),
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    }
}

/// U-7: for every row and query time, ((τ − τ₀) − (time_center − τ₀)) in float32 errs
/// from τ − time_center in float64 by less than 10⁻⁶ T_τ, and the float32 argument by
/// less than 10⁻⁶ T_τ/duration.
fn check(rows: &[(f64, f64)], taus: &[f64]) {
    let m = model(rows);
    m.validate().unwrap();
    let order: Vec<u32> = (0..rows.len() as u32).collect();
    let a = pack_for_gpu(&m, &order, &GpuLayoutOptions::default());
    for &tau in taus {
        let tau_rebased = (tau - TAU0) as f32;
        for (i, &(center, duration)) in rows.iter().enumerate() {
            let diff32 = tau_rebased - a.time_center[i];
            let arg32 = diff32 / a.duration[i];
            let diff64 = tau - center;
            let arg64 = diff64 / duration;
            let diff_err = (f64::from(diff32) - diff64).abs();
            let arg_err = (f64::from(arg32) - arg64).abs();
            assert!(
                diff_err < 1e-6 * SPAN,
                "row {i} at τ − τ₀ = {}: {diff_err}",
                tau - TAU0
            );
            assert!(
                arg_err < 1e-6 * SPAN / duration,
                "row {i} at τ − τ₀ = {}: argument error {arg_err}",
                tau - TAU0
            );
        }
    }
}

/// U-7: centers over the span (its ends included), durations from 0.05 s to 5 s, query
/// times over the span and 5 s beyond either end.
#[test]
fn u07_rebased_float32_argument_within_1e_6_span() {
    let mut rows = Vec::new();
    for c in 0..=40 {
        for d in [0.05, 0.3, 1.0, 5.0] {
            rows.push((
                TAU0 + SPAN * f64::from(c) / 40.0 + 1e-3 * f64::from(c % 7),
                d,
            ));
        }
    }
    let taus: Vec<f64> = (0..=200)
        .map(|k| TAU0 - 5.0 + 20.0 * f64::from(k) / 200.0 + 1.234_567e-4)
        .collect();
    check(&rows, &taus);
}

proptest! {
    /// U-7's property variant: the same bounds at random centers, durations and times
    /// within the row's domain.
    #[test]
    fn u07_rebased_float32_argument_property(
        center in 0.0f64..SPAN,
        duration in 0.01f64..10.0,
        tau in -5.0f64..SPAN + 5.0,
    ) {
        check(&[(TAU0 + center, duration)], &[TAU0 + tau]);
    }
}
