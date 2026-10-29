// SPDX-License-Identifier: Apache-2.0

//! U-14, the `ModelTime` part (E9, §1.6, D-32, D-9, `cpu` lane): τ_m for offsets and rates
//! (`s`, `ms`, an unknown unit → 1 with a warning); hold clamps, extrapolate flags; a single
//! timestamp gives τ₀; the frame step from a median spacing, from a checkpoint's Δt_f and the
//! 1/30 s default, with the [1/480 s, T_τ] bounds. Tolerance 10⁻¹², exact where the row says
//! so. The choices §1.6 leaves open follow Q-20.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use approx::assert_abs_diff_eq;
use ftgsxr_core::{EndRule, Model, ModelKind, ModelTime, RenderConventions, TimeSpan};

const TOL: f64 = 1e-12;

fn span(tau0: f64, tau1: f64, unit: &str) -> TimeSpan {
    TimeSpan {
        tau0,
        tau1,
        time_unit: unit.to_owned(),
    }
}

/// A model whose rows have the given time centers; only the centers, the span and the kind
/// matter to the frame step.
fn model(time: TimeSpan, centers: &[f64], kind: ModelKind) -> Model {
    let n = centers.len();
    Model {
        position: vec![[0.0; 3]; n],
        time_center: centers.to_vec(),
        duration: vec![1.0; n],
        velocity: vec![[0.0; 3]; n],
        scale: vec![[0.1; 3]; n],
        rotation: vec![[1.0, 0.0, 0.0, 0.0]; n],
        opacity: vec![0.5; n],
        sh: vec![0.0; n * 3],
        sh_degree: 0,
        time,
        conventions: RenderConventions::default(),
        producer: String::new(),
        identity: [0; 32],
        kind,
    }
}

/// Counts the `WARN` events emitted while it is the thread's default subscriber.
struct WarnCounter(Arc<AtomicUsize>);

impl tracing::Subscriber for WarnCounter {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        if *event.metadata().level() == tracing::Level::WARN {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// `ModelTime::from_span` and the number of warnings it emitted.
fn from_span_warnings(time: &TimeSpan) -> (ModelTime, usize) {
    let count = Arc::new(AtomicUsize::new(0));
    let mt = tracing::subscriber::with_default(WarnCounter(count.clone()), || {
        ModelTime::from_span(time)
    });
    (mt, count.load(Ordering::SeqCst))
}

#[test]
fn u14_rate_from_the_unit() {
    let (s, warned) = from_span_warnings(&span(0.0, 2.0, "s"));
    assert_eq!((s.rate, warned), (1.0, 0));
    let (ms, warned) = from_span_warnings(&span(0.0, 2000.0, "ms"));
    assert_eq!((ms.rate, warned), (1000.0, 0));
    assert_eq!(ms.offset_s, 0.0);
    assert_eq!(ms.end_rule, EndRule::Hold);
}

#[test]
fn u14_unknown_unit_is_rate_one_with_a_warning() {
    let time = span(0.0, 48.0, "frames");
    // The testable signal of PRD #1's decision: the span refuses the unit...
    assert_eq!(
        time.rate_per_second().expect_err("unknown unit").code(),
        "E-FILE-VALUE"
    );
    // ...and the mapping takes 1 and warns.
    let (mt, warned) = from_span_warnings(&time);
    assert_eq!(mt.rate, 1.0);
    assert_eq!(warned, 1);
    assert_eq!(mt.tau_at(3.0), (3.0, false));
}

#[test]
fn u14_tau_for_offsets_and_rates() {
    // τ_m = τ₀ + r (T − o): seconds with an offset of 1 s.
    let mut s = ModelTime::from_span(&span(10.0, 14.0, "s"));
    s.offset_s = 1.0;
    let (tau, flag) = s.tau_at(3.0);
    assert_abs_diff_eq!(tau, 12.0, epsilon = TOL);
    assert!(!flag);
    // Milliseconds with an offset of 0.5 s: 1000 (1.5 − 0.5) = 1000.
    let mut ms = ModelTime::from_span(&span(0.0, 2000.0, "ms"));
    ms.offset_s = 0.5;
    let (tau, flag) = ms.tau_at(1.5);
    assert_abs_diff_eq!(tau, 1000.0, epsilon = TOL);
    assert!(!flag);
    // A negative offset starts the instance before the clock's zero.
    ms.offset_s = -0.25;
    assert_abs_diff_eq!(ms.tau_at(0.0).0, 250.0, epsilon = TOL);
}

#[test]
fn u14_hold_clamps() {
    let mut mt = ModelTime::from_span(&span(10.0, 14.0, "s"));
    mt.offset_s = 1.0;
    assert_eq!(mt.tau_at(0.0), (10.0, false)); // 9 → τ₀
    assert_eq!(mt.tau_at(10.0), (14.0, false)); // 19 → τ₁
}

#[test]
fn u14_extrapolate_flags() {
    let mut mt = ModelTime::from_span(&span(10.0, 14.0, "s"));
    mt.offset_s = 1.0;
    mt.end_rule = EndRule::Extrapolate;
    assert_eq!(mt.tau_at(0.0), (9.0, true));
    assert_eq!(mt.tau_at(10.0), (19.0, true));
    // Q-20: inside the span (its ends included) nothing is flagged.
    assert_eq!(mt.tau_at(3.0), (12.0, false));
    assert_eq!(mt.tau_at(1.0), (10.0, false));
    assert_eq!(mt.tau_at(5.0), (14.0, false));
}

#[test]
fn u14_single_timestamp_gives_tau0() {
    // D-9: a single-timestamp model ignores τ, under either end rule (Q-20).
    let mut mt = ModelTime::from_span(&span(5.0, 5.0, "s"));
    mt.offset_s = 2.0;
    for rule in [EndRule::Hold, EndRule::Extrapolate] {
        mt.end_rule = rule;
        for t in [0.0, 2.0, 7.5, 100.0] {
            assert_eq!(mt.tau_at(t), (5.0, false));
        }
    }
}

#[test]
fn u14_length() {
    // o_m + T_τ / r_m.
    let mut s = ModelTime::from_span(&span(10.0, 14.0, "s"));
    s.offset_s = 1.0;
    assert_abs_diff_eq!(s.length_s(), 5.0, epsilon = TOL);
    let mut ms = ModelTime::from_span(&span(0.0, 2000.0, "ms"));
    ms.offset_s = 0.5;
    assert_abs_diff_eq!(ms.length_s(), 2.5, epsilon = TOL);
}

#[test]
fn u14_frame_step_from_the_median_spacing() {
    // Distinct centers 0, 0.1, 0.3, 0.6, 1.0 (duplicates ignored): gaps 0.1, 0.2, 0.3, 0.4;
    // Q-20: the even count averages the middle two, 0.25 s.
    let time = span(0.0, 1.0, "s");
    let centers = [0.3, 0.0, 0.6, 0.0, 1.0, 0.1, 0.3];
    let m = model(time.clone(), &centers, ModelKind::ExchangeFile);
    let step = ModelTime::from_span(&time).frame_step_s(&m);
    assert_abs_diff_eq!(step, 0.25, epsilon = TOL);
    // The same centers in milliseconds: converted to seconds by ÷ r.
    let time = span(0.0, 1000.0, "ms");
    let centers: Vec<f64> = centers.iter().map(|c| c * 1000.0).collect();
    let m = model(time.clone(), &centers, ModelKind::ExchangeFile);
    let step = ModelTime::from_span(&time).frame_step_s(&m);
    assert_abs_diff_eq!(step, 0.25, epsilon = TOL);
    // An odd count takes the middle gap: 0, 0.1, 0.3, 0.6 → 0.1, 0.2, 0.3 → 0.2.
    let time = span(0.0, 1.0, "s");
    let m = model(time.clone(), &[0.0, 0.1, 0.3, 0.6], ModelKind::Ply);
    assert_abs_diff_eq!(
        ModelTime::from_span(&time).frame_step_s(&m),
        0.2,
        epsilon = TOL
    );
}

#[test]
fn u14_frame_step_median_bounds_are_inclusive() {
    let time = span(0.0, 1.0, "s");
    let mt = ModelTime::from_span(&time);
    // Exactly 1/480 s and exactly T_τ are within the bounds.
    let low = model(time.clone(), &[0.0, 1.0 / 480.0], ModelKind::ExchangeFile);
    assert_eq!(mt.frame_step_s(&low), 1.0 / 480.0);
    let high = model(time.clone(), &[0.0, 1.0], ModelKind::ExchangeFile);
    assert_eq!(mt.frame_step_s(&high), 1.0);
}

#[test]
fn u14_frame_step_outside_the_bounds_falls_back() {
    let time = span(0.0, 2.0, "s");
    let mt = ModelTime::from_span(&time);
    // A median of 1 ms (below 1/480 s): the default, or the checkpoint's Δt_f·T_τ/r.
    let fine = [0.0, 0.001, 0.002, 0.003, 2.0];
    assert_eq!(
        mt.frame_step_s(&model(time.clone(), &fine, ModelKind::ExchangeFile)),
        1.0 / 30.0
    );
    let ckpt = ModelKind::Checkpoint {
        spatial_scale: 1.0,
        frame_dt: 0.1,
    };
    assert_abs_diff_eq!(
        mt.frame_step_s(&model(time.clone(), &fine, ckpt)),
        0.2,
        epsilon = TOL
    );
    // A median of 10 s, above T_τ/r = 2 s.
    let coarse = [-4.0, 6.0];
    assert_eq!(
        mt.frame_step_s(&model(time.clone(), &coarse, ModelKind::Ply)),
        1.0 / 30.0
    );
    // Fewer than two distinct centers.
    let one = [0.5, 0.5, 0.5];
    assert_eq!(
        mt.frame_step_s(&model(time.clone(), &one, ModelKind::ExchangeFile)),
        1.0 / 30.0
    );
    assert_abs_diff_eq!(
        mt.frame_step_s(&model(time, &one, ckpt)),
        0.2,
        epsilon = TOL
    );
}

#[test]
fn u14_frame_step_from_a_checkpoint_in_milliseconds() {
    // Δt_f·T_τ/r: 0.05 · 2000 ms / 1000 = 0.1 s.
    let time = span(0.0, 2000.0, "ms");
    let ckpt = ModelKind::Checkpoint {
        spatial_scale: 1.0,
        frame_dt: 0.05,
    };
    let m = model(time.clone(), &[700.0, 700.0], ckpt);
    assert_abs_diff_eq!(
        ModelTime::from_span(&time).frame_step_s(&m),
        0.1,
        epsilon = TOL
    );
}

#[test]
fn u14_frame_step_default() {
    // A single timestamp: no spacing, the default.
    let time = span(3.0, 3.0, "s");
    let mt = ModelTime::from_span(&time);
    assert_eq!(
        mt.frame_step_s(&model(time.clone(), &[3.0, 3.0], ModelKind::Ply)),
        1.0 / 30.0
    );
    // Q-20: a single-timestamp checkpoint's Δt_f·T_τ/r is 0, so the default too.
    let ckpt = ModelKind::Checkpoint {
        spatial_scale: 1.0,
        frame_dt: 1.0,
    };
    assert_eq!(mt.frame_step_s(&model(time, &[3.0], ckpt)), 1.0 / 30.0);
}

proptest::proptest! {
    /// U-14, proptest variant: E9's τ_m for offsets, rates and clock times drawn in their
    /// domains; hold stays in [τ₀, τ₁] unflagged, extrapolate flags exactly outside it.
    #[test]
    fn u14_prop_tau_at(
        tau0 in -1.0e3f64..1.0e3,
        width in 1.0e-3f64..100.0, // a temporal span; D-9's single timestamp is pinned above
        ms in proptest::bool::ANY,
        offset in -10.0f64..10.0,
        t in 0.0f64..200.0,
    ) {
        let unit = if ms { "ms" } else { "s" };
        let r = if ms { 1000.0 } else { 1.0 };
        let tau1 = tau0 + width;
        let mut mt = ModelTime::from_span(&span(tau0, tau1, unit));
        mt.offset_s = offset;
        let raw = tau0 + r * (t - offset);
        let tol = TOL * raw.abs().max(1.0); // 10⁻¹² relative to the raw τ
        let (held, flag) = mt.tau_at(t);
        proptest::prop_assert!(!flag);
        proptest::prop_assert!((tau0..=tau1).contains(&held));
        proptest::prop_assert!((held - raw.clamp(tau0, tau1)).abs() <= tol);
        mt.end_rule = EndRule::Extrapolate;
        let (free, flag) = mt.tau_at(t);
        proptest::prop_assert!((free - raw).abs() <= tol);
        proptest::prop_assert_eq!(flag, !(tau0..=tau1).contains(&free));
        proptest::prop_assert!((mt.length_s() - (offset + width / r)).abs() <= 1e-9);
    }
}
