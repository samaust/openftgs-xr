// SPDX-License-Identifier: Apache-2.0

//! U-8's window clauses (`cpu` lane): `motion::temporal_window`, E3's half-width, against
//! the exact test σ_eff < α_skip it stands for, with a proptest variant. The temporal
//! index's clauses (superset, bijection, persistent rows first) are the index ticket's.
//! α_skip enters as the float32 the model carries, promoted exactly to float64 (Q-16).

use ftgsxr_core::motion::temporal_window;
use proptest::prelude::*;

/// openftgs's α_skip as a model carries it: the float32 nearest 1/255 (D-39).
const ALPHA_SKIP: f32 = 1.0 / 255.0;

/// U-8: `None` for 255·opacity ≤ 1, widened or not; and (Q-16) at opacity = α_skip.
#[test]
fn u08_window_is_none_when_255_opacity_at_most_1() {
    let below = f32::from_bits(ALPHA_SKIP.to_bits() - 1);
    for opacity in [1.0e-7, 1.0e-3, 0.0039, below] {
        assert!(255.0 * f64::from(opacity) <= 1.0, "{opacity}");
        for widened in [false, true] {
            assert_eq!(
                temporal_window(opacity, 2.0, ALPHA_SKIP, 10.0, widened),
                None
            );
        }
    }
    for widened in [false, true] {
        assert_eq!(
            temporal_window(ALPHA_SKIP, 2.0, ALPHA_SKIP, 10.0, widened),
            None
        );
        // A producer's own α_skip (D-39).
        assert_eq!(temporal_window(0.1, 2.0, 0.1, 10.0, widened), None);
        assert_eq!(temporal_window(0.05, 2.0, 0.1, 10.0, widened), None);
    }
}

/// The exact test the window stands for, written from E2 and E3 in float64 apart from the
/// code under test: is σ_eff(τ) = exp(−½((τ − time_center)/duration)²)·opacity at least
/// α_skip, i.e. not culled by σ_eff < α_skip (all values here are finite)?
fn reaches_alpha_skip(opacity: f32, duration: f64, time_center: f64, tau: f64) -> bool {
    let x = (tau - time_center) / duration;
    let sigma_eff = (-0.5 * x * x).exp() * f64::from(opacity);
    sigma_eff >= f64::from(ALPHA_SKIP)
}

/// U-8's comparison for one (opacity, duration, τ): the unwidened window's verdict
/// |τ − time_center| ≤ w against the exact test, which may differ only within 10⁻⁹ of
/// the bound; with `None`, the exact test never passes. Returns whether τ is active.
fn check_window(opacity: f32, duration: f64, time_center: f64, tau: f64) -> Result<bool, String> {
    let active = reaches_alpha_skip(opacity, duration, time_center, tau);
    match temporal_window(opacity, duration, ALPHA_SKIP, 10.0, false) {
        None if active => Err(format!(
            "opacity {opacity} duration {duration} τ − center {}: None but active",
            tau - time_center
        )),
        None => Ok(false),
        Some(w) => {
            let distance = (tau - time_center).abs();
            let inside = distance <= w;
            if inside != active && (distance - w).abs() > 1e-9 {
                Err(format!(
                    "opacity {opacity} duration {duration}: |τ − center| {distance}, w {w}, \
                     inside {inside}, active {active}"
                ))
            } else {
                Ok(active)
            }
        }
    }
}

/// SplitMix64, a seeded generator for the 10⁵ draws, so the sweep is reproducible.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// U-8: the window's closed form against the σ_eff < 1/255 test on 10⁵ random
/// (opacity, duration, τ) in float64, agreeing except within 10⁻⁹ of the bound.
/// Opacity uniform in (0, 1), duration log-uniform in [10⁻³, 10²], the center uniform in
/// [−10³, 10³] and τ within ±4 durations of it, so the bound (at most √(2 ln 255) ≈ 3.33
/// durations) is crossed often; plus τ at 2·10⁻⁹ inside and outside the bound.
#[test]
fn u08_window_closed_form_matches_the_opacity_test() {
    let mut rng = SplitMix64(0x5EED_0008);
    let (mut active, mut culled) = (0, 0);
    for _ in 0..100_000 {
        let opacity = (rng.uniform(1.0e-7, 1.0 - 1.0e-7) as f32).clamp(1.0e-7, 1.0 - 1.0e-7);
        let duration = 10f64.powf(rng.uniform(-3.0, 2.0));
        let time_center = rng.uniform(-1.0e3, 1.0e3);
        let tau = time_center + rng.uniform(-4.0, 4.0) * duration;
        match check_window(opacity, duration, time_center, tau) {
            Ok(true) => active += 1,
            Ok(false) => culled += 1,
            Err(e) => panic!("{e}"),
        }
        // Random τ seldom fall near the bound, so each draw also probes 2·10⁻⁹ on either
        // side of it, just outside the band where the two may disagree.
        if let Some(w) = temporal_window(opacity, duration, ALPHA_SKIP, 10.0, false) {
            for offset in [-w - 2e-9, -w + 2e-9, w - 2e-9, w + 2e-9] {
                if let Err(e) = check_window(opacity, duration, time_center, time_center + offset) {
                    panic!("at the bound: {e}");
                }
            }
        }
    }
    // Both verdicts occur often enough for the comparison to mean something.
    assert!(
        active > 10_000 && culled > 10_000,
        "{active} active, {culled} culled"
    );
}

/// The distance from the center at which factor·σ_eff falls to α_skip, found by bisection
/// on the exact test (no closed form), to float64 resolution.
fn edge_by_bisection(factor: f64, opacity: f32, duration: f64) -> f64 {
    let reaches = |d: f64| {
        let x = d / duration;
        (-0.5 * x * x).exp() * f64::from(opacity) * factor >= f64::from(ALPHA_SKIP)
    };
    let (mut lo, mut hi) = (0.0, 10.0 * duration);
    assert!(reaches(lo) && !reaches(hi));
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if reaches(mid) { lo = mid } else { hi = mid }
    }
    lo
}

/// E3, D-10: the window is where σ_eff falls to α_skip; widened, it is where
/// (1 + 10⁻⁴)·σ_eff does, plus 10⁻⁶·T′, with T′ = T_τ = `span`, or 1 when `span` is 0
/// (a single timestamp; Q-16).
#[test]
fn u08_widened_window_adds_openftgs_margins() {
    for (opacity, duration) in [
        (0.5f32, 2.0),
        (0.9, 0.04),
        (0.0045, 1.5),
        (1.0 - 1e-7, 30.0),
    ] {
        let w = temporal_window(opacity, duration, ALPHA_SKIP, 10.0, false).unwrap();
        let edge = edge_by_bisection(1.0, opacity, duration);
        assert!(
            (w - edge).abs() <= 1e-12 * duration.max(1.0),
            "{opacity} {duration}: {w} {edge}"
        );
        let wide_edge = edge_by_bisection(1.0 + 1e-4, opacity, duration);
        assert!(wide_edge > edge);
        for (span, t_prime) in [(10.0, 10.0), (0.25, 0.25), (0.0, 1.0)] {
            let wide = temporal_window(opacity, duration, ALPHA_SKIP, span, true).unwrap();
            let margin = wide - wide_edge;
            assert!(
                (margin - 1e-6 * t_prime).abs() <= 1e-12 * duration.max(1.0),
                "{opacity} {duration} span {span}: margin {margin}"
            );
        }
        // Unwidened, `span` plays no part.
        assert_eq!(
            temporal_window(opacity, duration, ALPHA_SKIP, 0.0, false),
            Some(w)
        );
    }
}

proptest! {
    /// U-8's window clauses on proptest's draws over the sweep's domain: the closed form
    /// agrees with the exact test outside 10⁻⁹ of the bound; the widened window is at
    /// least 10⁻⁶·T′ wider and holds every active τ, with no band of disagreement.
    #[test]
    fn u08_proptest_window(
        opacity in 1.0e-7f32..(1.0 - 1.0e-7),
        log_duration in -3.0..2.0f64,
        time_center in -1.0e3..1.0e3f64,
        x in -4.0..4.0f64,
        span in prop_oneof![Just(0.0), 1.0e-2..1.0e2f64],
    ) {
        let duration = 10f64.powf(log_duration);
        let tau = time_center + x * duration;
        if let Err(e) = check_window(opacity, duration, time_center, tau) {
            return Err(TestCaseError::fail(e));
        }
        let t_prime = if span > 0.0 { span } else { 1.0 };
        match (
            temporal_window(opacity, duration, ALPHA_SKIP, span, false),
            temporal_window(opacity, duration, ALPHA_SKIP, span, true),
        ) {
            (None, None) => {}
            (Some(w), Some(wide)) => {
                prop_assert!(wide >= w + 1e-6 * t_prime, "w {} widened {}", w, wide);
                if reaches_alpha_skip(opacity, duration, time_center, tau) {
                    prop_assert!((tau - time_center).abs() <= wide);
                }
            }
            (w, wide) => prop_assert!(false, "unwidened {:?}, widened {:?}", w, wide),
        }
    }
}
