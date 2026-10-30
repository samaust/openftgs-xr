// SPDX-License-Identifier: Apache-2.0

//! U-8's index clauses (`cpu` lane): `TemporalIndex` (E3, D-10). On random models with N
//! up to 10⁵, `range(τ)` at 200 τ (τ₀, τ₁ and outside the span included) is a superset of
//! the active rows, computed here in float64 from E2 as the rows the exact cull keeps
//! (σ_eff ≥ α_skip); the permutation is a bijection; persistent rows come first, then the
//! transient rows by time center; a single-timestamp model is handled. Rows at or below
//! α_skip are placed as Q-28 decides.

use ftgsxr_core::motion::temporal_window;
use ftgsxr_core::{DispatchRange, Model, RenderConventions, TemporalIndex};
use ftgsxr_testkit::random_model;
use proptest::prelude::*;

/// A small seeded generator for the τ draws (SplitMix64).
struct Draws(u64);

impl Draws {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    fn index(&mut self, n: usize) -> usize {
        ((self.unit() * n as f64) as usize).min(n - 1)
    }
}

/// The oracle: whether file row `i` is active at τ, in float64 from E2 alone. The cull
/// keeps a row unless σ_eff < α_skip (E3, D-10), so a row is active when
/// opacity·σt(τ) ≥ α_skip, with σt ≡ 1 for a single timestamp (D-9). This is E3's active
/// set plus the rows with opacity = α_skip at their time center.
fn active(model: &Model, i: usize, tau: f64) -> bool {
    let alpha_skip = f64::from(model.conventions.alpha_skip);
    let sigma_t = if model.time.tau0 == model.time.tau1 {
        1.0
    } else {
        let x = (tau - model.time_center[i]) / model.duration[i];
        (-0.5 * x * x).exp()
    };
    f64::from(model.opacity[i]) * sigma_t >= alpha_skip
}

/// 200 τ for `model`: τ₀, τ₁, eight outside the span and the edges of the widest
/// transient window; then, for random rows, about a
/// third at an edge of the row's unwidened window (its center when it has none), a few at
/// its time center, and the rest uniform over the span widened by T_τ/2 on each side.
fn taus(model: &Model, seed: u64) -> Vec<f64> {
    let (tau0, tau1) = (model.time.tau0, model.time.tau1);
    let t = if tau1 > tau0 { tau1 - tau0 } else { 1.0 };
    let mut out = vec![tau0, tau1];
    for k in [0.01, 0.6, 3.0, 50.0] {
        out.push(tau0 - k * t);
        out.push(tau1 + k * t);
    }
    // The edges of the widest transient window, where w_max itself is the bound.
    let widest = (0..model.len())
        .filter(|&i| !covers_span(model, i))
        .filter_map(|i| unwidened(model, i).map(|w| (w, model.time_center[i])))
        .max_by(|a, b| a.0.total_cmp(&b.0));
    if let Some((w, tc)) = widest {
        out.extend([tc - w, tc + w]);
    }
    let mut draws = Draws(seed);
    while out.len() < 200 {
        let i = draws.index(model.len());
        let tc = model.time_center[i];
        let pick = draws.unit();
        let tau = if pick < 0.33 {
            match unwidened(model, i) {
                Some(w) if draws.unit() < 0.5 => tc - w,
                Some(w) => tc + w,
                None => tc,
            }
        } else if pick < 0.4 {
            tc
        } else {
            tau0 - 0.5 * t + draws.unit() * 2.0 * t
        };
        out.push(tau);
    }
    out
}

/// File row `i`'s E3 half-width w, unwidened.
fn unwidened(model: &Model, i: usize) -> Option<f64> {
    temporal_window(
        model.opacity[i],
        model.duration[i],
        model.conventions.alpha_skip,
        model.time.span(),
        false,
    )
}

/// Whether file row `i`'s E3 window (unwidened) covers \[τ₀, τ₁\]: the persistent rows.
fn covers_span(model: &Model, i: usize) -> bool {
    let tc = model.time_center[i];
    unwidened(model, i).is_some_and(|w| tc - model.time.tau0 <= w && model.time.tau1 - tc <= w)
}

/// Whether file row `i` is persistent: its window covers the span, or, with more than
/// one timestamp, its opacity is exactly α_skip (Q-28); every row with a single timestamp.
fn is_persistent(model: &Model, i: usize) -> bool {
    model.time.is_single()
        || model.opacity[i] == model.conventions.alpha_skip
        || covers_span(model, i)
}

/// U-8's index clauses on `model` at `taus`.
fn check(model: &Model, taus: &[f64]) -> Result<(), String> {
    let n = model.len();
    let (index, order) = TemporalIndex::build(model);
    if index.len() as usize != n || order.len() != n {
        return Err(format!(
            "len {} order {} for N {n}",
            index.len(),
            order.len()
        ));
    }
    // The permutation is a bijection: every file row appears once.
    let mut new_of = vec![u32::MAX; n];
    for (new, &file) in order.iter().enumerate() {
        let slot = new_of
            .get_mut(file as usize)
            .ok_or(format!("order[{new}] = {file} ≥ N {n}"))?;
        if *slot != u32::MAX {
            return Err(format!("file row {file} listed twice"));
        }
        *slot = new as u32;
    }

    let p = index.range(model.time.tau0).persistent.end as usize;
    // Persistent rows first: exactly the rows whose window covers the span, the rows at
    // opacity = α_skip and, with a single timestamp, every row (Q-28); then the
    // transient rows by time center.
    for (new, &file) in order.iter().enumerate() {
        let file = file as usize;
        let persistent = is_persistent(model, file);
        if persistent != (new < p) {
            return Err(format!(
                "file row {file} at new row {new}: persistent {persistent}, P = {p}"
            ));
        }
    }
    for pair in order[p.min(n)..].windows(2) {
        let (a, b) = (pair[0] as usize, pair[1] as usize);
        if model.time_center[a] > model.time_center[b] {
            return Err(format!("transient rows {a}, {b} out of time-center order"));
        }
    }

    let mut visited = vec![false; n];
    for &tau in taus {
        let DispatchRange {
            persistent,
            transient,
        } = index.range(tau);
        if persistent != (0..p as u32) {
            return Err(format!("τ = {tau}: persistent {persistent:?}, P = {p}"));
        }
        if transient.start < persistent.end
            || transient.start > transient.end
            || transient.end as usize > n
        {
            return Err(format!(
                "τ = {tau}: transient {transient:?}, P = {p}, N = {n}"
            ));
        }
        visited.fill(false);
        for r in persistent.chain(transient) {
            visited[r as usize] = true;
        }
        for (file, &new) in new_of.iter().enumerate() {
            if active(model, file, tau) && !visited[new as usize] {
                return Err(format!(
                    "τ = {tau}: active file row {file} (new row {new}) not visited"
                ));
            }
        }
    }
    Ok(())
}

/// `random_model`'s rows shifted by `offset` in raw time, with the span.
fn shifted(mut model: Model, offset: f64) -> Model {
    model.time.tau0 += offset;
    model.time.tau1 += offset;
    for tc in &mut model.time_center {
        *tc += offset;
    }
    model
}

/// `random_model(seed, n, 0)` turned into a single-timestamp model (§1.2: every time
/// center τ₀, velocities 0, durations 1).
fn single(seed: u64, n: usize) -> Model {
    let mut model = random_model(seed, n, 0);
    model.time.tau1 = model.time.tau0;
    model.time_center.fill(model.time.tau0);
    model.velocity.fill([0.0; 3]);
    model.duration.fill(1.0);
    model.validate().expect("a valid single-timestamp model");
    model
}

/// The new row of file row `i`.
fn new_row(order: &[u32], i: usize) -> u32 {
    order.iter().position(|&f| f as usize == i).unwrap() as u32
}

/// The float32 just below `x`.
fn below(x: f32) -> f32 {
    f32::from_bits(x.to_bits() - 1)
}

/// U-8: the superset, bijection and order clauses on random models with N up to 10⁵.
#[test]
fn u08_index_superset_on_random_models() {
    for (seed, n) in [(1, 1), (2, 2), (3, 7), (4, 100), (5, 5_000), (6, 100_000)] {
        let model = random_model(seed, n, 0);
        check(&model, &taus(&model, seed)).unwrap_or_else(|e| panic!("seed {seed}, N {n}: {e}"));
    }
}

/// U-8 on raw timestamps far from zero (an epoch in seconds): the margins still hold.
#[test]
fn u08_index_superset_far_from_zero() {
    let model = shifted(random_model(11, 20_000, 0), 1.7e9);
    check(&model, &taus(&model, 11)).unwrap();
}

/// Q-28: in a random model, the rows with opacity exactly α_skip, which the cull keeps
/// at their time center (σ_eff = α_skip is not < α_skip), are persistent, so `range`
/// visits them there; a row just below α_skip, never active, is transient.
#[test]
fn u08_index_row_at_the_bound_is_visited_at_its_center() {
    let mut model = random_model(21, 2_000, 0);
    let alpha_skip = model.conventions.alpha_skip;
    for (i, opacity) in [(10, alpha_skip), (11, below(alpha_skip)), (12, alpha_skip)] {
        model.opacity[i] = opacity;
    }
    model.validate().unwrap();
    let (index, order) = TemporalIndex::build(&model);
    let p = index.range(model.time.tau0).persistent.end;
    for i in [10usize, 12] {
        let tau = model.time_center[i];
        assert!(active(&model, i, tau));
        let new = new_row(&order, i);
        let range = index.range(tau);
        assert!(
            range.persistent.contains(&new),
            "row {i} (new {new}) at its center τ = {tau}: {range:?}"
        );
    }
    assert!(new_row(&order, 11) >= p, "row 11 below α_skip is transient");
    let mut taus = taus(&model, 21);
    taus.extend([10, 11, 12].map(|i| model.time_center[i]));
    check(&model, &taus).unwrap();
}

/// `random_model(seed, n, 0)` with every row persistent (opacity 0.9, duration 3·T_τ, so
/// E3's window covers the span) but row `k`, which takes `opacity`.
fn persistent_but(seed: u64, n: usize, k: usize, opacity: f32) -> Model {
    let mut model = random_model(seed, n, 0);
    let span = model.time.span();
    model.opacity.fill(0.9);
    model.duration.fill(3.0 * span);
    model.opacity[k] = opacity;
    model.duration[k] = 0.05 * span;
    model.validate().unwrap();
    for i in (0..n).filter(|&i| i != k) {
        assert!(covers_span(&model, i), "row {i} should cover the span");
    }
    model
}

/// Q-28: a row at opacity = α_skip, the only row whose window does not cover the span
/// (so w_max = 0), is persistent, so `range` visits it at its center and at
/// c ± 2⁻¹³·duration, inside the band where σt rounds to 1 in float32 and the cull keeps
/// it, though w_max reaches no further than its center.
#[test]
fn u08_index_row_at_the_bound_is_persistent() {
    let k = 17;
    let model = persistent_but(51, 300, k, RenderConventions::default().alpha_skip);
    let (index, order) = TemporalIndex::build(&model);
    let new = new_row(&order, k);
    assert_eq!(
        new, k as u32,
        "in file order with the other persistent rows"
    );
    let (c, d) = (model.time_center[k], model.duration[k]);
    let step = d * 2f64.powi(-13);
    assert_eq!(1.0f32, (-0.5 * (step / d) * (step / d)).exp() as f32);
    for tau in [c, c - step, c + step] {
        let range = index.range(tau);
        assert!(
            range.persistent.contains(&new),
            "τ = {tau}: new row {new}, {range:?}"
        );
    }
    assert_eq!(index.range(c).persistent, 0..300);
    check(&model, &taus(&model, 51)).unwrap();
}

/// Q-28: a row just below α_skip, the only row whose window does not cover the span, is
/// transient and adds nothing to w_max: w_max stays 0, so `range` visits it at its center
/// alone, not at c ± 2⁻¹³·duration. It passes the cull at no τ (opacity·σt ≤ opacity).
#[test]
fn u08_index_row_below_the_bound_is_transient() {
    let k = 17;
    let model = persistent_but(52, 300, k, below(RenderConventions::default().alpha_skip));
    let (index, order) = TemporalIndex::build(&model);
    let new = new_row(&order, k);
    assert_eq!(new, 299, "the one transient row comes last");
    let (c, d) = (model.time_center[k], model.duration[k]);
    let step = d * 2f64.powi(-13);
    let at_center = index.range(c);
    assert_eq!(at_center.persistent, 0..299);
    assert_eq!(at_center.transient, 299..300);
    for tau in [c - step, c + step] {
        let range = index.range(tau);
        assert!(range.transient.is_empty(), "τ = {tau}: {range:?}");
    }
    check(&model, &taus(&model, 52)).unwrap();
}

/// U-8 for a single-timestamp model: σt ≡ 1 (D-9), so a row's activity does not depend
/// on τ; every row is persistent and the permutation is the identity (Q-28), and the
/// range at any τ, inside or outside the span, covers every row, the row at the bound
/// included.
#[test]
fn u08_index_single_timestamp() {
    for n in [1, 3, 1_000] {
        let mut model = single(31, n);
        let alpha_skip = model.conventions.alpha_skip;
        model.opacity[0] = alpha_skip;
        if n > 1 {
            model.opacity[1] = below(alpha_skip);
        }
        model.validate().unwrap();
        let (index, order) = TemporalIndex::build(&model);
        assert_eq!(order, (0..n as u32).collect::<Vec<_>>());
        let tau0 = model.time.tau0;
        for tau in [tau0, tau0 - 1.0, tau0 + 0.5, tau0 + 1.0e6] {
            let range = index.range(tau);
            assert_eq!(range.persistent, 0..n as u32, "τ = {tau}");
            assert!(range.transient.is_empty(), "τ = {tau}: {range:?}");
        }
        check(&model, &taus(&model, 31)).unwrap();
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    /// U-8's proptest variant: random models with N up to 10⁵, some far from zero in raw
    /// time, some single-timestamp, each at 200 τ.
    #[test]
    fn u08_proptest_index(
        seed in any::<u64>(),
        n in prop_oneof![1usize..64, 64usize..=100_000],
        offset in prop_oneof![Just(0.0), -1.0e3..1.0e3f64, Just(1.7e9)],
        single_timestamp in prop::bool::weighted(0.2),
    ) {
        let model = if single_timestamp {
            single(seed, n)
        } else {
            shifted(random_model(seed, n, 0), offset)
        };
        if let Err(e) = check(&model, &taus(&model, seed ^ 0x5EED)) {
            return Err(TestCaseError::fail(e));
        }
    }
}

/// E3's index margins (D-10): the transient run reaches past the widest unwidened
/// window by the time margin, so the widest row is still visited 0.5·10⁻⁶·T′ beyond its
/// E3 edge, where the float64 oracle alone cannot tell the margins from none.
#[test]
fn u08_index_widened_by_the_margins() {
    let model = random_model(41, 10_000, 0);
    let (index, order) = TemporalIndex::build(&model);
    let (w, i) = (0..model.len())
        .filter(|&i| !covers_span(&model, i))
        .filter_map(|i| unwidened(&model, i).map(|w| (w, i)))
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap();
    let new = new_row(&order, i);
    let t_prime = model.time.span();
    for tau in [
        model.time_center[i] + w + 0.5e-6 * t_prime,
        model.time_center[i] - w - 0.5e-6 * t_prime,
    ] {
        let range = index.range(tau);
        assert!(range.transient.contains(&new), "τ = {tau}: {range:?}");
    }
}
