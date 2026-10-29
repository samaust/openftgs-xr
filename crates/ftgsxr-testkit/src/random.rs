// SPDX-License-Identifier: Apache-2.0

//! Synthetic random models (§2.7 Organisation, Q-1).

use ftgsxr_core::{Model, ModelKind, RenderConventions, TimeSpan};
use sha2::{Digest, Sha256};

use crate::exchange_bytes;

/// τ₀ and τ₁ of every random model: the probes' span (Q-10).
const TAU0: f64 = 0.0;
const TAU1: f64 = 1.9;
/// How many rows are persistent, at most.
const PERSISTENT: usize = 4;

/// SplitMix64, a small seeded generator (§2.3 has no random-number crate).
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
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// A uniformly random unit quaternion (w, x, y, z), normalized in float64, with the
    /// canonical sign of §1.2: w ≥ 0, and where w = 0 the first nonzero component positive.
    fn rotation(&mut self) -> [f32; 4] {
        loop {
            let q = [0; 4].map(|_| self.range(-1.0, 1.0));
            let norm2: f64 = q.iter().map(|x| x * x).sum();
            if !(1e-4..=1.0).contains(&norm2) {
                continue;
            }
            let norm = norm2.sqrt();
            let sign = match q.iter().find(|&&x| x != 0.0) {
                Some(&x) if q[0] < 0.0 || (q[0] == 0.0 && x < 0.0) => -1.0,
                _ => 1.0,
            };
            let out = q.map(|x| (sign * x / norm) as f32);
            if out[0] == 0.0 && out.iter().find(|&&x| x != 0.0).is_some_and(|&x| x < 0.0) {
                continue;
            }
            return out;
        }
    }
}

/// A random exchange model with `n` rows at SH degree `degree`, valid under §1.2
/// (`Model::validate`) and the same for the same arguments. τ₀ = 0, τ₁ = 1.9 (unit `s`),
/// openftgs's `render_conventions`, producer `ftgsxr-testkit`. Each row has a position
/// uniform in \[−1, 1\]³, a velocity uniform in \[−0.5, 0.5\]³ per unit of τ, scales
/// log-uniform in \[0.005, 0.1\], a uniformly random canonical unit rotation, a DC
/// coefficient uniform in \[−1, 1\] and higher coefficients in \[−0.25, 0.25\], and a time
/// center uniform over \[τ₀, τ₁\]. Up to four rows, at seeded positions, are persistent
/// (E3: opacity in \[0.6, 0.99\], duration 2–4 T_τ, so their window covers the span); the
/// others are transient (opacity in \[0.002, 0.99\], so a few never reach α_skip; duration
/// 0.01–0.1 T_τ). `identity` is the SHA-256 of `exchange_bytes` of the model (D-40), so a
/// reader of that file can return a model equal to this one; `kind` is `ExchangeFile`.
///
/// # Panics
///
/// When `n` is 0 or `degree` > 3.
pub fn random_model(seed: u64, n: usize, degree: u8) -> Model {
    assert!(n >= 1, "random_model: N ≥ 1 (§1.2)");
    assert!(degree <= 3, "random_model: degree {degree} > 3");
    let mut rng = SplitMix64(seed);
    let span = TAU1 - TAU0;

    let mut persistent = vec![false; n];
    let mut chosen = 0;
    while chosen < PERSISTENT.min(n) {
        let i = (rng.next_u64() % n as u64) as usize;
        if !persistent[i] {
            persistent[i] = true;
            chosen += 1;
        }
    }

    let k = (usize::from(degree) + 1).pow(2);
    let mut model = Model {
        position: Vec::with_capacity(n),
        time_center: Vec::with_capacity(n),
        duration: Vec::with_capacity(n),
        velocity: Vec::with_capacity(n),
        scale: Vec::with_capacity(n),
        rotation: Vec::with_capacity(n),
        opacity: Vec::with_capacity(n),
        sh: Vec::with_capacity(n * k * 3),
        sh_degree: degree,
        time: TimeSpan {
            tau0: TAU0,
            tau1: TAU1,
            time_unit: "s".to_owned(),
        },
        conventions: RenderConventions::default(),
        producer: "ftgsxr-testkit".to_owned(),
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    };
    let (ln_lo, ln_hi) = (0.005f64.ln(), 0.1f64.ln());
    for &keep in &persistent {
        model
            .position
            .push([0; 3].map(|_| rng.range(-1.0, 1.0) as f32));
        model
            .velocity
            .push([0; 3].map(|_| rng.range(-0.5, 0.5) as f32));
        model
            .scale
            .push([0; 3].map(|_| rng.range(ln_lo, ln_hi).exp() as f32));
        model.rotation.push(rng.rotation());
        model.time_center.push(rng.range(TAU0, TAU1));
        let (opacity, duration) = if keep {
            (rng.range(0.6, 0.99), rng.range(2.0, 4.0) * span)
        } else {
            (rng.range(0.002, 0.99), rng.range(0.01, 0.1) * span)
        };
        model.opacity.push(opacity as f32);
        model.duration.push(duration);
        for c in 0..k * 3 {
            let bound = if c < 3 { 1.0 } else { 0.25 };
            model.sh.push(rng.range(-bound, bound) as f32);
        }
    }
    model.identity = Sha256::digest(exchange_bytes(&model)).into();
    model
}
