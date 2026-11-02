// SPDX-License-Identifier: Apache-2.0

//! Dv-3 (§3.4, `dv` lane, D-33, Q-122, Q-T226.1): R-4's criteria (§3.2) on the Quest's
//! GPU. The driver writes R-4's two models as exchange files with `write_exchange` (Q-65)
//! from the testkit's `random_model`, reshaped as `ftgsxr-render`'s `r04_sort.rs` does:
//! `r4-random` at log-uniform random depths and `r4-repeated` over 1000 distinct depths,
//! 10⁶ rows each, every row visible from the runner's camera (the origin, looking along
//! +z, f = 400 px, 800 × 800 px). It pushes them into `dv/`; the device sorts them through
//! `encode_prepare`, checks the order, the stability, `Depth16`'s top 16 bits and the
//! dispatch arguments, and prints the verdict. Prerequisites: `dv_support`'s.

mod dv_support;

use std::time::Duration;

use dv_support::{apk, clear, device, install, launch, push, stop, verdict};
use ftgsxr_core::{Model, write_exchange};
use ftgsxr_testkit::random_model;

/// R-4's size, the runner's `N`.
const N: usize = 1_000_000;
/// The frame's time, every row's time center (the runner's `TAU`).
const TAU: f64 = 0.95;
/// The nearest and farthest depths, inside the default near (0.01) and far (10¹⁰).
const Z_MIN: f64 = 0.02;
const Z_MAX: f64 = 5.0e9;

/// SplitMix64, as `ftgsxr-render`'s `cull_support::Rng`.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64)
    }
}

/// The depth at `u` ∈ [0, 1] of the log-uniform range [`Z_MIN`, `Z_MAX`].
fn log_depth(u: f64) -> f32 {
    (Z_MIN * (Z_MAX / Z_MIN).powf(u)) as f32
}

/// `N` small, opaque, static rows at depths from `depth`, inside the runner's image
/// (|x|, |y| ≤ 0.4 z), their scale growing with the depth (R-4's model).
fn model(seed: u64, mut depth: impl FnMut(&mut Rng) -> f32) -> Model {
    let mut m = random_model(seed, N, 0);
    let mut rng = Rng(seed);
    for i in 0..N {
        let z = depth(&mut rng);
        let x = rng.range(-0.4, 0.4) as f32 * z;
        let y = rng.range(-0.4, 0.4) as f32 * z;
        m.position[i] = [x, y, z];
        m.velocity[i] = [0.0; 3];
        m.scale[i] = [0.0003 * z; 3];
        m.opacity[i] = 0.9;
        m.time_center[i] = TAU;
        m.duration[i] = 100.0;
    }
    m
}

#[test]
#[ignore = "dv"]
fn dv_dv3_the_sort_meets_r4_on_the_quest() {
    let apk = apk();
    let tmp = std::env::temp_dir().join(format!("ftgsxr-dv3-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("a scratch directory");
    let random = tmp.join("r4-random.safetensors");
    let repeated = tmp.join("r4-repeated.safetensors");
    write_exchange(
        &random,
        &model(0x4_01, |rng| log_depth(rng.range(0.0, 1.0))),
    )
    .expect("the random model writes");
    write_exchange(
        &repeated,
        &model(0x4_02, |rng| {
            log_depth((rng.next_u64() % 1000) as f64 / 999.0)
        }),
    )
    .expect("the repeated model writes");

    device();
    install(&apk);
    stop();
    clear();
    push(&random, "r4-random.safetensors");
    push(&repeated, "r4-repeated.safetensors");
    let _ = std::fs::remove_dir_all(&tmp);
    let start = launch("dv3");
    let result = verdict("dv3", start, Duration::from_secs(300));
    stop();
    let (pass, detail, _) = result.expect("a dv3 verdict line in logcat");
    assert!(pass, "dv3 FAIL {detail}");
}
