// SPDX-License-Identifier: Apache-2.0

//! R-4, the sort (§3.2; §1.4 stages 3 and 4, E8, D-33): 10⁶ random keys with payloads
//! sorted on the GPU against a stable CPU sort; stability on repeated keys; `Depth16`
//! agrees on the top 16 bits; the sizing pass's dispatch arguments equal ⌈count / block⌉.
//! The sort is reached through `encode_prepare` and `FrameBuffers::read_back` (Q-69): each
//! slot's key is recomputed from its record's depth with `sort_key` (`sort_key16` for
//! `Depth16`, Q-2), and `sorted` must equal the slots stably sorted by that key.

mod cull_support;

use cull_support::{Rng, draw, look_at, passes, prepare, sort_pose, upload, view};
use ftgsxr_core::{Camera, Model, Pose, Quat, UVec2, Vec3, sort_key, sort_key16};
use ftgsxr_render::{FrameBuffers, FrameContents, QualityTier, RenderMode, SortKey};
use ftgsxr_testkit::{gpu_context, layer_messages, random_model};

/// R-4's size.
const N: usize = 1_000_000;
/// The keys one workgroup of the sort's count pass takes (Q-85).
const SORT_BLOCK: u32 = 1024;
/// The frame's time: every row's time center, so every row is fully opaque in time.
const TAU: f64 = 0.95;

/// A camera at the origin looking along +z, 800 × 800 px, f = 400 px.
fn camera() -> Camera {
    look_at(
        Vec3::ZERO,
        Vec3::Z,
        400.0,
        UVec2::new(800, 800),
        [400.0, 400.0],
    )
}

/// `N` small, opaque, static rows in front of `camera()`, each at a depth from `depth`
/// and inside the image (|x|, |y| ≤ 0.4 z), so every row is visible.
fn model(seed: u64, mut depth: impl FnMut(&mut Rng) -> f32) -> Model {
    let mut m = random_model(seed, N, 0);
    let mut rng = Rng(seed);
    for i in 0..N {
        let z = depth(&mut rng);
        let x = rng.range(-0.4, 0.4) as f32 * z;
        let y = rng.range(-0.4, 0.4) as f32 * z;
        m.position[i] = [x, y, z];
        m.velocity[i] = [0.0; 3];
        m.scale[i] = [0.003; 3];
        m.opacity[i] = 0.9;
        m.time_center[i] = TAU;
        m.duration[i] = 100.0;
    }
    m
}

fn identity_pose() -> Pose {
    Pose {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: 1.0,
    }
}

/// One frame of `model` under `tier`, every row visible.
fn frame(model: &Model, tier: &QualityTier) -> FrameContents {
    let dev = gpu_context();
    let gpu_model = upload(&dev, model, tier);
    let passes = passes(&dev, tier);
    let frame = FrameBuffers::new(&dev, tier, N as u32);
    let cam = camera();
    let draws = [draw(&gpu_model, identity_pose(), TAU, RenderMode::Splats)];
    let got = prepare(
        &dev,
        &passes,
        &frame,
        &draws,
        &sort_pose(&cam),
        &[view(cam)],
    );
    assert!(dev.take_validation_messages().is_empty());
    got
}

/// Each slot's key from its record's depth (E8, Q-2), `digits` of it.
fn keys(got: &FrameContents, digits: impl Fn(u32) -> u32) -> Vec<u32> {
    got.records
        .iter()
        .map(|r| digits(sort_key(r.depth).expect("a visible record has z_c > 0")))
        .collect()
}

/// The slots 0, 1, … stably sorted by `keys` (Rust's `sort_by_key` is stable).
fn stable_order(keys: &[u32]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..keys.len() as u32).collect();
    order.sort_by_key(|&slot| keys[slot as usize]);
    order
}

fn assert_sorted(got: &FrameContents, keys: &[u32]) {
    assert_eq!(got.visible_splats as usize, N, "every row is visible");
    assert!(!got.overflow);
    // Q-85: the blocks of 1024 keys.
    assert_eq!(
        got.sort_dispatch,
        [got.visible_splats.div_ceil(SORT_BLOCK), 1, 1]
    );
    assert_eq!(got.sorted.len(), N);
    let expected = stable_order(keys);
    let first = got.sorted.iter().zip(&expected).position(|(a, b)| a != b);
    assert_eq!(
        first, None,
        "the GPU order departs from the stable CPU sort"
    );
}

#[test]
#[ignore = "gpu"]
fn gpu_r04_sort_orders_a_million_random_keys_like_a_stable_cpu_sort() {
    let _layer = layer_messages(); // R-16, Q-105
    let m = model(0x4_01, |rng| rng.range(2.0, 20.0) as f32);
    let got = frame(&m, &QualityTier::conformance());
    let k = keys(&got, |k| k);
    assert_sorted(&got, &k);
}

#[test]
#[ignore = "gpu"]
fn gpu_r04_sort_is_stable_on_repeated_keys() {
    let _layer = layer_messages(); // R-16, Q-105
    // 1000 distinct depths over 10⁶ rows: every key repeats about 1000 times.
    let m = model(0x4_02, |rng| 2.0 + (rng.next_u64() % 1000) as f32 * 0.018);
    let got = frame(&m, &QualityTier::conformance());
    let k = keys(&got, |k| k);
    let mut distinct = k.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(distinct.len() <= 1000, "{} distinct keys", distinct.len());
    assert_sorted(&got, &k);
}

#[test]
#[ignore = "gpu"]
fn gpu_r04_depth16_agrees_on_the_top_16_bits() {
    let _layer = layer_messages(); // R-16, Q-105
    let tier = QualityTier {
        sort_key: SortKey::Depth16,
        ..QualityTier::conformance()
    };
    let m = model(0x4_03, |rng| rng.range(2.0, 20.0) as f32);
    let got = frame(&m, &tier);
    let k = keys(&got, |k| u32::from(sort_key16(k)));
    assert_sorted(&got, &k);
}
