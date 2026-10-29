// SPDX-License-Identifier: Apache-2.0

//! U-13 (E8's CPU key, Q-2; `cpu` lane): on 10⁵ random positive depths ascending k̄ is
//! descending depth, strictly except at equal depths; `Depth16` (`sort_key16`) keeps the
//! order up to 16-bit ties; a depth ≤ 0 is never keyed. Exact.

use ftgsxr_core::{sort_key, sort_key16};
use proptest::prelude::*;

/// SplitMix64, a small deterministic generator for the row's 10⁵ draws.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform value in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Checks the row's order properties on `depths`, all finite and positive.
fn check_order(depths: &[f32]) -> Result<(), TestCaseError> {
    let keyed: Vec<(u32, f32)> = depths
        .iter()
        .map(|d| (sort_key(*d).expect("a positive depth is keyed"), *d))
        .collect();
    let mut sorted = keyed.clone();
    sorted.sort_by_key(|(k, _)| *k);
    for pair in sorted.windows(2) {
        let ((k0, d0), (k1, d1)) = (pair[0], pair[1]);
        // Ascending k̄ is descending depth; equal keys exactly at equal depths.
        prop_assert!(d0 >= d1, "k̄ {k0} < {k1} but depth {d0} < {d1}");
        prop_assert_eq!(k0 == k1, d0 == d1);
        // The 16-bit key never reverses the order; it may tie.
        prop_assert!(sort_key16(k0) <= sort_key16(k1));
    }
    Ok(())
}

#[test]
fn u13_key_of_depth_1_by_hand() {
    // bits(1.0) = 0x3F80_0000, k̄ = 0xFFFF_FFFF − 0x3F80_0000 = 0xC07F_FFFF; top 16 bits 0xC07F.
    assert_eq!(sort_key(1.0), Some(0xC07F_FFFF));
    assert_eq!(sort_key16(0xC07F_FFFF), 0xC07F);
    // bits(2.0) = 0x4000_0000: the farther primitive has the smaller key.
    assert_eq!(sort_key(2.0), Some(0xBFFF_FFFF));
}

#[test]
fn u13_ascending_key_is_descending_depth_on_100000_random_depths() {
    let mut rng = SplitMix64(0x5EED_0013);
    let mut depths = Vec::with_capacity(100_000);
    while depths.len() < 100_000 {
        let d = match rng.next() % 4 {
            // Room scale, where E8 meets most depths.
            0 => (0.01 + 20.0 * rng.unit()) as f32,
            // Log-uniform over the near and far planes' range.
            1 => 10f64.powf(-2.0 + 12.0 * rng.unit()) as f32,
            // Any positive finite float32 bit pattern, subnormals included.
            2 => f32::from_bits((rng.next() as u32) % 0x7F80_0000),
            // A repeat of an earlier depth, so that equal depths occur.
            _ => match depths.len() {
                0 => 1.0,
                n => depths[(rng.next() % n as u64) as usize],
            },
        };
        if d > 0.0 && d.is_finite() {
            depths.push(d);
        }
    }
    check_order(&depths).unwrap();
}

#[test]
fn u13_depth_at_or_below_0_or_not_finite_is_never_keyed() {
    for d in [
        0.0,
        -0.0,
        -1e-30,
        -1.0,
        -1e10,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::INFINITY,
    ] {
        assert_eq!(sort_key(d), None, "depth {d}");
    }
    // The smallest positive float32 is keyed.
    assert_eq!(sort_key(f32::from_bits(1)), Some(u32::MAX - 1));
}

proptest! {
    #[test]
    fn u13_proptest_key_order(depths in prop::collection::vec(
        prop_oneof![
            0.01f32..1e3,
            prop::num::f32::POSITIVE.prop_filter("finite, > 0", |d| d.is_finite() && *d > 0.0),
        ],
        1..2000,
    )) {
        check_order(&depths)?;
    }

    #[test]
    fn u13_proptest_non_positive_depth_is_never_keyed(d in prop::num::f32::ANY) {
        prop_assert_eq!(sort_key(d).is_some(), d > 0.0 && d.is_finite());
    }
}
