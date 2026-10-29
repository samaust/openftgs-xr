// SPDX-License-Identifier: Apache-2.0

//! `random_model` (Q-1), the synthetic models of U-8 and the loader's round trips: valid
//! under §1.2 (`Model::validate`), deterministic per seed, time centers over the span,
//! and a few persistent rows (E3's window covering [τ₀, τ₁]) among transient ones.

use ftgsxr_core::ModelKind;
use ftgsxr_testkit::{exchange_bytes, random_model};
use sha2::{Digest, Sha256};

/// The acceptance grid: seeds 0–9, N = 1, 1,000 and 100,000, every degree 0–3.
#[test]
fn random_model_is_valid_over_the_grid() {
    for seed in 0..10u64 {
        for n in [1usize, 1_000, 100_000] {
            for degree in 0..=3u8 {
                let m = random_model(seed, n, degree);
                assert_eq!(m.len(), n);
                assert_eq!(m.sh_degree, degree);
                if let Err(e) = m.validate() {
                    panic!("seed {seed}, n {n}, degree {degree}: {e:?}");
                }
            }
        }
    }
}

#[test]
fn random_model_is_deterministic_per_seed() {
    assert_eq!(random_model(3, 500, 2), random_model(3, 500, 2));
    assert_ne!(random_model(3, 500, 2), random_model(4, 500, 2));
}

#[test]
fn random_model_rotations_are_canonical_unit_quaternions() {
    let m = random_model(1, 10_000, 0);
    for q in &m.rotation {
        let norm = q.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "{q:?}");
        let first_nonzero = q.iter().find(|&&x| x != 0.0).unwrap();
        assert!(q[0] >= 0.0 && *first_nonzero > 0.0, "{q:?}");
    }
}

/// E3's half-window w = duration·√(2 ln(opacity/α_skip)), 0 when opacity ≤ α_skip.
fn window(m: &ftgsxr_core::Model, i: usize) -> f64 {
    let ratio = f64::from(m.opacity[i]) / f64::from(m.conventions.alpha_skip);
    if ratio <= 1.0 {
        0.0
    } else {
        m.duration[i] * (2.0 * ratio.ln()).sqrt()
    }
}

fn is_persistent(m: &ftgsxr_core::Model, i: usize) -> bool {
    let w = window(m, i);
    m.time_center[i] - w <= m.time.tau0 && m.time_center[i] + w >= m.time.tau1
}

#[test]
fn random_model_has_a_few_persistent_rows_among_transient_ones() {
    for seed in 0..10u64 {
        let m = random_model(seed, 1_000, 1);
        let persistent: Vec<usize> = (0..m.len()).filter(|&i| is_persistent(&m, i)).collect();
        assert!(
            (1..=m.len() / 20).contains(&persistent.len()),
            "seed {seed}: {} persistent rows",
            persistent.len()
        );
        assert_ne!(
            persistent,
            (0..persistent.len()).collect::<Vec<_>>(),
            "not simply the first rows"
        );
    }
    let one = random_model(0, 1, 0);
    assert!(is_persistent(&one, 0), "a one-row model is persistent");
}

#[test]
fn random_model_spreads_time_centers_over_the_span() {
    let m = random_model(5, 1_000, 0);
    assert!(!m.is_single_timestamp());
    let (tau0, tau1) = (m.time.tau0, m.time.tau1);
    assert!(m.time_center.iter().all(|&t| (tau0..=tau1).contains(&t)));
    let span = tau1 - tau0;
    for tenth in 0..10 {
        let lo = tau0 + span * f64::from(tenth) / 10.0;
        let hi = lo + span / 10.0;
        let count = m.time_center.iter().filter(|&&t| t >= lo && t < hi).count();
        assert!(count >= 50, "{count} centers in [{lo}, {hi})");
    }
}

/// `identity` is the SHA-256 of the model's own exchange file (D-40), so a reader of
/// `exchange_bytes(&m)` can return a model equal to `m`.
#[test]
fn random_model_identity_is_that_of_its_exchange_file() {
    let m = random_model(2, 100, 3);
    let digest: [u8; 32] = Sha256::digest(exchange_bytes(&m)).into();
    assert_eq!(m.identity, digest);
    assert_eq!(m.kind, ModelKind::ExchangeFile);
}
