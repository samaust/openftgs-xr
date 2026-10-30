// SPDX-License-Identifier: Apache-2.0

//! R-10, the tier bounds of D-12 (§3.2, §1.12, D-37): `quest-balanced`'s storage (half
//! precision for everything but position, time and duration, Q-22's layout, and its SH cap
//! of 2) against `conformance` on a degree-2 random model, PSNR ≥ 50 dB; and a degree-2 cap
//! on P5's degree-3 model against openftgs's reference rendered with `sh_degree` 2 (the
//! `R10_deg2` probe, Q-70), E15's tier.

mod probe_support;

use ftgsxr_core::Pose;
use ftgsxr_render::{HeadlessRenderer, QualityTier};
use ftgsxr_testkit::{layer_messages, metrics, probe, random_model};
use probe_support::{check_items, render_request};

/// R-10's bound between the two storages.
const MIN_STORAGE_PSNR: f64 = 50.0;

/// `conformance` with `tier`'s storage (§1.2 load options 1 and 2) and nothing else: the
/// headless renderer draws per view, and the Quest tiers' shared-center draw and E13's
/// occlusion are M-6's and M-8's (R-9, R-12), so R-10 changes only the storage it bounds.
fn conformance_with_storage_of(tier: QualityTier) -> QualityTier {
    QualityTier {
        layout: tier.layout,
        ..QualityTier::conformance()
    }
}

#[test]
#[ignore = "gpu"]
fn gpu_r10_quest_balanced_storage_within_50_db_of_conformance() {
    let _layer = layer_messages(); // R-16, Q-105
    let storage = conformance_with_storage_of(QualityTier::quest_balanced());
    assert!(storage.layout.half.sh && storage.layout.half.rotation);
    assert_eq!(storage.layout.sh_cap, Some(2));
    // A degree-2 random model of R-2's size, seen through R-2's random-model request
    // (three times, the probes' camera, which frames the model's [−1, 1]³).
    let model = random_model(0x10_10, 50_000, 2);
    let (_, request) = probe("R2_random");
    let mut full = HeadlessRenderer::new(QualityTier::conformance()).expect("an adapter");
    let mut half = HeadlessRenderer::new(storage).expect("an adapter");
    let full_id = full.load(&model, None).expect("the model loads");
    let half_id = half.load(&model, None).expect("the model loads");
    let mut failures = Vec::new();
    let mut min_psnr = f64::INFINITY;
    for item in &request.items {
        let render = |r: &mut HeadlessRenderer, id| {
            r.render(
                &[(id, Pose::IDENTITY)],
                &item.camera,
                item.timestamp,
                request.background,
            )
            .unwrap_or_else(|e| panic!("{}: {e}", item.id))
        };
        let a = render(&mut full, full_id);
        let b = render(&mut half, half_id);
        assert!(
            a.alpha.iter().any(|&v| v > 0.1),
            "{}: the model is in view",
            item.id
        );
        let psnr = metrics(&a, &b).psnr;
        min_psnr = min_psnr.min(psnr);
        if psnr.is_nan() || psnr < MIN_STORAGE_PSNR {
            failures.push(format!("{}: PSNR {psnr:.2} dB", item.id));
        }
    }
    eprintln!("R-10 quest-balanced storage: minimum PSNR {min_psnr:.2} dB");
    assert!(failures.is_empty(), "below 50 dB:\n{}", failures.join("\n"));
}

#[test]
#[ignore = "gpu"]
fn gpu_r10_degree_2_cap_against_the_degree_2_reference() {
    let _layer = layer_messages(); // R-16, Q-105
    // Q-70: P5's degree-3 model, capped at 2 by the tier (`GpuLayoutOptions::sh_cap`) and
    // loaded without a request degree, so the tier's cap alone sets L′ = 2.
    let (model, request) = probe("R10_deg2");
    assert_eq!(model.sh_degree, 3);
    assert_eq!(request.sh_degree, Some(2));
    let mut tier = QualityTier::conformance();
    tier.layout.sh_cap = Some(2);
    let mut renderer = HeadlessRenderer::new(tier).expect("an adapter");
    let id = renderer.load(&model, None).expect("the probe loads");
    let items = render_request(&mut renderer, &[(id, Pose::IDENTITY)], "R10_deg2");
    check_items("R10_deg2", &items, false);
}
