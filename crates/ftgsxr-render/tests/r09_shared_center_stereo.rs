// SPDX-License-Identifier: Apache-2.0

//! R-9, shared-center stereo (§3.2, D-4): every probe P1–P10 rendered for two eyes 64 mm
//! apart at 1 m (`R9_<P>`'s requests, Q-402, Q-415), per view and shared-center through
//! `HeadlessRenderer::render_stereo` (§2.1). The per-view renders pass E15's tier against
//! openftgs's reference renders of the same eyes, and the PSNR between the two modes, the
//! minimum over items, is ≥ 40 dB (calibrated). The minima are reported on standard error.
//! An added check (#720) runs the mode-to-mode comparison with the conformance tier and
//! `half_records` set, judged between the modes only.

use crate::probe_support;

use ftgsxr_core::{Pose, RenderImage, RenderRequest, read_request};
use ftgsxr_render::{HeadlessRenderer, QualityTier, SortMode};
use ftgsxr_testkit::{fixtures_dir, layer_messages, metrics, reference_render};
use probe_support::{check_items, probe};

/// R-9's calibrated bound on the PSNR between the two modes.
const MIN_MODE_PSNR: f64 = 40.0;

/// Probe `p`'s R-9 request, `probes/R9_<p>.request.json` (Q-415).
fn r9_request(p: &str) -> RenderRequest {
    let path = fixtures_dir().join(format!("probes/R9_{p}.request.json"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    read_request(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// One eye pair of probe `p`: the left item's id, both eyes' per-view and shared-center
/// renders.
struct Pair {
    ids: [String; 2],
    per_view: [RenderImage; 2],
    shared: [RenderImage; 2],
}

/// Every eye pair of `R9_<p>` (items `<id>L` then `<id>R`, Q-415) rendered from probe
/// `p`'s model at the identity pose, at the pair's τ over the request's background, in
/// both modes by one renderer of `tier`.
fn render_pairs(p: &str, tier: QualityTier) -> Vec<Pair> {
    let (model, _) = probe(p);
    let request = r9_request(p);
    let mut renderer = HeadlessRenderer::new(tier).expect("an adapter");
    let id = renderer
        .load(&model, request.sh_degree)
        .expect("the model loads");
    let instances = [(id, Pose::IDENTITY)];
    assert!(request.items.len().is_multiple_of(2) && !request.items.is_empty());
    request
        .items
        .chunks(2)
        .map(|pair| {
            let (l, r) = (&pair[0], &pair[1]);
            assert!(
                l.id.ends_with('L') && r.id.ends_with('R'),
                "{p}: {} {}",
                l.id,
                r.id
            );
            assert_eq!(l.timestamp, r.timestamp, "{p}: one τ per pair");
            let eyes = [l.camera, r.camera];
            let mut render = |sort| {
                renderer
                    .render_stereo(&instances, &eyes, l.timestamp, request.background, sort)
                    .unwrap_or_else(|e| panic!("{p} {}: {e}", l.id))
            };
            Pair {
                ids: [l.id.clone(), r.id.clone()],
                per_view: render(SortMode::PerView),
                shared: render(SortMode::SharedCenter),
            }
        })
        .collect()
}

/// The minimum over the pairs' items of the PSNR between the modes, the per-view render
/// as the reference, reported on standard error with the item that gave it.
fn mode_psnr(label: &str, pairs: &[Pair]) -> f64 {
    let mut min = (f64::INFINITY, String::new());
    for pair in pairs {
        for k in 0..2 {
            let psnr = metrics(&pair.per_view[k], &pair.shared[k]).psnr;
            // A NaN is the minimum, so it fails.
            if psnr.is_nan() || psnr < min.0 {
                min = (psnr, pair.ids[k].clone());
            }
        }
    }
    eprintln!(
        "{label}: per view against shared center, minimum PSNR {:.2} dB ({})",
        min.0, min.1
    );
    min.0
}

/// R-9 on probe `p` in the `conformance` tier.
fn check_r9(p: &str) {
    let pairs = render_pairs(p, QualityTier::conformance());
    let references = reference_render(&format!("R9_{p}"));
    let per_view: Vec<(String, RenderImage, RenderImage)> = pairs
        .iter()
        .flat_map(|pair| (0..2).map(move |k| (pair, k)))
        .map(|(pair, k)| {
            let id = &pair.ids[k];
            let reference = references
                .iter()
                .find(|(rid, _)| rid == id)
                .unwrap_or_else(|| panic!("R9_{p}: no reference for {id}"))
                .1
                .clone();
            (id.clone(), pair.per_view[k].clone(), reference)
        })
        .collect();
    assert_eq!(
        per_view.len(),
        references.len(),
        "R9_{p}: one reference per item"
    );
    let psnr = mode_psnr(&format!("R9_{p}"), &pairs);
    check_items(&format!("R9_{p} per view"), &per_view, false);
    assert!(
        psnr >= MIN_MODE_PSNR,
        "R9_{p}: the PSNR between the modes is {psnr:.2} dB, under {MIN_MODE_PSNR} dB"
    );
}

#[test]
#[ignore = "gpu"]
fn gpu_r09_probe_p1_shared_center_stereo() {
    let _layer = layer_messages(); // R-16, Q-105
    check_r9("P1");
}

#[test]
#[ignore = "gpu"]
fn gpu_r09_probe_p2_shared_center_stereo() {
    let _layer = layer_messages(); // R-16, Q-105
    check_r9("P2");
}

#[test]
#[ignore = "gpu"]
fn gpu_r09_probe_p3_shared_center_stereo() {
    let _layer = layer_messages(); // R-16, Q-105
    check_r9("P3");
}

#[test]
#[ignore = "gpu"]
fn gpu_r09_probe_p4_shared_center_stereo() {
    let _layer = layer_messages(); // R-16, Q-105
    check_r9("P4");
}

#[test]
#[ignore = "gpu"]
fn gpu_r09_probe_p5_shared_center_stereo() {
    let _layer = layer_messages(); // R-16, Q-105
    check_r9("P5");
}

#[test]
#[ignore = "gpu"]
fn gpu_r09_probe_p6_shared_center_stereo() {
    let _layer = layer_messages(); // R-16, Q-105
    check_r9("P6");
}

#[test]
#[ignore = "gpu"]
fn gpu_r09_probe_p7_shared_center_stereo() {
    let _layer = layer_messages(); // R-16, Q-105
    check_r9("P7");
}

#[test]
#[ignore = "gpu"]
fn gpu_r09_probe_p8_shared_center_stereo() {
    let _layer = layer_messages(); // R-16, Q-105
    check_r9("P8");
}

#[test]
#[ignore = "gpu"]
fn gpu_r09_probe_p9_shared_center_stereo() {
    let _layer = layer_messages(); // R-16, Q-105
    check_r9("P9");
}

#[test]
#[ignore = "gpu"]
fn gpu_r09_probe_p10_shared_center_stereo() {
    let _layer = layer_messages(); // R-16, Q-105
    check_r9("P10");
}

// #720: the 48-byte records with half-precision Σ^S (§1.12 `half_records`) in a tier built
// in the test, as Q-347's, judged between the modes only: the per-view render ignores the
// flag (Q-109).
#[test]
#[ignore = "gpu"]
fn gpu_r09_shared_center_stereo_against_per_view_with_half_records() {
    let _layer = layer_messages(); // R-16, Q-105
    let tier = QualityTier {
        half_records: true,
        ..QualityTier::conformance()
    };
    let mut failures = Vec::new();
    for p in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"] {
        let psnr = mode_psnr(
            &format!("R9_{p} half records"),
            &render_pairs(p, tier.clone()),
        );
        if psnr.is_nan() || psnr < MIN_MODE_PSNR {
            failures.push(format!("R9_{p}: {psnr:.2} dB"));
        }
    }
    assert!(
        failures.is_empty(),
        "under {MIN_MODE_PSNR} dB between the modes with half records:\n{}",
        failures.join("\n")
    );
}
