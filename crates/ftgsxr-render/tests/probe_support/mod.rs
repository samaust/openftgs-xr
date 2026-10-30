// SPDX-License-Identifier: Apache-2.0

//! Shared helpers of the probe tests (R-1, R-2, R-7, R-8, R-10): a committed probe's
//! request rendered item by item through `HeadlessRenderer` in the `conformance` tier and
//! compared with its reference render under E15's tier.

#![allow(dead_code, reason = "each test file uses a part of the helpers")]

use ftgsxr_core::{Pose, RenderImage};
use ftgsxr_render::{HeadlessRenderer, ModelId, QualityTier};
pub use ftgsxr_testkit::probe;
use ftgsxr_testkit::{Metrics, metrics, reference_render};

/// E15's tier: PSNR ≥ 45 dB, s ≤ 0.5 %, s_α ≤ 0.5 %.
pub const MIN_PSNR: f64 = 45.0;
pub const MAX_SHARE: f64 = 0.005;

/// Every item of probe `name` rendered at its timestamp from its camera, with the
/// request's background and SH degree, at the identity pose (§1.7 `render`), paired with
/// its reference render by id.
pub fn render_probe(name: &str) -> Vec<(String, RenderImage, RenderImage)> {
    let (model, request) = probe(name);
    let mut renderer = HeadlessRenderer::new(QualityTier::conformance()).expect("an adapter");
    let id = renderer
        .load(&model, request.sh_degree)
        .expect("the probe loads");
    render_request(&mut renderer, &[(id, Pose::IDENTITY)], name)
}

/// Every item of probe `name`'s request rendered by `renderer` with `instances` (each a
/// loaded model at its pose, E5) at the item's timestamp from its camera over the
/// request's background, paired with the probe's reference render by id (R-7, R-8: the
/// probe is the pre-posed merged model, Q-12).
pub fn render_request(
    renderer: &mut HeadlessRenderer,
    instances: &[(ModelId, Pose)],
    name: &str,
) -> Vec<(String, RenderImage, RenderImage)> {
    let (_, request) = probe(name);
    let references = reference_render(name);
    assert_eq!(
        references.len(),
        request.items.len(),
        "{name}: one reference per item"
    );
    request
        .items
        .iter()
        .map(|item| {
            let ours = renderer
                .render(instances, &item.camera, item.timestamp, request.background)
                .unwrap_or_else(|e| panic!("{name} {}: {e}", item.id));
            let reference = references
                .iter()
                .find(|(rid, _)| *rid == item.id)
                .unwrap_or_else(|| panic!("{name}: no reference for {}", item.id))
                .1
                .clone();
            (item.id.clone(), ours, reference)
        })
        .collect()
}

/// R-1 and R-2's check: every item of probe `name` passes the tier; the probe's minimum
/// PSNR is reported on standard error, with every failing item in the panic message.
pub fn check_probe(name: &str) {
    check_items(name, &render_probe(name));
}

/// E15's tier on every rendered item against its reference; the minimum PSNR on standard
/// error, every failing item in the panic message.
pub fn check_items(name: &str, items: &[(String, RenderImage, RenderImage)]) {
    let mut failures = Vec::new();
    let mut min_psnr = f64::INFINITY;
    for (id, ours, reference) in items {
        let m: Metrics = metrics(reference, ours);
        min_psnr = min_psnr.min(m.psnr);
        if !(m.psnr >= MIN_PSNR && m.share <= MAX_SHARE && m.alpha <= MAX_SHARE) {
            failures.push(format!(
                "{id}: PSNR {:.2} dB, s {:.4} %, s_α {:.4} %",
                m.psnr,
                100.0 * m.share,
                100.0 * m.alpha
            ));
        }
    }
    eprintln!(
        "{name}: {} items, minimum PSNR {min_psnr:.2} dB",
        items.len()
    );
    assert!(
        failures.is_empty(),
        "{name}: {} of {} items fail the tier:\n{}",
        failures.len(),
        items.len(),
        failures.join("\n")
    );
}
