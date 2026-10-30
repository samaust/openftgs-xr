// SPDX-License-Identifier: Apache-2.0

//! Shared helpers of the probe tests (R-1, R-2): a committed probe rendered item by item
//! through `HeadlessRenderer` in the `conformance` tier and compared with its reference
//! render under E15's tier.

#![allow(dead_code, reason = "each test file uses a part of the helpers")]

use ftgsxr_core::{Model, Pose, RenderImage, RenderRequest};
use ftgsxr_render::{HeadlessRenderer, QualityTier};
use ftgsxr_testkit::{Metrics, metrics, probe, reference_render};

/// E15's tier: PSNR ≥ 45 dB, s ≤ 0.5 %, s_α ≤ 0.5 %.
pub const MIN_PSNR: f64 = 45.0;
pub const MAX_SHARE: f64 = 0.005;

/// Every item of probe `name` rendered at its timestamp from its camera, with the
/// request's background and SH degree, at the identity pose (§1.7 `render`), paired with
/// its reference render by id.
pub fn render_probe(name: &str) -> Vec<(String, RenderImage, RenderImage)> {
    let (model, request) = probe(name);
    render_items(name, &model, &request, &reference_render(name))
}

/// Every item of `request` rendered from `model` through `HeadlessRenderer` in the
/// `conformance` tier, at the item's timestamp from its camera, with the request's
/// background and SH degree, at the identity pose (§1.7 `render`), paired with its
/// reference in `references` by id; `name` names the model in messages. The renderer and
/// its device are dropped before returning, so in a debug build any GPU validation message
/// fails the caller (§1.13, R-16, Q-74).
pub fn render_items(
    name: &str,
    model: &Model,
    request: &RenderRequest,
    references: &[(String, RenderImage)],
) -> Vec<(String, RenderImage, RenderImage)> {
    let mut renderer = HeadlessRenderer::new(QualityTier::conformance()).expect("an adapter");
    let id = renderer
        .load(model, request.sh_degree)
        .expect("the model loads");
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
                .render(
                    &[(id, Pose::IDENTITY)],
                    &item.camera,
                    item.timestamp,
                    request.background,
                )
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
    check_items(name, &render_probe(name), false);
}

/// E15's tier on every rendered item of `name`: the minimum PSNR goes to standard error,
/// every failing item into the panic message; with `report_each`, every item's PSNR, s
/// and s_α go to standard error too.
pub fn check_items(name: &str, items: &[(String, RenderImage, RenderImage)], report_each: bool) {
    let mut failures = Vec::new();
    let mut min_psnr = f64::INFINITY;
    for (id, ours, reference) in items {
        let m: Metrics = metrics(reference, ours);
        min_psnr = min_psnr.min(m.psnr);
        if report_each {
            eprintln!(
                "{name} {id}: PSNR {:.2} dB, s {:.4} %, s_α {:.4} %",
                m.psnr,
                100.0 * m.share,
                100.0 * m.alpha
            );
        }
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
